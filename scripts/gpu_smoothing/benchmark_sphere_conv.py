import sys, os, h5py, bz2, pylab as plt, importlib, itertools, numpy as np, time, psutil
from tqdm.auto import tqdm, trange
from collections import OrderedDict
import healpy as hp
from sklearn.neighbors import BallTree
import tensorflow as tf

# import pudb
# pudb.set_trace()

process = psutil.Process()
tf.config.run_functions_eagerly(False)
tf.config.set_soft_device_placement(False)

print("testing real space sphere convolution speed")
print("using batch-last order: n_channels, n_pix, batch_size")


# get healpix map

nside = 512
npix = hp.nside2npix(nside)
m = np.ones(npix)
cl = hp.alm2cl(hp.map2alm(m))
cl = 1 - np.arange(len(cl)) * 0.002
m = hp.synfast(cl, nside=nside, pixwin=True)
m = m_full = m - np.mean(m)
m = np.float32(m)


print(f"nside={nside} npix={npix}")
print(f"got heaplix map with sum={np.sum(m)} mean={np.mean(m)}")

# sigma_arcmin = 65
sigma_arcmin = 15
n_sigma_support = 3
sigma_rad = sigma_arcmin / 60 / 180 * np.pi

m_smooth_healpy = hp.sphtfunc.smoothing(m, sigma=sigma_rad)
print(f"smoothed map healpy sum={np.sum(m_smooth_healpy)} mean={np.mean(m_smooth_healpy)}")

# get coordinates of pixels
lon, lat = hp.pix2ang(nside, ipix=np.arange(npix), lonlat=True)
theta = np.vstack([np.radians(lat), np.radians(lon)]).T

print("using ~8000 deg**2 of sphere area")
ind_base = hp.ang2pix(nside=1, theta=lon, phi=lat, lonlat=True)
select = ind_base < 2
theta = theta[select, :]
m = m[select]
m_smooth_healpy = m_smooth_healpy[select]
npix_full = npix
npix = len(m)
print(f"new npix={npix} new area={npix/npix_full*44000}")

# get tree
# The first coordinate of each point is assumed to be the latitude, the second is the longitude, given in radians.
tree = BallTree(theta, metric="haversine")

load_dists = True

if load_dists:
    dist_k = np.load(f"dist_k_nside{nside}.npy")
    inds_k = np.load(f"inds_k_nside{nside}.npy")
    max_neighbours = inds_k.shape[1]

    print(f"loaded dist_k={dist_k.shape} inds_k={inds_k.shape}")

else:
    print(f"creating tree for {len(theta)} pixels and radius {n_sigma_support*sigma_arcmin} arcmin")

    inds_r, dist_r = tree.query_radius(theta, r=sigma_rad * n_sigma_support, return_distance=True, sort_results=True)
    n_neighbours = [len(i) for i in inds_r]
    max_neighbours = np.max(n_neighbours)
    print(f"max_neighbours={max_neighbours}")
    theta_split = np.array_split(theta, 100)
    list_dist_k, list_inds_k = [], []
    for theta_ in tqdm(theta_split):
        dist_k, inds_k = tree.query(theta_, k=max_neighbours, return_distance=True, sort_results=True)
        list_dist_k.append(dist_k)
        list_inds_k.append(inds_k)
    dist_k = np.concatenate(list_dist_k, axis=0)
    inds_k = np.concatenate(list_inds_k, axis=0)
    print(inds_k.shape, dist_k.shape)

    np.save(f"dist_k_nside{nside}.npy", dist_k)
    np.save(f"inds_k_nside{nside}.npy", inds_k)

    print(f"stored dist_k={dist_k.shape} inds_k={inds_k.shape}")


# kernel_func = lambda r: 1./np.sqrt(2.*np.pi*sigma_rad**2) * np.exp(-0.5/sigma_rad**2 * r**2)
kernel_func = lambda r: np.exp(-0.5 / sigma_rad**2 * r**2)

kernel = kernel_func(dist_k)
kernel = np.float32(kernel)
inds_k = np.int64(inds_k)

print(f"kernel size {kernel.nbytes/1e9:4.2f} GB dtype {kernel.dtype}")
print(f"index size {inds_k.nbytes/1e9:4.2f} GB dtype {inds_k.dtype} max_ind={np.max(inds_k)}")
print(f"single map tensor {npix * max_neighbours * 4/1e9:4.2f} GB")


n_channels = 8
batch_size = 2


print("===========================> loop sparse-dense convolution")

n_trials = 10

kernel_channels = []
map_channels_batch = []


with tf.device("cpu"):
    for i in range(n_channels):
        inds_r = tf.constant(np.arange(npix), dtype=tf.int64)
        inds_r = tf.expand_dims(inds_r, axis=-1)
        inds_r = tf.tile(inds_r, [1, max_neighbours])
        inds_c = tf.constant(inds_k, dtype=tf.int64)
        ind_coo = tf.concat([tf.reshape(inds_r, [-1, 1]), tf.reshape(inds_c, [-1, 1])], axis=-1)
        val_kernel = tf.reshape(kernel, [-1])
        m_batch = tf.concat([tf.expand_dims(m, axis=-1)] * batch_size, axis=-1)
        map_channels_batch.append(m_batch)

        sparse_kernel = tf.sparse.SparseTensor(indices=ind_coo, values=val_kernel, dense_shape=[npix, npix])
        sparse_kernel = tf.sparse.reorder(sparse_kernel)
        kernel_channels.append(sparse_kernel)

        print(f"========> channel {i}")
        print(
            f"ind_coo.shape={ind_coo.shape} ind_coo.size={np.array(ind_coo).nbytes/1e9:2.4f} GB val_kernel.shape={val_kernel.shape} val_kernel.size={np.array(val_kernel).nbytes/1e9:2.4f} GB"
        )
        print("memory used {:2.4f} GB".format(process.memory_info().rss / 1e9))  # in bytes


print(
    f"created {n_channels} sparse kernels with shape {kernel_channels[0].shape} and batched maps with size {map_channels_batch[0].shape}"
)

# maximum size allowed by tf.sparse.sparse_dense_matmul
op_size = len(kernel_channels[0].indices) * map_channels_batch[0].shape[1]
print(op_size < 2**31, op_size)

time_start = time.time()
with tf.device("gpu"):
    for j in range(n_trials):
        map_batch_conv = []
        for i in range(n_channels):
            m_conv = tf.sparse.sparse_dense_matmul(kernel_channels[i], map_channels_batch[i])
            map_batch_conv.append(m_conv)
        map_batch_conv = tf.stack(map_batch_conv)
time_elapsed = (time.time() - time_start) / n_trials
print(f"n_trials={n_trials} time per trial: {time_elapsed:2.6f} s")

m_smooth_conv = map_batch_conv[0]
print(f"smoothed map sparse-dense sum={np.sum(m_smooth_conv)} mean={np.mean(m_smooth_conv)}")

print("===========================> block matrix sparse-dense convolution")

ind_batch = []
val_batch = []
map_batch = []

with tf.device("cpu"):
    for i in trange(n_channels, desc="creating sparse kernels"):
        inds_r = tf.constant(np.arange(npix), dtype=tf.int64)
        inds_r = tf.expand_dims(inds_r, axis=-1)
        inds_r = tf.tile(inds_r, [1, max_neighbours])
        inds_c = tf.constant(inds_k, dtype=tf.int64)
        ind_coo = tf.concat([tf.reshape(inds_r, [-1, 1]), tf.reshape(inds_c, [-1, 1])], axis=-1)
        ind_coo = ind_coo + i * npix  # block-diag
        ind_batch.append(ind_coo)
        val_batch.append(tf.reshape(kernel, [-1]))

        m_batch = tf.concat([tf.expand_dims(m, axis=-1)] * batch_size, axis=-1)
        map_batch.append(m_batch)

    ind_batch = tf.concat(ind_batch, axis=0)
    val_batch = tf.concat(val_batch, axis=0)
    map_batch = tf.concat(map_batch, axis=0)

    sparse_kernel = tf.sparse.SparseTensor(
        indices=ind_batch, values=val_batch, dense_shape=[npix * n_channels, npix * n_channels]
    )
    sparse_kernel = tf.sparse.reorder(sparse_kernel)

    print(
        f"created block-sparse index shape={ind_batch.shape} max_val={np.max(ind_batch)} size={np.array(ind_batch).nbytes/1e9:2.4f} GB"
    )
    print("memory used {:2.4f} GB".format(process.memory_info().rss / 1e9))  # in bytes


time_start = time.time()
with tf.device("gpu"):
    for j in range(n_trials):
        m_batch_conv = tf.sparse.sparse_dense_matmul(sparse_kernel, map_batch)
        m_batch_conv = tf.reshape(m_batch_conv, [n_channels, -1, batch_size])

time_elapsed = (time.time() - time_start) / n_trials
print(f"n_trials={n_trials} time per trial: {time_elapsed:2.6f} s")

m_smooth_conv = m_batch_conv[0, :, 0]
print(f"smoothed map block sparse-dense sum={np.sum(m_smooth_conv)} mean={np.mean(m_smooth_conv)}")


def part_to_full(m_part):
    m_ = m_full * 0
    m_[select] = m_part
    return m_


np.save("m.npy", part_to_full(m))
np.save("m_smooth_conv.npy", part_to_full(m_smooth_conv))
np.save("m_smooth_healpy.npy", part_to_full(m_smooth_healpy))

print("===========================> healpy convolution CPU")

n_trials = 1

time_start = time.time()

for i in trange(n_channels, desc="smoothing channels"):
    for j in range(batch_size):
        m_smooth_healpy = hp.sphtfunc.smoothing(m_full, sigma=sigma_rad)

time_elapsed = (time.time() - time_start) / n_trials
print(f"n_trials={n_trials} time per trial: {time_elapsed:2.6f} s")
