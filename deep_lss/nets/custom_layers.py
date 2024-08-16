import tensorflow as tf


class MeanBinningLayer(tf.keras.layers.Layer):
    """
    Written by GPT-4o. This custom layer takes a 3D tensor as input and bins the values along axis 1 based on the
    provided bin_edges, similar to scipy.stats.binned_statistic for x=np.arange(input.shape[1]). The output is a 3D
    tensor with the same shape as the input, but the values are replaced by the mean of the values in each bin.
    """

    def __init__(self, bin_edges, **kwargs):
        super(MeanBinningLayer, self).__init__(**kwargs)
        self.bin_edges = bin_edges  # Store as a Python list
        self.num_bins = len(bin_edges) - 1

    def call(self, inputs):
        # inputs shape: (batch_size, dim1, dim2)
        batch_size = tf.shape(inputs)[0]
        dim1 = tf.shape(inputs)[1]
        dim2 = tf.shape(inputs)[2]

        # Create an array representing the indices along axis 1
        indices = tf.range(dim1, dtype=tf.float32)

        # Digitize the indices based on bin_edges
        bin_indices = tf.raw_ops.Bucketize(input=indices, boundaries=self.bin_edges[1:-1])

        # Handle values smaller than the lowest bin edge
        bin_indices = tf.where(indices < self.bin_edges[0], 0, bin_indices)

        # Handle values larger than the highest bin edge
        bin_indices = tf.where(indices >= self.bin_edges[-1], self.num_bins - 1, bin_indices)

        # Initialize a tensor to store the binned means
        binned_means = tf.TensorArray(dtype=tf.float32, size=self.num_bins)

        for i in range(self.num_bins):
            mask = tf.equal(bin_indices, i)
            mask = tf.cast(mask, dtype=tf.float32)
            mask = tf.expand_dims(mask, axis=-1)  # shape: (dim1, 1)

            # Apply the mask to inputs and calculate the mean along axis 1
            masked_values = inputs * mask
            sum_masked_values = tf.reduce_sum(masked_values, axis=1)
            sum_mask = tf.reduce_sum(mask, axis=0)

            # Avoid division by zero
            mean_masked_values = sum_masked_values / (sum_mask + 1e-8)

            binned_means = binned_means.write(i, mean_masked_values)

        # Stack the results along axis 1 to form the final output
        binned_means = binned_means.stack()
        binned_means = tf.transpose(binned_means, perm=[1, 0, 2])

        return binned_means

    def compute_output_shape(self, input_shape):
        return (input_shape[0], self.num_bins, input_shape[2])
