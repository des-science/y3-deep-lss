import os, argparse

parser = argparse.ArgumentParser()
parser.add_argument("--sbatch_output", type=str, default=None)
args = parser.parse_args()

print(args.sbatch_output)
print(os.path.abspath(args.sbatch_output))

sbatch_output = os.environ.get("SBATCH_OUTPUT", None)
srun_output = os.environ.get("SRUN_OUTPUT", None)
slurm_job_id = os.environ.get("SLURM_JOB_ID", None)

print(sbatch_output)
print(srun_output)
print(slurm_job_id)