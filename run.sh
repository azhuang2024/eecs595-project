#!/bin/bash
# The interpreter is used to execute the script

#"#SBATCH" directives that convey submission options:

#SBATCH --job-name=project
#SBATCH --account=eecs595w24\_class
#SBATCH --partition=spgpu
#SBATCH --gpus=1
#SBATCH --time=00-04:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --mem-per-cpu=16000m
#SBATCH --mail-type=BEGIN,END
#SBATCH --output=project.out

# The application(s) to execute along with its input arguments and options:

python3 project.py
