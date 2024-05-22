# Installation

1. Install the packages in requirements.txt
```
conda create -n cosine_fmh_env --file requirements.txt -c conda-forge -c bioconda
conda activate cosine_fmh_env
```
1. Install Simka:
```
wget https://github.com/GATB/simka/releases/download/v1.5.3/simka-v1.5.3-bin-Linux.tar.gz
tar -xvzf simka-v1.5.3-bin-Linux.tar.gz
```
Then, add the bin directory in the PATH. Make sure to give +x permission.
1. Install frac-kmc
(a) download binaries from https://github.com/KoslickiLab/frac-kmc/tree/main/wrappers/bin
(b) make sure to give +x (execution) permission.
(c) add the directory in the PATH environment

# Tables
After cloning this repository:
```
cd tables
python generate_tables.py
```

# Figure 1

# Figure 2