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
1. First obtain the data by following the instruction in fig_1/download.txt
1. Then, change fig_1/run_tools.py, where we need to write path to the files
1. Then, run the following:
```
cd fig_1
python run_tools.py
python plot_runtime_comparison.py
```

# Figure 2
1. First we need to obtain the data by following the instruction in fig_2/download.txt
1. Then, extract the downloaded data in fig_2
1. For the Ecoli dataset:
(a) `cd fig_2/ecoli3682`
(b) `bash run_tools.sh`
(c) `gunzip simka_results_*/*.gz`
1. For the HMP dataset:
(a) `cd fig_2/hmp_gut`
(b) `bash run_tools.sh`
(c) `gunzip simka_results_*/*.gz`
1. Make the comparisons using `python comparison.py`
1. Copy the files ecoli_combined, ecoli_runtime, hmp_combined, hmp_gut_runtime to fig_2/plotting
1. Plotting:
```
cd fig_2/plotting
python plot_results.py
```