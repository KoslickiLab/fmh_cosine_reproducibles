"""
In this script, we compare the running time of sourmash sketch and frackmcsketch on the same dataset.
"""


import os
import time
import numpy as np

working_dir = "/scratch/mbr5797/compare_fmh_simka_mash/data/hmp_gut/wgs"
filesize_to_filename = {
    1: working_dir + '/f2856_ihmp_IBD_PSM6XBTZ_P.fastq.gz',
    2: working_dir + '/f1072_ihmp_IBD_MSM5FZ9X_P.fastq.gz',
    3: working_dir + '/f1001_ihmp_IBD_ESM5MEE2_P.fastq.gz',
    4: working_dir + '/f1212_ihmp_IBD_CSM5FZ4E_P.fastq.gz',
    5: working_dir + '/f1253_ihmp_IBD_MSM5LLI8_P.fastq.gz'
}

num_readings = 3

f = open("runtime_comparison.csv", "w")
f.write("filesize,avg_sourmash_time,std_sourmash_time,avg_frackmc_time,std_frackmc_time,avg_mash_time,std_mash_ime\n")
for filesize, filename in filesize_to_filename.items():
    print(filename)
    # Run sourmash sketch, and record the time
    # command: sourmash sketch dna <filename> -p k=21,scaled=1000 -o <output_filename>
    time_needed_sourmash = []
    for _ in range(num_readings):
        start_time = time.time()
        os.system(f"sourmash sketch dna {filename} -p k=21,scaled=1000 -o {filename}.sourmash")
        end_time = time.time()
        time_needed_sourmash.append(end_time - start_time)

    time_needed_frackmc = []
    # Run frackmcsketch, and record the time
    # command: frackmcsketch <filename> -o <output_filename> --ksize 21 --scaled 1000 --fq --t 128
    for _ in range(num_readings):
        start_time = time.time()
        os.system(f"fracKmcSketch {filename} -o {filename}.frackmc --ksize 21 --scaled 1000 --fq --t 128")
        end_time = time.time()
        time_needed_frackmc.append(end_time - start_time)

    time_needed_mash = []
    # Run mash sketch, and record the time
    # command: mash sketch -s 1000 -k 21 -o <output_filename> <filename>
    for _ in range(num_readings):
        start_time = time.time()
        os.system(f"mash sketch -s 1000 -k 21 -o {filename}.mash {filename}")
        end_time = time.time()
        time_needed_mash.append(end_time - start_time)

    avg_sourmash_time = np.mean(time_needed_sourmash)
    std_sourmash_time = np.std(time_needed_sourmash)
    avg_frackmc_time = np.mean(time_needed_frackmc)
    std_frackmc_time = np.std(time_needed_frackmc)
    avg_mash_time = np.mean(time_needed_mash)
    std_mash_time = np.std(time_needed_mash)

    # print the following: filesize, avg sourmash time, std sourmash time, avg frackmc time, std frackmc time
    # print these in one line, separated by comma
    print(f"{filesize}, {avg_sourmash_time}, {std_sourmash_time}, {avg_frackmc_time}, {std_frackmc_time}")
    f.write(f"{filesize},{avg_sourmash_time},{std_sourmash_time},{avg_frackmc_time},{std_frackmc_time},{avg_mash_time},{std_mash_time}\n")

f.close()        