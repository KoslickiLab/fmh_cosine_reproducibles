import argparse
import os

from run_using_fmh.read_fmh_sketch import read_fmh_sig_file
from run_using_fmh.run_by_fmh_wrapper import compute_metric_for_a_pair_returns

def parse_args():
    # list of args
    # 1. path to the file list
    # 2. path to the output file
    # 3. kmer size (default 21)
    # 4. metric: cosine or braycurtis (default cosine)
    
    parser = argparse.ArgumentParser(description="generate ground truth using all kmers")
    parser.add_argument("file_list", type=str, help="Path to the file list")
    parser.add_argument("output_file", type=str, help="Path to the output file")
    parser.add_argument("--kmer_size", type=int, default=21, help="Kmer size")
    parser.add_argument("--metric", type=str, default="cosine", help="Metric to use")
    
    return parser.parse_args()
    
def main():
    args = parse_args()
    seed = 42
    scale_factor = 1
    
    # ensure that no empty lines are present in the file list
    with open(args.file_list) as f:
        files = f.readlines()
        files = [file.strip() for file in files if file.strip()]
    
    # if there is empty line, exit with an error message
    if len(files) == 0:
        print("No files in the file list")
        return
    
    # ensure that all files in the file list are present
    for file in files:
        assert os.path.exists(file)
    
    pair_to_metric = {}
    num_total_pairs = len(files) * (len(files) - 1) // 2
    num_pairs_completed = 0
    for i in range(len(files)):
        for j in range(i+1, len(files)):
            filename1 = files[i]
            filename2 = files[j]
            
            # read the contents of the files
            sketch_filename1 = f'{filename1}_{args.ksize}_{scale_factor}_{seed}.sig'
            sketch_filename2 = f'{filename2}_{args.ksize}_{scale_factor}_{seed}.sig'
            
            mins1 = read_fmh_sig_file(sketch_filename1, args.kmer_size, seed, scale_factor)
            mins2 = read_fmh_sig_file(sketch_filename2, args.kmer_size, seed, scale_factor)
            metric_value = compute_metric_for_a_pair_returns(mins1, mins2, args.metric)
            
            pair_to_metric[(filename1, filename2)] = metric_value
            
            num_pairs_completed += 1
            print(f"Completed {num_pairs_completed}/{num_total_pairs} pairs")
            
    with open(args.output_file, "w") as f:
        for pair, metric_value in pair_to_metric.items():
            f.write(f"{pair[0]}\t{pair[1]}\t{metric_value}\n")