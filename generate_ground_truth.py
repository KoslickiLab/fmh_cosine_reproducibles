import argparse
import os
import json
import numpy as np
from concurrent.futures import ProcessPoolExecutor
from tqdm import tqdm

"""
sig1 and sig2 are lists of tuples (min, abundance)
This function gets the dot product of the two sigs
The sigs are already sorted by min
"""
def get_dot_product(sig1, sig2):
    if len(sig1) == 0 or len(sig2) == 0:
        return 0
    
    i = 0
    j = 0
    dot_product = 0

    while i < len(sig1) and j < len(sig2):
        if sig1[i][0] == sig2[j][0]:
            dot_product += sig1[i][1] * sig2[j][1]
            i += 1
            j += 1
        elif sig1[i][0] < sig2[j][0]:
            i += 1
        else:
            j += 1

    return dot_product

"""
sig is a list of tuples (min, abundance)
This function computes the magnitude of the signature vector
"""
def compute_magnitute(sig):
    abundances_list = [abundance for min, abundance in sig]
    return sum([abundance**2 for abundance in abundances_list])**0.5


def get_num_common_using_abundances(sig1, sig2):
    # if either of the signatures is empty, return 0
    if len(sig1) == 0 or len(sig2) == 0:
        return 0

    i = 0
    j = 0
    num_common = 0

    while i < len(sig1) and j < len(sig2):
        if sig1[i][0] == sig2[j][0]:
            num_common += min(sig1[i][1], sig2[j][1])
            i += 1
            j += 1
        elif sig1[i][0] < sig2[j][0]:
            i += 1
        else:
            j += 1

    return num_common


def get_total_using_abundances(sig):
    return sum([abundance for min, abundance in sig])

def compute_metric_for_a_pair_returns(sig1, sig2, metric):
    # sig1 and sig2: list of tuples (min, abundance)
    if metric == 'cosine':
        # if either of the signatures is empty, return 0.0
        if len(sig1) == 0 or len(sig2) == 0:
            return 0.0

        # compute the dot product
        dot_product = get_dot_product(sig1, sig2)
        
        # compute the magnitudes
        magnitude1 = compute_magnitute(sig1)
        magnitude2 = compute_magnitute(sig2)
        
        # compute the cosine similarity
        return_value =  dot_product / (magnitude1 * magnitude2)
        return return_value
    elif metric == 'braycurtis':
        # if either of the signatures is empty, return 0.0
        if len(sig1) == 0 or len(sig2) == 0:
            return 0.0

        num_common = get_num_common_using_abundances(sig1, sig2)
        total1 = get_total_using_abundances(sig1)
        total2 = get_total_using_abundances(sig2)
        
        # compute the bray curtis similarity
        return_value = 1 - (2.0 * num_common / (total1 + total2))
        return return_value
    else:
        return -1

def read_fmh_sig_file(file, ksize, seed, scaled):
    # compute the correct max_hash
    theoretical_max_hash = np.longdouble(2**64 - 1)
    divide_by = np.longdouble(scaled)
    target_max_hash = round( theoretical_max_hash / divide_by)

    # first check that the input file exists
    if not os.path.exists(file):
        raise FileNotFoundError(f'Input file {file} does not exist')

    # read the file as json
    try:
        f = open(file, 'r')
        json_data = json.load(f)
        f.close()
    except json.JSONDecodeError as e:
        raise Exception(f'Error while reading the file {file}: {e}')
    
    json_data = json_data[0]
        
    # check that the json data has the required keys
    required_keys = ['signatures']
    if not all(key in json_data for key in required_keys):
        raise KeyError(f'File {file} does not have the required keys: {required_keys}')
        
    # check that the signatures are a list
    if not isinstance(json_data['signatures'], list):
        raise TypeError(f'Signatures in file {file} should be a list')
    
    # check that each entry in the signatures is a dictionary containing the following keys:
    # 'ksize': the kmer size, an integer
    # 'seed': the seed used to generate the sketch, an integer
    # 'max_hash': the maximum hash value, an integer
    # 'mins': a list of integers

    sigs = json_data['signatures']
    for sig in sigs:
        required_keys = ['ksize', 'seed', 'max_hash', 'mins']
        if not all(key in sig for key in required_keys):
            raise KeyError(f'Signature {sig} does not have the required keys: {required_keys}')
        
        # check that the values are of the correct type
        if not isinstance(sig['ksize'], int):
            raise TypeError(f'ksize in signature {sig} should be an integer')
        if not isinstance(sig['seed'], int):
            raise TypeError(f'seed in signature {sig} should be an integer')
        if not isinstance(sig['max_hash'], int):
            raise TypeError(f'max_hash in signature {sig} should be an integer')
        if not isinstance(sig['mins'], list):
            raise TypeError(f'mins in signature {sig} should be a list')

        # if this signature is the correct ksize, and correct seed, and correct max_hash, return the mins
        if sig['ksize'] == ksize and sig['seed'] == seed and sig['max_hash'] == target_max_hash:
            # if "abundances" is present, extract the abundances
            if 'abundances' in sig:
                return list(zip(sig['mins'], sig['abundances']))
            else:
                return list(zip(sig['mins'], [1.0] * len(sig['mins'])))
        
    # if we reach this point, we did not find the correct signature
    raise ValueError(f'Could not find the signature with ksize={ksize}, seed={seed}, and max_hash={target_max_hash}')



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
    parser.add_argument("--num_cores", type=int, default=20, help="Num of cores to parallelize over")
    
    return parser.parse_args()



def compute_metric_for_a_pair_returns(sketch_filename1, sketch_filename2, metric, kmer_size, seed, scale_factor):
    mins1 = read_fmh_sig_file(sketch_filename1, kmer_size, seed, scale_factor)
    mins2 = read_fmh_sig_file(sketch_filename2, kmer_size, seed, scale_factor)
    return compute_metric_for_a_pair_returns(mins1, mins2, metric)

    
    
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
    
    executor = ProcessPoolExecutor(max_workers=args.num_cores)
    filename_pairs = []
    
    for i in range(len(files)):
        filename1 = files[i]
        sketch_filename1 = f'{filename1}_{args.kmer_size}_{scale_factor}_{seed}.sig'
        for j in range(i+1, len(files)):
            filename2 = files[j]
            sketch_filename2 = f'{filename2}_{args.kmer_size}_{scale_factor}_{seed}.sig'
            filename_pairs.append((sketch_filename1, sketch_filename2))
            
    print (f"Computing metric for {len(filename_pairs)} pairs")
    returned_metrics = list(tqdm(executor.map(compute_metric_for_a_pair_returns, [pair[0] for pair in filename_pairs], [pair[1] for pair in filename_pairs], [args.metric] * len(filename_pairs), [args.kmer_size] * len(filename_pairs), [seed] * len(filename_pairs), [scale_factor] * len(filename_pairs))), total=len(filename_pairs))
    
    # populate the pair_to_metric dictionary
    for i in range(len(filename_pairs)):
        pair = filename_pairs[i]
        metric_value = returned_metrics[i]
        pair_to_metric[pair] = metric_value
        
    # write the pair_to_metric dictionary to the output file
    print (f"Writing the output to {args.output_file}")
            
    with open(args.output_file, "w") as f:
        for pair, metric_value in pair_to_metric.items():
            f.write(f"{pair[0]}\t{pair[1]}\t{metric_value}\n")
            
if __name__ == "__main__":
    main()