"""
This script will take the following inputs:
1. The path to a file that contains absolute path of the input files (files can be fasta, fastq, gz etc.)
1. The kmer size (default: 21)
1. The scale factor (scaled value)
1. The metric to compute (e.g. 'cosine', 'braycurtis', 'canberra', 'jaccard', 'kulsinski')
1. The number of cores to use
1. The output file name

The script will then do the following:
1. For all the input files, it will generate FMH sketch files
1. Using the sketches, the script will then compute pairwise metric values
1. The previous step will be parallelized using the number of cores specified
"""


# Requirements
# FracKmc needs to be in the PATH
# Download all binaries of FracKmc, and then add the path to the binaries to the PATH

# Usage
# python run_by_fmh_wrapper.py -i <input_file> -s <scale_factor> -m <metric> -w <weighted> -c <cores> -o <output_file>

import argparse
import os
import subprocess
import multiprocessing
import time

from concurrent.futures import ProcessPoolExecutor

from read_fmh_sketch import read_fmh_sig_file


def read_fmh_sig_file_single_process(file, ksize, seed, scaled, index, return_list):
    # call read_fmh_sig_file
    # get the return value
    # store in the return_list
    return_list[index] = read_fmh_sig_file(file, ksize, seed, scaled)


def parse_arguments():
    parser = argparse.ArgumentParser(description='Run FMH to compute metrics on input files')
    parser.add_argument('-i', '--input_file', type=str, help='Path to file containing input files')
    parser.add_argument('-k', '--ksize', type=int, help='Kmer size', default=21)
    parser.add_argument('-s', '--scale_factor', type=int, help='Scale factor', default=1000)
    parser.add_argument('-m', '--metric', type=str, help='Metric to compute', default='cosine')
    parser.add_argument('-c', '--cores', type=int, help='Number of cores', default=128)
    parser.add_argument('-o', '--output_file', type=str, help='Output file name')
    # add seed as an argument
    parser.add_argument('-S', '--seed', type=int, help='Seed', default=42)
    #add argument about whether to parallelize or not
    parser.add_argument('-p', '--parallelize', action='store_true', help='Whether to parallelize or not')

    # add two more arguments: how many cores to use for each instance, and how many instances to run in parallel
    parser.add_argument('-C', '--cores_each_instance', type=int, help='Number of cores to use for each instance', default=128)
    parser.add_argument('-P', '--num_instances_parallel', type=int, help='Number of instances to run in parallel', default=1)

    # add argument to use abundances
    parser.add_argument('-a', '--use_abund', action='store_true', help='Use abundances')

    # whether to sketch only (skip the metric computation)
    parser.add_argument('-sketch_only', action='store_true', help='Only generate sketches')

    # whether to skip sketch creation
    parser.add_argument('-skip_sketch', action='store_true', help='Skip sketch creation')

    # whether to not parallelize the metric calculation
    parser.add_argument('-np', '--no_parallelize_metric', action='store_true', help='Do not parallelize metric calculation')

    # whether to process the first n files only
    parser.add_argument('-n', '--num_files', type=int, help='Number of files to process', default=-1)
    
    args = parser.parse_args()
    return args


def check_input_files_exist(input_file):
    # first check that the input file exists
    if not os.path.exists(input_file):
        raise FileNotFoundError(f'Input file {input_file} does not exist')

    # next, check that all the files in the input file exist
    with open(input_file, 'r') as f:
        for line in f:
            if not os.path.exists(line.strip()):
                raise FileNotFoundError(f'File {line.strip()} does not exist')


def check_num_cores(cores):
    if cores < 1:
        raise ValueError('Number of cores should be greater than 0')
    
    # check that the machine has enough cores
    num_cores = multiprocessing.cpu_count()
    if cores > num_cores:
        raise ValueError(f'Machine does not have enough cores. Number of cores requested: {cores}, Number of cores available: {num_cores}')


def generate_fmh_sketch(file, scale_factor, ksize, output_file, is_fasta, num_cores, seed=42, use_abund=False):
    # use the proper command
    # command: fracKmcSketch <input_filename> <sketch_filename> --ksize <ksize> --scaled <scaled> --seed 42 --fa/--fq
    if is_fasta:
        cmd = f'fracKmcSketch {file} {output_file} --ksize {ksize} --scaled {scale_factor} --seed {seed} --fa'
    else:
        cmd = f'fracKmcSketch {file} {output_file} --ksize {ksize} --scaled {scale_factor} --seed {seed} --fq'

    cmd += ' --t ' + str(num_cores)

    if use_abund:
        cmd += ' --a'

    # show the command
    print(cmd)

    # run the command and check for errors, make silent
    try:
        subprocess.run(cmd, shell=True, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    except subprocess.CalledProcessError as e:
        raise Exception(f'Error while generating sketch for file {file}: {e}')


"""
list1 and list2 are sorted lists of integers
This function should return the number of common elements between the two lists
"""
def count_num_common(list1, list2):
    
    # check if either of the lists is empty
    if len(list1) == 0 or len(list2) == 0:
        return 0
    
    # use merge sort like operation to find the common elements
    i = 0
    j = 0
    count = 0

    while i < len(list1) and j < len(list2):
        if list1[i] == list2[j]:
            count += 1
            i += 1
            j += 1
        elif list1[i] < list2[j]:
            i += 1
        else:
            j += 1

    return count


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

        # compute the dot product
        dot_product = get_dot_product(sig1, sig2)
        
        # compute the magnitudes
        magnitude1 = compute_magnitute(sig1)
        magnitude2 = compute_magnitute(sig2)
        
        # compute the bray curtis similarity
        return_value = 1 - (2.0 * dot_product / (magnitude1 + magnitude2))
        return return_value
    else:
        return -1



def compute_metric_for_a_pair(sig1, sig2, metric, return_list, index):
    # sig1 and sig2: list of tuples (min, abundance)
    if metric == 'cosine':
        # if either of the signatures is empty, return 0.0
        if len(sig1) == 0 or len(sig2) == 0:
            return_list[index] = 0.0
            return

        # compute the dot product
        dot_product = get_dot_product(sig1, sig2)
        
        # compute the magnitudes
        magnitude1 = compute_magnitute(sig1)
        magnitude2 = compute_magnitute(sig2)
        
        # compute the cosine similarity
        return_value =  dot_product / (magnitude1 * magnitude2)
        return_list[index] = return_value
    elif metric == 'braycurtis':
        # if either of the signatures is empty, return 0.0
        if len(sig1) == 0 or len(sig2) == 0:
            return_list[index] = 0.0
            return

        # compute the dot product
        dot_product = get_dot_product(sig1, sig2)
        
        # compute the magnitudes
        magnitude1 = compute_magnitute(sig1)
        magnitude2 = compute_magnitute(sig2)
        
        # compute the bray curtis similarity
        return_value = 1 - (2.0 * dot_product / (magnitude1 + magnitude2))
        return_list[index] = return_value
    else:
        return_list[index] = -1


def compute_metric_for_range_of_pairs(i_j_pairs_to_work_on, all_sketches, start_index, end_index, return_list, metric, show_progress):
    for index in range(start_index, end_index):
        i, j = i_j_pairs_to_work_on[index-start_index]
        sigs_and_abundances1 = all_sketches[i]
        sigs_and_abundances2 = all_sketches[j]

        # compute the metric
        compute_metric_for_a_pair(sigs_and_abundances1, sigs_and_abundances2, metric, return_list, index)

        # show progress
        if not show_progress:
            continue
        
        progress_percentage = 100*(index-start_index+1)/(end_index-start_index)
        print(f'{progress_percentage:.3f}% completed..', end='\r')


def main():
    args = parse_arguments()
    check_input_files_exist(args.input_file)
    check_num_cores(args.cores)
    
    # read the input file
    input_files = []
    with open(args.input_file, 'r') as f:
        for line in f:
            if line.strip() == '':
                continue

            input_files.append(line.strip())

    # check if the user wants to process only a few files
    if args.num_files != -1:
        input_files = input_files[:args.num_files]
    
    # see if more than 4 threads are available
    if args.parallelize:
        cores_each_instance = args.cores_each_instance
        num_processes_in_parallel = args.num_instances_parallel
    else:
        cores_each_instance = args.cores
        num_processes_in_parallel = 1


    sketch_files = []
    filename_to_sketch_name = {}
    is_fasta_list = []
    for file in input_files:
        # sketch filename format: <input_filename>_ksize_scaled_seed.sig
        sketch_filename = f'{file}_{args.ksize}_{args.scale_factor}_{args.seed}.sig'
        sketch_files.append(sketch_filename)
        filename_to_sketch_name[file] = sketch_filename
        is_fasta = file.endswith('.fa') or file.endswith('.fasta') or file.endswith('.fna')
        is_fasta_list.append(is_fasta)    

    if not args.skip_sketch:
        # generate sketches
        print('Generating sketches')
        with ProcessPoolExecutor(max_workers=num_processes_in_parallel) as executor:
            executor.map(generate_fmh_sketch, input_files, [args.scale_factor]*len(input_files), [args.ksize]*len(input_files), sketch_files, is_fasta_list, [cores_each_instance]*len(input_files), [args.seed]*len(input_files), [args.use_abund]*len(input_files))
        
    # check if the user only wants to sketch
    if args.sketch_only:
        print('Done')
        return
    
    if args.no_parallelize_metric:
        pair_to_metric_dict = {}
        num_completed = 0
        num_total_pairs = len(input_files) * (len(input_files) - 1) // 2
        for i in range(len(input_files)):
            sketch1_name = filename_to_sketch_name[input_files[i]]
            sigs_and_abundances1 = read_fmh_sig_file(sketch1_name, args.ksize, args.seed, args.scale_factor)
            for j in range(i+1, len(input_files)):
                #print(f'Computing metric for {input_files[i]} and {input_files[j]}')
                
                sketch2_name = filename_to_sketch_name[input_files[j]]
                sigs_and_abundances2 = read_fmh_sig_file(sketch2_name, args.ksize, args.seed, args.scale_factor)

                # compute the metric
                return_list = [-1]
                index = 0
                compute_metric_for_a_pair(sigs_and_abundances1, sigs_and_abundances2, args.metric, return_list, index)
                pair_to_metric_dict[(input_files[i], input_files[j])] = return_list[index]

                #print(f'Computed metric for {input_files[i]} and {input_files[j]}')
                num_completed += 1
                # show percentage progress
                print(f'{100*num_completed/num_total_pairs:.3f}% completed..', end='\r')

        print('')

        # write the output to a file
        with open(args.output_file, 'w') as f:
            for pair, metric in pair_to_metric_dict.items():
                f.write(f'{pair[0]}\t{pair[1]}\t{metric}\n')

        print('Done')
        return


    # measure time for rest of the code
    start_time = time.time()

    with ProcessPoolExecutor(max_workers=num_processes_in_parallel) as executor:
        all_sketches = list(executor.map(read_fmh_sig_file, sketch_files, [args.ksize]*len(sketch_files), [args.seed]*len(sketch_files), [args.scale_factor]*len(sketch_files)))

    if all_sketches is None:
        print('Error while reading the sketch files')
        return

    # compute pairwise metrics
    pair_to_metric_dict = {}
    print('Computing pairwise metrics')
    
    # use all the cores to compute the metrics
    num_processes_in_parallel = args.cores
    
    process_list = []
    num_files = len(input_files)
    num_pairs = num_files * (num_files - 1) // 2
    return_list = multiprocessing.Manager().list([-1] * num_pairs)

    all_i_j_pairs = []
    for i in range(len(input_files)):
        for j in range(i+1, len(input_files)):
            all_i_j_pairs.append((i, j))

    for i in range(num_processes_in_parallel):
        start_index = i * num_pairs // num_processes_in_parallel
        end_index = (i+1) * num_pairs // num_processes_in_parallel
        
        if i == num_processes_in_parallel - 1:
            end_index = num_pairs

        p = multiprocessing.Process(target=compute_metric_for_range_of_pairs, args=(all_i_j_pairs, all_sketches, start_index, end_index, return_list, args.metric, i==num_processes_in_parallel-1))
        process_list.append(p)
        p.start()

    for p in process_list:
        p.join()
    
    print('Done computing pairwise metrics, now writing to file')
    
    # extract the values from the return_list
    index = 0
    for i in range(len(input_files)):
        for j in range(i+1, len(input_files)):
            pair_to_metric_dict[(input_files[i], input_files[j])] = return_list[index]
            index += 1

    # write the output to a file
    with open(args.output_file, 'w') as f:
        for pair, metric in pair_to_metric_dict.items():
            f.write(f'{pair[0]}\t{pair[1]}\t{metric}\n')
                
    print('Done')

    # print the time taken
    print('Time taken only for similarity calculation:', time.time() - start_time)

    # create a new file with the sketch files as the filelist
    with open('sketch_files.txt', 'w') as f:
        for file in sketch_files:
            f.write(file + '\n')

if __name__ == '__main__':
    main()



# sample: python ../../run_using_fmh/run_by_fmh_wrapper.py -i filelist_testrun_fmh -k 21 -s 1000 -m cosine -c 128 -o temp
