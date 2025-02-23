"""
In this script, we will run mash to compute the results.
"""

import os
import argparse
import json
import multiprocessing



def process_a_range_of_pairs(filenames, filenames_to_hashes, all_i_j_pairs, start_index, end_index, return_list, show_progress):
    """
    Compute the pairwise cosines for a range of pairs of files
    """
    completed = 0
    for index in range(start_index, end_index):
        i, j = all_i_j_pairs[index]
        filename1 = filenames[i]
        filename2 = filenames[j]
        hash1 = filenames_to_hashes[filename1]
        hash2 = filenames_to_hashes[filename2]
        
        dot_product = set(hash1).intersection(hash2)
        cosine = len(dot_product) / (len(hash1)**0.5 * len(hash2)**0.5)
        bray_curtis = (len(hash1) + len(hash2) - 2 * len(dot_product)) / (len(hash1) + len(hash2))
        return_list[index] = (cosine, bray_curtis)

        completed += 1
        
        if not show_progress:
            continue
        
        percentage_progress = 100 * completed / (end_index - start_index)
        print(f"Completed {percentage_progress:.3f}%", end='\r')


"""
arguments:
1. file list that contains many files
2. output file name
3. kmer size (default 21)
4. sketch size (default 10000)
"""
def parse_args():
    parser = argparse.ArgumentParser(description="Run Mash")
    parser.add_argument("file_list", type=str, help="Path to the file list")
    parser.add_argument("output_file", type=str, help="Path to the output file")
    parser.add_argument("--kmer_size", type=int, default=21, help="Kmer size")
    parser.add_argument("--sketch_size", type=int, default=10000, help="Sketch size")
    parser.add_argument("--num_cores", type=int, default=128, help="Num of cores to parallelize over")
    parser.add_argument('--skip_sketch', dest='skip_sketch', action='store_true', help='Skip sketching')
    parser.add_argument('--skip_similarity', dest='skip_similarity', action='store_true', help='Skip similarity computation')
    return parser.parse_args()

def main():
    args = parse_args()

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
    
    if not args.skip_sketch:
        print('*****************************')
        print('Running Mash')
        print('*****************************')

        # create a sketch for each file in the file list
        # command: mash sketch -k kmer_size -s sketch_size -o mash_sketch -l filelist
        # the generated output file: mash_sketch.msh
        os.system(f"mash sketch -k {args.kmer_size} -s {args.sketch_size} -o mash_sketch -l {args.file_list} -p {args.num_cores}")

        print('*****************************')
        print('Sketching completed, creating json')
        print('*****************************')

        # dump the .msh file into a json file
        # command: mash info mash_sketch.msh -d > mash_sketch.json
        return_code = os.system(f"mash info mash_sketch.msh -d > mash_sketch.json")
        if return_code != 0:
            print("Error in creating json file (check Mash version)")
            return

        print('*****************************')
        print('Json created')
        print('*****************************')


    if args.skip_similarity:
        print('*****************************')
        print('Skipping similarity computation')
        print('*****************************')
        return

    # read the json file to obtain the min hash values
    # format: data['sketches']['name'] is the filename
    # data['sketches']['hashes'] is the list of the min hash values
    filenames_to_hashes = {}
    with open("mash_sketch.json") as f:
        data = json.load(f)
        for i in range( len(data['sketches']) ):
            filename = str(data['sketches'][i]['name'])
            filenames_to_hashes[filename] = list(data['sketches'][i]['hashes'])

    print('*****************************')
    print('Hashes read, computing pairwise cosines')
    print('*****************************')
                

    all_i_j_pairs = []
    for i in range(len(files)):
        for j in range(i+1, len(files)):
            all_i_j_pairs.append((i, j))

    # create a list using multiprocessing manager
    # this list will be shared among all processes
    return_list = multiprocessing.Manager().list([-1] * len(all_i_j_pairs))

    list_processes = []
    show_progress = False
    for i in range(args.num_cores):
        start_index = i * len(all_i_j_pairs) // args.num_cores
        end_index = (i + 1) * len(all_i_j_pairs) // args.num_cores
        if i == args.num_cores - 1:
            end_index = len(all_i_j_pairs)
            show_progress = True
        p = multiprocessing.Process(target=process_a_range_of_pairs, args=(files, filenames_to_hashes, all_i_j_pairs, start_index, end_index, return_list, show_progress))
        p.start()
        list_processes.append(p)

    for p in list_processes:
        p.join()

    print('*****************************')
    print('Computing completed, writing to file')
    print('*****************************')

    # write the results to the output file
    index = 0
    with open(args.output_file, "w") as f:
        f.write("file1,file2,cosine_similarity,bray_curtis\n")
        for i in range(len(files)):
            for j in range(i+1, len(files)):
                cosine, bray_curtis = return_list[index]
                index += 1
                f.write(f"{files[i]},{files[j]},{cosine},{bray_curtis}\n")



if __name__ == "__main__":
    main()