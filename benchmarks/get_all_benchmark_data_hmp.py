from get_metrics import get_metrics
import sys

if __name__ == "__main__":
    
    print("Usage: python get_all_benchmark_data_hmp.py <directory_where_benchmark_files> <output_filename> <fmh/mash>")
    
    directory_where_benchmark_files = sys.argv[1]
    output_filename = sys.argv[2]
    method_name = sys.argv[3]
    
    if method_name != "fmh" and method_name != "mash":
        raise ValueError("The third argument must be either 'fmh' or 'mash'")
    
    # output file columns: num_genomes, complete_run_cputime, complete_run_walltime, complete_run_mem, sketch_only_cputime, sketch_only_walltime, sketch_only_mem
    with open(output_filename, 'w') as f:
        f.write("num_genomes,complete_run_cputime,complete_run_walltime,complete_run_mem,sketch_only_cputime,sketch_only_walltime,sketch_only_mem\n")

    num_genomes_in_files = [100, 150, 200, 250, 300]
    for num_genomes in num_genomes_in_files:
        log_filename_complete_run = f"{directory_where_benchmark_files}/{method_name}_complete_run_{num_genomes}.log"
        complete_run_cputime, complete_run_walltime, complete_run_mem = get_metrics(log_filename_complete_run)
        
        sketch_only_filename = f"{directory_where_benchmark_files}/{method_name}_skip_similarity_run_{num_genomes}.log"
        sketch_only_cputime, sketch_only_walltime, sketch_only_mem = get_metrics(sketch_only_filename)
        
        print(f"completed processing for {num_genomes} genomes")
        
        with open(output_filename, 'a') as f:
            f.write(f"{num_genomes},{complete_run_cputime},{complete_run_walltime},{complete_run_mem},{sketch_only_cputime},{sketch_only_walltime},{sketch_only_mem}\n")