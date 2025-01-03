import sys

"""
Given a filename, read the end of the file, and return
CPU time, wall time, and memory usage.
CPU time is the summation of user and system time.
"""


def get_metrics(filename):
    with open (filename, 'r') as f:
        lines = f.readlines()
        for line in reversed(lines):
            if "Maximum resident set size" in line:
                mem = int(line.split()[-1])
                # this is kbytes, convert to gb
                mem_in_gb = mem / 1024 / 1024
            elif "Elapsed (wall clock) time" in line:
                wall_time_str = str(line.split()[-1])
                # this is in the format of h:mm:ss or m:ss
                try:
                    wall_time = sum([float(x) * 60 ** i for i, x in enumerate(reversed(wall_time_str.split(':')))])
                except:
                    raise ValueError("Wall time is not in the expected format")
            elif "System time" in line:
                system_time = float(line.split()[-1])
            elif "User time" in line:
                user_time = float(line.split()[-1])
                break
            
        cpu_time = system_time + user_time
        return cpu_time, wall_time, mem_in_gb
    
    
if __name__ == "__main__":
    filename = sys.argv[1]
    print(get_metrics(filename))
        