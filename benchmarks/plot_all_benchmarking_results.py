"""
In this script, we plot the results.
"""


import matplotlib.pyplot as plt
import numpy as np
import random
import pandas as pd
import seaborn as sns
import matplotlib.patches as mpatches


simka_color = '#377eb8'
mash_color = '#ff7f00'
frackmc_color = '#99DC97'

frackmc_color_sketch_only = '#B9FCB7'
mash_color_sketch_only = '#FF9F30'


def plot_actual_errors_hmp(ax, metric):
    df_gt = pd.read_csv(f"hmp/ground_truth_{metric}", delim_whitespace=True, header=None)
    df_gt.columns = ['f1', 'f2', 'metric']
    f1_f2_to_metric_gt = { (f1, f2): metric for f1, f2, metric in zip(df_gt['f1'], df_gt['f2'], df_gt['metric']) }
    
    df_fmh = pd.read_csv(f"hmp/fmh_{metric}", delim_whitespace=True, header=None)
    df_fmh.columns = ['f1', 'f2', 'metric']
    f1_f2_to_metric_fmh = { (f1, f2): metric for f1, f2, metric in zip(df_fmh['f1'], df_fmh['f2'], df_fmh['metric']) }
    
    df_mash = pd.read_csv(f"hmp/mash_combined")
    f1_f2_to_metric_mash = { (f1, f2): metric for f1, f2, metric in zip(df_mash['file1'], df_mash['file2'], df_mash[metric]) }
    
    true_metrics = []
    fmh_values = []
    mash_values = []
    
    for f1, f2 in f1_f2_to_metric_gt.keys():
        true_metrics.append(f1_f2_to_metric_gt[(f1, f2)])
        fmh_values.append(f1_f2_to_metric_fmh[(f1, f2)])
        mash_values.append(f1_f2_to_metric_mash[(f1, f2)])
    

    # show a line y = x
    ax.plot([0, 1], [0, 1], color='grey', linestyle='--', linewidth=1)

    # plot mash and frackmc ecoli values againts the true values in a scatter plot
    ax.scatter(true_metrics, mash_values, label='Mash', color=mash_color, alpha=0.45, marker='s')
    ax.scatter(true_metrics, fmh_values, label='frac-kmc', color=frackmc_color, alpha=0.45, marker='o')

    if metric == 'cosine':
        metric = 'Cosine'
    elif metric == 'braycurtis':
        metric = 'Bray-Curtis'

    # add labels
    ax.set_xlabel(f"True {metric} (HMP)")
    ax.set_ylabel(f"Est. {metric} (HMP)")

    # add legend
    ax.legend()

    # flip order of legends
    handles, labels = ax.get_legend_handles_labels()
    #ax.legend(handles[::-1], labels[::-1])

    # set legend's alpha=1
    for lh in ax.legend().legendHandles:
        lh.set_alpha(1)

    # tight layout
    plt.tight_layout()

    # add legend
    mash_patch = mpatches.Patch(facecolor=mash_color, label='Mash')
    frackmc_patch = mpatches.Patch(facecolor=frackmc_color, label='frac-kmc')
    ax.legend(handles=[frackmc_patch, mash_patch])

    #plt.show()



def plot_actual_errors_ecoli(ax, metric):
    df_gt = pd.read_csv(f"ecoli/ground_truth_{metric}", delim_whitespace=True, header=None)
    df_gt.columns = ['f1', 'f2', 'metric']
    f1_f2_to_metric_gt = { (f1, f2): metric for f1, f2, metric in zip(df_gt['f1'], df_gt['f2'], df_gt['metric']) }
    
    df_fmh = pd.read_csv(f"ecoli/fmh_{metric}", delim_whitespace=True, header=None)
    df_fmh.columns = ['f1', 'f2', 'metric']
    f1_f2_to_metric_fmh = { (f1, f2): metric for f1, f2, metric in zip(df_fmh['f1'], df_fmh['f2'], df_fmh['metric']) }
    
    df_mash = pd.read_csv(f"ecoli/mash_combined")
    f1_f2_to_metric_mash = { (f1, f2): metric for f1, f2, metric in zip(df_mash['file1'], df_mash['file2'], df_mash[metric]) }
    
    true_metrics = []
    fmh_values = []
    mash_values = []
    
    for f1, f2 in f1_f2_to_metric_gt.keys():
        true_metrics.append(f1_f2_to_metric_gt[(f1, f2)])
        fmh_values.append(f1_f2_to_metric_fmh[(f1, f2)])
        mash_values.append(f1_f2_to_metric_mash[(f1, f2)])
    

    # show a line y = x
    ax.plot([0, 1], [0, 1], color='grey', linestyle='--', linewidth=1)

    # plot mash and frackmc ecoli values againts the true values in a scatter plot
    ax.scatter(true_metrics, mash_values, label='Mash', color=mash_color, alpha=0.45, marker='s')
    ax.scatter(true_metrics, fmh_values, label='frac-kmc', color=frackmc_color, alpha=0.45, marker='o')

    if metric == 'cosine':
        metric = 'Cosine'
    elif metric == 'braycurtis':
        metric = 'Bray-Curtis'

    # add labels
    ax.set_xlabel(f"True {metric} (Ecoli)")
    ax.set_ylabel(f"Est. {metric} (Ecoli)")

    # add legend
    ax.legend()

    # flip order of legends
    handles, labels = ax.get_legend_handles_labels()
    #ax.legend(handles[::-1], labels[::-1])

    # set legend's alpha=1
    for lh in ax.legend().legendHandles:
        lh.set_alpha(1)

    # tight layout
    plt.tight_layout()

    # add legend
    mash_patch = mpatches.Patch(facecolor=mash_color, label='Mash')
    frackmc_patch = mpatches.Patch(facecolor=frackmc_color, label='frac-kmc')
    ax.legend(handles=[frackmc_patch, mash_patch])



def plot_violins(ax, metric):
    df_gt = pd.read_csv(f"ecoli/ground_truth_{metric}", delim_whitespace=True, header=None)
    df_gt.columns = ['f1', 'f2', 'metric']
    f1_f2_to_metric_gt = { (f1, f2): metric for f1, f2, metric in zip(df_gt['f1'], df_gt['f2'], df_gt['metric']) }
    
    df_fmh = pd.read_csv(f"ecoli/fmh_{metric}", delim_whitespace=True, header=None)
    df_fmh.columns = ['f1', 'f2', 'metric']
    f1_f2_to_metric_fmh = { (f1, f2): metric for f1, f2, metric in zip(df_fmh['f1'], df_fmh['f2'], df_fmh['metric']) }
    
    df_mash = pd.read_csv(f"ecoli/mash_combined")
    f1_f2_to_metric_mash = { (f1, f2): metric for f1, f2, metric in zip(df_mash['file1'], df_mash['file2'], df_mash[metric]) }
    
    errors_mash_ecoli = []
    errors_frackmc_ecoli = []
    true_metrics_ecoli = []
    
    for f1, f2 in f1_f2_to_metric_gt.keys():
        errors_mash_ecoli.append(-f1_f2_to_metric_gt[(f1, f2)] + f1_f2_to_metric_mash[(f1, f2)])
        errors_frackmc_ecoli.append(-f1_f2_to_metric_gt[(f1, f2)] + f1_f2_to_metric_fmh[(f1, f2)])
        true_metrics_ecoli.append(f1_f2_to_metric_gt[(f1, f2)])
        
    df_gt = pd.read_csv(f"hmp/ground_truth_{metric}", delim_whitespace=True, header=None)
    df_gt.columns = ['f1', 'f2', 'metric']
    f1_f2_to_metric_gt = { (f1, f2): metric for f1, f2, metric in zip(df_gt['f1'], df_gt['f2'], df_gt['metric']) }
    
    df_fmh = pd.read_csv(f"hmp/fmh_{metric}", delim_whitespace=True, header=None)
    df_fmh.columns = ['f1', 'f2', 'metric']
    f1_f2_to_metric_fmh = { (f1, f2): metric for f1, f2, metric in zip(df_fmh['f1'], df_fmh['f2'], df_fmh['metric']) }
    
    df_mash = pd.read_csv(f"hmp/mash_combined")
    f1_f2_to_metric_mash = { (f1, f2): metric for f1, f2, metric in zip(df_mash['file1'], df_mash['file2'], df_mash[metric]) }
    
    errors_mash_hmp = []
    errors_frackmc_hmp = []
    true_metrics_hmp = []
    
    for f1, f2 in f1_f2_to_metric_gt.keys():
        errors_mash_hmp.append(-f1_f2_to_metric_gt[(f1, f2)] + f1_f2_to_metric_mash[(f1, f2)])
        errors_frackmc_hmp.append(-f1_f2_to_metric_gt[(f1, f2)] + f1_f2_to_metric_fmh[(f1, f2)])
        true_metrics_hmp.append(f1_f2_to_metric_gt[(f1, f2)])
        

    errors_mash_ecoli_pctg = [ 100*x/t for x, t in zip(errors_mash_ecoli, true_metrics_ecoli) ]
    errors_frackmc_ecoli_pctg = [ 100*x/t for x, t in zip(errors_frackmc_ecoli, true_metrics_ecoli) ]
    
    errors_mash_hmp_pctg = [ 100*x/t for x, t in zip(errors_mash_hmp, true_metrics_hmp) ]
    errors_frackmc_hmp_pctg = [ 100*x/t for x, t in zip(errors_frackmc_hmp, true_metrics_hmp) ]

    datasets = ['Ecoli', 'HMP']

    # Plotting
    #fig, ax = plt.subplots()
    

    #bplot1 = plt.boxplot([errors_mash_ecoli_pctg, errors_frackmc_ecoli_pctg], labels=['Mash', 'FrackMC'], notch=True, patch_artist=True, positions=[0.9, 1.1], showfliers=False)
    vplot1 = ax.violinplot([errors_mash_ecoli_pctg, errors_frackmc_ecoli_pctg], positions=[0.9, 1.1], showextrema=False, showmedians=False, widths=0.25)
    #bplot2 = plt.boxplot([errors_mash_hmp_pctg, errors_frackmc_hmp_pctg], labels=['Mash', 'FrackMC'], notch=True, patch_artist=True, positions=[1.4, 1.6], showfliers=False)
    vplot2 = ax.violinplot([errors_mash_hmp_pctg, errors_frackmc_hmp_pctg], positions=[1.9, 2.1], showextrema=False, showmedians=False, widths=0.25)

    colors = [mash_color, frackmc_color, mash_color, frackmc_color]

    i = 0
    for patch in vplot1['bodies'] + vplot2['bodies']:
        patch.set_facecolor(colors[i])
        if i % 2 == 0:
            patch.set_alpha(0.99)
        else:
            patch.set_alpha(1)
        i += 1

    # add labels
    ax.set_xticks([1, 2])
    ax.set_xticklabels(datasets)
    
    if metric == 'cosine':
        loc = 'lower center'
    elif metric == 'braycurtis':
        loc = 'upper center'

    # add legend
    mash_patch = mpatches.Patch(facecolor=mash_color, label='Mash')
    frackmc_patch = mpatches.Patch(facecolor=frackmc_color, label='frac-kmc')
    ax.legend(handles=[frackmc_patch, mash_patch], loc=loc)

    if metric == 'cosine':
        metric = 'Cosine'
    elif metric == 'braycurtis':
        metric = 'Bray-Curtis'

    # add y-axis label
    ax.set_ylabel(f"Error (%, {metric})")
    
    # restrict y axis to -40% to 40%
    #ax.set_ylim(-40, 40)

    # show y grid
    #ax.grid(axis="y")


def plot_ecoli_cputime_large(ax):
    df = pd.read_csv("ecoli_benchmarks_for_fmh_compiled")
    frackmc_times_complete = df['complete_run_cputime'].tolist()
    frackmc_times_sketch_only = df['sketch_only_cputime'].tolist()
    
    df = pd.read_csv("ecoli_benchmarks_for_mash_compiled")
    mash_times_complete = df['complete_run_cputime'].tolist()
    mash_times_sketch_only = df['sketch_only_cputime'].tolist()

    index = df['num_genomes'].tolist()
    index = np.array(index)
    
    bar_width = 160
    
    # show bar plots for complete times
    ax.bar(index - bar_width/2, mash_times_complete, bar_width, label="Mash", color=mash_color, edgecolor='black')
    ax.bar(index + bar_width/2, frackmc_times_complete, bar_width, label="frac-kmc", color=frackmc_color, edgecolor='black')
    
    # add legend
    ax.legend()
    
    ax.bar(index - bar_width/2, mash_times_sketch_only, bar_width, label="Mash", color=mash_color_sketch_only, edgecolor='black')
    ax.bar(index + bar_width/2, frackmc_times_sketch_only, bar_width, label="frac-kmc", color=frackmc_color_sketch_only, edgecolor='black')
    
    # show line plots for sketch only times
    #ax.plot(index, mash_times_sketch_only, label="Mash (sketch only)", color=mash_color, marker = 's', markersize=7, linestyle='--')
    #ax.plot(index, frackmc_times_sketch_only, label="frac-kmc (sketch only)", color=frackmc_color, marker = 'o', markersize=7, linestyle='--')


    # add grid
    ax.grid(axis="y")

    ax.set_xticks(index)
    ax.set_xticklabels(index)

    # add labels
    ax.set_xlabel("Num. samples")

    # add y-axis label
    ax.set_ylabel("CPU time time (Ecoli, s)")

    
    #handles, labels = ax.get_legend_handles_labels()
    #ax.legend(handles[::-1], labels[::-1])

    # tight layout
    plt.tight_layout()


def plot_ecoli_walltime_large(ax):
    #fig, ax = plt.subplots()

    df = pd.read_csv("ecoli_benchmarks_for_fmh_compiled")
    frackmc_times_complete = df['complete_run_walltime'].tolist()
    frackmc_times_sketch_only = df['sketch_only_walltime'].tolist()
    
    df = pd.read_csv("ecoli_benchmarks_for_mash_compiled")
    mash_times_complete = df['complete_run_walltime'].tolist()
    mash_times_sketch_only = df['sketch_only_walltime'].tolist()

    index = df['num_genomes'].tolist()
    index = np.array(index)
    
    bar_width = 160
    
    # show bar plots for complete times
    ax.bar(index - bar_width/2, mash_times_complete, bar_width, label="Mash", color=mash_color, edgecolor='black')
    ax.bar(index + bar_width/2, frackmc_times_complete, bar_width, label="frac-kmc", color=frackmc_color, edgecolor='black')
    
    # add legend
    ax.legend()
    
    ax.bar(index - bar_width/2, mash_times_sketch_only, bar_width, label="Mash", color=mash_color_sketch_only, edgecolor='black')
    ax.bar(index + bar_width/2, frackmc_times_sketch_only, bar_width, label="frac-kmc", color=frackmc_color_sketch_only, edgecolor='black')
    
    # show line plots for sketch only times
    #ax.plot(index, mash_times_sketch_only, label="Mash (sketch only)", color=mash_color, marker = 's', markersize=7, linestyle='--')
    #ax.plot(index, frackmc_times_sketch_only, label="frac-kmc (sketch only)", color=frackmc_color, marker = 'o', markersize=7, linestyle='--')


    # add grid
    ax.grid(axis="y")

    ax.set_xticks(index)
    ax.set_xticklabels(index)

    # add labels
    ax.set_xlabel("Num. samples")

    # add y-axis label
    ax.set_ylabel("Wall-clock time (Ecoli, s)")

    
    #handles, labels = ax.get_legend_handles_labels()
    #ax.legend(handles[::-1], labels[::-1])

    # tight layout
    plt.tight_layout()

    #plt.show()


def plot_ecoli_walltime_small(ax):
    #fig, ax = plt.subplots()

    df = pd.read_csv("ecoli_small_runs_combined")
    
    mash_times = df[df["method"] == "mash"]["walltime"].tolist()
    simka_times = df[df["method"] == "simka"]["walltime"].tolist()
    frackmc_times = df[df["method"] == "frackmc"]["walltime"].tolist()

    bar_width = 6

    index = df['num_files'].tolist()[ :len(df['num_files'].tolist())//3 ]
    index = np.array(index)
    
    bar2 = ax.bar(index - bar_width, simka_times, bar_width, label="Simka", color=simka_color, edgecolor='black')
    bar3 = ax.bar(index, frackmc_times, bar_width, label="frac-kmc", color=frackmc_color, edgecolor='black')
    bar1 = ax.bar(index + bar_width, mash_times, bar_width, label="Mash", color=mash_color, edgecolor='black')
    

    # add grid
    ax.grid(axis="y")

    # write a letter X at 125
    ax.text(125-bar_width, 0, "X", fontsize=13, ha='center', va='bottom', color='red')

    ax.set_xticks(index)
    ax.set_xticklabels(index)

    # add labels
    ax.set_xlabel("Num. samples")

    # add y-axis label
    ax.set_ylabel("Wall-clock time (Ecoli, s)")

    # add legend
    ax.legend()

    # tight layout
    plt.tight_layout()

    #plt.show()


def plot_hmp_walltime_small(ax):
    #fig, ax = plt.subplots()

    df = pd.read_csv("hmp_small_runs_combined")
    
    mash_times = df[df["method"] == "mash"]["walltime"].tolist()
    simka_times = df[df["method"] == "simka"]["walltime"].tolist()
    frackmc_times = df[df["method"] == "frackmc"]["walltime"].tolist()

    bar_width = 6

    index = df['num_files'].tolist()[ :len(df['num_files'].tolist())//3 ]
    index = np.array(index)
    
    bar2 = ax.bar(index - bar_width, simka_times, bar_width, label="Simka", color=simka_color, edgecolor='black')
    bar3 = ax.bar(index, frackmc_times, bar_width, label="frac-kmc", color=frackmc_color, edgecolor='black')
    bar1 = ax.bar(index + bar_width, mash_times, bar_width, label="Mash", color=mash_color, edgecolor='black')
    

    # add grid
    ax.grid(axis="y")

    # write a letter X at 125
    ax.text(125-bar_width, 0, "X", fontsize=13, ha='center', va='bottom', color='red')

    ax.set_xticks(index)
    ax.set_xticklabels(index)

    # add labels
    ax.set_xlabel("Num. samples")

    # add y-axis label
    ax.set_ylabel("Wall-clock time (HMP, s)")

    # add legend to top left
    ax.legend(loc='upper left')

    # tight layout
    plt.tight_layout()

    #plt.show()


def plot_hmp_walltime_large(ax):
    df = pd.read_csv("hmp_benchmarks_for_fmh_compiled")
    frackmc_times = df['complete_run_walltime'].tolist()
    
    df = pd.read_csv("hmp_benchmarks_for_mash_compiled")
    mash_times = df['complete_run_walltime'].tolist()

    index = df['num_genomes'].tolist()
    index = np.array(index)
    
    # show times for frac-kmc
    ax.plot(index, frackmc_times, label="frac-kmc", color=frackmc_color, marker = 'o', markersize=7, linestyle='-')

    # show times for mash
    ax.plot(index, mash_times, label="Mash", color=mash_color, marker = 's', markersize=7, linestyle='-')

    # add grid
    ax.grid(axis="y")

    ax.set_xticks(index)
    ax.set_xticklabels(index)

    # add labels
    ax.set_xlabel("Num. samples")

    # add y-axis label
    ax.set_ylabel("Wall-clock time (HMP, s)")

    # add legend
    ax.legend()

    # tight layout
    plt.tight_layout()

    #plt.show()


def plot_hmp_cputime_large(ax):
    df = pd.read_csv("hmp_benchmarks_for_fmh_compiled")
    frackmc_times = df['complete_run_cputime'].tolist()
    
    df = pd.read_csv("hmp_benchmarks_for_mash_compiled")
    mash_times = df['complete_run_cputime'].tolist()

    index = df['num_genomes'].tolist()
    index = np.array(index)
    
    # show times for frac-kmc
    ax.plot(index, frackmc_times, label="frac-kmc", color=frackmc_color, marker = 'o', markersize=7, linestyle='-')

    # show times for mash
    ax.plot(index, mash_times, label="Mash", color=mash_color, marker = 's', markersize=7, linestyle='-')

    # add grid
    ax.grid(axis="y")

    ax.set_xticks(index)
    ax.set_xticklabels(index)

    # add labels
    ax.set_xlabel("Num. samples")

    # add y-axis label
    ax.set_ylabel("CPU time (HMP, s)")

    # add legend
    ax.legend()

    # tight layout
    plt.tight_layout()

    #plt.show()



if __name__ == "__main__":
    # use aptos font
    plt.rcParams["font.family"] = "Arial"

    # set font size to 9
    plt.rcParams.update({'font.size': 10})


    fig, axs = plt.subplots(2, 4, figsize=(10, 4))

    # use ggplot style
    #plt.style.use('ggplot')

    plot_ecoli_walltime_small(axs[0, 0])
    plot_violins(axs[0, 1], 'cosine')
    plot_actual_errors_ecoli(axs[0, 2], 'cosine')
    plot_actual_errors_hmp(axs[0, 3], 'cosine')
    
    plot_hmp_walltime_small(axs[1, 0])
    plot_violins(axs[1, 1], 'braycurtis')
    plot_actual_errors_ecoli(axs[1, 2], 'braycurtis')
    plot_actual_errors_hmp(axs[1, 3], 'braycurtis')
    
    plt.tight_layout()
    plt.savefig('small_runs_and_errors.pdf', format='pdf')
    
    
    
    plt.rcParams.update({'font.size': 9})
    
    # clear the figure
    plt.clf()
    
    fig, axs = plt.subplots(2, 3, figsize=(8, 4))
    plot_ecoli_walltime_large(axs[0, 0])
    plot_ecoli_cputime_large(axs[0, 1])
    
    plot_hmp_walltime_large(axs[1, 0])
    plot_hmp_cputime_large(axs[1, 1])
    
    plt.tight_layout()
    plt.show()
    