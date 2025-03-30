# ===================================
# run_4_3_final.py -- 完整满足 4.3
# ===================================

import os
import torch
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm

from train import Arguments, train_m_models, load_trial_metrics, get_extrema_performance_steps

import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--model', type=str, default='lstm', choices=['lstm', 'gpt'], help='Model to use (lstm or gpt)')
args_cli = parser.parse_args()



# ============ CONFIG ==============
r_train_list = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
seeds = [0, 42]
exp_base = "4.3_LSTM"
log_dir = "./log4_3"
os.makedirs(log_dir, exist_ok=True)

metric_keys = ["Ltrain", "tf(Ltrain)", "Ltest", "tf(Ltest)", "Atrain", "tf(Atrain)", "Atest", "tf(Atest)"]

# ============ Data Store =============
summary_txt = open(os.path.join(log_dir, f"{exp_base}_summary.txt"), "w")
summary_tex = open(os.path.join(log_dir, f"{exp_base}_table.tex"), "w")

summary_tex.write("\\begin{tabular}{c|" + "c"*len(metric_keys) + "}\n")
summary_tex.write("rtrain & " + " & ".join(metric_keys) + " \\\\\n\\hline\n")

all_results = {}

# ======== TRAIN & COLLECT ========

exp_counter = 0
all_results = {}

for r_train in r_train_list:
    summary_txt.write(f"\n=== r_train = {r_train:.1f} ===\n")
    latex_row = [f"{r_train:.1f}"]
    all_metrics_seeds = []
    metrics_per_seed = {k: [] for k in metric_keys}

    args = Arguments()
    args.model = args_cli.model 
    args.r_train = r_train
    args.exp_name = f"{exp_base}_rtrain={r_train}"
    args.log_dir = log_dir

    # train M=2 models
    all_models_per_trials, all_metrics, all_checkpoint_paths = train_m_models(args, M=2, seeds=[0,42])

    for path in all_checkpoint_paths:
        # path is like log4_3/0 or log4_3/1
        trial_metrics = load_trial_metrics(path, args.exp_name)  # 自动去 path/exp_name.pth
        extrema = get_extrema_performance_steps(trial_metrics)
        all_metrics_seeds.append(trial_metrics)

        for k in metric_keys:
            metrics_per_seed[k].append(extrema[k])

    for k in metric_keys:
        arr = np.array(metrics_per_seed[k])
        mean, std = arr.mean(), arr.std()
        summary_txt.write(f"{k}: {mean:.4f} ± {std:.4f}\n")
        latex_row.append(f"{mean:.4f} ± {std:.4f}")

    summary_tex.write(" & ".join(latex_row) + " \\\\\n")
    all_results[r_train] = all_metrics_seeds



summary_txt.close()
summary_tex.write("\\end{tabular}\n")
summary_tex.close()
print("✅ Summary + LaTeX table saved")

# ============ 4.3 (a) Curve: Step → Metrics, Colorbar = r_train =============

print("✅ Drawing 4.3(a)")

colors = cm.viridis(np.linspace(0, 1, len(r_train_list)))
metric_names = ["train.loss", "test.loss", "train.accuracy", "test.accuracy"]
ylabels = ["Train Loss", "Validation Loss", "Train Accuracy", "Validation Accuracy"]

for metric_name, ylabel in zip(metric_names, ylabels):
    plt.figure(figsize=(8,6))
    for r_train, color in zip(r_train_list, colors):
        curves = []
        for metrics in all_results[r_train]:
            curves.append(metrics["train"]["loss"] if metric_name=="train.loss" else
                          metrics["test"]["loss"] if metric_name=="test.loss" else
                          metrics["train"]["accuracy"] if metric_name=="train.accuracy" else
                          metrics["test"]["accuracy"])
        steps = np.array(metrics["all_steps"])
        mean_curve = np.mean(np.stack(curves), axis=0)
        plt.plot(steps, mean_curve, label=f"r_train={r_train:.1f}", color=color)
    sm = plt.cm.ScalarMappable(cmap=cm.viridis, norm=plt.Normalize(min(r_train_list), max(r_train_list)))
    plt.gcf().colorbar(sm, ax=plt.gca(), label="r_train")
    plt.legend()
    plt.xlabel("Training Step")
    plt.ylabel(ylabel)
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(log_dir, f"{exp_base}_4.3a_{metric_name}.png"))
    plt.close()

# ============ 4.3 (b) Curve: r_train → Metrics, Errorbar = std =============

print("✅ Drawing 4.3(b)")

for idx, k in enumerate(metric_keys):
    plt.figure(figsize=(6,4))
    means, stds = [], []
    for r_train in r_train_list:
        vals = []
        for metrics in all_results[r_train]:
            extrema = get_extrema_performance_steps(metrics)
            vals.append(extrema[k])
        arr = np.array(vals)
        means.append(arr.mean())
        stds.append(arr.std())
    plt.errorbar(r_train_list, means, yerr=stds, fmt='-o', capsize=3)
    plt.xlabel("r_train")
    plt.ylabel(k)
    if "L" in k: plt.yscale("log")  # log-scale for Loss
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(log_dir, f"{exp_base}_4.3b_{k}.png"))
    plt.close()

print("🎉 All Done! (Summary, Table, 4.3(a), 4.3(b))")
