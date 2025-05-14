"""
File for extracting some results from the jsons of the human evaluation dashboard.
"""

from collections import defaultdict
import json
from itertools import product
import matplotlib.pyplot as plt
import numpy as np
import os
import seaborn as sns
from scipy.stats import bootstrap, binomtest

MODELS = ["ours", "ref", "dpo"] 
FORMATTED_MODELS = ["Ours", "Reference", "DPO"]

def read_jsons(folder: str) -> list:
    json_files = [f for f in os.listdir(folder)]
    json_data_list = []
    for file in json_files:
        with open(os.path.join(folder, file), 'r') as f:
            json_data_list += json.load(f)['preferences']

    return json_data_list

def get_win_rate_dict(json_data_list: list) -> dict:
    win_rate_dict = defaultdict(lambda: defaultdict(list))
    overall_win_rate_dict = defaultdict(list)

    # Group entries by image_id
    image_groups = defaultdict(list)
    for entry in json_data_list:
        image_groups[entry['image_id']].append(entry)
    
    # Calculate win rates for each image and pair
    for image_id, entries in image_groups.items():
        # Initialize counters for this image
        image_wins = defaultdict(lambda: defaultdict(int))
        image_total = defaultdict(lambda: defaultdict(int))
        
        for entry in entries:
            preference = entry["preference"]
            winner = entry[f"model_{preference}"]
            loser = entry[f"model_{1 - preference}"]
            
            image_wins[winner][loser] += 1
            image_total[winner][loser] += 1
            image_total[loser][winner] += 1
        
        # Calculate win rates for each pair for this image
        for model_a in MODELS:
            for model_b in MODELS:
                if image_total[model_a][model_b] > 0:
                    win_rate_dict[model_a][model_b].append(image_wins[model_a][model_b] / image_total[model_a][model_b] * 100)
                    overall_win_rate_dict[model_a].append(image_wins[model_a][model_b] / image_total[model_a][model_b] * 100)
        
    win_rate_dict_avg = defaultdict(lambda: defaultdict(lambda: 50.0))  # Default to 50.0 for same-model comparisons
    for model_a, model_b in product(MODELS, repeat=2):
        if model_a != model_b:
            win_rate_dict_avg[model_a][model_b] = np.mean(win_rate_dict[model_a][model_b])

    overall_win_rate_dict_avg = defaultdict(lambda: 50.0)  # Default to 50.0 for same-model comparisons
    for model in MODELS:
        overall_win_rate_dict_avg[model] = np.mean(overall_win_rate_dict[model])
    
    return win_rate_dict, overall_win_rate_dict, win_rate_dict_avg, overall_win_rate_dict_avg

def get_latex_table(win_rate_dict: dict, overall_wins_dict: dict) -> str:
    latex = []
    latex.append("\\begin{table}[ht]")
    latex.append("\\centering")
    latex.append("\\rowcolors{2}{white}{NeutralColor}")
    latex.append("\\begin{tabular}{c" + "|c" * len(MODELS) + "|c}")
    latex.append("\\toprule")

    # Header row
    header = "\\textbf{Model} & " + " & ".join(f"\\textbf{{{m}}}" for m in FORMATTED_MODELS) + " & \\textbf{Overall} \\\\"
    latex.append(header)
    latex.append("\\midrule")

    # Data rows
    for model_a, formatted_model_a in zip(MODELS, FORMATTED_MODELS):
        row = [f"\\textbf{{{formatted_model_a}}}"]
        for model_b in MODELS:
            if model_a == model_b:
                cell = "--"
            else:
                rate = win_rate_dict[model_a][model_b]
                color = "DarkWinText" if rate > 50 else "DarkLoseText" if rate < 50 else "black"
                cell = f"\\textcolor{{{color}}}{{{rate:.2f}\\%}}"
            row.append(cell)

        rate = overall_wins_dict[model_a]
        color = "WinColor" if rate > 50 else "LoseColor" if rate < 50 else "NeutralColor"
        overall = f"\\cellcolor{{{color}}}\\textbf{{{rate:.2f}\\%}}"
        row.append(overall)
        latex.append(" & ".join(row) + " \\\\")

    latex.append("\\bottomrule")
    latex.append("\\end{tabular}")
    latex.append("\\caption{Win rates between different models (row model vs column model).}")
    latex.append("\\label{tab:win_rates}")
    latex.append("\\end{table}")

    return "\n".join(latex)


def generate_win_rate_histograms(json_data_list: list):
    # Group entries by image_id
    image_groups = defaultdict(list)
    for entry in json_data_list:
        image_groups[entry['image_id']].append(entry)
    
    # Prepare data for histograms
    pair_results = {
        ('ours', 'ref'): [],
        ('ours', 'dpo'): [],
        ('ref', 'dpo'): []
    }
    
    # Calculate win rates for each image and pair
    for image_id, entries in image_groups.items():
        # Initialize counters for this image
        image_wins = defaultdict(lambda: defaultdict(int))
        image_total = defaultdict(lambda: defaultdict(int))
        
        for entry in entries:
            preference = entry["preference"]
            winner = entry[f"model_{preference}"]
            loser = entry[f"model_{1 - preference}"]
            
            image_wins[winner][loser] += 1
            image_total[winner][loser] += 1
            image_total[loser][winner] += 1
        
        # Calculate win rates for each pair for this image
        for pair in pair_results.keys():
            a, b = pair
            if image_total[a][b] > 0:
                win_rate = image_wins[a][b] / image_total[a][b] * 100
                pair_results[pair].append(win_rate)
    
    # Plot histograms
    plt.figure(figsize=(15, 5))
    
    for i, (pair, win_rates) in enumerate(pair_results.items()):
        plt.subplot(1, 3, i+1)
        plt.hist(win_rates, bins=20, range=(0, 100), alpha=0.7)
        plt.title(f'{pair[0]} vs {pair[1]}')
        plt.xlabel('Win Rate (%)')
        plt.ylabel('Number of Images')
        plt.grid(True)
    
    plt.tight_layout()
    plt.savefig('win_rate_histograms.png')
    plt.close()
    print("Saved win rate histograms to 'win_rate_histograms.png'")

def analyze_preference_deviation(json_data_list: list):
    # Group entries by image_id and model pair
    pair_groups = defaultdict(lambda: defaultdict(list))
    
    for entry in json_data_list:
        model_a = entry['model_0']
        model_b = entry['model_1']
        if model_a > model_b:  # Ensure consistent ordering
            model_a, model_b = model_b, model_a
            preference = 1 - entry['preference']
        else:
            preference = entry['preference']
        
        pair = (model_a, model_b)
        pair_groups[pair][entry['image_id']].append(preference)
    
    # Calculate statistics for each pair
    deviation_results = {}
    
    for pair, image_prefs in pair_groups.items():
        deviations = []
        for image_id, prefs in image_prefs.items():
            if len(prefs) > 1:  # Only consider images with multiple judgments
                mean_pref = np.mean(prefs)
                deviations.extend([abs(p - mean_pref) for p in prefs])
        
        if deviations:
            deviation_results[pair] = {
                'mean_deviation': np.mean(deviations),
                'max_deviation': np.max(deviations),
                'num_images': len(image_prefs),
                'num_judgments': sum(len(p) for p in image_prefs.values())
            }
    
    # Print results
    print("\nPreference Deviation Analysis:")
    print("=" * 50)
    for pair, stats in deviation_results.items():
        print(f"\nPair: {pair[0]} vs {pair[1]}")
        print(f"Images: {stats['num_images']}")
        print(f"Total judgments: {stats['num_judgments']}")
        print(f"Mean absolute deviation: {stats['mean_deviation']:.3f}")
        print(f"Max deviation: {stats['max_deviation']:.3f}")
    
    # Plot deviation distribution
    plt.figure(figsize=(10, 6))
    all_deviations = []
    pair_labels = []
    
    for pair, stats in deviation_results.items():
        deviations = []
        for image_id, prefs in pair_groups[pair].items():
            if len(prefs) > 1:
                mean_pref = np.mean(prefs)
                deviations.extend([abs(p - mean_pref) for p in prefs])
        
        if deviations:
            all_deviations.append(deviations)
            pair_labels.append(f"{pair[0]} vs {pair[1]}")
    
    plt.boxplot(all_deviations, labels=pair_labels)
    plt.title('Deviation in Preferences by Model Pair')
    plt.ylabel('Absolute Deviation from Mean Preference')
    plt.xticks(rotation=45)
    plt.grid(True)
    plt.tight_layout()
    plt.savefig('preference_deviation.png')
    plt.close()
    print("\nSaved preference deviation plot to 'preference_deviation.png'")

def plot_preference_variance(overall_win_rate_dict, output_dir="plots"):
    """
    Saves a bar plot of standard deviation of win rates per model (preference variance).
    """
    os.makedirs(output_dir, exist_ok=True)
    stds = [np.std(overall_win_rate_dict[m]) for m in MODELS]
    # print(stds)

    plt.figure(figsize=(8, 5))
    sns.barplot(x=MODELS, y=stds, palette="viridis", hue=MODELS, legend=False)
    plt.xticks(ticks=range(len(FORMATTED_MODELS)), labels=FORMATTED_MODELS)
    plt.title("Preference Variance (Standard Deviation of Wins per Image)")
    plt.ylabel("Std Dev of Win Rate")
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "preference_variance.png"))
    plt.close()

def plot_bootstrap_confidence_intervals(overall_win_rate_dict, output_dir="plots", n_resamples=5000):
    """
    Saves a bar plot of mean win rates with bootstrapped 95% confidence intervals.
    """
    os.makedirs(output_dir, exist_ok=True)
    ci_low, ci_high, means = [], [], []

    for model in MODELS:
        data = np.array(overall_win_rate_dict[model])
        means.append(np.mean(data))

        res = bootstrap((data,), np.mean, confidence_level=0.95,
                        n_resamples=n_resamples, method='percentile')
        ci_low.append(res.confidence_interval.low)
        ci_high.append(res.confidence_interval.high)

    yerr = [np.array(means) - np.array(ci_low), np.array(ci_high) - np.array(means)]

    plt.figure(figsize=(8, 5))
    colors = sns.color_palette("viridis", n_colors=len(MODELS))
    plt.bar(x=MODELS, height=means, yerr=yerr, capsize=0.1, tick_label=FORMATTED_MODELS, color=colors)
    plt.title("Bootstrapped Confidence Intervals for Win Rate per Model")
    plt.ylabel("Mean Win Rate")
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "bootstrap_confidence_intervals.png"))
    plt.close()

def plot_statistical_significance_matrix(win_rate_dict, output_dir="plots"):
    """
    Saves a heatmap of p-values from pairwise binomial tests between models.
    """
    os.makedirs(output_dir, exist_ok=True)
    p_matrix = np.zeros((len(MODELS), len(MODELS)))

    for i, m1 in enumerate(MODELS):
        for j, m2 in enumerate(MODELS):
            if i == j:
                p_matrix[i, j] = np.nan
            else:
                wins = win_rate_dict[m1][m2]
                total = len(wins)
                successes = sum(np.array(wins) > 0.5)
                p = binomtest(successes, total, p=0.5, alternative='greater').pvalue
                p_matrix[i, j] = p

    plt.figure(figsize=(8, 6))
    sns.heatmap(p_matrix, annot=True, fmt=".3f", xticklabels=FORMATTED_MODELS, yticklabels=FORMATTED_MODELS,
                cmap="coolwarm", mask=np.isnan(p_matrix))
    plt.title("P-value Heatmap (Binomial Test for Pairwise Preference)")
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "statistical_significance_matrix_.png"))
    plt.close()


def main():
    json_data_list = read_jsons("jsons")
    wr_dict, overall_wr_dict, wr_dict_avg, overall_wr_dict_avg = get_win_rate_dict(json_data_list)
    latex_table = get_latex_table(wr_dict_avg, overall_wr_dict_avg)
    print(latex_table)

    # generate_win_rate_histograms(json_data_list)
    # analyze_preference_deviation(json_data_list)

    plot_preference_variance(overall_wr_dict)
    plot_bootstrap_confidence_intervals(overall_wr_dict)
    plot_statistical_significance_matrix(wr_dict)


if __name__ == "__main__":
    main()
