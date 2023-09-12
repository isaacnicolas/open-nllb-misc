import torch

import seaborn as sns
import matplotlib.pyplot as plt

import pandas as pd
import numpy as np

import os
import shutil

# Load tokens length
D = torch.load("spm200_tokenizer_FLORES200.pt")

# Get statistics for each language
stats = list()
for lang in D.keys():
    stats.append([np.mean(D[lang]), np.std(D[lang]), np.median(D[lang]), lang, min(D[lang]), max(D[lang])])
stats.sort(key=lambda x: x[0])
for stat in stats:
    print(f"{stat[3]} - {stat[0]:.2f} ± {stat[1]:.2f} (median: {stat[2]} min: {stat[4]}, max: {stat[5]})")

# File path for the stats.txt
stats_file_path = os.path.join(os.getcwd(), "mean_token_lengths.txt")

# Write the stats to the file
with open(stats_file_path, "w") as file:
    for stat in stats:
        line = f"{stat[3]} - {stat[0]:.2f} ± {stat[1]:.2f} (median: {stat[2]} min: {stat[4]}, max: {stat[5]})\n"
        file.write(line)
        print(line)  # This will also print the line to the console as in your original code

# Visualize
token_plots_path = "token_len_plots"

# Check if the directory exists
if os.path.exists(token_plots_path):
    # Remove the directory and all its content
    shutil.rmtree(token_plots_path)
    print(f"Directory {token_plots_path} removed.")

os.mkdir(token_plots_path)

grouped_langs = [stats[i:i+10] for i in range(0, len(stats), 10)]

for idx, group in enumerate(grouped_langs):
    # Extract the languages from the current group
    langs_in_group = [entry[3] for entry in group]

    # Prepare data for visualization
    lengths_group, langs_group = [], []

    for lang in langs_in_group:
        langs_group.append(lang)
        lengths_group.append(D[lang])
    df_group = pd.DataFrame(lengths_group).T
    df_group.columns = langs_group

    # Create the plot
    sns.set_theme(style="whitegrid")
    sns.displot(df_group, kind="kde")
    plt.xlabel("Number of tokens per sentence")
    plt.title(f"Group {idx + 1}")
    plt.savefig(os.path.join(os.getcwd(),token_plots_path,f"spm200_tokenizer_FLORES200_group{idx + 1}.png"), bbox_inches='tight')
    plt.close()
