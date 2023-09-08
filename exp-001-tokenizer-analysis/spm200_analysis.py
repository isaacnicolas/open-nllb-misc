import torch
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

D = torch.load("spm200_tokenizer_MASSIVE.pt")
# get statistics for each language
stats = list()
for lang in D.keys():
    stats.append([np.mean(D[lang]), np.std(D[lang]), lang, min(D[lang]), max(D[lang])])
stats.sort(key=lambda x: x[0])
for stat in stats:
    print(f"{stat[2]} - {stat[0]:.2f} ± {stat[1]:.2f} (min: {stat[3]}, max: {stat[4]})")
assert False
# visualize
lengths, langs = [], []
for lang in D.keys():
    langs.append(lang)
    lengths.append(D[lang])
df = pd.DataFrame(lengths).T
df.columns = langs # each column is for a language
sns.set_theme(style="whitegrid")
sns.displot(df, kind="kde")
plt.xlabel("Number of tokens per utterance")
plt.savefig("spm200_tokenizer_MASSIVE.png", bbox_inches='tight')