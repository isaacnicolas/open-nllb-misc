import torch
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd

D = torch.load("spm200_tokenizer_MASSIVE.pt")
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