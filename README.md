# Open-nllb-misc
Tokenization analysis for the 204 languages on [FLORES-200](https://huggingface.co/datasets/facebook/flores) (based on https://github.com/yongzx/open-nllb-misc.git)

> "*The creation of FLORES-200 doubles the existing language coverage of FLORES-101. Given the nature of the new languages, which have less standardization and require more specialized professional translations, the verification process became more complex. This required modifications to the translation workflow. FLORES-200 has several languages which were not translated from English. Specifically, several languages were translated from Spanish, French, Russian and Modern Standard Arabic. Moreover, FLORES-200 also includes two script alternatives for four languages. FLORES-200 consists of translations from 842 distinct web articles, totaling 3001 sentences. These sentences are divided into three splits: dev, devtest, and test (hidden). On average, sentences are approximately 21 words long.*"


## 1. Install
```bash
git clone https://github.com/isaacnicolas/open-nllb-misc.git
cd open-nllb-misc/exp-001-tokenizer-analysis
pip install -r requirements.txt
```

## 2. Usage
```bash
python spm200_tokenize.py
python spm200_analysis.py
```
First, the **spm200_tokenize.py** script loads the FLORES-200 dataset and uses the *facebook/nllb-200-distilled-600M* Sentencepiece tokenizer on the sentences (997 per language).  
Then the **spm200_analysis.py** script gets stats for the token length of sentences for each language (mean, std, median, min and max) and writes it in a .txt file. The script also saves the FLORES200.tok.csv file containing the tokenized sentences and their counts. Finally, distribution plots of groups of 10 languages grouped by ascending mean are saved as .png.

```plaintext
ind_Latn - 29.15 ± 9.31 (median: 28.0 min: 8, max: 72)
zsm_Latn - 30.13 ± 9.70 (median: 29.0 min: 10, max: 69)
eng_Latn - 30.99 ± 9.49 (median: 30.0 min: 9, max: 80)
jpn_Jpan - 31.39 ± 10.52 (median: 30.0 min: 8, max: 78)
jav_Latn - 31.94 ± 10.24 (median: 31.0 min: 10, max: 79))
```

<p align="center">
  <img src="exp-001-tokenizer-analysis\token_len_plots\spm200_tokenizer_FLORES200_group10.png" alt="Alt Text" width="some-width"/>
</p>