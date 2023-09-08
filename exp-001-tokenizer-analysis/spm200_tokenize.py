from transformers import NllbTokenizer
import datasets
import torch
import tqdm

LANGS = [
    "af-ZA", "am-ET", "ar-SA", "az-AZ", "bn-BD", "ca-ES", "zh-CN", "zh-TW", "da-DK", "de-DE", "el-GR", "en-US", "es-ES", "fa-IR", "fi-FI", "fr-FR", "he-IL", "hu-HU", "hy-AM", "id-ID", "is-IS", "it-IT", "ja-JP", "jv-ID", "ka-GE", "km-KH", "ko-KR", "lv-LV", "mn-MN", "ms-MY", "my-MM", "nb-NO", "nl-NL", "pl-PL", "pt-PT", "ro-RO", "ru-RU", "sl-SL", "sq-AL", "sv-SE", "sw-KE", "hi-IN", "kn-IN", "ml-IN", "ta-IN", "te-IN", "th-TH", "tl-PH", "tr-TR", "ur-PK", "vi-VN", "cy-GB"
]

D = {}

for lang in tqdm.tqdm(LANGS):
    data = datasets.load_dataset("AmazonScience/massive", lang, split="train")
    tokenizer = NllbTokenizer.from_pretrained("facebook/nllb-200-distilled-600M")
    D[lang] = list(map(len, tokenizer(data['utt'])['input_ids']))

torch.save(D, "spm200_tokenizer_MASSIVE.pt")


