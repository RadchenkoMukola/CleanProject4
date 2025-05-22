import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import BertTokenizer, BertForSequenceClassification, LongformerTokenizer, LongformerForSequenceClassification,RobertaForSequenceClassification,RobertaTokenizer
from tqdm import tqdm
import gc
import os
import pickle

# ====== Налаштування ======
device = torch.device("cuda")
batch_size = 8
max_length = 512
chunk_size = 10000
start_chunk = 0

# ====== Завантаження моделі ======
model_path = "results_bert/checkpoint-3690"
model = BertForSequenceClassification.from_pretrained(model_path)
tokenizer = BertTokenizer.from_pretrained("bert-base-uncased",use_fast=True)
#model_path = "LongFormer_bias_1085_each"
#model = LongformerForSequenceClassification.from_pretrained(model_path)
#tokenizer = LongformerTokenizer.from_pretrained(model_path,use_fast=True)

#model_path = "RoBERTa_bias_1085_each"
#model = RobertaForSequenceClassification.from_pretrained(model_path)
#tokenizer = RobertaTokenizer.from_pretrained(model_path,use_fast=True)

model.to(device)
model.eval()

# ====== Завантаження даних ======
df = pd.read_csv("data/allsides_news_standardized.csv", encoding='utf-8')
df = df[df["text"].apply(lambda x: isinstance(x, str))].copy()
df = df[df["text"].str.strip() != ""]
df = df[df["text"].str.split().str.len() > 50]

def truncate_by_words(text, max_words=512):
    words = text.split()
    return " ".join(words[:max_words])

df["text"] = df["text"].apply(lambda x: truncate_by_words(x, max_words=max_length))
texts = df["text"].tolist()

# ====== Токенізація з кешем ======
def stepwise_tokenize_with_cache(texts, tokenizer, max_length=512, cache_path="pred_BERT_encodings.pkl"):
    if os.path.exists(cache_path) and os.path.getsize(cache_path) > 0:
        print(f"[✓] Завантажено кеш: {cache_path}")
        with open(cache_path, "rb") as f:
            return pickle.load(f)

    print("[⏳] Токенізую тексти...")
    input_ids, attention_mask = [], []
    for i, text in enumerate(texts):
        try:
            encoding = tokenizer(
                text,
                truncation=True,
                padding='max_length',
                max_length=max_length
            )
            input_ids.append(encoding['input_ids'])
            attention_mask.append(encoding['attention_mask'])
        except Exception as e:
            print(f"[!] Помилка в тексті {i}:", e)
            input_ids.append([0] * max_length)
            attention_mask.append([0] * max_length)

        if i % 500 == 0:
            print(f"[{i}/{len(texts)}] Tokenized...")
        if i % 100 == 0:
            torch.cuda.empty_cache()
            gc.collect()

    encodings = {'input_ids': input_ids, 'attention_mask': attention_mask}
    with open(cache_path, "wb") as f:
        pickle.dump(encodings, f)
        print(f"[💾] Збережено в: {cache_path}")

    return encodings

encodings = stepwise_tokenize_with_cache(texts, tokenizer, max_length=max_length)

# ====== Dataset ======
class PredictionDataset(Dataset):
    def __init__(self, encodings):
        self.encodings = encodings
    def __len__(self):
        return len(self.encodings['input_ids'])
    def __getitem__(self, idx):
        return {
            'input_ids': torch.tensor(self.encodings['input_ids'][idx]),
            'attention_mask': torch.tensor(self.encodings['attention_mask'][idx])
        }

# ====== Інференс по чанках ======
for i in range(start_chunk * chunk_size, len(texts), chunk_size):
    print(f"\n🧩 Обробка частини {i // chunk_size + 1}...")
    chunk_df = df.iloc[i:i + chunk_size].copy()
    chunk_encodings = {
        'input_ids': encodings['input_ids'][i:i + chunk_size],
        'attention_mask': encodings['attention_mask'][i:i + chunk_size]
    }

    dataset = PredictionDataset(chunk_encodings)
    dataloader = DataLoader(dataset, batch_size=batch_size, num_workers=0, pin_memory=True)

    chunk_preds = []
    chunk_probs = []

    with torch.no_grad():
        for batch in tqdm(dataloader, desc=f"🔍 Частина {i // chunk_size + 1}", ncols=100):
            batch = {k: v.to(device) for k, v in batch.items()}
            outputs = model(**batch)
            probs = torch.nn.functional.softmax(outputs.logits, dim=-1)
            preds = torch.argmax(probs, dim=-1)

            chunk_preds.extend(preds.cpu().numpy())
            chunk_probs.extend(probs.cpu().numpy())

            del batch, outputs, probs, preds
            torch.cuda.empty_cache()

    chunk_df["predicted_label"] = chunk_preds
    chunk_df["predicted_probs"] = chunk_probs
    chunk_df.to_csv(f"allsides_predictions_chunk_BERTv3_{i//chunk_size+1}.csv", index=False)

    del dataset, dataloader, chunk_preds, chunk_probs, chunk_df
    gc.collect()
    torch.cuda.empty_cache()

print("✅ Готово!")
