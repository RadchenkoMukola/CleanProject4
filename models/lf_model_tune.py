import gc
import pandas as pd
import torch
from torch.utils.data import Dataset
from transformers import LongformerTokenizer, LongformerForSequenceClassification, Trainer, TrainingArguments
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import numpy as np
from sklearn.metrics import accuracy_score, f1_score
import pickle
import os

os.chdir("..")
# ========== 1. Clear Memory ==========
torch.cuda.empty_cache()
gc.collect()

# ========== 2. Load Gold Standard Dataset ==========
df = pd.read_csv("data/allsides_news_marked_2.csv")
#df["text_len"] = df["text"].astype(str).apply(len)
#df = df[df["text_len"] > 50]

# ========== 3. Prepare Labels (-1, 0, 1 as Left, Center, Right) ==========
label_map = {-1: "Left", 0: "Center", 1: "Right"}
df["label_str"] = df["label"].map(label_map)

# ========== 4. Encode Labels ==========
label_encoder = LabelEncoder()
df["label_id"] = label_encoder.fit_transform(df["label_str"])  # 0=Left, 1=Center, 2=Right

# ========== 5. Train-Test Split ==========
train_texts, val_texts, train_labels, val_labels = train_test_split(
    df["text"].astype(str).tolist(),
    df["label_id"].tolist(),
    test_size=0.2,
    random_state=42
)

# ========== 6. Tokenization ==========
tokenizer = LongformerTokenizer.from_pretrained("allenai/longformer-base-4096", use_fast=False)
max_length = 1024

torch.cuda.empty_cache()
gc.collect()
# Функція для підрахунку довжини токенів
def get_token_lengths(texts, tokenizer, batch_size=64):
    token_lengths = []
    for i in range(0, len(texts), batch_size):
        batch_texts = texts[i:i+batch_size]
        try:
            encodings = tokenizer(batch_texts, truncation=False, padding=False)
            batch_lens = [len(input_ids) for input_ids in encodings["input_ids"]]
            token_lengths.extend(batch_lens)
        except Exception as e:
            print(f"[!] Проблема з батчем {i}-{i+batch_size}:", e)
            for text in batch_texts:
                try:
                    token_lengths.append(len(tokenizer.encode(text)))
                except:
                    token_lengths.append(0)

        if i % 500 == 0:
            print(f"[{i}/{len(texts)}] done...")
    return token_lengths

# ========== 7. Перевірка наявності збереженого файлу ==========
if os.path.exists("gold_marked_2.csv"):
    print("[✓] Файл 'gold_marked_2.csv' вже існує — пропускаємо токенізацію за розміром.")
    df = pd.read_csv("gold_marked_2.csv")
else:
    texts = df["text"].astype(str).tolist()
    df["token_len"] = get_token_lengths(texts, tokenizer)
    df = df[df["token_len"] <= 1024]

    print("Максимальна довжина токенів:", df["token_len"].max())
    print("Середня довжина токенів:", df["token_len"].mean())

    df.to_csv("gold_marked_2.csv", index=False)
    print("[✓] Збережено в 'gold_marked_2.csv'")


def stepwise_tokenize(texts, tokenizer, max_length=1024):
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

            if i % 500 == 0:
                print(f"[{i}/{len(texts)}] Tokenized...")

        except Exception as e:
            print(f"[!] Проблема з текстом {i}:", text[:100], "|", e)

        # Примусове очищення (коли GPU працює дуже тісно)
        if i % 100 == 0:
            torch.cuda.empty_cache()
            gc.collect()

    return {'input_ids': input_ids, 'attention_mask': attention_mask}


def stepwise_tokenize_with_cache(texts, tokenizer, max_length=1024, cache_path="encodings_cache.pkl"):
    # Якщо файл існує і не порожній — завантажуємо
    if os.path.exists(cache_path) and os.path.getsize(cache_path) > 0:
        print(f"[✓] Завантажую збережений файл: {cache_path}")
        with open(cache_path, "rb") as f:
            return pickle.load(f)

    # Інакше — токенізуємо і зберігаємо
    print(f"[⏳] Токенізую... Файл не знайдено або порожній: {cache_path}")
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

            if i % 500 == 0:
                print(f"[{i}/{len(texts)}] Tokenized...")

        except Exception as e:
            print(f"[!] Проблема з текстом {i}:", text[:100], "|", e)

        if i % 100 == 0:
            torch.cuda.empty_cache()
            gc.collect()

    encodings = {'input_ids': input_ids, 'attention_mask': attention_mask}

    with open(cache_path, "wb") as f:
        pickle.dump(encodings, f)
        print(f"[💾] Збережено токенізовані дані в: {cache_path}")

    return encodings

train_encodings = stepwise_tokenize_with_cache(train_texts, tokenizer, max_length)
#val_encodings = tokenizer(val_texts, truncation=True, padding='max_length', max_length=max_length)
val_encodings = stepwise_tokenize_with_cache(val_texts, tokenizer, max_length)

# ========== 7. Torch Dataset ==========
class NewsDataset(Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels
    def __len__(self):
        return len(self.labels)
    def __getitem__(self, idx):
        item = {
            'input_ids': torch.tensor(self.encodings['input_ids'][idx]),
            'attention_mask': torch.tensor(self.encodings['attention_mask'][idx]),
            'labels': torch.tensor(self.labels[idx]),
        }
        global_attention = torch.zeros(max_length, dtype=torch.long)
        global_attention[0] = 1
        item['global_attention_mask'] = global_attention
        return item



train_dataset = NewsDataset(train_encodings, train_labels)

val_dataset = NewsDataset(val_encodings, val_labels)

# ========== 8. Load Model ==========
model = LongformerForSequenceClassification.from_pretrained("results_confident/checkpoint-2211", num_labels=3)

# ========== 9. Training Args ==========
training_args = TrainingArguments(
    output_dir='./results_confident',
    eval_strategy="epoch",
    save_strategy="epoch",
    num_train_epochs=5,
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    gradient_accumulation_steps=4,
    warmup_steps=50,
    weight_decay=0.01,
    logging_dir='./logs_confident',
    logging_steps=100,
    load_best_model_at_end=True,
    metric_for_best_model="accuracy",
    fp16=True
)

# ========== 10. Metrics ==========
def compute_metrics(pred):
    labels = pred.label_ids
    preds = np.argmax(pred.predictions, axis=1)
    acc = accuracy_score(labels, preds)
    f1 = f1_score(labels, preds, average='macro')
    return {"accuracy": acc, "f1": f1}

# ========== 11. Trainer ==========
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
    compute_metrics=compute_metrics
)

# ========== 12. Train ==========
torch.cuda.empty_cache()
gc.collect()
print(torch.cuda.memory_summary())
#trainer.train()
trainer.train(resume_from_checkpoint="results_confident/checkpoint-2211")