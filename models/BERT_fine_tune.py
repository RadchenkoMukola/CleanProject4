
import gc
import pandas as pd
import torch
from torch.utils.data import Dataset
from transformers import BertTokenizer, BertForSequenceClassification, Trainer, TrainingArguments
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import numpy as np
from sklearn.metrics import accuracy_score, f1_score
import os
import pickle
import os

os.chdir("..")
# ========== 1. Clear Memory ==========
torch.cuda.empty_cache()
gc.collect()

# ========== 2. Load Dataset ==========
df = pd.read_csv("data/allsides_news_marked_2.csv")

# ========== 3. Map Labels ==========
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

# ========== 6. Tokenizer ==========
tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
max_length = 512

def stepwise_tokenize_with_cache(texts, tokenizer, max_length=512, cache_path="encodings/bert_cache.pkl"):
    if os.path.exists(cache_path) and os.path.getsize(cache_path) > 0:
        print(f"[✓] Cache found: {cache_path}")
        with open(cache_path, "rb") as f:
            return pickle.load(f)

    print(f"[⏳] Tokenizing... {cache_path}")
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
            print(f"[!] Error at {i}:", text[:100], "|", e)

        if i % 100 == 0:
            torch.cuda.empty_cache()
            gc.collect()

    encodings = {'input_ids': input_ids, 'attention_mask': attention_mask}
    with open(cache_path, "wb") as f:
        pickle.dump(encodings, f)
        print(f"[💾] Saved cache: {cache_path}")
    return encodings

train_encodings = stepwise_tokenize_with_cache(train_texts, tokenizer, max_length, "encodings/bert_train_cache.pkl")
val_encodings = stepwise_tokenize_with_cache(val_texts, tokenizer, max_length, "encodings/bert_val_cache.pkl")

# ========== 7. Dataset ==========
class NewsDataset(Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels
    def __len__(self):
        return len(self.labels)
    def __getitem__(self, idx):
        return {
            'input_ids': torch.tensor(self.encodings['input_ids'][idx]),
            'attention_mask': torch.tensor(self.encodings['attention_mask'][idx]),
            'labels': torch.tensor(self.labels[idx]),
        }

train_dataset = NewsDataset(train_encodings, train_labels)
val_dataset = NewsDataset(val_encodings, val_labels)

# ========== 8. Model ==========
model = BertForSequenceClassification.from_pretrained("results_bert/checkpoint-1845", num_labels=3)

# ========== 9. Training Args ==========
training_args = TrainingArguments(
    output_dir='./results_bert',
    eval_strategy="epoch",
    save_strategy="epoch",
    num_train_epochs=10,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=16,
    warmup_steps=50,
    weight_decay=0.01,
    logging_dir='./logs_bert',
    logging_steps=100,
    load_best_model_at_end=True,
    metric_for_best_model="accuracy",
    fp16=False,
    learning_rate=1e-5

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
trainer.train(resume_from_checkpoint="results_bert/checkpoint-1845")
