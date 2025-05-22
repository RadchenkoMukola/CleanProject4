import gc
import pandas as pd
import torch
from torch.utils.data import Dataset
from transformers import LongformerTokenizer, LongformerForSequenceClassification, Trainer, TrainingArguments, \
    EarlyStoppingCallback
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import numpy as np
from sklearn.metrics import accuracy_score, f1_score
import os

os.chdir("..")
# ========== 1. Load Dataset ==========
torch.cuda.empty_cache()
gc.collect()

df = pd.read_csv("data/bbc_news_marked_v2.csv")

# ========== 2. Convert Score to Class ==========
def score_to_label(score):
    if score <= -0.33:
        return "Left"
    elif score >= 0.33:
        return "Right"
    else:
        return "Center"

df["label"] = df["bias_score_weighted"].apply(score_to_label)

label_encoder = LabelEncoder()
df["label_id"] = label_encoder.fit_transform(df["label"])  # Left=0, Center=1, Right=2

# ========== 3. Split ==========
df = df.dropna(subset=["text"])
train_texts, val_texts, train_labels, val_labels = train_test_split(
    df["text"].astype(str).tolist(), df["label_id"].tolist(), test_size=0.2, random_state=42
)

# ========== 4. Tokenization ==========
tokenizer = LongformerTokenizer.from_pretrained("allenai/longformer-base-4096", use_fast=True)
max_length = 1024

train_encodings = tokenizer(train_texts, truncation=True, padding='max_length', max_length=max_length)
val_encodings = tokenizer(val_texts, truncation=True, padding='max_length', max_length=max_length)
lens = [len(tokenizer.encode(t)) for t in df["text"].tolist()]
print("Середня довжина токенів:", np.mean(lens))
# ========== 5. Torch Dataset with Global Attention ==========
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
        # Додаємо global attention: лише на [CLS] (тобто перший токен — 1, решта — 0)
        global_attention = torch.zeros(max_length, dtype=torch.long)
        global_attention[0] = 1
        item['global_attention_mask'] = global_attention
        return item

train_dataset = NewsDataset(train_encodings, train_labels)
val_dataset = NewsDataset(val_encodings, val_labels)

# ========== 6. Load Model ==========
model = LongformerForSequenceClassification.from_pretrained("allenai/longformer-base-4096", num_labels=3)

# ========== 7. Training Args ==========
training_args = TrainingArguments(
    output_dir='./results',
    eval_strategy="epoch",
    save_strategy="epoch",
    num_train_epochs=5,
    per_device_train_batch_size=2,
    per_device_eval_batch_size=2,
    gradient_accumulation_steps=4,
    warmup_steps=50,
    weight_decay=0.01,
    logging_dir='./logs',
    logging_steps=100,
    load_best_model_at_end=True,
    metric_for_best_model="accuracy",
    fp16=True
)

# ========== 8. Metrics ==========
def compute_metrics(pred):
    labels = pred.label_ids
    preds = np.argmax(pred.predictions, axis=1)
    acc = accuracy_score(labels, preds)
    f1 = f1_score(labels, preds, average='macro')
    return {"accuracy": acc, "f1": f1}

# ========== 9. Trainer ==========
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
    compute_metrics=compute_metrics,
#    callbacks=[EarlyStoppingCallback(early_stopping_patience=5)]
)


torch.cuda.empty_cache()
gc.collect()
print(torch.cuda.memory_summary())
trainer.train()
