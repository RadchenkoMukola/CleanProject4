import gc
import pandas as pd
import torch
from torch.utils.data import Dataset
from transformers import BertTokenizer, BertForSequenceClassification, Trainer, TrainingArguments
from sklearn.model_selection import train_test_split
import numpy as np
from sklearn.metrics import accuracy_score, f1_score
import os
import pickle

# ========== 1. Clear Memory ==========
torch.cuda.empty_cache()
gc.collect()

# ========== 2. Load Dataset ==========
df = pd.read_csv("data/allsides_news_standardized.csv")

# ========== 3. Keep Original Labels ==========
# label values: -1 (left), 0 (center), 1 (right)
labels = df["label"].tolist()
texts = df["text"].astype(str).tolist()

# ========== 4. Train-Test Split ==========
train_texts, val_texts, train_labels, val_labels = train_test_split(
    texts,
    labels,
    test_size=0.2,
    random_state=42,
    stratify=labels
)

# ========== 5. Tokenizer ==========
tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
max_length = 512

def stepwise_tokenize_with_cache(texts, tokenizer, max_length=512, cache_path="cache.pkl"):
    if os.path.exists(cache_path) and os.path.getsize(cache_path) > 0:
        with open(cache_path, "rb") as f:
            return pickle.load(f)

    input_ids, attention_mask = [], []
    for i, text in enumerate(texts):
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
            torch.cuda.empty_cache()
            gc.collect()

    encodings = {'input_ids': input_ids, 'attention_mask': attention_mask}
    with open(cache_path, "wb") as f:
        pickle.dump(encodings, f)
    return encodings

train_encodings = stepwise_tokenize_with_cache(train_texts, tokenizer, max_length, "encodings/train_cache.pkl")
val_encodings = stepwise_tokenize_with_cache(val_texts, tokenizer, max_length, "encodings/val_cache.pkl")

# ========== 6. Custom Dataset ==========
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
            'labels': torch.tensor(self.labels[idx] + 1),  # Shift to 0,1,2 for internal loss
        }

train_dataset = NewsDataset(train_encodings, train_labels)
val_dataset = NewsDataset(val_encodings, val_labels)

# ========== 7. Load and Freeze BERT ==========
model = BertForSequenceClassification.from_pretrained("bert-base-uncased", num_labels=3)

# Freeze all layers except classifier head
for param in model.bert.parameters():
    param.requires_grad = False

# ========== 8. Compute Class Weights ==========
from sklearn.utils.class_weight import compute_class_weight

class_weights = compute_class_weight(class_weight='balanced', classes=[-1, 0, 1], y=train_labels)
class_weights_tensor = torch.tensor(class_weights, dtype=torch.float)

# ========== 9. Custom Trainer with Weighted Loss ==========
from transformers import Trainer
import torch.nn as nn

class WeightedTrainer(Trainer):
    def compute_loss(self, model, inputs, return_outputs=False):
        labels = inputs.get("labels")
        outputs = model(**inputs)
        logits = outputs.get("logits")
        loss_fct = nn.CrossEntropyLoss(weight=class_weights_tensor.to(model.device))
        loss = loss_fct(logits, labels)
        return (loss, outputs) if return_outputs else loss

# ========== 10. Training Args ==========
training_args = TrainingArguments(
    output_dir='./results_bert_frozen_weighted',
    eval_strategy="epoch",
    save_strategy="epoch",
    num_train_epochs=10,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=16,
    warmup_steps=50,
    weight_decay=0.01,
    logging_dir='./logs_bert_frozen_weighted',
    logging_steps=100,
    load_best_model_at_end=True,
    metric_for_best_model="accuracy",
    learning_rate=1e-5,
    fp16=False
)

# ========== 11. Metrics ==========
def compute_metrics(pred):
    labels = pred.label_ids - 1  # Map back to original (-1, 0, 1)
    preds = np.argmax(pred.predictions, axis=1) - 1
    acc = accuracy_score(labels, preds)
    f1 = f1_score(labels, preds, average='macro')
    return {"accuracy": acc, "f1": f1}

# ========== 12. Train ==========
trainer = WeightedTrainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
    compute_metrics=compute_metrics
)

torch.cuda.empty_cache()
gc.collect()
trainer.train()
