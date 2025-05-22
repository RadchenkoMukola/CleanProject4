from transformers import BertTokenizer, BertForSequenceClassification, LongformerForSequenceClassification, \
    LongformerTokenizer, RobertaForSequenceClassification, RobertaTokenizer
import os

os.chdir("..")
# Шлях до останнього чекпоінта (модель)
checkpoint_path = "results_longformer_balanced/checkpoint-9580"

tokenizer_path = "allenai/longformer-base-4096"

# Куди зберігати все разом
save_path = "LF_bias_v2_each"

# Завантажити модель
model = LongformerForSequenceClassification.from_pretrained(checkpoint_path)
#model = BertForSequenceClassification.from_pretrained(checkpoint_path)
#model = RobertaForSequenceClassification.from_pretrained(checkpoint_path)
# Завантажити tokenizer
tokenizer = LongformerTokenizer.from_pretrained(tokenizer_path)
#tokenizer = RobertaTokenizer.from_pretrained("roberta-large")
#tokenizer = BertTokenizer.from_pretrained("bert-base-uncased",use_fast=True)
# Зберегти модель і токенайзер
model.save_pretrained(save_path)
tokenizer.save_pretrained(save_path)

print(f"✅ Модель та токенайзер збережено в: {save_path}")
