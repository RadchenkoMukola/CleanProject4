import pandas as pd
from sklearn.utils import resample
import json
import os

os.chdir("..")
# 1. Завантаження файлу
df = pd.read_csv("data/allsides_news_with_text.csv")

# 2. Перейменування колонок
df = df.rename(columns={
    "bias_label": "label",
    "link": "url"
})

# 3. Мапінг текстових лейблів у числові
label_map = {
    "left": -1,
    "center": 0,
    "right": 1
}

df["label"] = df["label"].map(label_map)

def truncate_by_words(text, max_words=1024):
    words = text.split()
    return " ".join(words[:max_words])

# Спершу відфільтруй дуже короткі (якщо ще не зроблено)
df["text"] = df["text"].astype(str)
df = df[df["text"].str.split().str.len() > 50]

# Потім обріжемо довгі
df["text"] = df["text"].apply(lambda x: truncate_by_words(x, max_words=1024))

# 4. Збереження результату
df.to_csv("allsides_news_standardized.csv", index=False)

with open("allsides_news_standardized.json", "w", encoding="utf-8") as f:
    json.dump(df[["title", "url", "text", "label"]].to_dict(orient="records"), f, ensure_ascii=False, indent=4)
