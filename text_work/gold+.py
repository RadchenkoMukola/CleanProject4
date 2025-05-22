
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

# Фільтрація та обробка тексту
df["text"] = df["text"].astype(str)
df = df[df["text"].str.split().str.len() > 50]
df["text"] = df["text"].apply(lambda x: truncate_by_words(x, max_words=1024))

# 4. Збереження попередньо обробленого датасету
df.to_csv("allsides_news_standardized.csv", index=False)
with open("allsides_news_standardized.json", "w", encoding="utf-8") as f:
    json.dump(df[["title", "url", "text", "label"]].to_dict(orient="records"), f, ensure_ascii=False, indent=4)

# 5. Балансування шляхом апсемплінгу (додавання прикладів)
print("Початковий розподіл міток:")
print(df["label"].value_counts())

df_left = df[df.label == -1]
df_center = df[df.label == 0]
df_right = df[df.label == 1]

max_len = max(len(df_left), len(df_center), len(df_right))

df_balanced = pd.concat([
    resample(df_left, replace=True, n_samples=max_len, random_state=42),
    resample(df_center, replace=True, n_samples=max_len, random_state=42),
    resample(df_right, replace=True, n_samples=max_len, random_state=42)
]).sample(frac=1).reset_index(drop=True)

# Збереження збалансованого датасету
df_balanced.to_csv("allsides_news_marked.csv", index=False)
with open('allsides_news_marked.json', 'w', encoding='utf-8') as f:
    json.dump(df_balanced[["title", "url", "text", "label"]].to_dict(orient="records"), f, ensure_ascii=False, indent=2)

print("Розподіл після апсемплінгу:")
print(df_balanced["label"].value_counts())