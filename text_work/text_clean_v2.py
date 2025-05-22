import re
import html
import pandas as pd
import json
import os

os.chdir("..")

def clean_text(text):
    if pd.isna(text):
        return ""

    text = html.unescape(text)  # Декодування HTML-ентітетів (&nbsp;, &rsquo; → ', &copy; → © тощо)
    text = text.replace("\xa0", " ")  # Видаляємо неперервні пробіли
    text = re.sub(r'<.*?>', '', text)  # Видаляємо HTML-теги
    text = re.sub(r'[^a-zA-Z0-9\s.,!?;:\'\"()\[\]{}\-–—]', '', text)
    text = re.sub(r'\s+', ' ', text)  # Видаляємо зайві пробіли
    # Видаляємо зайві символи
    text = text.replace("\n", " ").replace("\r", "")  # заміняємо нові рядки
    # Видаляємо невидимі символи та непотрібні лапки
    text = re.sub(r"[\"\ /\\ ]", " ", text)  # заміняємо " / \ на пробіл
    # Заміняємо всі зайві пробіли на один
    text = re.sub(r'\s+', ' ', text).strip()

    return text

df = pd.read_csv("data/bbc_news_with_text.csv")
df["text"] = df["text"].apply(clean_text)
df.to_csv("bbc_news_cleaned_2.csv", index=False)
print("✅ Очищений файл збережено!")

output_json = df[["title", "url", "text"]].to_dict(orient="records")
with open("bbc_cleaned_news_2.json", "w", encoding="utf-8") as f:
    json.dump(output_json, f, ensure_ascii=False, indent=4)
print("✅ Файл bbc_cleaned_news_2.json збережено!")