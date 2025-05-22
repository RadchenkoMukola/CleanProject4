import pandas as pd
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
import html
import json
import os

os.chdir("..")
# Завантажуємо дані
df = pd.read_csv("data/bbc_news_with_text.csv")

# Завантажуємо список стоп-слів (англійська мова)
nltk.download("stopwords")
stop_words = set(stopwords.words("english"))

def clean_text(text):
    if pd.isna(text):  # Перевірка на NaN
        return ""

    text = html.unescape(text)  # Декодування HTML-ентітетів (&nbsp;, &rsquo; → ', &copy; → © тощо)
    text = text.replace("\xa0", " ")  # Видаляємо неперервні пробіли
    text = re.sub(r'<.*?>', '', text)  # Видаляємо HTML-теги
    text = re.sub(r'[^a-zA-Z0-9\s\'"-]', '', text)  # Дозволяємо літери, цифри, апострофи та дефіси
    text = re.sub(r'\s+', ' ', text)  # Видаляємо зайві пробіли
    text = text.lower().strip()  # Змінюємо регістр і прибираємо зайві пробіли
    # Видаляємо зайві символи
    text = text.replace("\n", " ").replace("\r", "")  # заміняємо нові рядки
    # Видаляємо невидимі символи та непотрібні лапки
    text = re.sub(r"[\"\'/\\ -]", " ", text)  # заміняємо " ' - / \ на пробіл
    # Заміняємо всі зайві пробіли на один
    text = re.sub(r'\s+', ' ', text).strip()
    # Видалення стоп-слів
    words = text.split()
    filtered_words = [word for word in words if word not in stop_words]
    text = " ".join(filtered_words)

    return text

# Застосовуємо до всіх текстів
df["text"] = df["text"].apply(clean_text)

df.to_csv("bbc_news_cleaned.csv", index=False)
print("✅ Очищений файл збережено!")
# Зберігаємо у JSON
output_json = df[["title", "url","text"]].to_dict(orient="records")
with open("bbc_cleaned_news_2.json", "w", encoding="utf-8") as f:
    json.dump(output_json, f, ensure_ascii=False, indent=4)

print("✅ Файл bbc_cleaned_news.json збережено!")