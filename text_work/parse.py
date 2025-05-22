import requests
from bs4 import BeautifulSoup
import pandas as pd
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm
import os

os.chdir("..")
# Завантажуємо очищений датасет
df = pd.read_csv("data/cleaned_bbc_news_links.csv")
news_data = []

# Функція для парсингу тексту новини
def get_news_text(row):
    url, title = row["link"], row["title"]
    try:
        headers = {"User-Agent": "Mozilla/5.0"}
        response = requests.get(url, headers=headers, timeout=5)  # Додаємо тайм-аут
        if response.status_code == 200:
            soup = BeautifulSoup(response.text, "html.parser")
            paragraphs = soup.find_all("p")
            article_text = " ".join([p.get_text() for p in paragraphs])
            return {"title": title, "url": url, "text": article_text}
    except Exception as e:
        return None  # Якщо помилка, повертаємо None

# Використання багатопотоковості
num_threads = 10  # Залежить від потужності процесора та інтернет-з'єднання

with ThreadPoolExecutor(max_workers=num_threads) as executor:
    future_to_url = {executor.submit(get_news_text, row): row for _, row in df.iterrows()}
    for future in tqdm(as_completed(future_to_url), total=len(df)):
        result = future.result()
        if result:
            news_data.append(result)

# Збереження в CSV та JSON
news_df = pd.DataFrame(news_data)
news_df.to_csv("bbc_news_with_text.csv", index=False)
news_df.to_json("bbc_news_with_text.json", orient="records", indent=2)

print("✅ Швидкий парсинг завершено!")
