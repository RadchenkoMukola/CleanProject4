import pandas as pd
from newspaper import Article
import concurrent.futures
from tqdm import tqdm
import json
import os

os.chdir("..")
# Завантаження CSV
df = pd.read_csv("data/allsides_news.csv")

# Перейменування колонок
df = df[["Links", "bias_rating", "heading"]].rename(columns={
    "Links": "link",
    "bias_rating": "bias_label",
    "heading": "title"
}).dropna()

# Функція для парсингу однієї статті
def get_article_text(url):
    try:
        article = Article(url)
        article.download()
        article.parse()
        return article.text
    except:
        return None

# Обгортка з індексом для відстеження
def fetch_text(row):
    idx, link = row
    return idx, get_article_text(link)

# Паралельний збір тексту
results = []
with concurrent.futures.ThreadPoolExecutor(max_workers=32) as executor:
    futures = {executor.submit(fetch_text, row): row[0] for row in df["link"].items()}
    for future in tqdm(concurrent.futures.as_completed(futures), total=len(futures)):
        idx, text = future.result()
        results.append((idx, text))

text_df = pd.DataFrame(results, columns=["idx", "text"]).set_index("idx")
df["text"] = text_df["text"]

# Видаляємо неуспішно спаршені
df = df.dropna(subset=["text"])

# Збереження результату
df.to_csv("allsides_news_with_text.csv", index=False)
with open("allsides_news_with_text.json", "w", encoding="utf-8") as f:
    json.dump(df[["title", "link", "text", "bias_label"]].to_dict(orient="records"), f, ensure_ascii=False, indent=4)

print("✅ Парсинг завершено. JSON і CSV збережено.")
