import pandas as pd
import os

os.chdir("..")
# Завантажуємо датасет
df = pd.read_csv("data/bbc_news.csv")  # Замінити на свій файл

# Залишаємо тільки потрібні колонки
df = df[["title", "link"]]

# Видаляємо дублікати
df = df.drop_duplicates()

# Видаляємо рядки, де немає посилань
df = df.dropna(subset=["link"])

# Зберігаємо оновлений датасет
df.to_csv("cleaned_bbc_news_links.csv", index=False)

print("Датасет очищено та збережено!")
