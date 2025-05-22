import pandas as pd
import json
import os

os.chdir("..")
# Завантаження CSV файлу
df = pd.read_csv('data/allsides_predictions_chunk_LFv2_1.csv')

# Перетворення в JSON
json_data = df.to_dict(orient='records')

# Збереження у файл JSON
with open('allsides_predictions_chunk_LFv2_1.json', 'w', encoding='utf-8') as f:
    json.dump(json_data, f, ensure_ascii=False, indent=2)

print("✅ JSON файл створено!")
