import pandas as pd
from sklearn.utils import resample
import json
from collections import Counter
import re
import matplotlib.pyplot as plt
import os

os.chdir("..")
right_terms = [
    # США
    "free speech", "cancel culture", "wokeism", "anti-woke", "First Amendment", "censorship resistance",
    "religious liberty", "traditional values", "gender ideology", "CRT", "DEI", "Second Amendment",
    "gun rights", "build the wall", "illegal immigration", "America First", "fiscal conservatism",
    "anti-socialism", "Western civilization", "Christian nationalism", "deep state", "entitlement reform",

    # Велика Британія
    "pro-Brexit", "Remoaner", "Global Britain", "sovereignty", "mass immigration", "small boats", "Channel crossings",
    "British values", "grooming gangs", "Islamism", "leftist media", "cultural Marxism", "nanny state",
    "political correctness gone mad", "Union Jack", "anti-EU", "pro-Brexit", "British jobs for British workers",
    "woke left", "statues debate", "Church of England collapse", "liberal elite", "Great British culture",
    "English identity", "heritage protection", "Royal Family under attack", "Londonistan", "small state"

    # Континентальна Європа
    "great replacement", "no-go zones", "Sharia zones", "Islamization of Europe", "Fortress Europe",
    "EU overreach", "Brussels bureaucrats", "anti-globalism", "multicultural failure", "migrant crisis",
    "sovereign borders", "traditional Europe", "European civilization", "anti-NGO ships", "Christian Europe",
    "anti-migrant violence", "white genocide", "cultural suicide", "depopulation", "gender propaganda in schools",
    "national revival", "anti-EU", "climate authoritarianism", "wokeness from the EU", "drag shows for kids",
    "NGO invasion", "leftist judiciary", "progressive tyranny",
    
    #захист еліт/порядку
    "aspirational success", "free enterprise", "merit-based society", "reward for hard work", "business-friendly", "wealth creators", "regulation burdens", "entrepreneurial spirit", "private initiative", "global competitiveness", "incentivizing growth",

    "Oxbridge elite", "the Establishment", "Brexit betrayal", "sovereignty", "woke agenda", "British values", "nation first", "Great Replacement", "London-centric", "white working class", "patriotic education", "BBC bias", "grooming gangs", "leftist media"
]

right_leaning_terms = [
    # Спільні праві ідеї
    "freedom of speech", "religious freedom", "patriot", "conservative", "liberty", "traditionalism",
    "small government", "pro-life", "gun ownership", "family values", "school choice", "national pride",
    "border security", "law and order", "Christian values", "heritage", "private property", "anti-communism",
    "national sovereignty", "economic freedom", "anti-globalism", "personal responsibility",

    # Британський дискурс
    "Brexit", "British exceptionalism", "Church and Crown", "lawful Britain", "anti-immigration", "border controls",
    "freedom of the press", "save the monarchy", "leftist overreach", "British heritage", "bring back grammar schools",
    "Unionism", "England first", "UK independence", "British jobs", "anti-woke curriculum",

    # Європейський контекст
    "Christian Europe", "European heritage", "national identity", "EU dictatorship", "sovereign states",
    "pro-natalist policy", "demographic collapse", "cultural sovereignty", "stop Islamization", "anti-NGO",
    "traditional gender roles", "pro-family", "ethno-cultural continuity", "decolonization of Europe",
    "anti-progressivism", "anti-EU", "no-go areas", "restore the West", "white identity politics",
    "traditional schooling", "patriotic education", "defend the homeland", "values of the nation",

    #Праві слова:
    "common sense", "radical left", "out of touch", "hardworking taxpayers", "traditional values"

]

left_leaning_terms = [
    # Економіка
    "economic opportunity", "fair taxation", "closing the wealth gap", "corporate accountability",
    "responsible capitalism", "consumer protections", "public investment", "economic mobility",
    "support for working families", "strengthening the middle class",

    # Соціальна справедливість
    "equal opportunity", "inclusive policies", "gender equality", "diversity and representation",
    "civil rights protections", "equity initiatives", "social inclusion", "LGBTQ+ inclusion",
    "women’s rights", "non-discrimination laws", "individual dignity", "cultural awareness",

    # Імміграція/міграція
    "fair immigration policies", "pathways to citizenship", "welcoming communities",
    "strengthening immigrant contributions", "humanitarian protections", "refugee integration",

    # Охорона здоров’я
    "healthcare access", "mental health care", "reproductive healthcare", "medical privacy",
    "personal choice", "public health funding", "women’s healthcare",

    # Клімат
    "climate awareness", "environmental responsibility", "clean energy transition",
    "renewable innovation", "sustainable policies", "conservation efforts", "green economy",

    # Британські/Європейські теми
    "rail nationalisation", "NHS protection", "rent freeze", "free university education",
    "childcare expansion", "worker co-ops", "EU cooperation", "liberal democracy", "press pluralism",
    "anti-corruption", "housing as a right", "right to protest", "fair asylum policy", "affordable transport",

    # Управління/уряд
    "government accountability", "responsible governance", "evidence-based policy",
    "common good", "modernizing institutions", "participatory democracy", "community engagement",

    #Ліві тональні слова
    "alarming", "exploitative", "unfair", "underserved", "disproportionate", "systemic"
]

left_terms = [
    # Економічна справедливість
    "wealth redistribution", "progressive taxation", "tax the rich", "universal basic income", "living wage",
    "fair wages", "workers' rights", "trade union power", "economic justice", "public ownership",
    "anti-austerity", "nationalizing utilities", "rent control", "affordable housing", "economic democracy",

    # Соціальні права
    "universal healthcare", "Medicare for all", "free healthcare", "healthcare as a right",
    "Planned Parenthood", "abortion access", "pro-choice", "reproductive rights", "bodily autonomy",
    "LGBTQ+ rights", "gender equity", "equal pay", "feminist policies", "trans inclusion",
    "non-binary recognition", "intersectionality", "safe spaces", "anti-harassment policies",

    # Расові/етнічні питання
    "systemic racism", "racial justice", "white privilege", "decolonization", "anti-racism", "institutional discrimination",
    "reparations", "Black Lives Matter", "police brutality", "defund the police", "abolish ICE", "immigrant justice",

    # Клімат/довкілля
    "Green New Deal", "climate justice", "environmental justice", "corporate polluters", "net zero",
    "carbon neutrality", "climate crisis", "fossil fuel divestment", "sustainable development",
    "eco-socialism", "environmental reparations",

    # Європейський/Британський контекст
    "public NHS", "free school meals", "refugee solidarity", "anti-Brexit", "freedom of movement",
    "asylum rights", "sanctuary cities", "no borders", "pro-EU", "people's vote", "Workers' Party", "Corbynism",
    "renters’ rights", "basic income trials", "anti-monarchism", "republicanism UK", "green industrial strategy",

    # Політична структура/активізм
    "anti-fascism", "direct democracy", "horizontal organizing", "participatory budgeting", "mutual aid",
    "grassroots organizing", "climate strike", "student activism", "mass mobilization", "occupy movement",

    # Анти-елітарні фрейми
    "wealthy elites", "political donations", "corporate lobbying", "speaking fees", "post-political career", "private jets", "donor class", "old boys network", "offshore accounts", "corporate influence in politics", "revolving door politics", "privileged insiders", "plutocracy",

    "NHS underfunding", "austerity impact", "Grenfell justice", "food banks", "social housing crisis", "Tory cuts", "climate refugees", "worker exploitation", "benefit sanctions", "wealth inequality in Britain"
]

df = pd.read_csv("data/bbc_news_cleaned_2.csv")

sample_df = df.sample(n=30000, random_state=42).reset_index(drop=True)
sample_df[["title", "text"]].head(10)
# Тепер повторно запускаємо оцінку
bias_dict = {
    -1: left_terms,
    -0.5: left_leaning_terms,
    0.5: right_leaning_terms,
    1: right_terms
}

def weighted_bias_score(text):
    text = str(text).lower()
    word_counts = Counter(re.findall(r'\w+', text))
    total_score = 0
    for weight, terms in bias_dict.items():
        for term in terms:
            term_lower = term.lower()
            term_count = word_counts[term_lower]
            if term_count > 0:
                total_score += weight * term_count
    return round(total_score, 3)

def label_bias(score):
    if score <= -0.5:
        return -1  # Left lean
    elif score >= 0.5:
        return 1   # Right
    else:
        return 0   # Center

# Застосування до вибірки
sample_df["bias_score_weighted"] = sample_df["text"].apply(weighted_bias_score)

sample_df["label"] = sample_df["bias_score_weighted"].apply(label_bias)

# Показати перші 10 результатів
#print(sample_df[["title", "bias_score_weighted", "label"]].head(10))
print(sample_df["label"].value_counts())
sample_df.to_csv("bbc_news_marked_no_balance.csv", index=False)

#sample_df_left = sample_df[sample_df.label == -2]
sample_df_left_lean = sample_df[sample_df.label == -1]
sample_df_center = sample_df[sample_df.label == 0]
sample_df_right_lean = sample_df[sample_df.label == 1]
#sample_df_right = sample_df[sample_df.label == 2]

min_len = min(len(sample_df_left_lean), len(sample_df_center), len(sample_df_right_lean))

df_balanced = pd.concat([
    #resample(sample_df_left, replace=False, n_samples=min_len, random_state=42),
    resample(sample_df_left_lean, replace=False, n_samples=min_len, random_state=42),
    resample(sample_df_center, replace=False, n_samples=min_len, random_state=42),
    resample(sample_df_right_lean, replace=False, n_samples=min_len, random_state=42),
    #resample(sample_df_right, replace=False, n_samples=min_len, random_state=42),
]).sample(frac=1).reset_index(drop=True)

df_balanced.to_csv("bbc_news_marked_v2.csv", index=False)
json_data = df_balanced.to_dict(orient='records')

# Збереження у файл JSON
with open('bbc_news_marked_v2.json', 'w', encoding='utf-8') as f:
    json.dump(json_data, f, ensure_ascii=False, indent=2)

print(df_balanced["label"].value_counts())

def count_term_matches(articles, term_list):
    term_counter = Counter()
    for article in articles:
        for term in term_list:
            if re.search(r'\b{}\b'.format(re.escape(term)), article, flags=re.IGNORECASE):
                term_counter[term] += 1
    return term_counter

#left_count = count_term_matches(sample_df.text, left_terms)
#print(left_count)
#left_lean_count = count_term_matches(sample_df.text, left_leaning_terms)
#print(left_lean_count)
#right_lean_count = count_term_matches(sample_df.text, right_leaning_terms)
#print(right_lean_count)
#right_count = count_term_matches(sample_df.text, right_terms)
#print(right_count)
