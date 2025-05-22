import pandas as pd
from sklearn.metrics import (
    accuracy_score, precision_recall_fscore_support, classification_report,
    balanced_accuracy_score, matthews_corrcoef, confusion_matrix
)
import seaborn as sns
import matplotlib.pyplot as plt
import os

os.chdir("..")
# Шляхи до файлів
bert_file = 'predictions/allsides_predictions_chunk_BERTv3_1.csv'
roberta_file = 'predictions/allsides_predictions_chunk_RoBERTa_1.csv'
longformer_file = 'predictions/allsides_predictions_chunk_LFv2_1.csv'

# Завантаження CSV
bert_df = pd.read_csv(bert_file)
roberta_df = pd.read_csv(roberta_file)
longformer_df = pd.read_csv(longformer_file)


def evaluate_model(df, name):
    print(f"\n======= 🔍 Evaluating {name} =======")

    # Перетворення label: -1→0, 0→1, 1→2
    y_true_full = df['label'] + 1
    y_pred_full = df['predicted_label']

    print("\n📊 Actual label distribution:")
    print(df['label'].value_counts())
    print("\n📊 Predicted label distribution:")
    print(df['predicted_label'].value_counts())

    # === FULL DATASET METRICS ===
    print("\n=== 📈 Full Dataset Evaluation ===")
    acc = accuracy_score(y_true_full, y_pred_full)
    print("Accuracy:", round(acc, 4))
    print("Balanced Accuracy:", round(balanced_accuracy_score(y_true_full, y_pred_full), 4))
    print("Matthews Corr:", round(matthews_corrcoef(y_true_full, y_pred_full), 4))
    print(classification_report(y_true_full, y_pred_full, target_names=['left', 'center', 'right']))

    # === BALANCED DATASET ===
    print("\n=== Balanced Subset Evaluation ===")
    min_count = df['label'].value_counts().min()
    balanced_df = pd.concat([
        df[df['label'] == -1].sample(min_count, random_state=42),
        df[df['label'] == 0].sample(min_count, random_state=42),
        df[df['label'] == 1].sample(min_count, random_state=42)
    ])
    y_true_bal = balanced_df['label'] + 1
    y_pred_bal = balanced_df['predicted_label']

    print("Balanced Accuracy:", round(balanced_accuracy_score(y_true_bal, y_pred_bal), 4))
    print("Matthews Corr:", round(matthews_corrcoef(y_true_bal, y_pred_bal), 4))
    print(classification_report(y_true_bal, y_pred_bal, target_names=['left', 'center', 'right']))

    # Confusion matrix
    cm = confusion_matrix(y_true_bal, y_pred_bal)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=['left', 'center', 'right'],
                yticklabels=['left', 'center', 'right'])
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.title(f"Confusion Matrix ({name}) — Balanced Subset")
    plt.show()

    # Показати приклади помилок
    mismatches = df[y_true_full != y_pred_full]
    if not mismatches.empty:
        print("\nПриклади помилкових передбачень:")
        print(mismatches[['label', 'predicted_label']].sample(5, random_state=42))
    else:
        print("✅ Немає помилок (дуже підозріло)")

    # Повертаємо короткі метрики для таблиці
    precision, recall, f1, _ = precision_recall_fscore_support(
        y_true_full, y_pred_full, average='weighted', zero_division=0
    )

    return {
        'Model': name,
        'Accuracy': round(acc, 4),
        'Precision': round(precision, 4),
        'Recall': round(recall, 4),
        'F1 Score': round(f1, 4)
    }


# Збір результатів
results = [
    evaluate_model(bert_df, 'BERT'),
    evaluate_model(roberta_df, 'RoBERTa-large'),
    evaluate_model(longformer_df, 'Longformer')
]

# Підсумкова таблиця
results_df = pd.DataFrame(results)
print("\n📋 Summary of Models:")
print(results_df)
