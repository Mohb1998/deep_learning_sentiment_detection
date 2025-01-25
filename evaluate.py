import matplotlib.pyplot as plt
import seaborn as sns

import pandas as pd
import torch
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
)


def predict_label(phrase, model, tokenizer, device):
    inputs = tokenizer(
        phrase,
        return_tensors="pt",
        truncation=True,
        padding=True,
        max_length=128,
    ).to(device)

    with torch.no_grad():
        outputs = model(**inputs)
        logits = outputs.logits
        predicted_class = torch.argmax(logits, dim=1).item()
    return predicted_class


def evaluate_model(model, tokenizer, device):

    eval_data = pd.read_csv("./data/eval.csv", names=["phrase", "true_label"])

    label_mapping = {
        0: "joy",
        1: "sadness",
        2: "anger",
        3: "fear",
        4: "love",
        5: "surprise",
    }
    reverse_label_mapping = {v: k for k, v in label_mapping.items()}
    label_names = list(label_mapping.values())

    # Map true labels to numeric values
    eval_data["true_label"] = eval_data["true_label"].map(reverse_label_mapping)

    # Predict labels for all phrases
    eval_data["predicted_label"] = eval_data["phrase"].apply(
        lambda x: predict_label(x, model, tokenizer, device)
    )

    # Extract true and predicted labels
    y_true = eval_data["true_label"]
    y_pred = eval_data["predicted_label"]

    # Compute metrics
    accuracy = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred, average="weighted")
    recall = recall_score(y_true, y_pred, average="weighted")
    f1 = f1_score(y_true, y_pred, average="weighted")

    # Classification report and confusion matrix
    classification_report_str = classification_report(
        y_true, y_pred, target_names=label_names
    )
    conf_matrix = confusion_matrix(y_true, y_pred)

    # Plot confusion matrix
    plt.figure(figsize=(10, 8))
    sns.heatmap(
        conf_matrix,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=label_names,
        yticklabels=label_names,
    )
    plt.title("Confusion Matrix")
    plt.xlabel("Predicted Labels")
    plt.ylabel("True Labels")
    plt.show()

    # Print metrics
    print(f"Accuracy: {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1 Score: {f1:.4f}")
    print("Classification Report:")
    print(classification_report_str)

    # Save results
    eval_data["true_label_name"] = eval_data["true_label"].map(label_mapping)
    eval_data["predicted_label_name"] = eval_data["predicted_label"].map(label_mapping)
    eval_data.to_csv("results/eval_results.csv", index=False)

    # Save metrics
    with open("results/eval_metrics.txt", "w") as f:
        f.write(f"Accuracy: {accuracy:.4f}\n")
        f.write(f"Precision: {precision:.4f}\n")
        f.write(f"Recall: {recall:.4f}\n")
        f.write(f"F1 Score: {f1:.4f}\n\n")
        f.write("Classification Report:\n")
        f.write(classification_report_str)
        f.write("\nConfusion Matrix:\n")
        f.write(str(conf_matrix))

    # Bar plot for precision, recall, and F1-score
    report = classification_report(
        y_true, y_pred, target_names=label_names, output_dict=True
    )
    report_df = pd.DataFrame(report).transpose()
    report_df = report_df.iloc[:-3, :-1]  # Exclude averages and support
    report_df.plot(kind="bar", figsize=(12, 6), title="Evaluation Metrics by Class")
    plt.xticks(rotation=45, ha="right")
    plt.ylabel("Score")
    plt.xlabel("Classes")
    plt.tight_layout()
    plt.show()
