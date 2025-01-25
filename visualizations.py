import matplotlib.pyplot as plt
import seaborn as sns
import torch
from sklearn.metrics import confusion_matrix
import pandas as pd
from matplotlib.patches import FancyBboxPatch
from collections import Counter
import string
from nltk.corpus import stopwords
import nltk

nltk.download("stopwords")
stop_words = set(stopwords.words("english"))
with open("data/rouge_155.txt", "r") as f:
    file_stop_words = {line.strip().lower() for line in f}
stop_words.update(file_stop_words)


def preprocess_text(text):
    text = text.lower()
    text = text.translate(str.maketrans("", "", string.punctuation))
    tokens = text.split()
    tokens = [word for word in tokens if word not in stop_words]
    return tokens


def plot_data_distribution(train_data, label_mapping):
    emotion_counts = train_data["label"].value_counts().rename(index=label_mapping)
    plt.figure(figsize=(10, 6))
    plt.bar(emotion_counts.index, emotion_counts.values, color="skyblue")
    plt.title("Frequency Distribution of Emotions", fontsize=16)
    plt.xlabel("Emotion", fontsize=12)
    plt.ylabel("Frequency", fontsize=12)
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig("emotion_frequency.png")
    plt.show()


def plot_confusion_matrix(y_true, y_pred, labels):
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(
        cm, annot=True, fmt="d", cmap="Blues", xticklabels=labels, yticklabels=labels
    )
    plt.title("Confusion Matrix")
    plt.xlabel("Predicted Labels")
    plt.ylabel("True Labels")
    plt.show()


# Visualization of Bert Model Architecture
def visualize_model_architecture():
    fig, ax = plt.subplots(figsize=(12, 10))
    ax.axis("off")

    # Add BertModel and components
    ax.add_patch(
        FancyBboxPatch(
            (0.1, 0.8),
            0.8,
            0.15,
            boxstyle="round,pad=0.1",
            edgecolor="black",
            facecolor="#d0e1f9",
        )
    )
    ax.text(
        0.5,
        0.875,
        "BertModel",
        ha="center",
        va="center",
        fontsize=14,
        fontweight="bold",
    )

    # Embeddings block
    ax.add_patch(
        FancyBboxPatch(
            (0.15, 0.72),
            0.7,
            0.05,
            boxstyle="round,pad=0.1",
            edgecolor="black",
            facecolor="#f7d8ba",
        )
    )
    ax.text(
        0.5,
        0.745,
        "BertEmbeddings (Word, Position, TokenType)",
        ha="center",
        va="center",
        fontsize=12,
    )

    # Encoder block
    ax.add_patch(
        FancyBboxPatch(
            (0.15, 0.6),
            0.7,
            0.1,
            boxstyle="round,pad=0.1",
            edgecolor="black",
            facecolor="#d4e5b2",
        )
    )
    ax.text(0.5, 0.65, "BertEncoder (12 Layers)", ha="center", va="center", fontsize=12)

    # Pooler block
    ax.add_patch(
        FancyBboxPatch(
            (0.15, 0.5),
            0.7,
            0.05,
            boxstyle="round,pad=0.1",
            edgecolor="black",
            facecolor="#fde0d0",
        )
    )
    ax.text(
        0.5,
        0.525,
        "BertPooler (Tanh Activation)",
        ha="center",
        va="center",
        fontsize=12,
    )

    # Dropout layer
    ax.add_patch(
        FancyBboxPatch(
            (0.15, 0.4),
            0.7,
            0.05,
            boxstyle="round,pad=0.1",
            edgecolor="black",
            facecolor="#ddd6f3",
        )
    )
    ax.text(0.5, 0.425, "Dropout (p=0.1)", ha="center", va="center", fontsize=12)

    # Classifier layer
    ax.add_patch(
        FancyBboxPatch(
            (0.15, 0.3),
            0.7,
            0.05,
            boxstyle="round,pad=0.1",
            edgecolor="black",
            facecolor="#f5b7b1",
        )
    )
    ax.text(
        0.5,
        0.325,
        "Classifier (Linear Layer, 6 Classes)",
        ha="center",
        va="center",
        fontsize=12,
    )

    # Input tokens and prediction workflow
    ax.text(
        0.5,
        0.2,
        "Input Tokens: ['[CLS]', 'i', \"'\", 'm', 'feeling', ...]",
        ha="center",
        va="center",
        fontsize=10,
    )
    ax.text(0.5, 0.15, "Logits: tensor([...])", ha="center", va="center", fontsize=10)
    ax.text(
        0.5,
        0.1,
        "Prediction: joy (Confidence: 1.00)",
        ha="center",
        va="center",
        fontsize=10,
    )

    plt.title(
        "Visualization of BertForSequenceClassification Architecture",
        fontsize=16,
        fontweight="bold",
    )
    plt.tight_layout()
    plt.show()


def plot_common_words_per_emotion(data, label_mapping):

    df = pd.read_csv("data/train.csv", names=["phrase", "label"])

    label_word_counts = {}

    # Process each label
    for label in df["label"].unique():
        # Filter rows for the current label
        label_phrases = df[df["label"] == label]["phrase"]

        # Tokenize and count words
        all_words = []
        for phrase in label_phrases:
            all_words.extend(preprocess_text(phrase))

        # Count word frequencies
        word_counts = Counter(all_words)

        # Store the 10 most common words
        label_word_counts[label] = word_counts.most_common(10)

    # Display the results
    for label, common_words in label_word_counts.items():
        print(f"\nLabel: {label}")
        for word, count in common_words:
            print(f"{word}: {count}")

            # Prepare data for visualization

    label_word_counts = {}
    for label in df["label"].unique():
        label_phrases = df[df["label"] == label]["phrase"]
        all_words = []
        for phrase in label_phrases:
            all_words.extend(preprocess_text(phrase))
        word_counts = Counter(all_words)
        label_word_counts[label] = word_counts.most_common(10)

    # Visualize the results
    for label, common_words in label_word_counts.items():
        words, counts = zip(*common_words)
        plt.figure(figsize=(10, 6))
        plt.barh(words, counts, color="skyblue")
        plt.title(f"Most Common Words for {label.capitalize()}", fontsize=16)
        plt.xlabel("Frequency", fontsize=12)
        plt.ylabel("Words", fontsize=12)
        plt.tight_layout()
        plt.show()


def visualize_word_length_distribution():
    data = pd.read_csv("data/train.csv", names=["phrase", "label"])

    for label in data["label"].unique():
        label_phrases = data[data["label"] == label]["phrase"]
        word_lengths = [
            len(word)
            for phrase in label_phrases.dropna()
            for word in preprocess_text(phrase)
        ]
        plt.figure(figsize=(8, 5))
        sns.histplot(word_lengths, bins=20, kde=True, color="skyblue")
        plt.title(f"Word Length Distribution for {label.capitalize()}", fontsize=16)
        plt.xlabel("Word Length", fontsize=12)
        plt.ylabel("Frequency", fontsize=12)
        plt.tight_layout()
        plt.show()


# Visualize the distribution of labels
def visualize_label_distribution():

    data = pd.read_csv("data/train.csv", names=["phrase", "label"])

    label_counts = data["label"].value_counts()
    plt.figure(figsize=(8, 5))
    sns.barplot(x=label_counts.index, y=label_counts.values, palette="viridis")
    plt.title("Distribution of Labels in Dataset", fontsize=16)
    plt.xlabel("Labels", fontsize=12)
    plt.ylabel("Count", fontsize=12)
    plt.tight_layout()
    plt.show()


def visualizations(train_data, label_mapping):
    eval_data = pd.read_csv("./results/eval_results.csv")
    visualize_label_distribution()
    visualize_word_length_distribution()
    plot_common_words_per_emotion(train_data, label_mapping)
    visualize_model_architecture()
    plot_data_distribution(train_data, label_mapping)
    plot_confusion_matrix(
        eval_data["true_label"],
        eval_data["predicted_label"],
        list(label_mapping.values()),
    )
