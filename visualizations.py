import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import nltk
from collections import Counter
import string
from nltk.corpus import stopwords

nltk.download("stopwords")
stop_words = set(stopwords.words("english"))
with open("data/rouge_155.txt", "r") as f:
    file_stop_words = {
        line.strip().translate(str.maketrans("", "", string.punctuation)) for line in f
    }
stop_words.update(file_stop_words)

stop_words.update(
    [
        "feel",
        "feeling",
        "time",
        "people",
        "life",
        "bit",
        "make",
        "things",
        "day",
        "back",
        "didnt",
        "made",
        "feels",
        "feelings",
        "today",
        "work",
        "days",
        "felt",
        "makes",
        "fucked",
        "year",
        "lot",
        "find",
    ]
)


def preprocess_text(text):
    tokens = text.split()
    tokens = [word for word in tokens if word not in stop_words]
    return tokens


def common_words(data):
    label_word_counts = {}

    for label in data["label"].unique():
        label_phrases = data[data["label"] == label]["text"]

        all_words = []
        for phrase in label_phrases:
            all_words.extend(preprocess_text(phrase))

        word_counts = Counter(all_words)
        label_word_counts[label] = word_counts.most_common(10)

    fig, axes = plt.subplots(2, 3, figsize=(10, 5))
    for i, label_common_words in enumerate(label_word_counts.items()):
        label, common_words = label_common_words[0], label_common_words[1]
        words, counts = zip(*common_words)
        ax = axes.flat[i]
        ax.barh(words, counts, color="skyblue")
        ax.set_title(f"{label}", fontsize=12)
        ax.invert_yaxis()
    fig.supylabel("Words", fontsize=12)
    plt.tight_layout()
    plt.savefig("results/common_words.png")
    plt.show()


def phrase_length_distribution(data):
    fig, axes = plt.subplots(2, 3, figsize=(10, 5))
    for i, label in enumerate(data["label"].unique()):
        label_phrases = data[data["label"] == label]["text"]
        phrase_lengths = [len(phrase.split()) for phrase in label_phrases]
        avg_length = np.mean(phrase_lengths)
        ax = axes.flat[i]
        sns.histplot(
            phrase_lengths,
            bins=list(range(0, 40, 2)),
            kde=False,
            color="skyblue",
            ax=ax,
        )
        ax.text(
            0.95,
            0.95,
            f"Avg: {avg_length:.2f}",
            transform=ax.transAxes,
            fontsize=8,
            verticalalignment="top",
            horizontalalignment="right",
            bbox=dict(facecolor="white", alpha=0.7, edgecolor="black"),
        )
        ax.set_title(f"{label}")
    fig.suptitle("Phrase Length", fontsize=14)
    fig.supxlabel("Number of words", fontsize=12)
    plt.tight_layout()
    plt.savefig("results/phrase_length_distribution.png")
    plt.show()


def class_distribution(data):

    label_counts = data["label"].value_counts()
    plt.figure(figsize=(8, 5))
    sns.barplot(x=label_counts.index, y=label_counts.values, palette="viridis")
    plt.title("Class Distribution in Train Dataset", fontsize=14)
    plt.xlabel("Class", fontsize=12)
    plt.ylabel("Count", fontsize=12)
    plt.tight_layout()
    plt.savefig("results/class_distribution.png")
    plt.show()


def visualizations(train_data):
    class_distribution(train_data)
    phrase_length_distribution(train_data)
    common_words(train_data)
