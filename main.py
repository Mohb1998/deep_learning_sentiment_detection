from transformers import BertTokenizer, BertForSequenceClassification
import pandas as pd
import torch
import os

from model_building import build_and_train_model
from evaluate import evaluate_model
from visualizations import visualizations
from ui import run_app


def main():

    train_data = pd.read_csv("./data/train.csv", sep=",", names=["text", "label"])
    test_data = pd.read_csv("./data/test.csv", sep=",", names=["text", "label"])
    eval_data = pd.read_csv("./data/eval.csv", names=["phrase", "true_label"])

    label_mapping = {
        "joy": 0,
        "sadness": 1,
        "anger": 2,
        "fear": 3,
        "love": 4,
        "surprise": 5,
    }

    model_path = "./model"

    if not os.path.exists(model_path):
        print("Starting Model Building and Training...")
        model, tokenizer = build_and_train_model(train_data, test_data, label_mapping)

        os.makedirs(model_path, exist_ok=True)
        model.save_pretrained(model_path)
        tokenizer.save_pretrained(model_path)
        print("Model Building and Training Completed.\n")
    else:
        print("Pretrained model found. Skipping Model Building and Training phase.")
        tokenizer = BertTokenizer.from_pretrained(model_path)
        model = BertForSequenceClassification.from_pretrained(model_path)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    print("Starting Model Evaluation...")
    evaluate_model(model, tokenizer, device, eval_data)
    print("Model Evaluation Completed.")

    print("Starting visualization process...")
    visualizations(train_data)

    print("Opening UI...")
    run_app(model, tokenizer, device)


if __name__ == "__main__":
    main()
