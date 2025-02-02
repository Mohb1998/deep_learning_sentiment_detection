from transformers import DebertaTokenizer, DebertaForSequenceClassification
from torch.utils.data import DataLoader, Dataset
from torch.optim.lr_scheduler import ReduceLROnPlateau
from sklearn.metrics import accuracy_score
import torch
from torch.optim import AdamW
from sklearn.utils.class_weight import compute_class_weight
import numpy as np


class EmotionDetectionDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_length):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = str(self.texts[idx])
        label = self.labels[idx]
        encoding = self.tokenizer(
            text,
            max_length=self.max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )
        return {
            "input_ids": encoding["input_ids"].squeeze(0),
            "attention_mask": encoding["attention_mask"].squeeze(0),
            "labels": torch.tensor(label, dtype=torch.long),
        }


def build_and_train_model(train_data, val_data, label_mapping):
    max_length = 256
    batch_size = 8
    epochs = 20
    learning_rate = 2e-6

    train_data["label"] = train_data["label"].map(label_mapping)
    val_data["label"] = val_data["label"].map(label_mapping)
    tokenizer = DebertaTokenizer.from_pretrained("microsoft/deberta-base")

    train_dataset = EmotionDetectionDataset(
        train_data["text"].tolist(), train_data["label"].tolist(), tokenizer, max_length
    )
    val_dataset = EmotionDetectionDataset(
        val_data["text"].tolist(), val_data["label"].tolist(), tokenizer, max_length
    )

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size)

    class_weights = compute_class_weight(
        class_weight="balanced",
        classes=np.arange(len(label_mapping)),
        y=train_data["label"].values,
    )
    class_weights = torch.tensor(class_weights, dtype=torch.float)

    model = DebertaForSequenceClassification.from_pretrained(
        "microsoft/deberta-base", num_labels=len(label_mapping)
    )
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    optimizer = AdamW(model.parameters(), lr=learning_rate)
    scheduler = ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=2)
    loss_fn = torch.nn.CrossEntropyLoss(weight=class_weights.to(device))

    best_val_loss = float("inf")
    best_model_state = None

    early_stopping_patience = 5
    no_improvement = 0

    for epoch in range(epochs):
        model.train()
        total_loss = 0
        for _, batch in enumerate(train_loader):
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)

            optimizer.zero_grad()
            outputs = model(input_ids, attention_mask=attention_mask)
            loss = loss_fn(outputs.logits, labels)
            total_loss += loss.item()
            loss.backward()
            optimizer.step()

        avg_loss = total_loss / len(train_loader)

        model.eval()
        val_loss = 0
        val_accuracy = 0
        with torch.no_grad():
            for batch in val_loader:
                input_ids = batch["input_ids"].to(device)
                attention_mask = batch["attention_mask"].to(device)
                labels = batch["labels"].to(device)

                outputs = model(input_ids, attention_mask=attention_mask)
                loss = loss_fn(outputs.logits, labels)
                val_loss += loss.item()
                preds = torch.argmax(outputs.logits, dim=1)
                val_accuracy += accuracy_score(labels.cpu(), preds.cpu())

        val_loss /= len(val_loader)
        val_accuracy /= len(val_loader)
        scheduler.step(val_loss)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_model_state = model.state_dict()
            no_improvement = 0
        else:
            no_improvement += 1

        if no_improvement >= early_stopping_patience:
            break

        torch.cuda.empty_cache()

    if best_model_state:
        model.load_state_dict(best_model_state)

    return model, tokenizer
