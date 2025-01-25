from transformers import BertTokenizer, BertForSequenceClassification, AdamW
from torch.utils.data import DataLoader, Dataset
from sklearn.metrics import accuracy_score
import torch


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
    batch_size = 16
    epochs = 5

    train_data["label"] = train_data["label"].map(label_mapping)
    val_data["label"] = val_data["label"].map(label_mapping)

    tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")

    # Create datasets
    train_dataset = EmotionDetectionDataset(
        train_data["text"].tolist(), train_data["label"].tolist(), tokenizer, max_length
    )
    val_dataset = EmotionDetectionDataset(
        val_data["text"].tolist(), val_data["label"].tolist(), tokenizer, max_length
    )

    # Data loaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size)

    # Model
    model = BertForSequenceClassification.from_pretrained(
        "bert-base-uncased", num_labels=len(label_mapping)
    )
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    # Optimizer
    optimizer = AdamW(model.parameters(), lr=5e-5)

    # Training loop
    for epoch in range(epochs):
        model.train()
        total_loss = 0
        for batch in train_loader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)

            optimizer.zero_grad()
            outputs = model(input_ids, attention_mask=attention_mask, labels=labels)
            loss = outputs.loss
            total_loss += loss.item()
            loss.backward()
            optimizer.step()

        avg_loss = total_loss / len(train_loader)
        print(f"Epoch {epoch + 1}/{epochs}, Training Loss: {avg_loss:.4f}")

        # Validation
        model.eval()
        val_accuracy = 0
        with torch.no_grad():
            for batch in val_loader:
                input_ids = batch["input_ids"].to(device)
                attention_mask = batch["attention_mask"].to(device)
                labels = batch["labels"].to(device)

                outputs = model(input_ids, attention_mask=attention_mask)
                preds = torch.argmax(outputs.logits, dim=1)
                val_accuracy += accuracy_score(labels.cpu(), preds.cpu())

        val_accuracy /= len(val_loader)
        print(f"Validation Accuracy: {val_accuracy:.4f}")

    return model, tokenizer
