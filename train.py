import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

import json
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, random_split
from sklearn.preprocessing import LabelEncoder
import numpy as np
import pickle
import random

POSES_DIR = r"Data\Poses"
ANNOTATIONS_FILE = r"Data\shot_annotations.json"
MODEL_PATH = "model.pth"
BATCH_SIZE = 8
EPOCHS = 10
LEARNING_RATE = 1e-5
SEQ_LEN = 45

LABELS = [
    "Serve", "Forehand", "Backhand", "Forehand Slice",
    "Backhand Slice", "Forehand Volley", "Backhand Volley", "Smash"
]

class PoseSequenceDataset(Dataset):
    def __init__(self, poses_dir, annotations_file, label_encoder, seq_len=SEQ_LEN, selected_labels=None):
        with open(annotations_file, "r") as f:
            annotations = json.load(f)
        self.samples = []
        self.labels = []
        self.filenames = [] 
        print(f"Total annotations: {len(annotations)}")
        for fname, label in annotations.items():
            if selected_labels is not None and label not in selected_labels:
                continue 
            base = fname.replace('.mp4', '').replace('.txt', '').replace('_mirrored', '')
            files_to_load = [f" {base}.txt", f" {base}_mirrored.txt"]
            for pose_file in files_to_load:
                pose_path = os.path.join(poses_dir, pose_file)
                if not os.path.exists(pose_path):
                    print(f"Missing pose file: {pose_path}")
                    continue
                with open(pose_path, "r") as pf:
                    lines = pf.readlines()
                    lines = lines[1:]
                    features = []
                    for line in lines:
                        parts = line.strip().split('\t')
                        coords = parts[2:]
                        xy = [float(v) for i, v in enumerate(coords) if i % 3 != 0]
                        if len(xy) == 44:
                            features.append(xy)
                    for i in range(0, len(features) - seq_len + 1, seq_len):
                        seq = features[i:i+seq_len]
                        if len(seq) == seq_len:
                            self.samples.append(seq)
                            self.labels.append(label)
                            self.filenames.append(pose_file)
        print(f"Loaded {len(self.samples)} sequences from pose files.")
        self.samples = np.array(self.samples, dtype=np.float32)
        self.labels = label_encoder.transform(self.labels)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        return torch.tensor(self.samples[idx]), torch.tensor(self.labels[idx], dtype=torch.long)

class BiLSTMClassifier(nn.Module):
    def __init__(self, input_dim=44, hidden_dim=128, num_layers=2, num_classes=8, bidirectional=True, dropout=0.5):
        super().__init__()
        self.lstm = nn.LSTM(
            input_dim, hidden_dim, num_layers=num_layers,
            batch_first=True, bidirectional=bidirectional
        )
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_dim * 2 if bidirectional else hidden_dim, num_classes)

    def forward(self, x):
        lstm_out, _ = self.lstm(x)
        out = lstm_out[:, -1, :]
        out = self.dropout(out)
        return self.fc(out)

def evaluate(model, dataloader, device):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for X, y in dataloader:
            X, y = X.to(device), y.to(device)
            outputs = model(X)
            _, predicted = torch.max(outputs, 1)
            correct += (predicted == y).sum().item()
            total += y.size(0)
    return correct / total if total > 0 else 0

def main():
    # Define which labels to use for training
    selected_labels = [
        "Serve", "Forehand", "Backhand"
    ]

    label_encoder = LabelEncoder()
    label_encoder.fit(selected_labels)

    with open("label_encoder.pkl", "wb") as f:
        pickle.dump(label_encoder, f)

    dataset = PoseSequenceDataset(POSES_DIR, ANNOTATIONS_FILE, label_encoder, selected_labels=selected_labels)
    print(f"Loaded {len(dataset)} samples in dataset.")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    num_samples = len(dataset)
    all_indices = list(range(num_samples))
    random.shuffle(all_indices)

    # Always use 100 samples for test set
    test_indices = all_indices[:100]
    trainval_indices_full = all_indices[100:]

    total_trainval_pool_size = len(trainval_indices_full)
    if total_trainval_pool_size < len(trainval_indices_full):
        trainval_indices = trainval_indices_full[:total_trainval_pool_size]
    else:
        trainval_indices = trainval_indices_full

    test_set = torch.utils.data.Subset(dataset, test_indices)
    test_loader = DataLoader(test_set, batch_size=BATCH_SIZE, shuffle=False)

    model = BiLSTMClassifier(num_classes=len(selected_labels)).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

    epoch = 0
    run_val_accs = []

    while True:
        # --- Reshuffle and split train/val for each run ---
        random.shuffle(trainval_indices)
        # Use 20% for validation, 80% for training
        val_size = max(1, len(trainval_indices) // 5)
        val_indices = trainval_indices[:val_size]
        train_indices = trainval_indices[val_size:]

        train_set = torch.utils.data.Subset(dataset, train_indices)
        val_set = torch.utils.data.Subset(dataset, val_indices)

        train_loader = DataLoader(train_set, batch_size=BATCH_SIZE, shuffle=True)
        val_loader = DataLoader(val_set, batch_size=BATCH_SIZE, shuffle=False)

        val_accs = []
        for _ in range(EPOCHS):
            model.train()
            total_loss = 0
            for X, y in train_loader:
                X, y = X.to(device), y.to(device)
                optimizer.zero_grad()
                outputs = model(X)
                loss = criterion(outputs, y)
                loss.backward()
                optimizer.step()
                total_loss += loss.item()
            val_acc = evaluate(model, val_loader, device)
            val_accs.append(val_acc)
            epoch += 1
            print(f"Epoch {epoch}, Loss: {total_loss/len(train_loader):.4f}, Val Acc: {val_acc:.4f}")

        avg_val_acc = sum(val_accs) / len(val_accs)
        run_val_accs.append(avg_val_acc)

        with open("val_acc_log.txt", "w") as logf:
            for i, acc in enumerate(run_val_accs):
                logf.write(f"{i+1}\t{acc:.6f}\n")

        if all(acc > 0.995 for acc in val_accs):
            print(f"All {EPOCHS} epochs in the last run achieved validation accuracy > 0.995. Training complete.")
            break
        else:
            print(f"Not all {EPOCHS} epochs achieved validation accuracy > 0.995. Reshuffling and continuing training...")

    print("Saving model.")
    torch.save(model.state_dict(), MODEL_PATH)

    # --- Evaluate on entire dataset and test set, save results ---
    model.eval()
    all_results = []
    test_results = []

    with torch.no_grad():
        # Evaluate on entire dataset
        for idx in range(len(dataset)):
            X, y = dataset[idx]
            X = X.unsqueeze(0).to(device)
            y = y.item()
            output = model(X)
            pred = torch.argmax(output, dim=1).item()
            fname = dataset.filenames[idx] if hasattr(dataset, "filenames") else f"sample_{idx}"
            all_results.append((fname, pred, y))

        # Evaluate on test set
        for idx in test_indices:
            X, y = dataset[idx]
            X = X.unsqueeze(0).to(device)
            y = y.item()
            output = model(X)
            pred = torch.argmax(output, dim=1).item()
            fname = dataset.filenames[idx] if hasattr(dataset, "filenames") else f"sample_{idx}"
            test_results.append((fname, pred, y))

    with open("validation_and_test_results.txt", "w") as f:
        f.write("=== Validation on Entire Dataset ===\n")
        f.write("filename\tprediction\tground_truth\n")
        for fname, pred, gt in all_results:
            f.write(f"{fname}\t{pred}\t{gt}\n")
        f.write("\n=== Test Set Results ===\n")
        f.write("filename\tprediction\tground_truth\n")
        for fname, pred, gt in test_results:
            f.write(f"{fname}\t{pred}\t{gt}\n")

    # Write only incorrect predictions, first for test set, then for entire dataset, with label names
    with open("incorrect_predictions.txt", "w") as f:
        # Test set incorrect predictions
        f.write("=== Incorrect Predictions on Test Set ===\n")
        f.write("filename\tprediction\tground_truth\n")
        for fname, pred, gt in test_results:
            if pred != gt:
                pred_label = label_encoder.inverse_transform([pred])[0]
                gt_label = label_encoder.inverse_transform([gt])[0]
                f.write(f"{fname}\t{pred_label}\t{gt_label}\n")

        # All dataset incorrect predictions
        f.write("\n=== Incorrect Predictions on Entire Dataset ===\n")
        f.write("filename\tprediction\tground_truth\n")
        for fname, pred, gt in all_results:
            if pred != gt:
                pred_label = label_encoder.inverse_transform([pred])[0]
                gt_label = label_encoder.inverse_transform([gt])[0]
                f.write(f"{fname}\t{pred_label}\t{gt_label}\n")

    # Evaluate on test set
    test_acc = evaluate(model, test_loader, device)
    print(f"Test Accuracy: {test_acc:.4f}")

    # Evaluate on entire dataset
    all_loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=False)
    total_acc = evaluate(model, all_loader, device)
    print(f"Total Dataset Accuracy: {total_acc:.4f}")

    train_labels = [dataset.labels[i] for i in train_set.indices]
    val_labels = [dataset.labels[i] for i in val_set.indices]
    test_labels = [dataset.labels[i] for i in test_set.indices]

    print("\nClass distribution in the training set:")
    class_counts = np.bincount(train_labels, minlength=len(selected_labels))
    for idx, count in enumerate(class_counts):
        label = label_encoder.inverse_transform([idx])[0]
        print(f"{label}: {count} instances")

    print("\nClass distribution in the validation set:")
    class_counts_val = np.bincount(val_labels, minlength=len(selected_labels))
    for idx, count in enumerate(class_counts_val):
        label = label_encoder.inverse_transform([idx])[0]
        print(f"{label}: {count} instances")

    print("\nClass distribution in the test set:")
    class_counts_test = np.bincount(test_labels, minlength=len(selected_labels))
    for idx, count in enumerate(class_counts_test):
        label = label_encoder.inverse_transform([idx])[0]
        print(f"{label}: {count} instances")

if __name__ == "__main__":
    main()