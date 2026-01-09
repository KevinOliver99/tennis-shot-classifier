import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

import torch
import numpy as np
import sys
import json
import pickle
from train import BiLSTMClassifier, SEQ_LEN

MODEL_PATH = "model.pth"
with open("label_encoder.pkl", "rb") as f:
    label_encoder = pickle.load(f)
LABELS = list(label_encoder.classes_)

def load_pose_sequence(pose_file, seq_len=SEQ_LEN):
    with open(pose_file, "r") as pf:
        lines = pf.readlines()[1:]
        features = []
        for line in lines:
            parts = line.strip().split('\t')
            coords = parts[2:]
            xy = [float(v) for i, v in enumerate(coords) if i % 3 != 0]
            if len(xy) == 44:
                features.append(xy)
    if len(features) < seq_len:
        raise ValueError(f"Pose file has only {len(features)} frames, but {seq_len} are required.")
    seq = features[:seq_len]
    return np.array(seq, dtype=np.float32)

def main():
    if len(sys.argv) != 2:
        print("Usage: python validate.py <pose_file.txt>")
        sys.exit(1)
    pose_file = sys.argv[1]

    seq = load_pose_sequence(pose_file)
    seq_tensor = torch.tensor(seq).unsqueeze(0)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = BiLSTMClassifier(input_dim=44, num_classes=len(LABELS))
    model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
    model.to(device)
    model.eval()

    with torch.no_grad():
        seq_tensor = seq_tensor.to(device)
        outputs = model(seq_tensor)
        pred = torch.argmax(outputs, dim=1).item()
        print(f"Predicted class: {LABELS[pred]}")

if __name__ == "__main__":
    main()