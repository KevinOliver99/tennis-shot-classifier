import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

import matplotlib.pyplot as plt
import numpy as np
from scipy.ndimage import uniform_filter1d

runs = []
val_accs = []
with open("val_acc_log.txt", "r") as f:
    for line in f:
        if line.strip():
            parts = line.strip().split('\t')
            if len(parts) == 2:
                run, acc = parts
                runs.append(int(run))
                val_accs.append(float(acc))

if len(val_accs) > 1:
    window_size = max(1, min(5, len(val_accs) // 3))
    smoothed_accs = uniform_filter1d(val_accs, size=window_size)
else:
    smoothed_accs = val_accs

plt.figure(figsize=(10, 6))
plt.plot(runs, val_accs, marker='o', alpha=0.7, label='Actual Validation Accuracy')
if len(val_accs) > 1:
    plt.plot(runs, smoothed_accs, linewidth=2, color='red', label='Smoothed Curve')
plt.axhline(y=0.995, color='green', linestyle='--', linewidth=2, label='Training Threshold (0.995)')
plt.title("Average Validation Accuracy per Training Run")
plt.xlabel("Training Run")
plt.ylabel("Average Validation Accuracy")
plt.ylim(0, 1.05)
plt.grid(True, alpha=0.3)
plt.legend(loc='lower right')
plt.tight_layout()
plt.show()