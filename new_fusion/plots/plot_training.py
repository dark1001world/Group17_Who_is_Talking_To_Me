#!/usr/bin/env python3
"""
Script to plot training and validation loss from fusion training log.
"""

import matplotlib.pyplot as plt
import numpy as np

# Data extracted from train_fusion_log2.txt
epochs = np.arange(1, 11)

# Training loss (final step loss for each epoch)
train_loss = np.array([0.7678, 0.5861, 0.5442, 0.5334, 0.5138, 0.5208, 0.5152, 0.5160, 0.5083, 0.4859])

# Validation loss
val_loss = np.array([0.7495, 0.7479, 0.7460, 0.7444, 0.7434, 0.7427, 0.7423, 0.7419, 0.7418, 0.7418])

# Create the plot
plt.figure(figsize=(10, 6))
plt.plot(epochs, train_loss, 'b-o', label='Training Loss', linewidth=2, markersize=8)

plt.xlabel('Epoch', fontsize=12)
plt.ylabel('Loss', fontsize=12)
plt.title('Fusion Training: Training Loss Only', fontsize=14)
plt.legend(['Training Loss'], fontsize=11)
plt.grid(True, alpha=0.3)
plt.xticks(epochs)

# Add annotations for final values
plt.annotate(f'{train_loss[-1]:.4f}', xy=(epochs[-1], train_loss[-1]), 
             xytext=(epochs[-1]+0.3, train_loss[-1]+0.01),
             fontsize=10, color='blue')

plt.tight_layout()
plt.savefig('/DATA/G17/Group17_Who_is_Talking_To_Me/new_fusion/plots/training_validation_loss.png', dpi=150)
plt.close()

print("Plot saved to: /DATA/G17/Group17_Who_is_Talking_To_Me/new_fusion/plots/training_validation_loss.png")