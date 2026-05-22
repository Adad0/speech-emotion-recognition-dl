# -*- coding: utf-8 -*-
# =============================================================================
#  BLOCK 1 — Standalone CNN   (Author: Ahmad, retrained by Adad)
#  Speech Emotion Recognition on RAVDESS
#  Input: Mel Spectrogram (128 x 130 x 1)
# =============================================================================

import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import (
    Conv2D, MaxPooling2D, BatchNormalization,
    Dropout, GlobalAveragePooling2D, Dense, Input
)
from tensorflow.keras.callbacks import EarlyStopping

# =============================================================================
# CONFIG — update DATA_DIR to your local path
# =============================================================================
DATA_DIR   = "./processed_data"   # folder containing .npy files
OUTPUT_DIR = "./models"

# =============================================================================
# LOAD DATA
# Data is already normalized (StandardScaler applied in preprocess_ravdess.py)
# DO NOT normalize again.
# =============================================================================
X_train = np.load(f"{DATA_DIR}/X_train_mel.npy")
X_val   = np.load(f"{DATA_DIR}/X_val_mel.npy")
y_train = np.load(f"{DATA_DIR}/y_train.npy")
y_val   = np.load(f"{DATA_DIR}/y_val.npy")

print("X_train shape:", X_train.shape)
print("X_val shape  :", X_val.shape)
print(f"mean={X_train.mean():.4f}  std={X_train.std():.4f}  (should be ≈0 and ≈1)")

# =============================================================================
# MODEL
# Architecture:
#   Conv2D(32) + BN + MaxPool + Dropout(0.25)
#   Conv2D(64) + BN + MaxPool + Dropout(0.25)
#   Conv2D(128) + BN + MaxPool + Dropout(0.25)
#   GlobalAveragePooling2D
#   Dense(128, relu) + Dropout(0.3)
#   Dense(8, softmax)
#
# Justification:
#   2D convolutions treat mel spectrograms like images — local frequency
#   patterns (harmonics, formants) are captured by spatial filters.
#   BatchNorm stabilizes training; Dropout prevents overfitting.
#   GlobalAveragePooling replaces Flatten to reduce parameters.
# =============================================================================
tf.keras.backend.clear_session()

model = Sequential([
    Input(shape=(128, 130, 1)),

    # Block 1
    Conv2D(32, (3, 3), activation='relu', padding='same'),
    BatchNormalization(),
    MaxPooling2D((2, 2)),
    Dropout(0.25),

    # Block 2
    Conv2D(64, (3, 3), activation='relu', padding='same'),
    BatchNormalization(),
    MaxPooling2D((2, 2)),
    Dropout(0.25),

    # Block 3
    Conv2D(128, (3, 3), activation='relu', padding='same'),
    BatchNormalization(),
    MaxPooling2D((2, 2)),
    Dropout(0.25),

    GlobalAveragePooling2D(),
    Dense(128, activation='relu'),
    Dropout(0.3),
    Dense(8, activation='softmax')
], name="cnn_block")

model.summary()

# =============================================================================
# TRAIN
# Hyperparameters:
#   optimizer : Adam (lr=1e-3) — standard adaptive optimizer
#   loss      : sparse_categorical_crossentropy — integer labels
#   epochs    : 100 with EarlyStopping (patience=10)
#   batch_size: 32
# =============================================================================
model.compile(
    optimizer='adam',
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

early_stop = EarlyStopping(
    monitor='val_loss',
    patience=10,
    restore_best_weights=True,
    verbose=1
)

history = model.fit(
    X_train, y_train,
    validation_data=(X_val, y_val),
    epochs=100,
    batch_size=32,
    callbacks=[early_stop],
    verbose=1
)

# =============================================================================
# EVALUATE
# =============================================================================
val_loss, val_acc = model.evaluate(X_val, y_val, verbose=0)
print(f"\nValidation Accuracy : {val_acc:.4f}")
print(f"Validation Loss     : {val_loss:.4f}")

# =============================================================================
# SAVE
# =============================================================================
import os
os.makedirs(OUTPUT_DIR, exist_ok=True)
model.save(f"{OUTPUT_DIR}/cnn_block.keras")
print("Model saved to models/cnn_block.keras")
