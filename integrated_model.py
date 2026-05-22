# -*- coding: utf-8 -*-
# =============================================================================
#  BLOCK 5 — Integration + Classifier   (Author: Adad)
#  Speech Emotion Recognition on RAVDESS
#  Connects: CNN (Ahmad) + CNN+BiLSTM (İpek) + AE latent (Omar) + Attention (Omar)
# =============================================================================

import os, sys, json
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.layers import Input, Dense, Dropout, BatchNormalization, Activation
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from sklearn.metrics import accuracy_score, f1_score, classification_report, confusion_matrix

# =============================================================================
# CONFIG — update paths to your local setup
# =============================================================================
DATA_DIR   = "./processed_data"
MODELS_DIR = "./models"

# =============================================================================
# Custom Transformer Layer (required to load Omar's attention model)
# =============================================================================
class TransformerEncoderBlock(tf.keras.layers.Layer):
    def __init__(self, d_model, num_heads=4, ffn_units=256, dropout=0.1, **kwargs):
        super().__init__(**kwargs)
        self.d_model = d_model; self.num_heads = num_heads
        self.ffn_units = ffn_units; self.dropout_rate = dropout
        self.mha = tf.keras.layers.MultiHeadAttention(num_heads=num_heads, key_dim=d_model//num_heads, dropout=dropout)
        self.dropout1 = tf.keras.layers.Dropout(dropout)
        self.norm1 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        self.ffn_dense1 = tf.keras.layers.Dense(ffn_units, activation="relu")
        self.ffn_dense2 = tf.keras.layers.Dense(d_model, activation="linear")
        self.dropout2 = tf.keras.layers.Dropout(dropout)
        self.norm2 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
    def call(self, x, training=False):
        attn_out = self.mha(x, x, x, training=training)
        attn_out = self.dropout1(attn_out, training=training)
        x = self.norm1(x + attn_out)
        ffn_out = self.ffn_dense2(self.ffn_dense1(x))
        ffn_out = self.dropout2(ffn_out, training=training)
        return self.norm2(x + ffn_out)
    def get_config(self):
        cfg = super().get_config()
        cfg.update({"d_model": self.d_model, "num_heads": self.num_heads, "ffn_units": self.ffn_units, "dropout": self.dropout_rate})
        return cfg

CUSTOM = {"TransformerEncoderBlock": TransformerEncoderBlock}

# =============================================================================
# STEP 1 — Load data
# =============================================================================
def _load(name): return np.load(os.path.join(DATA_DIR, name))

y_train = _load("y_train.npy"); y_val = _load("y_val.npy"); y_test = _load("y_test.npy")
X_train_mel  = _load("X_train_mel.npy");  X_val_mel  = _load("X_val_mel.npy");  X_test_mel  = _load("X_test_mel.npy")
X_train_mfcc = _load("X_train_mfcc.npy"); X_val_mfcc = _load("X_val_mfcc.npy"); X_test_mfcc = _load("X_test_mfcc.npy")
ae_train   = _load("ae_latent_train.npy");   ae_val   = _load("ae_latent_val.npy");   ae_test   = _load("ae_latent_test.npy")
attn_train = _load("attn_features_train.npy"); attn_val = _load("attn_features_val.npy"); attn_test = _load("attn_features_test.npy")
print("Data loaded ✅")

# =============================================================================
# STEP 2 — Extract CNN features (Ahmad's model)
# =============================================================================
print("Loading CNN model...")
cnn_model = load_model(f"{MODELS_DIR}/cnn_block.keras", custom_objects=CUSTOM)
cnn_model(tf.zeros((1, 128, 130, 1)))
cnn_input = tf.keras.Input(shape=(128, 130, 1))
cnn_out = cnn_input
for layer in cnn_model.layers[:-2]:
    cnn_out = layer(cnn_out)
cnn_feat = tf.keras.Model(cnn_input, cnn_out)
cnn_train = cnn_feat.predict(X_train_mel, verbose=0)
cnn_val   = cnn_feat.predict(X_val_mel,   verbose=0)
cnn_test  = cnn_feat.predict(X_test_mel,  verbose=0)
print("CNN features extracted ✅", cnn_train.shape)

# =============================================================================
# STEP 3 — Extract BiLSTM features (İpek's model)
# =============================================================================
print("Loading BiLSTM model...")
bilstm_model = load_model(f"{MODELS_DIR}/cnn_bilstm.keras", custom_objects=CUSTOM)
bilstm_feat  = Model(bilstm_model.input, bilstm_model.get_layer("dense1").output)
bilstm_train = bilstm_feat.predict(X_train_mfcc, verbose=0)
bilstm_val   = bilstm_feat.predict(X_val_mfcc,   verbose=0)
bilstm_test  = bilstm_feat.predict(X_test_mfcc,  verbose=0)
print("BiLSTM features extracted ✅", bilstm_train.shape)

# =============================================================================
# STEP 4 — Concatenate all features
# CNN(128) + BiLSTM(64) + AE(64) + Attention(128) = 384
# =============================================================================
X_train_int = np.concatenate([cnn_train, bilstm_train, ae_train, attn_train], axis=1)
X_val_int   = np.concatenate([cnn_val,   bilstm_val,   ae_val,   attn_val],   axis=1)
X_test_int  = np.concatenate([cnn_test,  bilstm_test,  ae_test,  attn_test],  axis=1)
INPUT_DIM = X_train_int.shape[1]
print(f"Concatenated dim: {INPUT_DIM}")

# =============================================================================
# STEP 5 — Build Integration Model
# Architecture (from README):
#   Dense(256) + BatchNorm + ReLU + Dropout(0.4)
#   Dense(128) + ReLU + Dropout(0.3)
#   Dense(8, softmax)
# =============================================================================
inp = Input(shape=(INPUT_DIM,))
x   = Dense(256)(inp); x = BatchNormalization()(x); x = Activation("relu")(x); x = Dropout(0.4)(x)
x   = Dense(128, activation="relu")(x); x = Dropout(0.3)(x)
out = Dense(8, activation="softmax")(x)
integration_model = tf.keras.Model(inp, out, name="integration_block")
integration_model.summary()

# =============================================================================
# STEP 6 — Train
# Hyperparameters:
#   optimizer: Adam (lr=1e-3)
#   loss: sparse_categorical_crossentropy
#   epochs: 100 with EarlyStopping (patience=15)
#   batch_size: 32
# =============================================================================
integration_model.compile(
    optimizer=tf.keras.optimizers.Adam(1e-3),
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"]
)
cb = [
    EarlyStopping(monitor="val_loss", patience=15, restore_best_weights=True, verbose=1),
    ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=5, min_lr=1e-6, verbose=1)
]
history = integration_model.fit(
    X_train_int, y_train,
    validation_data=(X_val_int, y_val),
    epochs=100, batch_size=32, callbacks=cb, verbose=1
)

# =============================================================================
# STEP 7 — Evaluate
# =============================================================================
y_pred    = integration_model.predict(X_test_int, verbose=0).argmax(axis=1)
test_acc  = accuracy_score(y_test, y_pred)
test_f1   = f1_score(y_test, y_pred, average="macro")
print(f"\nTest Accuracy : {test_acc:.4f}")
print(f"Test Macro-F1 : {test_f1:.4f}")

try:
    with open(os.path.join(DATA_DIR, "label_map.json")) as f:
        LABEL_MAP = json.load(f)
    CLASS_NAMES = [LABEL_MAP[str(i)] for i in range(8)]
except:
    CLASS_NAMES = ["neutral","calm","happy","sad","angry","fearful","disgust","surprised"]

print(classification_report(y_test, y_pred, target_names=CLASS_NAMES, digits=3, zero_division=0))

# =============================================================================
# STEP 8 — Save
# =============================================================================
os.makedirs(MODELS_DIR, exist_ok=True)
integration_model.save(f"{MODELS_DIR}/integrated_model.keras")

results = {
    "test_accuracy": round(float(test_acc), 4),
    "test_macro_f1": round(float(test_f1), 4),
    "input_dim": int(INPUT_DIM),
    "feature_breakdown": {
        "CNN": int(cnn_train.shape[1]),
        "BiLSTM": int(bilstm_train.shape[1]),
        "AE": int(ae_train.shape[1]),
        "Attention": int(attn_train.shape[1])
    }
}
with open("integration_results.json", "w") as f:
    json.dump(results, f, indent=2)
print("✅ integrated_model.keras saved!")
