# -*- coding: utf-8 -*-
# =============================================================================
#  ABLATION STUDY   (Author: Adad)
#  Speech Emotion Recognition on RAVDESS
#  Removes one block at a time and measures Test Accuracy + Macro F1
# =============================================================================

import os, json
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.layers import Input, Dense, Dropout, BatchNormalization, Activation
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from sklearn.metrics import accuracy_score, f1_score

DATA_DIR   = "./processed_data"
MODELS_DIR = "./models"

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
        x = self.norm1(x + self.dropout1(attn_out, training=training))
        return self.norm2(x + self.dropout2(self.ffn_dense2(self.ffn_dense1(x)), training=training))
    def get_config(self):
        cfg = super().get_config()
        cfg.update({"d_model": self.d_model, "num_heads": self.num_heads, "ffn_units": self.ffn_units, "dropout": self.dropout_rate})
        return cfg

CUSTOM = {"TransformerEncoderBlock": TransformerEncoderBlock}

# Load data
def _load(name): return np.load(os.path.join(DATA_DIR, name))
y_train = _load("y_train.npy"); y_val = _load("y_val.npy"); y_test = _load("y_test.npy")
X_train_mel  = _load("X_train_mel.npy");  X_val_mel  = _load("X_val_mel.npy");  X_test_mel  = _load("X_test_mel.npy")
X_train_mfcc = _load("X_train_mfcc.npy"); X_val_mfcc = _load("X_val_mfcc.npy"); X_test_mfcc = _load("X_test_mfcc.npy")
ae_train = _load("ae_latent_train.npy"); ae_val = _load("ae_latent_val.npy"); ae_test = _load("ae_latent_test.npy")
attn_train = _load("attn_features_train.npy"); attn_val = _load("attn_features_val.npy"); attn_test = _load("attn_features_test.npy")

# CNN features
cnn_model = load_model(f"{MODELS_DIR}/cnn_block.keras", custom_objects=CUSTOM)
cnn_model(tf.zeros((1, 128, 130, 1)))
cnn_in = tf.keras.Input(shape=(128, 130, 1))
cnn_out = cnn_in
for layer in cnn_model.layers[:-2]: cnn_out = layer(cnn_out)
cnn_feat = tf.keras.Model(cnn_in, cnn_out)
cnn_train = cnn_feat.predict(X_train_mel, verbose=0)
cnn_val   = cnn_feat.predict(X_val_mel,   verbose=0)
cnn_test  = cnn_feat.predict(X_test_mel,  verbose=0)

# BiLSTM features
bilstm_model = load_model(f"{MODELS_DIR}/cnn_bilstm.keras", custom_objects=CUSTOM)
bilstm_feat  = Model(bilstm_model.input, bilstm_model.get_layer("dense1").output)
bilstm_train = bilstm_feat.predict(X_train_mfcc, verbose=0)
bilstm_val   = bilstm_feat.predict(X_val_mfcc,   verbose=0)
bilstm_test  = bilstm_feat.predict(X_test_mfcc,  verbose=0)

print("All features ready ✅")

def run_ablation(name, tr, va, te):
    inp = Input(shape=(tr.shape[1],))
    x = Dense(256)(inp); x = BatchNormalization()(x); x = Activation("relu")(x); x = Dropout(0.4)(x)
    x = Dense(128, activation="relu")(x); x = Dropout(0.3)(x)
    out = Dense(8, activation="softmax")(x)
    m = tf.keras.Model(inp, out)
    m.compile(optimizer=tf.keras.optimizers.Adam(1e-3), loss="sparse_categorical_crossentropy", metrics=["accuracy"])
    cb = [EarlyStopping(monitor="val_loss", patience=15, restore_best_weights=True, verbose=0),
          ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=5, min_lr=1e-6, verbose=0)]
    m.fit(tr, y_train, validation_data=(va, y_val), epochs=100, batch_size=32, callbacks=cb, verbose=0)
    y_pred = m.predict(te, verbose=0).argmax(axis=1)
    acc = accuracy_score(y_test, y_pred)
    f1  = f1_score(y_test, y_pred, average="macro")
    print(f"  {name}: Acc={acc:.4f}  F1={f1:.4f}")
    return {"configuration": name, "test_accuracy": round(acc, 4), "macro_f1": round(f1, 4)}

results = []
results.append(run_ablation("Full model (all blocks)",
    np.concatenate([cnn_train, bilstm_train, ae_train, attn_train], axis=1),
    np.concatenate([cnn_val,   bilstm_val,   ae_val,   attn_val],   axis=1),
    np.concatenate([cnn_test,  bilstm_test,  ae_test,  attn_test],  axis=1)))

results.append(run_ablation("w/o CNN",
    np.concatenate([bilstm_train, ae_train, attn_train], axis=1),
    np.concatenate([bilstm_val,   ae_val,   attn_val],   axis=1),
    np.concatenate([bilstm_test,  ae_test,  attn_test],  axis=1)))

results.append(run_ablation("w/o BiLSTM",
    np.concatenate([cnn_train, ae_train, attn_train], axis=1),
    np.concatenate([cnn_val,   ae_val,   attn_val],   axis=1),
    np.concatenate([cnn_test,  ae_test,  attn_test],  axis=1)))

results.append(run_ablation("w/o Autoencoder",
    np.concatenate([cnn_train, bilstm_train, attn_train], axis=1),
    np.concatenate([cnn_val,   bilstm_val,   attn_val],   axis=1),
    np.concatenate([cnn_test,  bilstm_test,  attn_test],  axis=1)))

results.append(run_ablation("w/o Attention",
    np.concatenate([cnn_train, bilstm_train, ae_train], axis=1),
    np.concatenate([cnn_val,   bilstm_val,   ae_val],   axis=1),
    np.concatenate([cnn_test,  bilstm_test,  ae_test],  axis=1)))

results.append(run_ablation("CNN only (baseline)", cnn_train, cnn_val, cnn_test))

with open("ablation_results.json", "w") as f:
    json.dump(results, f, indent=2)

print("\n=== README Markdown Table ===")
print("| Configuration | Test Acc | Macro F1 |")
print("|---|---|---|")
for r in results:
    print(f"| {r['configuration']} | {r['test_accuracy']:.4f} | {r['macro_f1']:.4f} |")

# Bar chart
labels = [r["configuration"] for r in results]
accs   = [r["test_accuracy"] for r in results]
f1s    = [r["macro_f1"] for r in results]
x = np.arange(len(labels)); width = 0.35
fig, ax = plt.subplots(figsize=(14, 6))
ax.bar(x - width/2, accs, width, label="Test Accuracy", color="#4C72B0")
ax.bar(x + width/2, f1s,  width, label="Macro F1",      color="#DD8452")
ax.set_title("Ablation Study"); ax.set_ylabel("Score")
ax.set_xticks(x); ax.set_xticklabels(labels, rotation=30, ha="right", fontsize=9)
ax.set_ylim(0, 1.05); ax.legend(); ax.grid(axis="y", alpha=0.3)
plt.tight_layout()
plt.savefig("ablation_results.png", dpi=150, bbox_inches="tight")
plt.show()
print("✅ ablation_results.json and ablation_results.png saved!")
