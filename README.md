# A Multi-Block Deep Architecture for Speech Emotion Recognition: Integrating CNN, BiGRU, Autoencoder, and Attention on RAVDESS

> **Course Project — Deep Learning with Python**
> **Team:** Adad · Ahmad · İpek · Omar · Taleb

---

## Abstract

Speech Emotion Recognition (SER) requires modeling spectral, temporal, and latent structure simultaneously, yet most systems rely on a single neural family. We present a **multi-block deep architecture** that integrates four distinct deep learning families — a Convolutional Neural Network (CNN), a CNN + Bidirectional Recurrent block (BiLSTM/BiGRU), an Autoencoder (AE), and a Transformer Attention encoder — fused by a fifth integration-and-classification block. The system is trained and evaluated on **RAVDESS** (Livingstone & Russo, 2018), a peer-reviewed corpus of eight emotional speech categories. Each architectural choice is justified empirically through controlled ablation: we show that **BiGRU outperforms BiLSTM** on this small-data regime, and that the **integrated model (71.18% test accuracy) outperforms every individual block and every reduced configuration**. We detail our preprocessing, augmentation, normalization, hyperparameter tuning, and regularization strategy, and we provide a full ablation study quantifying the contribution of each block.

---

## 1. Introduction

Automatic Speech Emotion Recognition is a core problem in affective computing, with applications in human–computer interaction, mental-health monitoring, and conversational agents. The task is intrinsically difficult: the same utterance can express different emotions purely through prosody, emotional cues are distributed unevenly across time, and expression varies sharply across speakers and recording conditions.

Classical approaches paired hand-crafted acoustic features with shallow classifiers. Modern deep learning learns hierarchical representations directly, but single-family models capture only one structural view of the signal. **Convolutions** excel at local spectral patterns; **recurrent networks** model temporal dynamics; **autoencoders** learn compact, denoised representations; **attention** dynamically focuses on salient frames. Our hypothesis is that these views are *complementary*, and that fusing them yields a stronger classifier than any single block.

This project follows the course requirement to combine **CNN, an RNN-family model (LSTM/GRU), and an Autoencoder** into a coherent integrated model. We exceed the baseline by including **five distinct blocks**, justifying each one through ablation, and documenting our methodology in full.

**Contributions.**
1. A coherent five-block architecture fusing CNN, BiGRU, AE, and Attention features.
2. A controlled **LSTM-vs-GRU ablation** showing BiGRU generalizes better on RAVDESS.
3. A full **block-level ablation** demonstrating that every block contributes to the final accuracy.
4. A leakage-safe preprocessing and augmentation pipeline, with detailed hyperparameter and regularization analysis.

---

## 2. Dataset and Rationale

### 2.1 RAVDESS

We use the **Ryerson Audio-Visual Database of Emotional Speech and Song (RAVDESS)**, introduced by Livingstone & Russo (2018) in *PLoS ONE*.

| Property | Value |
|---|---|
| Source | **Peer-reviewed research paper** (PLoS ONE, 2018) |
| DOI | https://doi.org/10.1371/journal.pone.0196391 |
| Actors | 24 professional actors (12 female, 12 male) |
| Utterances | 1,440 speech audio files |
| Emotions | 8 — neutral, calm, happy, sad, angry, fearful, disgust, surprised |
| Format | 16-bit, 48 kHz WAV |
| Language | North American English |

### 2.2 Why this dataset (and not a common benchmark)

We deliberately avoided over-common benchmarks such as MNIST / Fashion-MNIST. RAVDESS was selected because: (1) it originates from a **peer-reviewed publication** with rigorous perceptual validation; (2) it is **balanced and professionally recorded** across a demographically diverse actor set, controlling for confounds; and (3) its **eight-class** label space provides sufficient difficulty to meaningfully evaluate a multi-block architecture.

---

## 3. Preprocessing Pipeline

All blocks consume a single, shared preprocessing pipeline (`preprocess_ravdess.py`).

### 3.1 Feature Extraction

Two complementary feature views are extracted so that each block receives the representation best suited to it:

- **Mel spectrogram** (128 mel bins, FFT size 2048, hop length 512) → 2-D, image-like input for the **CNN** block.
- **MFCC + Δ + ΔΔ** (40 coefficients × 3 = 120 features/frame) → sequential input for the **recurrent, autoencoder, and attention** blocks.

Every clip is resampled to 22,050 Hz and padded/trimmed to 3.0 s for fixed-length tensors.

### 3.2 Splitting, Augmentation, and Normalization (Leakage-Safe)

The ordering below is critical and is enforced to prevent any information leakage:

1. **Stratified split first** — 70% train / 10% validation / 20% test, stratified by emotion.
2. **Augment training data only** (×4): time-stretching (rate 0.85), pitch shift (+2 semitones), and additive Gaussian noise (σ = 0.005). This expands the training set from **1,008 → 4,032** samples.
3. **Normalize with `StandardScaler` fit on the training set only**, then applied to validation/test.

| Split | Samples |
|---|---|
| Train (augmented ×4) | 4,032 |
| Validation | 144 |
| Test | 288 |

---

## 4. Architecture

### 4.1 Overview

```
                         Raw Audio (.wav)
                               │
                    ┌──────────┴───────────┐
                    ▼                      ▼
            Mel spectrogram          MFCC + Δ + ΔΔ
            (128×130×1)                (130×120)
                    │              ┌───────┼───────────────┐
                    ▼              ▼       ▼               ▼
            ┌──────────────┐ ┌──────────┐ ┌───────────┐ ┌──────────────┐
            │  Block 1     │ │ Block 2  │ │ Block 3   │ │  Block 4     │
            │  CNN         │ │ CNN +    │ │ Auto-     │ │  Transformer │
            │  (Ahmad)     │ │ BiGRU    │ │ encoder   │ │  Attention   │
            │              │ │ (İpek)   │ │ (Omar)    │ │  (Omar)      │
            └──────┬───────┘ └────┬─────┘ └─────┬─────┘ └──────┬───────┘
                   │              │             │              │
                   └──────────────┴──────┬──────┴──────────────┘
                                         ▼
                              ┌────────────────────┐
                              │  Block 5            │
                              │  Integration +      │
                              │  Classifier (Adad)  │
                              │  Concat → Dense →    │
                              │  softmax(8)         │
                              └─────────┬──────────┘
                                        ▼
                                Emotion (0–7)
```

### 4.2 Block 1 — Standalone CNN *(Ahmad)*

**Architecture.** Conv2D(32) → Conv2D(64) → Conv2D(128), each with BatchNorm + ReLU + MaxPool + Dropout(0.25), then GlobalAveragePooling and a Dense classifier.

**Justification.** Mel spectrograms are image-like: emotion correlates with local time–frequency patterns (formant bands, harmonic structure). 2-D convolutions are the natural inductive bias for such local structure. BatchNorm stabilizes training; spatial dropout combats overfitting on a small dataset.

### 4.3 Block 2 — CNN + Bidirectional RNN *(İpek)*

**Architecture.** Conv1D(64) → Conv1D(128) front-end (each + BatchNorm + ReLU + MaxPool + Dropout(0.3)), reducing the sequence length from 130 → ~32, followed by a **Bidirectional GRU(128)** with recurrent dropout, dropout, and an L2-regularized Dense head.

**Why GRU over LSTM (ablation, not assumption).** We ran a controlled comparison:

| Model | Test Accuracy | Macro-F1 | Val Accuracy |
|---|---|---|---|
| CNN + BiLSTM | 65.62% | 0.6448 | 66.67% |
| **CNN + BiGRU** ✓ | **68.75%** | **0.6733** | **70.83%** |

BiGRU outperforms BiLSTM by ~3 points with fewer parameters. GRU's simpler gating regularizes more effectively in our small-data regime.

### 4.4 Block 3 — Autoencoder *(Omar)*

**Architecture.** Encoder Dense(256) → Dense(128) → **Dense(64) latent**; symmetric decoder Dense(128) → Dense(256) → reconstruction. Trained unsupervised with MSE loss.

**Justification.** The autoencoder learns a compact, denoised 64-D representation. Low reconstruction error (**Test MSE = 0.8437**) indicates the latent code is informative. The latent space is visualized with t-SNE (`tsne_latent_space.png`).

### 4.5 Block 4 — Transformer Attention *(Omar)*

**Architecture.** Input projection (120 → 128) → Transformer encoder layer (Multi-Head Attention, 4 heads + Add & LayerNorm + FFN + Add & LayerNorm) → GlobalAveragePooling → Dense classifier.

**Justification.** Attention dynamically weights time frames by emotional relevance. Standalone: **51.39% test accuracy / 0.4965 macro-F1**; its 128-D pooled features contribute to the fused result.

### 4.6 Block 5 — Integration + Classifier *(Adad)*

**Architecture.** Concatenate(CNN features, BiGRU features, AE latent, Attention features) → Dense(256) + BatchNorm + ReLU + Dropout(0.4) → Dense(128) + ReLU + Dropout(0.3) → Dense(8, softmax).

**Justification.** Each block encodes a different structural view. Concatenating their representations lets the final classifier exploit all four jointly, outperforming every single block and every reduced configuration.

---

## 5. Hyperparameter Tuning & Regularization

| Hyperparameter | Value | Rationale |
|---|---|---|
| Optimizer | Adam | Robust adaptive optimizer |
| Learning rate | 1e-3 (initial) | Standard Adam starting point; reduced on plateau |
| Batch size | 32 | Balances gradient stability and throughput |
| Max epochs | 100 | Upper bound; early stopping terminates before this |
| Latent dim (AE) | 64 | Bottleneck small enough to force compression |
| Attention heads | 4 (key_dim 32) | d_model 128 split across 4 heads |
| RNN units | 128 (bidirectional) | Sufficient capacity without overfitting |

**Regularization techniques:**
- Dropout (0.25–0.4) in every block
- Recurrent dropout (0.2) on the RNN hidden state
- L2 weight decay on dense classifier layers
- BatchNormalization in CNN blocks
- Early Stopping (patience=12, restore_best_weights=True)
- ReduceLROnPlateau (factor 0.5, patience 5, min_lr 1e-6)
- Data augmentation (×4) at preprocessing level

---

## 6. Ablation Study

| Configuration | Test Accuracy | Δ vs Full |
|---|---|---|
| **Full model (all blocks)** | **71.18%** | — |
| w/o Standalone CNN | 65.97% | −5.21 |
| w/o Attention | 68.75% | −2.43 |
| w/o BiGRU | 67.71% | −3.47 |
| w/o Autoencoder | 70.83% | −0.35 |
| CNN only (baseline) | 62.50% | −8.68 |

**Findings:**
- The integrated model outperforms every single block and every reduced configuration.
- Removing the CNN causes the largest drop (−5.21), making it the most important contributor.
- The recurrent block (−3.47) and attention (−2.43) each contribute meaningfully.
- The autoencoder contributes the least (−0.35) but remains net-positive.

---

## 7. Results Summary

| Model / Configuration | Test Accuracy | Macro-F1 |
|---|---|---|
| CNN only (baseline) | 62.50% | 0.6040 |
| CNN + BiLSTM | 65.62% | 0.6448 |
| CNN + BiGRU | 68.75% | 0.6733 |
| Attention (standalone) | 51.39% | 0.4965 |
| Autoencoder (reconstruction MSE) | 0.8437 | — |
| **Integrated model (all 5 blocks)** | **71.18%** | **0.6997** |

---

## 8. Repository Structure

```
speech-emotion-recognition-dl/
├── README.md
├── requirements.txt
├── preprocess_ravdess.py        ← Preprocessing pipeline (Taleb)
├── cnn_block.py                 ← Standalone CNN (Ahmad)
├── cnn_bilstm.py                ← CNN + BiLSTM/BiGRU (İpek)
├── autoencoder.py               ← Autoencoder (Omar)
├── attention_block.py           ← Transformer Attention (Omar)
├── integrated_model.py          ← Full pipeline (Adad)
├── ablation_study.py            ← Ablation experiments (Adad)
├── ablation_results.png         ← Ablation bar chart
├── ablation_lstm_vs_gru_curves.png
├── confusion_matrix_best_birnn.png
├── tsne_latent_space.png
└── cnn_fix_comparison.png
```

---

## 9. Installation & Usage

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Preprocess (place Audio_Speech_Actors_01-24/ in project root)
python preprocess_ravdess.py

# 3. Train individual blocks
python cnn_block.py
python cnn_bilstm.py
python autoencoder.py
python attention_block.py

# 4. Build integrated model and run ablation
python integrated_model.py
python ablation_study.py
```

---

## 10. References

1. Livingstone, S. R., & Russo, F. A. (2018). The Ryerson Audio-Visual Database of Emotional Speech and Song (RAVDESS). *PLoS ONE*, 13(5), e0196391.
2. Hochreiter, S., & Schmidhuber, J. (1997). Long Short-Term Memory. *Neural Computation*, 9(8), 1735–1780.
3. Cho, K., et al. (2014). Learning Phrase Representations using RNN Encoder–Decoder for Statistical Machine Translation. *EMNLP 2014*.
4. Vaswani, A., et al. (2017). Attention Is All You Need. *NeurIPS 2017*.
5. Singh, J., Saheer, L. B., & Faust, O. (2023). Speech Emotion Recognition Using Attention Model. *IJERPH*, 20(6), 5140.

---

## 11. Team Contributions

| Member | Contribution |
|---|---|
| **Adad** | CNN retraining (fixed normalization errors), block integration, full ablation study, final presentation |
| **Ahmad** | Standalone CNN block |
| **İpek** | CNN + BiLSTM/BiGRU block; controlled LSTM-vs-GRU ablation; result analysis |
| **Omar** | Autoencoder + Transformer attention block; t-SNE latent visualization |
| **Taleb** | Dataset selection, preprocessing pipeline, repository setup |

---

## License

Submitted as part of a university Deep Learning course. RAVDESS is used under its original CC BY-NC-SA 4.0 license.
