<div align="center">

# 🌿🦋 Flora & Fauna Image Classification 🐸🍄
### 🧠 A 45M-parameter ResNeXt-style CNN trained **completely from scratch**

**Deep Learning Practice (DLP) · IIT Madras BS Degree Program**

![Python](https://img.shields.io/badge/Python-3.10+-3776AB?style=for-the-badge&logo=python&logoColor=white)
![PyTorch](https://img.shields.io/badge/PyTorch-EE4C2C?style=for-the-badge&logo=pytorch&logoColor=white)
![Kaggle](https://img.shields.io/badge/Kaggle-20BEFF?style=for-the-badge&logo=kaggle&logoColor=white)
![Task](https://img.shields.io/badge/Task-Image_Classification-success?style=for-the-badge)
![Classes](https://img.shields.io/badge/Classes-10-orange?style=for-the-badge)
![Scratch](https://img.shields.io/badge/Trained-From_Scratch-critical?style=for-the-badge)
![No Transfer Learning](https://img.shields.io/badge/Transfer_Learning-None_🚫-lightgrey?style=for-the-badge)
![Val F1](https://img.shields.io/badge/Val_F1-0.4807-brightgreen?style=for-the-badge)

</div>

---

## 🎯 Problem Statement

Classify images from the world of **flora and fauna** 🌍 into **10 biological categories** with the best **F1 score**.

🚫 **Rules:** no transfer learning and no transformer models — everything here is **trained from scratch** with randomly initialised weights.

---

## 🗂️ Classes

| 🔢 Label | 🏷️ Class | 🎨 | 🔢 Label | 🏷️ Class | 🎨 |
|---|---|---|---|---|---|
| 0 | Amphibia | 🐸 | 5 | Insecta | 🦋 |
| 1 | Animalia | 🐾 | 6 | Mammalia | 🦁 |
| 2 | Arachnida | 🕷️ | 7 | Mollusca | 🐌 |
| 3 | Aves | 🐦 | 8 | Plantae | 🌱 |
| 4 | Fungi | 🍄 | 9 | Reptilia | 🦎 |

---

## 📦 Dataset

| 📁 Split | 🔢 Size |
|---|---|
| 🏋️ Train | 8,499 images (85%) |
| 🧪 Validation | 1,500 images (15%) |
| 🚀 Test | 2,000 images |

📏 **Metric:** weighted **F1 score (micro)**  
📤 **Submission:** CSV with `Image_ID`, `Label` (0–9)

---

## 🏆 Results

| 📊 Metric | 🎯 Score |
|---|---|
| 🥇 **Best validation F1 (micro)** | **0.4807** (epoch 74 / 80) |
| Macro F1 | 0.4771 |
| Weighted F1 | 0.4793 |

### 🔬 Per-class F1 on validation

| Class | Precision | Recall | F1 |
|---|---|---|---|
| 🕷️ Arachnida | 0.7477 | 0.5646 | **0.6434** 🥇 |
| 🐦 Aves | 0.6024 | 0.5848 | 0.5935 |
| 🌱 Plantae | 0.5045 | 0.6957 | 0.5849 |
| 🐾 Animalia | 0.5000 | 0.5321 | 0.5155 |
| 🍄 Fungi | 0.5556 | 0.4610 | 0.5039 |
| 🦁 Mammalia | 0.4286 | 0.5538 | 0.4832 |
| 🦋 Insecta | 0.4362 | 0.4545 | 0.4452 |
| 🐌 Mollusca | 0.5309 | 0.2792 | 0.3660 |
| 🦎 Reptilia | 0.2607 | 0.4044 | 0.3170 |
| 🐸 Amphibia | 0.3945 | 0.2671 | 0.3185 |

> 💡 Amphibia, Reptilia and Mollusca are the hardest classes — they look very alike (and often blend into their surroundings), which is tough for a from-scratch model.

📈 Val F1 climbed steadily from **0.208 → 0.481** over 80 epochs, with no sign of collapse.

---

## 🧠 Model Architecture

A **ResNeXt / Wide-ResNet style** network with **grouped bottleneck blocks** — all weights randomly initialised (Kaiming) 🎲

```text
🖼️ Input 224×224×3
   │
   ▼
🔰 Stem: 7×7 conv (stride 2) → BN → ReLU → MaxPool
   │
   ▼
🧱 Layer 1: 3 × Bottleneck  (64  → 256 ch)
🧱 Layer 2: 4 × Bottleneck  (128 → 512 ch,  stride 2)
🧱 Layer 3: 6 × Bottleneck  (256 → 1024 ch, stride 2)
🧱 Layer 4: 3 × Bottleneck  (512 → 2048 ch, stride 2)
   │        (grouped 3×3 conv, groups = 2)
   ▼
🌐 Global Average Pooling
   │
   ▼
🎯 Head: Dropout 0.4 → Linear 2048→512 → LayerNorm → ReLU → Dropout 0.3 → Linear 512→10
```

| ⚙️ Property | Value |
|---|---|
| 🔢 Parameters | **45.25M** |
| 🎲 Initialisation | Kaiming normal (convs), Xavier (linear) |
| 🚫 Pretrained weights | **None** |

---

## 🛠️ Training Recipe

| 🎛️ Setting | ✅ Value |
|---|---|
| 📐 Image size | 224 (resize 256 → crop) |
| 📦 Batch size | 96 |
| 🔁 Epochs | 80 |
| 🎓 Optimizer | AdamW (`lr = 3e-4`, weight decay `1e-4`) |
| 🌀 Scheduler | Cosine annealing → `1e-6` (single cycle) |
| 📉 Loss | Cross-Entropy + **label smoothing 0.1** + **class weights** |
| ⚡ Precision | Mixed precision (AMP) |
| ✂️ Grad clipping | 1.0 |
| 💾 Checkpointing | Best validation F1 |
| 🎲 Seed | 42 |

### 🎨 Data Augmentation

- 🔄 Horizontal / vertical flips, rotation up to 30°
- 🔍 RandomResizedCrop (scale 0.6 – 1.0)
- 🌈 ColorJitter + RandomGrayscale
- 🩹 RandomErasing
- 🧪 **MixUp** (α = 0.4) and ✂️ **CutMix** (α = 1.0) after epoch 5 — applied to ~40% / ~40% of batches

### 🔮 Test-Time Augmentation (TTA)

Predictions average softmax probabilities over **5 views** 🖼️:
original · horizontal flip · vertical flip · larger-scale crop · 90° rotation

---

## 🌟 Key Features

- 🚫 **100% from scratch** — complies with the no-transfer-learning rule
- ⚖️ **Class-weighted loss** to handle class imbalance
- 🧪 **MixUp + CutMix** for strong regularisation
- 🔮 **5-view TTA** at inference
- 🖥️ Multi-GPU ready (`DataParallel`) with mixed precision
- 📊 Full classification report on validation

---

## 💡 Key Learnings

- 🌱 Training a deep CNN from scratch on ~8.5K images is hard — **heavy augmentation and regularisation are essential**.
- 📈 A **long cosine schedule (80 epochs)** kept improving validation F1 until the very end.
- ⚖️ Fine-grained, visually similar classes (Reptilia / Amphibia / Mollusca) dominate the errors.
- 🔮 TTA is a cheap way to squeeze extra accuracy at inference time.

---

## 🔮 Future Work

- 🖼️ Higher input resolution (e.g. 288 / 320)
- 🤝 **Ensembling** several from-scratch models / seeds
- 🧬 Deeper SE / attention blocks or different from-scratch backbones
- 🧹 Hard-class mining for Amphibia, Reptilia and Mollusca
- 🔁 Longer schedules with SGD + warm restarts

---

## 🚀 How to Run

1. 📥 Open the notebook on **Kaggle** and add the competition dataset
2. ⚙️ Settings → Accelerator → **GPU** 🎮
3. ▶️ **Run All** — best checkpoint is saved as `best_model.pth`, predictions in `submission.csv`

```bash
pip install torch torchvision scikit-learn pandas pillow tqdm
```

---

## 🗂️ Repository Structure

```text
📦 DLP-Flora-Fauna-Image-Classification
 ┣ 📔 dlp-flora-fauna-classification-scratch.ipynb
 ┗ 📄 README.md
```

---

<div align="center">

### 👨‍💻 Author
**Saini** · (CyberSoul) 🎓

⭐ If this helped you, drop a star on the repo! ⭐

*Made with ❤️, ☕ and a lot of GPU hours* 🔥

</div>
