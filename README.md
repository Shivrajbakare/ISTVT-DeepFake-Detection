# ISTVT-Transformer-For-Deepfake-deatection


This project implements a deepfake detection pipeline using a custom transformer-based architecture, **ISTVT (Interpretable Spatial-Temporal Video Transformer)**. It combines spatial artifact detection and temporal inconsistency analysis to identify manipulated facial videos with high accuracy and interpretability.

---

##  Objective

To develop a robust and interpretable system for detecting deepfake videos by leveraging spatial and temporal attention mechanisms that highlight manipulation cues in facial regions over time.

---

## Methodology

### 1. **Preprocessing**
- Applied **MTCNN** to detect facial landmarks and extract aligned face crops.
- Extracted face frames at a fixed rate (e.g., 7 FPS) for temporal consistency.
- Created fixed-length face sequences (e.g., 6 frames) normalized for model input.

### 2. **Feature Extraction**
- Used **Xception** (pretrained on ImageNet) to extract high-dimensional spatial embeddings from each frame.

### 3. **Tokenization**
- Constructed spatio-temporal token sequences, including:
  - Frame-wise classification tokens.
  - Spatial and temporal positional embeddings.

### 4. **Model Architecture**
- Passed token sequences through stacked **ISTVT blocks**, combining:
  - **Decomposed Spatial and Temporal Self-Attention**
  - **Residual Feature Subtraction** for detecting fine manipulation patterns.

### 5. **Training**
- Trained using **Binary Cross-Entropy Loss**.
- Optimized with **Adam** optimizer and scheduled learning rate.
- Tracked performance using **AUC** and **Accuracy** metrics.

### 6. **Interpretability**
- Visualized attention heatmaps to highlight model focus on tampered regions.
- Identified spatial and temporal inconsistencies contributing to predictions.

---

## Results

- Achieved test **accuracy of [95.32]%** and **AUC of [97.23]%** on [ FaceForensics++] dataset.
- Model consistently focused on manipulated facial areas such as **eyes, mouth**, and **jawline**.
- Detected **temporal anomalies** like unnatural blinking or inconsistent expressions.
- Generalized well across different compression levels and unseen video samples.
- Attention visualizations validated the **interpretability** of predictions.
- Inference time was suitable for real-time applications on short clips.

---

## Libraries and Dependencies

- `torch`, `torchvision`
- `torchaudio`
- `facenet-pytorch`
- `opencv-python`
- `numpy`, `scipy`
- `matplotlib`
- `tqdm`

---


