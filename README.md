<p align="center">
  <img src="logo/UFC_TB_logo_final.png" width="600">
</p>

<h1 align="center">UFC-TB</h1>

<p align="center">
  <b>Unified Framework Classification for Tuberculosis</b>
</p>

<p align="center">
A lightweight Convolutional Neural Network (CNN) for tuberculosis classification from chest X-ray images using residual connections, depthwise convolution, and Grad-CAM interpretability.
</p>

<p align="center">

![Python](https://img.shields.io/badge/Python-3.10-blue)
![PyTorch](https://img.shields.io/badge/PyTorch-2.5.1-red)
![Status](https://img.shields.io/badge/Status-Completed-success)
![License](https://img.shields.io/badge/License-Academic-green)

</p>

---

# Overview

UFC-TB (*Unified Framework Classification for Tuberculosis*) is a lightweight deep learning framework developed from scratch for binary classification of chest X-ray images into **Normal** and **Tuberculosis (TB)** classes.

The architecture combines modern CNN design principles including residual connections and depthwise convolutions to achieve strong classification performance while maintaining computational efficiency.

In addition to classification performance, UFC-TB incorporates Grad-CAM visualization to provide insight into image regions contributing to model predictions.

---

# Key Features

- Lightweight CNN architecture built from scratch
- Residual Connection mechanism
- Depthwise Convolution for computational efficiency
- Inverted Residual Blocks
- Dual-Dropout Classifier Head
- Mixed Precision Training (FP16)
- Grad-CAM Interpretability
- Binary Chest X-ray Classification
- Class Imbalance Handling Strategy

---

# Dataset

This project uses the publicly available:

**Tuberculosis (TB) Chest X-ray Database**

Developed by:

> Rahman et al. (2020)

Dataset composition:

| Class | Images |
|---------|---------:|
| Normal | 3500 |
| Tuberculosis | 700 |
| Total | 4200 |

### Data Split (Stratified)

| Subset | Images |
|---------|---------:|
| Training | 2939 |
| Validation | 630 |
| Testing | 631 |

A stratified split was applied to preserve the original class distribution due to the approximately 1:5 imbalance ratio between tuberculosis and normal cases.

---

# Model Architecture

The UFC-TB architecture consists of:

- Enhanced Stem Block
- Stage 1 Inverted Residual Blocks
- Stage 2 Inverted Residual Blocks
- Stage 3 Inverted Residual Blocks
- Global Average Pooling
- Fully Connected Classifier Head

Total Trainable Parameters:

**1,164,386**

<p align="center">
  <img src="logo/Architecture.png" width="200">
</p>

<p align="center">
Architecture of UFC-TB
</p>

---

# Data Preprocessing

Input images are:

- Resized to 224 × 224 pixels
- Converted to RGB format
- Normalized using ImageNet statistics

### ImageNet Normalization

```python
mean = [0.485, 0.456, 0.406]
std  = [0.229, 0.224, 0.225]
```

---

# Data Augmentation

Data augmentation was applied exclusively to the training set.

Augmentation techniques include:

- Horizontal Flip
- Rotation (±10°)
- Translation (5%)
- Brightness Adjustment (±20%)
- Contrast Adjustment (±20%)

These transformations improve model generalization and reduce overfitting.

---

# Class Imbalance Handling

The dataset contains approximately five times more normal images than tuberculosis images.

To address this issue, UFC-TB employs a dual-strategy approach:

### WeightedRandomSampler

Balances mini-batches by increasing the probability of sampling minority-class images.

### Class-Weighted CrossEntropyLoss

Assigns larger penalties to misclassified tuberculosis samples during optimization.

---

# Training Configuration

| Parameter | Value |
|------------|------------|
| Optimizer | AdamW |
| Learning Rate | 1×10⁻⁴ |
| Weight Decay | 1×10⁻⁴ |
| Scheduler | ReduceLROnPlateau |
| Scheduler Patience | 3 |
| Scheduler Factor | 0.5 |
| Early Stopping | 5 |
| Epochs | 50 |
| Batch Size | 32 |
| Mixed Precision | FP16 |
| Device | NVIDIA GeForce RTX 2050 |

---

# Performance Evaluation

Model performance was evaluated using:

- Accuracy
- Precision
- Recall (Sensitivity)
- Specificity
- F1 Score
- AUROC
- Matthews Correlation Coefficient (MCC)

### Test Set Results

| Metric | Score |
|------------|------------:|
| Accuracy | 0.9810 |
| Precision | 0.9346 |
| Recall | 0.9524 |
| Specificity | 0.9867 |
| F1 Score | 0.9434 |
| AUROC | 0.9972 |
| MCC | 0.9320 |

---

# Training Curves

<p align="center">
  <img src="outputs/training_curves.png" width="900">
</p>

The learning curves demonstrate stable convergence throughout training with no significant indication of overfitting.

---

# Confusion Matrix

<p align="center">
  <img src="outputs/confusion_matrix.png" width="500">
</p>

The confusion matrix indicates strong classification performance across both classes with only a small number of false positives and false negatives.

---

# ROC Curve

<p align="center">
  <img src="outputs/roc_curve.png" width="500">
</p>

The model achieved an AUROC score of **0.9972**, indicating excellent discrimination capability between normal and tuberculosis chest X-ray images.

---

# Grad-CAM Visualization

Grad-CAM was employed to visualize image regions contributing to classification decisions.

<p align="center">
  <img src="outputs/gradcam.png" width="1000">
</p>

The resulting heatmaps highlight regions within the lungs that contribute most strongly to model predictions, providing additional interpretability and insight into the decision-making process.

---

# Repository Structure

```text
UFC_TB/
│
├── checkpoints/                # Saved model checkpoints
├── dataset/                    # Chest X-ray dataset
├── laporan/                    # Project report and documentation
├── logo/                       # UFC-TB logo assets
├── outputs/                    # Generated plots and visualizations
├── referensi/                  # Research papers and references
│
├── main.py                     # Main training/evaluation script
├── UFC_TB.ipynb                # Development notebook
├── UFC_TB_FINAL_IND.ipynb      # Final experiment notebook
│
├── pyproject.toml              # Dependency configuration
├── .python-version
├── .gitignore
└── README.md
```

---

# References

Andreu, J., Cáceres, J., Pallisa, E., & Martinez-Rodriguez, M. (2004). Radiological manifestations of pulmonary tuberculosis. European Journal of Radiology, 51(2), 139–149. https://doi.org/10.1016/j.ejrad.2004.03.009

Bhalla, A. S., Goyal, A., Guleria, R., & Gupta, A. K. (2015). Chest tuberculosis: Radiological review and imaging recommendations. Indian Journal of Radiology and Imaging, 25(3), 213–225. https://doi.org/10.4103/0971-3026.161431

Chen, J., Mei, J., Li, X., Lu, Y., Yu, Q., Wei, Q., Luo, X., Xie, Y., Adeli, E., Wang, Y., Lungren, M. P., Zhang, S., Xing, L., Lu, L., Yuille, A., & Zhou, Y. (2024). TransUNet: Rethinking the U-Net architecture design for medical image segmentation through the lens of transformers. Medical Image Analysis, 97. https://doi.org/10.1016/j.media.2024.103280

Deepak, G. D., & Bhat, S. K. (2025). A multi-stage deep learning approach for comprehensive lung disease classification from x-ray images. Expert Systems with Applications, 277. https://doi.org/10.1016/j.eswa.2025.127220

Esbergenov, H. S., & Baymuratovich, N. P. (2026). LIGHTWEIGHT CNN ARCHITECTURES FOR THREE-CLASS TUBERCULOSIS X-RAY CLASSIFICATION. Sun’iy Intellekt Ilmiy Jurnali, 4(2).

Fati, S. M., Senan, E. M., & ElHakim, N. (2022). Deep and Hybrid Learning Technique for Early Detection of Tuberculosis Based on X-ray Images Using Feature Fusion. Applied Sciences (Switzerland), 12(14). https://doi.org/10.3390/app12147092

Fawad, M., Baig, A. ullah, & Usmani, I. (2026). TB-LiteNet tuberculosis detection with a lightweight scratch-trained CNN using heterogeneous chest X-ray data. IMAGING. https://doi.org/10.1556/1647.2026.00376

Hammad, M. M. (2024). Artificial Neural Network and Deep Learning Fundamentals and Theory. https://orcid.org/0000-0003-0306-9719

Hansun, S., Argha, A., Liaw, S. T., Celler, B. G., & Marks, G. B. (2023). Machine and Deep Learning for Tuberculosis Detection on Chest X-Rays: Systematic Literature Review. In Journal of Medical Internet Research (Vol. 25). JMIR Publications Inc. https://doi.org/10.2196/43154

Kemenkes. (2024). LAPORAN HASIL STUDI INVENTORI TUBERKULOSIS INDONESIA 2023-2024.

Kocak, B., Klontzas, M. E., Stanzione, A., Meddeb, A., Demircioğlu, A., Bluethgen, C., Bressem, K. K., Ugga, L., Mercaldo, N., Díaz, O., & Cuocolo, R. (2025). Evaluation metrics in medical imaging AI: fundamentals, pitfalls, misapplications, and recommendations. European Journal of Radiology Artificial Intelligence, 3, 100030. https://doi.org/10.1016/j.ejrai.2025.100030

Murphy, K., Habib, S. S., Zaidi, S. M. A., Khowaja, S., Khan, A., Melendez, J., Scholten, E. T., Amad, F., Schalekamp, S., Verhagen, M., Philipsen, R. H. H. M., Meijers, A., & van Ginneken, B. (2020). Computer aided detection of tuberculosis on chest radiographs: An evaluation of the CAD4TB v6 system. Scientific Reports, 10(1). https://doi.org/10.1038/s41598-020-62148-y

Ou, C. Y., Chen, I. Y., Chang, H. T., Wei, C. Y., Li, D. Y., Chen, Y. K., & Chang, C. Y. (2024). Deep Learning-Based Classification and Semantic Segmentation of Lung Tuberculosis Lesions in Chest X-ray Images. Diagnostics, 14(9). https://doi.org/10.3390/diagnostics14090952

Owda, M., Abumihsan, A., Owda, A. Y., & Abumohsen, M. (2025). A Lightweight Hybrid Deep Learning Model for Tuberculosis Detection from Chest X-Rays. Diagnostics, 15(24). https://doi.org/10.3390/diagnostics15243216

Park, M., Lee, Y., Kim, S., Kim, Y. J., Kim, S. Y., Kim, Y., & Kim, H. M. (2023). Distinguishing nontuberculous mycobacterial lung disease and Mycobacterium tuberculosis lung disease on X-ray images using deep transfer learning. BMC Infectious Diseases, 23(1). https://doi.org/10.1186/s12879-023-07996-5

Powers, D. M. W., & Ailab. (2011). EVALUATION: FROM PRECISION, RECALL AND F-MEASURE TO ROC, INFORMEDNESS, MARKEDNESS & CORRELATION. 2(1), 37–63. http://www.bioinfo.in/contents.php?id=51

Rahman, T., Khandakar, A., Kadir, M. A., Islam, K. R., Islam, K. F., Mazhar, R., Hamid, T., Islam, M. T., Kashem, S., Mahbub, Z. Bin, Ayari, M. A., & Chowdhury, M. E. H. (2020). Reliable tuberculosis detection using chest X-ray with deep learning, segmentation and visualization. IEEE Access, 8, 191586–191601. https://doi.org/10.1109/ACCESS.2020.3031384

Rajpurohit, S. S., Shirsath, V. A., Parvat, T., Patil, P. R., Nagarhalli, T. P., & Save, A. M. (2025). Convolutional neural networks for automated tuberculosis detection in chest X-ray screening programs. In Indian Journal of Tuberculosis. Tuberculosis Association of India. https://doi.org/10.1016/j.ijtb.2025.11.010

Selvaraju, R. R., Cogswell, M., Das, A., Vedantam, R., Parikh, D., & Batra, D. (2017). Grad-CAM: Visual Explanations from Deep Networks via Gradient-based Localization. http://gradcam.cloudcv.org

Shapiro, L., & Stockman, G. (2000). Computer Vision.

Sokolova, M., & Lapalme, G. (2009). A systematic analysis of performance measures for classification tasks. Information Processing and Management, 45(4), 427–437. https://doi.org/10.1016/j.ipm.2009.03.002

Szeliski, R. (2021). Computer Vision: Algorithms and Applications 2nd Edition. https://szeliski.org/Book,
WHO. (2025). Global tuberculosis report 2025.

Wira, J., & Putra, G. (2020). Pengenalan Konsep Pembelajaran Mesin dan Deep Learning Edisi 1.4.
 
---

# License

This project is intended for educational, academic, and research purposes.

---

# Author

**Alfian**

Applied Statistics and Computing 

Universitas Negeri Semarang

Project: UFC-TB (*Unified Framework Classification for Tuberculosis*)
```