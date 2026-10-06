# Explainable Multimodal AI for Precision Pathology

**Integrating Histopathology and Transcriptomics with Late Fusion and Fuzzy Logic for Survival Prediction in TCGA-KIRC**

## 📌 Overview & Clinical Utility

Predicting patient survival in **Clear Cell Renal Cell Carcinoma (TCGA-KIRC)** requires capturing multifaceted biological mechanisms that single-modality models often miss. This project addresses the "black-box" dilemma and the curse of dimensionality in precision oncology by introducing a lightweight, multimodal deep learning pipeline.

By integrating transcriptomics with statistical histopathological features using a **Late Fusion** strategy—coupled with a **Mamdani Fuzzy Inference System**—the framework translates raw neural probabilities into interpretable, actionable **Clinical Urgency Scores** (0–100).

> **🔗 Interactive CDSS Dashboard:** An interactive web application built with Streamlit has been implemented to simulate and visualize clinical risk scores, fuzzy membership functions, and model explainability.

## 🎯 Research Objectives

* **Multimodal Integration:** Bridge transcriptomic profiles (True HVGs) and whole-slide image (WSI) statistical descriptors using an asymmetric dual-branch neural architecture.
* **Dimensionality & Overhead Mitigation:** Bypass multi-gigabyte WSI memory overhead and high omics dimensionality via variance filtering and targeted statistical-textural image extraction.
* **Explainable AI (XAI) & Fuzzy Decision Support:** Replace opaque thresholding with an interpretable **Neuro-Fuzzy Clinical Decision Support System (CDSS)** using rule-based Mamdani inference to assist clinical decision-making.

## 🛠️️ Computational Pipeline

* **Data Acquisition & Curation:** Filtered and preprocessed **510 paired patient records** from the TCGA-KIRC cohort. Handled missing values using median imputation and isolated the top **1,000 Highly Variable Genes (HVGs)** via population variance.
* **Pathology Feature Extraction:** Extracted **13 statistical-textural descriptors**—including RGB channel mean, standard deviation, skewness, kurtosis, and Shannon entropy—to represent tissue heterogeneity without excessive computational overhead.
* **Dual-Branch Architecture & Late Fusion:**
  * **Omics Branch:** Built on a **Self-Normalizing Network (SNN)** featuring linear layers, `SELU` activation, and `AlphaDropout` (p = 0.1) to preserve gradient dynamics.
  * **Vision Branch:** Developed an **Attention-based Multiple Instance Learning (AMIL)** module projecting 13-dimensional morphological metrics into a 512-dimensional latent feature space.
  * **Classifier:** Concatenates both latent spaces and feeds them into a dense classifier with `Sigmoid` activation to predict binary survival risk.
* **Training & Regularization:** Trained with the **Adam optimizer** (learning rate: `5e-4`, weight decay: `1e-3`) and **Binary Cross-Entropy (BCE)** loss over 10 epochs. Integrated gradient clipping, probability clamping, and feature-level noise injection for stabilization.
* **Neuro-Fuzzy Inference:** Extracted global biomarker weights from the projection layer, mapping risk predictions and biomarker influence into triangular membership functions to generate crisp urgency ratings.

## 📊 Key Findings & Benchmarks

* **Biological Cohort Structure:** High-dimensional manifold projection using **t-SNE** confirmed significant genomic structure and variance capture across the top 1,000 HVGs.
* **Proof-of-Concept Baseline:** The model achieved **66.67% overall accuracy** on the 20% validation split (102 samples).
* **Class Imbalance Dynamics:** Highlighting typical challenges in proof-of-concept survival pipelines, class imbalance yielded an **ROC-AUC of 0.5000**, establishing an empirical baseline for future survival-loss formulations and real-world WSI integration.
* **Explainable Output Translation:** The fuzzy decision engine demonstrated precise, non-linear risk interpretation (e.g., converting an AI risk of `0.10` paired with high biomarker influence of `1.0` into a **Clinical Urgency Score of 83.33/100**).

## 📁 Repository Structure


├── Data/                   # Clinical and processed RNA-seq datasets (TCGA-KIRC)
├── Models/                 # Trained PyTorch weights (.pt) for SNN, AMIL, and late fusion
├── Notebooks/              # End-to-end training, preprocessing, and t-SNE projection notebooks
├── Results/                # Visualizations (t-SNE plots, fuzzy membership graphs, confusion matrices)
├── cdss_fuzzy/             # Mamdani fuzzy inference system scripts (scikit-fuzzy rules)
├── app.py                  # Interactive Streamlit dashboard for real-time risk assessment
├── requirements.txt        # Python dependencies (PyTorch, scikit-fuzzy, scikit-learn, etc.)
└── README.md               # Project documentation
