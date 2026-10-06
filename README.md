# Explainable-Multimodal-PathoOmics-KIRC

##🧬 Explainable Multimodal AI for Precision Pathology: Integrating Histopathology and Transcriptomics in Renal Carcinoma (TCGA-KIRC)
An end-to-end, production-grade Deep Learning proof-of-concept (PoC) pipeline that integrates high-dimensional transcriptomic profiles with statistical histopathological features for prognostic risk prediction in Clear Cell Renal Cell Carcinoma (TCGA-KIRC). This framework bridges the gap between deep learning black-box predictions and clinical translation using Explainable AI (XAI) and Neuro-Fuzzy logic.
📂 Repository Structure
├── app.py                            # Streamlit Clinical Decision Support System (CDSS) Dashboard
├── real_paired_tcga_kirc_1000hvg.csv # Aligned TCGA-KIRC clinical & transcriptomic dataset (1000 HVGs)
├── sample_dataloader_batch.pt        # Validated PyTorch multimodal batch tensor artifact
├── untrained_multimodal_model.pth    # Initialized model state dictionary
├── trained_multimodal_model.pth      # Optimized weights after 10 epochs of training
├── training_history.csv              # Training and validation loss tracking logs
├── clinical_evaluation_metrics.pdf   # Publication-ready ROC-AUC and Confusion Matrix vector plots
├── xai_feature_importance_weights.pdf# Global weight-based attribution plot for pathology biomarkers
└── fuzzy_clinical_inference.pdf      # Mamdani Fuzzy Inference System membership & urgency plots


##🚀 Key Architectural Highlights
Strict Reproducibility & Data Integrity: Implements rigorous global seed setting across NumPy, Random, and PyTorch (cudnn.deterministic = True), relying exclusively on authentic, paired TCGA-KIRC clinical and multi-omics GDC data.
Variance-Based Feature Selection (True HVGs): Bypasses the curse of dimensionality by cleaning and filtering low-variance features to extract the top 1,000 Highly Variable Genes.
Statistical Pathology Feature Extraction: Bypasses heavy multi-gigabyte WSI downloads in cloud notebooks by extracting 13 core statistical and textural moments (Mean, Std, Skewness, Kurtosis across RGB channels + Shannon Entropy).
Multimodal Late Fusion Architecture:
Omics Branch: Self-Normalizing Network (SNN) utilizing SELU activations and AlphaDropout.
Vision Branch: Attention-based Multiple Instance Learning (AMIL) specialized for high-dimensional feature projection and attention weighting.
Explainable AI (XAI): Global weight-based attribution analysis mapping learned network representations directly back to structural pathology signatures.
Neuro-Fuzzy Clinical Decision Support System (CDSS): Implements a Mamdani Fuzzy Inference System (scikit-fuzzy) utilizing triangular membership functions and clinical IF-THEN rules to translate raw neural network risk probabilities and biomarker impacts into a crisp Clinical Urgency Score (0-100).
##🛠️ Installation & Requirements
Ensure you have Python 3.11 or higher installed. Clone the repository and install the mandatory dependencies:
git clone https://github.com/your-username/multimodal-oncology-kirc.git
cd multimodal-oncology-kirc
pip install torch torchvision pandas numpy scikit-learn scipy scikit-image matplotlib seaborn streamlit scikit-fuzzy openslide-python


##🖥️ Running the Clinical Dashboard (Streamlit)
To launch the interactive Clinical Decision Support System dashboard locally:
streamlit run app.py


The application provides:
Patient selection and real metadata inspection.
Interactive 13-dimensional statistical pathology signature visualization.
Real-time computation of the Neuro-Fuzzy Clinical Urgency Score and automated medical intervention recommendations.
📊 Training Pipeline Summary
The model is trained using MLOps stability controls (gradient clipping, probability clamping, and lightweight data augmentation).
Optimizer: Adam (, )
Loss Function: Binary Cross-Entropy (BCE) Loss
Data Split: 80% Training (408 patients) | 20% Validation (102 patients)
📜 License
This project is released under the MIT License. See LICENSE for more details.
Developed as a high-performance scientific proof-of-concept for international peer-reviewed biomedical journals.
