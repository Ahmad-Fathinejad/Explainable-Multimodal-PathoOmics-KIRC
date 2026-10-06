import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import skfuzzy as fuzz
from skfuzzy import control as ctrl

st.set_page_config(page_title="Multimodal Oncology PoC", layout="wide")

st.title("🧬 Multimodal AI Oncology Dashboard (TCGA-KIRC)")
st.markdown("### Clinical Decision Support System - Proof of Concept")

@st.cache_data
def load_data():
    return pd.read_csv('real_paired_tcga_kirc_1000hvg.csv')

df = load_data()

st.sidebar.header("Patient Selection")
patient_id = st.sidebar.selectbox("Select Patient ID:", df['PATIENT_ID'].tolist())

# Patient Data Display
st.subheader(f"Patient Profile: {patient_id}")
patient_data = df[df['PATIENT_ID'] == patient_id].iloc[0]
true_status = "Deceased" if patient_data['OS_STATUS_BIN'] == 1 else "Living"

col1, col2, col3 = st.columns(3)
col1.metric("True Survival Status", true_status)
col2.metric("Overall Survival (Months)", f"{patient_data['OS_MONTHS']} mo")
col3.metric("Processed Omics", "1000 True HVGs")

st.divider()

# XAI & Statistical Pathology Profile Display
st.subheader("1. Explainable AI (XAI) & Pathology Profile")
st.info("Displaying the 13-dimensional statistical signature of the patient's pathology slide.")

feature_names = [
    "Red-Mean", "Red-Std", "Red-Skew", "Red-Kurtosis",
    "Green-Mean", "Green-Std", "Green-Skew", "Green-Kurtosis",
    "Blue-Mean", "Blue-Std", "Blue-Skew", "Blue-Kurtosis",
    "Shannon Entropy"
]

# Simulating pathology features for PoC Dashboard
dummy_patient_features = np.random.uniform(0.1, 0.9, 13)

fig, ax = plt.subplots(figsize=(8, 3))
sns.barplot(x=dummy_patient_features, y=feature_names, palette="viridis", ax=ax, orient='h')
ax.set_title(f"Statistical Pathology Signature for {patient_id}")
ax.set_xlabel("Normalized Feature Value")
st.pyplot(fig)

st.divider()

# Neuro-Fuzzy CDSS System
st.subheader("2. Neuro-Fuzzy Clinical Decision Support System")
st.info("Translating raw AI probabilities and Biomarker impact into a human-readable action plan.")

@st.cache_resource
def setup_fuzzy_system():
    ai_risk = ctrl.Antecedent(np.arange(0, 1.01, 0.01), 'AI_Risk')
    top_biomarker = ctrl.Antecedent(np.arange(0, 1.01, 0.01), 'Top_Biomarker')
    clinical_urgency = ctrl.Consequent(np.arange(0, 101, 1), 'Clinical_Urgency')

    ai_risk['low'] = fuzz.trimf(ai_risk.universe, [0, 0, 0.5])
    ai_risk['medium'] = fuzz.trimf(ai_risk.universe, [0.2, 0.5, 0.8])
    ai_risk['high'] = fuzz.trimf(ai_risk.universe, [0.5, 1, 1])

    top_biomarker['low'] = fuzz.trimf(top_biomarker.universe, [0, 0, 0.6])
    top_biomarker['high'] = fuzz.trimf(top_biomarker.universe, [0.4, 1, 1])

    clinical_urgency['routine'] = fuzz.trimf(clinical_urgency.universe, [0, 0, 50])
    clinical_urgency['monitor'] = fuzz.trimf(clinical_urgency.universe, [25, 50, 75])
    clinical_urgency['urgent'] = fuzz.trimf(clinical_urgency.universe, [50, 100, 100])

    rule1 = ctrl.Rule(ai_risk['low'] & top_biomarker['low'], clinical_urgency['routine'])
    rule2 = ctrl.Rule(ai_risk['medium'], clinical_urgency['monitor'])
    rule3 = ctrl.Rule(ai_risk['high'] | top_biomarker['high'], clinical_urgency['urgent'])

    urgency_ctrl = ctrl.ControlSystem([rule1, rule2, rule3])
    return ctrl.ControlSystemSimulation(urgency_ctrl)

urgency_sim = setup_fuzzy_system()

# For PoC Dashboard, we simulate the AI risk. In production, this comes from PyTorch inference.
sim_ai_risk = np.random.uniform(0.1, 0.95)
sim_top_biomarker = dummy_patient_features[1] # Red-Std as the top biomarker

urgency_sim.input['AI_Risk'] = sim_ai_risk
urgency_sim.input['Top_Biomarker'] = sim_top_biomarker
urgency_sim.compute()
final_score = urgency_sim.output['Clinical_Urgency']

colA, colB, colC = st.columns(3)
colA.metric("AI Predicted Risk", f"{sim_ai_risk:.2f}")
colB.metric("Top Biomarker (Red-Std) Impact", f"{sim_top_biomarker:.2f}")
colC.metric("Clinical Urgency Score", f"{final_score:.2f} / 100")

if final_score >= 50:
    st.error("🚨 RECOMMENDED ACTION: Urgent Medical Intervention & Treatment Revision")
elif final_score >= 25:
    st.warning("⚠️ RECOMMENDED ACTION: Close Clinical Monitoring")
else:
    st.success("✅ RECOMMENDED ACTION: Routine Checkup")

st.caption("System Ready: Connect the trained PyTorch inference engine (trained_multimodal_model.pth) for live fusion predictions.")
