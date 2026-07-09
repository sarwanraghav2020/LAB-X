# =========================================================
# Imports
# =========================================================

import os
import time
import shutil
from pathlib import Path
import numpy as np
import streamlit as st
from PIL import Image
from dotenv import load_dotenv

from utils import (
    Document,
    MultimodalPredictor,
    MedicalLiteratureDB,
    DocumentSplitter,
    MockVectorDB,
    ClinicalRAG,
)

# Load environment variables if any
load_dotenv()

# =========================================================
# Page Configuration
# =========================================================

st.set_page_config(
    page_title="LAB X",
    page_icon="🩻",
    layout="wide",
    initial_sidebar_state="expanded"
)

# =========================================================
# Apple-Level UI Styling (Awwwards-Level Aesthetics)
# =========================================================

st.markdown(
    """
<style>
/* Custom Fonts */
@import url('https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;500;600;700&family=Plus+Jakarta+Sans:wght@300;400;500;600;700&family=JetBrains+Mono:wght@400;500&display=swap');

/* Force Dark Mode & Premium Typography globally */
html, body, [data-testid="stAppViewContainer"] {
    background-color: #07080B !important;
    background-image: radial-gradient(circle at 50% 0%, #151926 0%, #07080b 75%) !important;
    color: #F5F5F7 !important;
    font-family: 'Plus Jakarta Sans', -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif !important;
}

/* Adjust Streamlit block containers */
.main .block-container {
    padding-top: 1.5rem !important;
    padding-bottom: 2rem !important;
    max-width: 1280px !important;
}

/* Title & Header Section */
.hero-header {
    text-align: center;
    position: relative;
    padding: 3.5rem 1.5rem;
    margin-bottom: 2.5rem;
    background: rgba(10, 12, 20, 0.5) !important;
    border: 1px solid rgba(255, 255, 255, 0.05) !important;
    box-shadow: 0 24px 80px rgba(0, 0, 0, 0.6), inset 0 1px 1px rgba(255, 255, 255, 0.08) !important;
    border-radius: 28px;
    backdrop-filter: blur(40px);
    -webkit-backdrop-filter: blur(40px);
    overflow: hidden;
}

.hero-glow {
    position: absolute;
    top: -250px;
    left: 50%;
    transform: translateX(-50%);
    width: 800px;
    height: 500px;
    background: radial-gradient(circle, rgba(0, 113, 227, 0.22) 0%, rgba(138, 48, 253, 0.07) 50%, rgba(0, 0, 0, 0) 75%) !important;
    filter: blur(80px);
    z-index: 0;
    pointer-events: none;
}

.hero-badge {
    display: inline-block;
    padding: 6px 16px;
    background: rgba(41, 151, 255, 0.06) !important;
    border: 1px solid rgba(41, 151, 255, 0.25) !important;
    border-radius: 980px;
    color: #2997FF;
    font-size: 0.72rem;
    font-weight: 700;
    margin-bottom: 1.5rem;
    letter-spacing: 0.1em;
    text-transform: uppercase;
    z-index: 1;
    position: relative;
    box-shadow: 0 0 15px rgba(41, 151, 255, 0.15);
}

.hero-title {
    font-family: 'Outfit', sans-serif !important;
    font-size: 5.5rem !important;
    font-weight: 900 !important;
    background: linear-gradient(135deg, #FFFFFF 15%, #D2D2D7 45%, #2997FF 75%, #BF5AF2 100%) !important;
    -webkit-background-clip: text !important;
    -webkit-text-fill-color: transparent !important;
    margin-bottom: 0.8rem !important;
    letter-spacing: -0.06em !important;
    filter: drop-shadow(0 4px 30px rgba(41, 151, 255, 0.2)) !important;
    line-height: 1.05 !important;
    z-index: 1;
    position: relative;
}

.hero-subtitle {
    font-size: 1.05rem !important;
    color: #8E8E93 !important;
    max-width: 750px;
    margin: 0 auto !important;
    line-height: 1.6 !important;
    z-index: 1;
    position: relative;
    letter-spacing: -0.01em;
}

.hero-subtitle b {
    color: #ffffff;
    font-weight: 600;
}

/* Sidebar Custom Overrides */
[data-testid="stSidebar"] {
    background-color: rgba(10, 11, 16, 0.95) !important;
    backdrop-filter: blur(25px) !important;
    -webkit-backdrop-filter: blur(25px) !important;
    border-right: 1px solid rgba(255, 255, 255, 0.05) !important;
}

[data-testid="stSidebar"] .stMarkdown h1, 
[data-testid="stSidebar"] .stMarkdown h2, 
[data-testid="stSidebar"] .stMarkdown h3 {
    color: #FFFFFF !important;
    font-family: 'Outfit', sans-serif !important;
    font-weight: 600 !important;
    letter-spacing: -0.01em;
}

/* Glassmorphic Containers */
.glass-container {
    background: rgba(255, 255, 255, 0.01) !important;
    border: 1px solid rgba(255, 255, 255, 0.04) !important;
    border-radius: 20px !important;
    padding: 24px !important;
    margin-bottom: 24px !important;
    backdrop-filter: blur(20px) !important;
    -webkit-backdrop-filter: blur(20px) !important;
    box-shadow: 0 10px 30px rgba(0, 0, 0, 0.3) !important;
    position: relative;
}

.glass-container:hover {
    border-color: rgba(255, 255, 255, 0.08) !important;
    box-shadow: 0 15px 40px rgba(0, 0, 0, 0.4) !important;
}

/* Scanning Box Animation */
.scanning-box {
    position: relative;
    border: 1px solid rgba(0, 113, 227, 0.15);
    background: rgba(0, 0, 0, 0.2);
    border-radius: 16px;
    overflow: hidden;
    height: 380px;
    display: flex;
    justify-content: center;
    align-items: center;
}

.scan-line {
    position: absolute;
    width: 100%;
    height: 3px;
    background: linear-gradient(to right, rgba(0, 113, 227, 0), rgba(0, 113, 227, 0.7), rgba(0, 113, 227, 0));
    animation: scan 2.5s infinite linear;
    z-index: 5;
}

@keyframes scan {
    0% { top: 0%; }
    50% { top: 100%; }
    100% { top: 0%; }
}

/* Dynamic Medical Progress Bars */
.confidence-container {
    margin-bottom: 1rem;
}

.confidence-header {
    display: flex;
    justify-content: space-between;
    margin-bottom: 4px;
    font-size: 0.85rem;
    font-weight: 600;
}

.confidence-bar-bg {
    width: 100%;
    height: 8px;
    background: rgba(255, 255, 255, 0.04);
    border-radius: 980px;
    overflow: hidden;
}

.confidence-bar-fill {
    height: 100%;
    border-radius: 980px;
    box-shadow: 0 0 10px rgba(0, 113, 227, 0.5);
    transition: width 0.8s ease-in-out;
}

/* Custom Warning / Info / Success Boxes */
.custom-warning {
    background: rgba(255, 159, 10, 0.06) !important;
    border-left: 4px solid #FF9F0A !important;
    border-radius: 8px !important;
    padding: 14px 18px !important;
    margin: 16px 0 !important;
    color: #FF9F0A !important;
    font-size: 0.9rem;
    line-height: 1.5;
}

.custom-success {
    background: rgba(48, 209, 88, 0.06) !important;
    border-left: 4px solid #30D158 !important;
    border-radius: 8px !important;
    padding: 14px 18px !important;
    margin: 16px 0 !important;
    color: #30D158 !important;
    font-size: 0.9rem;
    line-height: 1.5;
}

.custom-info {
    background: rgba(10, 132, 255, 0.06) !important;
    border-left: 4px solid #0A84FF !important;
    border-radius: 8px !important;
    padding: 14px 18px !important;
    margin: 16px 0 !important;
    color: #0A84FF !important;
    font-size: 0.9rem;
    line-height: 1.5;
}

.custom-critical {
    background: rgba(255, 69, 58, 0.07) !important;
    border-left: 4px solid #FF453A !important;
    border-radius: 8px !important;
    padding: 14px 18px !important;
    margin: 16px 0 !important;
    color: #FF453A !important;
    font-size: 0.9rem;
    line-height: 1.5;
}

/* Buttons Styling */
div.stButton > button {
    background: linear-gradient(135deg, #0071e3 0%, #2997ff 100%) !important;
    color: #ffffff !important;
    font-weight: 600 !important;
    font-family: 'Plus Jakarta Sans', sans-serif !important;
    border: none !important;
    border-radius: 10px !important;
    padding: 10px 20px !important;
    width: 100% !important;
    transition: all 0.3s cubic-bezier(0.16, 1, 0.3, 1) !important;
    box-shadow: 0 4px 15px rgba(0, 113, 227, 0.2) !important;
}

div.stButton > button:hover {
    transform: translateY(-1px) !important;
    box-shadow: 0 8px 25px rgba(0, 113, 227, 0.35) !important;
    filter: brightness(1.05) !important;
}

div.stButton > button:active {
    transform: translateY(0) !important;
}

/* Reset / Clear Buttons */
div.stButton > button[key*="clear"], 
div.stButton > button[key*="reset"],
[data-testid="stSidebar"] div.stButton:nth-child(2) > button, 
[data-testid="stSidebar"] div.stButton:nth-child(3) > button {
    background: rgba(255, 255, 255, 0.03) !important;
    border: 1px solid rgba(255, 255, 255, 0.08) !important;
    color: #f5f5f7 !important;
    box-shadow: none !important;
}

[data-testid="stSidebar"] div.stButton:nth-child(2) > button:hover {
    background: rgba(255, 255, 255, 0.07) !important;
    border-color: rgba(255, 255, 255, 0.12) !important;
}

[data-testid="stSidebar"] div.stButton:nth-child(3) > button:hover {
    background: rgba(255, 69, 58, 0.12) !important;
    border-color: rgba(255, 69, 58, 0.25) !important;
    color: #ff453a !important;
}

/* Chat Message overrides */
div[data-testid="stChatMessage"] {
    background-color: rgba(255, 255, 255, 0.015) !important;
    border: 1px solid rgba(255, 255, 255, 0.04) !important;
    border-radius: 16px !important;
    padding: 16px !important;
    margin-bottom: 12px !important;
    box-shadow: 0 4px 15px rgba(0, 0, 0, 0.15) !important;
}

div[data-testid="chatAvatarIcon-user"] {
    background-color: #0071e3 !important;
}

div[data-testid="chatAvatarIcon-assistant"] {
    background: linear-gradient(135deg, #a259ff 0%, #6600ff 100%) !important;
}

/* Expanders */
[data-testid="stExpander"] {
    background: rgba(255, 255, 255, 0.01) !important;
    border: 1px solid rgba(255, 255, 255, 0.04) !important;
    border-radius: 12px !important;
    box-shadow: 0 4px 20px rgba(0, 0, 0, 0.15) !important;
}

[data-testid="stExpander"] > details > summary {
    font-weight: 600 !important;
    color: #ffffff !important;
}

/* Metric custom styling override */
div[data-testid="stMetric"] {
    background: rgba(255, 255, 255, 0.01) !important;
    border: 1px solid rgba(255, 255, 255, 0.03) !important;
    border-radius: 12px !important;
    padding: 12px 16px !important;
}

div[data-testid="stMetricValue"] > div {
    font-family: 'Outfit', sans-serif !important;
    font-weight: 700 !important;
    font-size: 1.5rem !important;
}

/* Status Indicator Dot */
.pulse-dot {
    display: inline-block;
    width: 8px;
    height: 8px;
    background-color: #30D158;
    border-radius: 50%;
    margin-right: 8px;
    box-shadow: 0 0 8px #30D158;
    animation: pulse 2s infinite;
}

.pulse-dot-waiting {
    background-color: #FF9F0A;
    box-shadow: 0 0 8px #FF9F0A;
}

@keyframes pulse {
    0% {
        transform: scale(0.95);
        box-shadow: 0 0 0 0 rgba(48, 209, 88, 0.7);
    }
    70% {
        transform: scale(1);
        box-shadow: 0 0 0 6px rgba(48, 209, 88, 0);
    }
    100% {
        transform: scale(0.95);
        box-shadow: 0 0 0 0 rgba(48, 209, 88, 0);
    }
}

/* Horizontal line custom styling */
hr {
    border: 0 !important;
    height: 1px !important;
    background: linear-gradient(to right, rgba(255,255,255,0), rgba(255,255,255,0.06), rgba(255,255,255,0)) !important;
    margin: 2.0rem 0 !important;
}

/* Premium Footer */
.premium-footer {
    margin-top: 4rem;
    padding: 2.5rem 1.5rem;
    border-top: 1px solid rgba(255, 255, 255, 0.03);
    font-size: 0.8rem;
    color: #8E8E93;
}

.premium-footer h3 {
    font-family: 'Outfit', sans-serif;
    color: #ffffff;
    font-size: 0.95rem;
    margin-bottom: 0.8rem;
    font-weight: 600;
}
</style>
""",
    unsafe_allow_html=True
)

# =========================================================
# Header Hero Section
# =========================================================

st.markdown(
    """
<div class="hero-header">
    <div class="hero-glow"></div>
    <div class="hero-badge">🧬 Neural Imaging Intelligence</div>
    <div class="hero-title">LAB X</div>
    <div class="hero-subtitle">
        Deep Learning Visual Inference combined with Retrieval-Augmented Clinical Knowledge
        <br>Powered by <b>EfficientNet-B4 • Grad-CAM • FAISS • LangChain • Llama 3</b>
    </div>
</div>
""",
    unsafe_allow_html=True
)

# =========================================================
# Sidebar - Engine Parameters
# =========================================================

st.sidebar.title("🩻 Diagnostic Settings")
st.sidebar.divider()

# ---------------------------------------------------------
# Scan Modality Selection
# ---------------------------------------------------------

modality = st.sidebar.selectbox(
    "📐 Scan Modality",
    [
        "Chest X-Ray",
        "Brain MRI",
        "Abdominal CT",
        "Knee MRI"
    ]
)

# ---------------------------------------------------------
# Upload Scan Image
# ---------------------------------------------------------

uploaded_scan = st.sidebar.file_uploader(
    "📸 Upload Scan Image",
    type=["png", "jpg", "jpeg"]
)

# ---------------------------------------------------------
# Deep Learning Backbone Select
# ---------------------------------------------------------

dl_model = st.sidebar.selectbox(
    "🤖 Deep Learning Backbone",
    [
        "EfficientNet-B4 (Recommended)",
        "EfficientNet-B7",
        "EfficientNet-B0",
        "ResNet-50",
        "DenseNet-121"
    ]
)

# ---------------------------------------------------------
# Clinical Case Overrider (For Debugging / Demos)
# ---------------------------------------------------------

# Define pathologies based on selected modality
modality_scenarios = {
    "Chest X-Ray": ["Auto-Detect", "Pneumonia", "Cardiomegaly", "Pleural Effusion", "Normal"],
    "Brain MRI": ["Auto-Detect", "Glioma", "Meningioma", "Pituitary Tumor", "Healthy Brain"],
    "Abdominal CT": ["Auto-Detect", "Appendicitis", "Kidney Stone", "Liver Lesion", "Healthy Abdomen"],
    "Knee MRI": ["Auto-Detect", "ACL Tear", "Meniscus Tear", "Osteoarthritis", "Healthy Knee"]
}

scenarios = modality_scenarios.get(modality, ["Auto-Detect"])

scenario = st.sidebar.selectbox(
    "🎯 Clinical Case Scenario",
    scenarios,
    help="Force a pathology simulation to inspect Grad-CAM focus and vector database retrieval outputs."
)

# ---------------------------------------------------------
# LLM Clinical Decoder
# ---------------------------------------------------------

llm_provider = st.sidebar.selectbox(
    "🧠 LLM Clinical Decoder",
    ["Llama 3 (Recommended)", "Google Gemini", "Mistral-7B"]
)

# ---------------------------------------------------------
# Vector Database
# ---------------------------------------------------------

vector_db_choice = st.sidebar.selectbox(
    "🗄️ Knowledge Vector Store",
    ["FAISS", "ChromaDB"]
)

# ---------------------------------------------------------
# Tuning Parameters Collapsible
# ---------------------------------------------------------

with st.sidebar.expander("🛠️ Advanced Settings"):
    chunk_size = st.slider(
        "Chunk Size",
        min_value=400,
        max_value=1200,
        value=800,
        step=100
    )

    chunk_overlap = st.slider(
        "Chunk Overlap",
        min_value=50,
        max_value=300,
        value=150,
        step=50
    )

    top_k = st.slider(
        "Literature Retrieve K",
        min_value=1,
        max_value=5,
        value=3
    )

st.sidebar.divider()

# ---------------------------------------------------------
# Sidebar Command Action Buttons
# ---------------------------------------------------------

build_database_btn = st.sidebar.button(
    "🚀 Build Database",
    use_container_width=True
)

clear_chat_btn = st.sidebar.button(
    "🧹 Clear Chat Console",
    use_container_width=True
)

reset_system_btn = st.sidebar.button(
    "🗑 Reset System State",
    use_container_width=True
)

st.sidebar.divider()

# System Information Panel
st.sidebar.markdown(
    """
<div style="font-size: 0.8rem; color: #86868b; line-height: 1.4;">
    <h4 style="color: #ffffff; font-size: 0.9rem; margin-bottom: 0.4rem;">🔬 Core Diagnosis Node</h4>
    <p><b>Pipeline Version:</b> 3.0.0 (Multimodal Clinical Simulation)</p>
    <p><b>Image Size:</b> 512x512 standard resize</p>
    <p style="color: #ff9f0a; font-style: italic; margin-top: 0.4rem;">⚠️ Warning: Diagnostic support tool only. Clinician supervision required.</p>
</div>
""",
    unsafe_allow_html=True
)

# =========================================================
# Session State & DB Initialization
# =========================================================

# Database Built & Analysis Ready Status
if "database_built" not in st.session_state:
    st.session_state.database_built = False

# RAG Chain
if "rag" not in st.session_state:
    st.session_state.rag = None

# Chat History
if "messages" not in st.session_state:
    st.session_state.messages = []

# Retrieved Evidence
if "retrieved_docs" not in st.session_state:
    st.session_state.retrieved_docs = []

# Cached Inference Outputs
if "inference_run" not in st.session_state:
    st.session_state.inference_run = False
if "predicted_condition" not in st.session_state:
    st.session_state.predicted_condition = ""
if "predictions" not in st.session_state:
    st.session_state.predictions = {}
if "gradcam_img" not in st.session_state:
    st.session_state.gradcam_img = None
if "observations" not in st.session_state:
    st.session_state.observations = []
if "report_summary" not in st.session_state:
    st.session_state.report_summary = ""
if "summary_generated" not in st.session_state:
    st.session_state.summary_generated = False

# Track settings changes to reset state
if "last_scan_name" not in st.session_state:
    st.session_state.last_scan_name = ""
if "last_scenario" not in st.session_state:
    st.session_state.last_scenario = ""
if "last_modality" not in st.session_state:
    st.session_state.last_modality = ""

# ---------------------------------------------------------
# Action Handlers
# ---------------------------------------------------------

if clear_chat_btn:
    st.session_state.messages = []
    st.session_state.retrieved_docs = []
    st.rerun()

if reset_system_btn:
    st.session_state.rag = None
    st.session_state.database_built = False
    st.session_state.messages = []
    st.session_state.retrieved_docs = []
    st.session_state.inference_run = False
    st.session_state.predicted_condition = ""
    st.session_state.predictions = {}
    st.session_state.gradcam_img = None
    st.session_state.observations = []
    st.session_state.report_summary = ""
    st.session_state.summary_generated = False
    st.session_state.last_scan_name = ""
    st.session_state.last_scenario = ""
    st.session_state.last_modality = ""
    st.rerun()

# ---------------------------------------------------------
# Core DB Building Helper
# ---------------------------------------------------------

def build_vector_database(provider, model_name, db_choice, size, overlap):
    """
    Builds the vector indexing pipeline using standard clinical literature articles.
    """
    articles = MedicalLiteratureDB.get_articles()
    raw_docs = []
    for art in articles:
        content = f"Document ID: {art['id']}\nTitle: {art['title']}\nAuthors: {art['authors']}\nJournal: {art['journal']} ({art['year']})\nTopic: {art['topic']}\n\nAbstract Details:\n{art['content']}"
        metadata = {
            "id": art["id"],
            "title": art["title"],
            "authors": art["authors"],
            "journal": art["journal"],
            "year": art["year"],
            "topic": art["topic"],
            "source": f"PubMed://{art['id']}"
        }
        raw_docs.append(Document(page_content=content, metadata=metadata))

    # Split into optimized retrieval chunks
    splitter = DocumentSplitter(chunk_size=size, chunk_overlap=overlap)
    chunks = splitter.split(raw_docs)

    # Build Mock FAISS / ChromaDB similarity node
    db = MockVectorDB(chunks)

    # Setup Clinical RAG Chain
    rag = ClinicalRAG(vector_db=db, provider=provider, model_name=f"{provider}-Instruct")

    # Save to state
    st.session_state.rag = rag
    st.session_state.database_built = True

# Handle settings changes to trigger resets
if uploaded_scan is not None and st.session_state.last_scan_name != uploaded_scan.name:
    st.session_state.database_built = False
    st.session_state.inference_run = False
    st.session_state.summary_generated = False
    st.session_state.report_summary = ""
    st.session_state.messages = []
    st.session_state.retrieved_docs = []
    st.session_state.last_scan_name = uploaded_scan.name

if st.session_state.last_modality != modality:
    st.session_state.database_built = False
    st.session_state.inference_run = False
    st.session_state.summary_generated = False
    st.session_state.report_summary = ""
    st.session_state.messages = []
    st.session_state.retrieved_docs = []
    st.session_state.last_modality = modality

# Trigger Database build when user presses "Build Database" button
if build_database_btn:
    if uploaded_scan is None:
        st.sidebar.error("⚠️ Please upload a scan image first!")
    else:
        with st.sidebar:
            status_placeholder = st.empty()
            status_placeholder.info("Building Knowledge Database...")
            
        build_vector_database(
            provider=llm_provider,
            model_name=f"{llm_provider}-Instruct",
            db_choice=vector_db_choice,
            size=chunk_size,
            overlap=chunk_overlap
        )
        
        # Run deep learning model prediction
        predictor = MultimodalPredictor(model_name=dl_model)
        cond, preds, cam_img, obs = predictor.predict(
            uploaded_scan, 
            modality=modality, 
            scenario=scenario
        )
        
        st.session_state.predicted_condition = cond
        st.session_state.predictions = preds
        st.session_state.gradcam_img = cam_img
        st.session_state.observations = obs
        st.session_state.inference_run = True
        st.session_state.last_scenario = scenario
        st.session_state.retrieved_docs = st.session_state.rag.vector_db.similarity_search(cond, k=top_k)
        
        st.balloons()
        st.sidebar.success("✨ Database successfully built!")

# =========================================================
# Main Area Rendering Logic
# =========================================================

# Check if X-ray/Scan file is uploaded
if uploaded_scan is None:
    # -----------------------------------------------------
    # LANDING STATE - Standby Awaiting Image
    # -----------------------------------------------------
    
    st.markdown(
        f"""
        <div class="glass-container" style="text-align: center; padding: 4.5rem 2rem; margin-top: 1rem;">
            <div style="font-size: 4.5rem; margin-bottom: 1.5rem; position: relative; display: inline-block;">
                🩻
                <div class="scan-line" style="width: 140%; left: -20%;"></div>
            </div>
            <h3 style="font-family: 'Outfit', sans-serif; font-weight: 600; font-size: 2.2rem; margin-bottom: 0.8rem; color: white; letter-spacing: -0.02em;">Awaiting {modality} Scan Input</h3>
            <p style="color: #8E8E93; max-width: 580px; margin: 0 auto 2.5rem auto; line-height: 1.6; font-size: 1.05rem;">
                Upload a <b>{modality}</b> scan file in the sidebar panel to run deep learning classification models and initiate clinical RAG references.
            </p>
            <div style="display: inline-flex; align-items: center; gap: 8px; color: #FF9F0A; font-weight: 600; font-size: 0.9rem; background: rgba(255, 159, 10, 0.08); padding: 8px 18px; border-radius: 980px; border: 1px solid rgba(255, 159, 10, 0.2);">
                <span class="pulse-dot pulse-dot-waiting"></span> Diagnostic Node Online • System Awaiting {modality} File
            </div>
        </div>
        """,
        unsafe_allow_html=True
    )
    
    # Showcase standard clinical database statistics
    st.subheader("📚 Active Diagnostic Knowledge Base")
    col1, col2, col3 = st.columns(3)
    
    articles = MedicalLiteratureDB.get_articles()
    with col1:
        st.metric(
            label="📖 Indexed Literature Reviews",
            value=f"{len(articles)} Publications"
        )
    with col2:
        st.metric(
            label="🧬 Diagnostic Pathology Nodes",
            value="13 Categories"
        )
    with col3:
        st.metric(
            label="🗄️ Knowledge Vector Store",
            value=vector_db_choice
        )
        
    st.markdown(
        f"""
        <div class="glass-container" style="padding: 20px 24px; margin-top: 1rem; border-color: rgba(10, 132, 255, 0.15);">
            <h4 style="margin-top:0; font-family: 'Outfit', sans-serif; font-size: 1.1rem; color: #0A84FF;">⚡ Multimodal Deep Learning Simulation Active</h4>
            <p style="margin: 0; color: #8E8E93; line-height: 1.5; font-size: 0.95rem;">
                This system runs a high-fidelity visual diagnostic pipeline supporting <b>Chest X-Rays, Brain MRIs, Abdominal CTs, and Knee MRIs</b>. 
                When you upload a scan, our simulated <b>EfficientNet-B4</b> backbone computes pathology classification vectors, 
                and overlays a customized <b>Grad-CAM heatmap image</b> directly over key anatomical points corresponding to the pathology.
            </p>
        </div>
        """,
        unsafe_allow_html=True
    )

elif not st.session_state.database_built:
    # -----------------------------------------------------
    # FILE UPLOADED BUT DATABASE NOT BUILT STATE
    # -----------------------------------------------------
    st.markdown(
        f"""
        <div class="glass-container" style="text-align: center; padding: 4.5rem 2rem; margin-top: 1rem; border-color: rgba(255, 159, 10, 0.2);">
            <div style="font-size: 4rem; margin-bottom: 1.5rem;">📄</div>
            <h3 style="font-family: 'Outfit', sans-serif; font-weight: 600; font-size: 2.0rem; margin-bottom: 0.8rem; color: white;">{modality} Scan Uploaded Successfully</h3>
            <p style="color: #8E8E93; max-width: 550px; margin: 0 auto 2.2rem auto; line-height: 1.6; font-size: 1.05rem;">
                File <b>{uploaded_scan.name}</b> is staged. Please click the <b>"🚀 Build Database"</b> button in the sidebar to run the visual diagnostic predictor and index clinical papers.
            </p>
            <div style="display: inline-flex; align-items: center; gap: 8px; color: #FF9F0A; font-weight: 600; font-size: 0.9rem; background: rgba(255, 159, 10, 0.08); padding: 8px 18px; border-radius: 980px; border: 1px solid rgba(255, 159, 10, 0.2);">
                <span class="pulse-dot pulse-dot-waiting"></span> Standby - Press "Build Database" to Proceed
            </div>
        </div>
        """,
        unsafe_allow_html=True
    )

else:
    # -----------------------------------------------------
    # ACTIVE DIAGNOSTIC DASHBOARD STATE (DATABASE READY)
    # -----------------------------------------------------

    # Handle parameter modifications when DB is already active
    if st.session_state.last_scenario != scenario:
        # Re-run inference with the modified scenario
        predictor = MultimodalPredictor(model_name=dl_model)
        cond, preds, cam_img, obs = predictor.predict(
            uploaded_scan, 
            modality=modality, 
            scenario=scenario
        )
        
        st.session_state.predicted_condition = cond
        st.session_state.predictions = preds
        st.session_state.gradcam_img = cam_img
        st.session_state.observations = obs
        st.session_state.last_scenario = scenario
        st.session_state.summary_generated = False
        st.session_state.report_summary = ""
        st.session_state.retrieved_docs = st.session_state.rag.vector_db.similarity_search(cond, k=top_k)

    # Online System Status Alert
    st.markdown(
        f'<div class="custom-success" style="padding: 10px 18px; font-weight: 600; display: flex; align-items: center; margin-bottom: 1.5rem;"><span class="pulse-dot"></span> {modality} Database Built Successfully! System Online • Active Inference: <b>{dl_model}</b></div>',
        unsafe_allow_html=True
    )

    # Main columns layout: Visuals vs Metrics/Observations
    main_col_left, main_col_right = st.columns([1.1, 0.9], gap="large")

    # =====================================================
    # LEFT COLUMN: Visual Radiography Panel
    # =====================================================
    with main_col_left:
        st.markdown(
            f"""
            <div style="display: flex; align-items: center; gap: 10px; margin-bottom: 0.8rem;">
                <span style="font-size: 1.4rem;">🩻</span>
                <h3 style="font-family: 'Outfit', sans-serif; font-weight: 600; font-size: 1.3rem; margin: 0; color: white;">Visual Diagnostic Scans</h3>
            </div>
            """,
            unsafe_allow_html=True
        )
        
        # Display side-by-side: Input Scan and Grad-CAM Heatmap
        visual_tab_orig, visual_tab_cam = st.tabs([f"📸 Original {modality} Scan", "🔥 Neural Grad-CAM Attention Map"])
        
        with visual_tab_orig:
            st.image(
                uploaded_scan, 
                caption=f"Original {modality} View ({uploaded_scan.name})", 
                use_container_width=True
            )
            
        with visual_tab_cam:
            st.image(
                st.session_state.gradcam_img, 
                caption=f"Grad-CAM Heatmap Overlay ({dl_model} Backbone)", 
                use_container_width=True
            )

        # Highlight focus regions explanation
        st.markdown("#### 🔬 Neural Highlight Analysis")
        cond_key = st.session_state.predicted_condition
        
        # Determine warning display depending on the active pathology
        if "Normal" in cond_key or "Healthy" in cond_key:
            st.markdown(
                f"""
                <div class="custom-success">
                    <b>Baseline Structural Markers:</b> Gradients demonstrate minor baseline highlights around the central scan coordinates. There are no abnormal focal consolidations, mass lesions, calcifications, or ligament tears detected in the <b>{modality}</b>.
                </div>
                """, 
                unsafe_allow_html=True
            )
        elif cond_key == "Pneumonia":
            st.markdown(
                """
                <div class="custom-critical">
                    <b>Bilateral Consolidation Focus:</b> The neural activation map indicates intensive high-gradient vectors centering in the <b>lower left lung lobe (155, 340)</b> and <b>lower right lung lobe (350, 360)</b>. This correlates with alveolar air space opacity and exudative fluid collection typical of bacterial lobar pneumonia.
                </div>
                """, 
                unsafe_allow_html=True
            )
        elif cond_key == "Cardiomegaly":
            st.markdown(
                """
                <div class="custom-warning">
                    <b>Cardiac Silhouette Encroachment:</b> The Grad-CAM highlights a broad, heavy focus centering in the <b>mediastinum and leftward cardiac area (260, 330)</b>. The model highlights the lateral boundaries of the heart shadow, indicating an enlarged cardiothoracic ratio (CTR > 0.60) suggesting cardiomyopathy.
                </div>
                """, 
                unsafe_allow_html=True
            )
        elif cond_key == "Pleural Effusion":
            st.markdown(
                """
                <div class="custom-warning">
                    <b>Costophrenic Recess Occlusion:</b> Highlights are concentrated specifically at the <b>base corners of the thoracic cavity (110, 420) and (400, 430)</b>. This represents neural focus on the blunting of the costophrenic recesses where fluid accumulation obscures normal sharp diaphragmatic contours.
                </div>
                """, 
                unsafe_allow_html=True
            )
        elif cond_key == "Glioma":
            st.markdown(
                """
                <div class="custom-critical">
                    <b>Intra-axial Glial Hyperintensity:</b> Grad-CAM highlights focus intensively in the <b>right frontal/temporal lobe (330, 210)</b>. This highlights the region of heterogeneous mass density and surrounding vasogenic edema in the brain parenchyma.
                </div>
                """, 
                unsafe_allow_html=True
            )
        elif cond_key == "Meningioma":
            st.markdown(
                """
                <div class="custom-warning">
                    <b>Dural-based Extra-axial Mass:</b> Highlights focus along the <b>right parietal convex vault (360, 160)</b>, identifying a well-demarcated dural-based enhancement zone corresponding to the meningioma.
                </div>
                """, 
                unsafe_allow_html=True
            )
        elif cond_key == "Pituitary Tumor":
            st.markdown(
                """
                <div class="custom-warning">
                    <b>Sellar / Suprasellar Expansion:</b> The neural network is focusing on the <b>central skull base and sella turcica (256, 310)</b>, highlighting mass enlargement at the pituitary gland.
                </div>
                """, 
                unsafe_allow_html=True
            )
        elif cond_key == "Appendicitis":
            st.markdown(
                """
                <div class="custom-critical">
                    <b>Cecal Base Appendiceal Inflammatory Focus:</b> Strong highlights are centered in the <b>lower right abdominal quadrant (350, 370)</b>. This correlates with the thick-walled appendix and periappendiceal fat stranding.
                </div>
                """, 
                unsafe_allow_html=True
            )
        elif cond_key == "Kidney Stone":
            st.markdown(
                """
                <div class="custom-warning">
                    <b>Renal Pelvis Hyperdensity:</b> Neural attention points directly to the <b>left renal pelvis (175, 280)</b>, isolating the hyperdense calcification stone that is obstructing urine flow.
                </div>
                """, 
                unsafe_allow_html=True
            )
        elif cond_key == "Liver Lesion":
            st.markdown(
                """
                <div class="custom-warning">
                    <b>Segment IV Hepatic Hypodensity:</b> Highlights center in the <b>upper right quadrant of the liver (320, 210)</b>, identifying the focal boundaries of the liver lesion.
                </div>
                """, 
                unsafe_allow_html=True
            )
        elif cond_key == "ACL Tear":
            st.markdown(
                """
                <div class="custom-critical">
                    <b>Cruciate Ligament Notch Notching:</b> Highlights are concentrated in the <b>center of the knee joint notch (256, 250)</b>, identifying fiber discontinuity of the anterior cruciate ligament.
                </div>
                """, 
                unsafe_allow_html=True
            )
        elif cond_key == "Meniscus Tear":
            st.markdown(
                """
                <div class="custom-warning">
                    <b>Joint Line Meniscal Signal:</b> The model highlights the <b>lateral knee joint line (190, 265)</b>, corresponding to the posterior horn medial meniscus horizontal tear.
                </div>
                """, 
                unsafe_allow_html=True
            )
        elif cond_key == "Osteoarthritis":
            st.markdown(
                """
                <div class="custom-warning">
                    <b>Femorotibial Joint Space Narrowing:</b> Gradients are concentrated at the <b>medial femorotibial joint line (310, 260)</b>, corresponding to bone-on-bone joint space narrowing and marginal osteophytes.
                </div>
                """, 
                unsafe_allow_html=True
            )

    # =====================================================
    # RIGHT COLUMN: Metrics, Probabilities & Executive Summary
    # =====================================================
    with main_col_right:
        # Diagnostic Classification Probabilities
        st.markdown(
            """
            <div style="display: flex; align-items: center; gap: 10px; margin-bottom: 0.8rem;">
                <span style="font-size: 1.4rem;">📊</span>
                <h3 style="font-family: 'Outfit', sans-serif; font-weight: 600; font-size: 1.3rem; margin: 0; color: white;">Pathology Probabilities</h3>
            </div>
            """,
            unsafe_allow_html=True
        )
        
        # Display custom HTML progress bars with Apple Aesthetics
        predictions = st.session_state.predictions
        for condition, prob in predictions.items():
            percentage = prob * 100
            
            # Determine color theme based on prediction values and condition type
            is_predicted = (condition == st.session_state.predicted_condition)
            if is_predicted:
                if "Normal" in condition or "Healthy" in condition:
                    color_hex = "#30D158" # Green
                    glow_rgba = "rgba(48, 209, 88, 0.4)"
                elif condition in ["Pneumonia", "Glioma", "Appendicitis", "ACL Tear"]:
                    color_hex = "#FF453A" # Red
                    glow_rgba = "rgba(255, 69, 58, 0.4)"
                else:
                    color_hex = "#FF9F0A" # Orange
                    glow_rgba = "rgba(255, 159, 10, 0.4)"
            else:
                color_hex = "#2997FF" # Blue
                glow_rgba = "rgba(41, 151, 255, 0.1)"
                
            st.markdown(
                f"""
                <div class="confidence-container">
                    <div class="confidence-header">
                        <span style="color: {'#ffffff' if is_predicted else '#8E8E93'}; font-weight: {700 if is_predicted else 500};">
                            {condition} {'🏷️ Primary Diagnosis' if is_predicted else ''}
                        </span>
                        <span style="color: {color_hex}; font-weight: 700;">{percentage:.1f}%</span>
                    </div>
                    <div class="confidence-bar-bg">
                        <div class="confidence-bar-fill" style="width: {percentage}%; background-color: {color_hex}; box-shadow: 0 0 8px {glow_rgba};"></div>
                    </div>
                </div>
                """,
                unsafe_allow_html=True
            )
            
        st.divider()

        # Synthesis Summary Block (Now explicitly manual as requested)
        st.markdown(
            """
            <div style="display: flex; align-items: center; gap: 10px; margin-bottom: 0.5rem;">
                <span style="font-size: 1.4rem;">📋</span>
                <h3 style="font-family: 'Outfit', sans-serif; font-weight: 600; font-size: 1.3rem; margin: 0; color: white;">Executive Synthesis</h3>
            </div>
            """,
            unsafe_allow_html=True
        )
        st.caption("Synthesize a clinical summary of findings combined with medical literature guidelines.")
        
        if not st.session_state.summary_generated:
            summary_btn = st.button("📋 Run Summary Synthesis", use_container_width=True)
            if summary_btn:
                with st.spinner("Generating synthesis..."):
                    summary_text = st.session_state.rag.summarize_diagnosis(
                        st.session_state.predicted_condition, 
                        st.session_state.observations
                    )
                    st.session_state.report_summary = summary_text
                    st.session_state.summary_generated = True
                    st.rerun()
            
            st.markdown(
                """
                <div class="custom-info" style="text-align: center; padding: 20px;">
                    Click "Run Summary Synthesis" to synthesize visual findings with clinical knowledge.
                </div>
                """,
                unsafe_allow_html=True
            )
        else:
            st.markdown(
                f"""
                <div class="glass-container" style="background: rgba(255, 255, 255, 0.015); border-color: rgba(0, 113, 227, 0.15); padding: 20px;">
                    {st.session_state.report_summary}
                </div>
                """,
                unsafe_allow_html=True
            )

    st.divider()

    # =====================================================
    # LOWER AREA: Evidence Retrieval Dashboard & Chat
    # =====================================================
    
    col_bottom_left, col_bottom_right = st.columns([1.1, 0.9], gap="large")
    
    # -----------------------------------------------------
    # BOTTOM LEFT: Retrieved Evidence References (FAISS)
    # -----------------------------------------------------
    with col_bottom_left:
        st.markdown(
            """
            <div style="display: flex; align-items: center; gap: 10px; margin-bottom: 0.8rem;">
                <span style="font-size: 1.4rem;">📚</span>
                <h3 style="font-family: 'Outfit', sans-serif; font-weight: 600; font-size: 1.3rem; margin: 0; color: white;">RAG Evidence Literature (FAISS Index)</h3>
            </div>
            """,
            unsafe_allow_html=True
        )
        st.caption("Sub-document fragments matched semantic-distance vectors in local FAISS literature storage.")
        
        # Search Box
        st.markdown("<span style='font-size:0.85rem; color:#8E8E93;'>🔍 Query Literature Store Manually</span>", unsafe_allow_html=True)
        search_query = st.text_input(
            "Query Literature Store Manually",
            placeholder="e.g., glioma radiotherapy, appendectomy guidelines, ACL surgery PT, light's criteria",
            label_visibility="collapsed"
        )
        
        search_btn = st.button("Search Knowledge Index", use_container_width=True)
        
        if search_btn and search_query.strip():
            with st.spinner("Computing cosine scores in FAISS index..."):
                results = st.session_state.rag.vector_db.similarity_search(search_query, k=top_k)
                st.session_state.retrieved_docs = results
                st.toast(f"Retrieved {len(results)} matches for '{search_query[:15]}...'", icon="📂")
        
        # Display the documents
        if len(st.session_state.retrieved_docs) > 0:
            for index, doc in enumerate(st.session_state.retrieved_docs, start=1):
                pmcid = doc.metadata.get("id", "N/A")
                title = doc.metadata.get("title", "Clinical Article")
                authors = doc.metadata.get("authors", "Unknown Authors")
                journal = doc.metadata.get("journal", "Medical Review")
                year = doc.metadata.get("year", "N/A")
                
                with st.expander(f"📖 Reference {index} | {pmcid} | {title[:40]}..."):
                    st.markdown(f"#### **{title}**")
                    st.markdown(f"**Authors:** {authors} | **Journal:** *{journal} ({year})*")
                    
                    st.markdown(
                        f"""
                        <div style="background: rgba(0, 0, 0, 0.3); padding: 14px; border-radius: 8px; border: 1px solid rgba(255, 255, 255, 0.04); font-family: monospace; font-size: 0.85rem; line-height: 1.4; color: #C7C7CC; margin-top: 10px;">
                            {doc.page_content}
                        </div>
                        """,
                        unsafe_allow_html=True
                    )
                    
                    st.markdown("##### Metadata Tags")
                    st.json(doc.metadata)
        else:
            st.info("No medical evidence currently retrieved.")
            
    # -----------------------------------------------------
    # BOTTOM RIGHT: Conversational Clinical assistant (Llama 3)
    # -----------------------------------------------------
    with col_bottom_right:
        st.markdown(
            """
            <div style="display: flex; align-items: center; gap: 10px; margin-bottom: 0.8rem;">
                <span style="font-size: 1.4rem;">💬</span>
                <h3 style="font-family: 'Outfit', sans-serif; font-weight: 600; font-size: 1.3rem; margin: 0; color: white;">Clinical Q&A Terminal</h3>
            </div>
            """,
            unsafe_allow_html=True
        )
        st.caption("Pose conversational clinical questions to Llama 3 backed by retrieved guidelines.")
        
        # Chat Input Console
        question = st.text_input(
            "Enter your clinical question...",
            placeholder="e.g., What is the chemotherapy dosage for high grade gliomas? ACL surgery vs physical therapy?",
            key="chat_input_text"
        )
        
        ask_btn = st.button("🤖 Ask Assistant Engine", use_container_width=True)
        
        if ask_btn and question.strip():
            with st.spinner("Querying Llama 3 context layers..."):
                start_time = time.time()
                answer, docs = st.session_state.rag.ask(question, k=top_k)
                end_time = time.time()
                duration = end_time - start_time
                
                # Append user and assistant interactions
                st.session_state.messages.append({"role": "user", "content": question})
                st.session_state.messages.append({
                    "role": "assistant", 
                    "content": f"{answer}\n\n*⚡ Llama 3 Inference latency: {duration:.2f} seconds | Retrieved {len(docs)} supportive papers.*"
                })
                
                # Update retrieved evidence documents
                st.session_state.retrieved_docs = docs
                st.rerun()

        # Render conversation history (newest first for clean visual hierarchy)
        if len(st.session_state.messages) > 0:
            st.markdown("<span style='font-size:0.85rem; color:#8E8E93; font-weight:600; display:block; margin-bottom:8px;'>TERMINAL CHAT HISTORY</span>", unsafe_allow_html=True)
            for msg in reversed(st.session_state.messages):
                with st.chat_message(msg["role"]):
                    st.markdown(msg["content"])

# =========================================================
# Medical Disclaimer & FDA Regulatory Fine-Print
# =========================================================

st.divider()

st.markdown(
    """
    <div class="custom-warning" style="background: rgba(255, 69, 58, 0.03) !important; border-left-color: #FF453A !important; color: #ff6961 !important; margin-bottom: 2rem;">
        <h4 style="color: #ffffff; font-family: 'Outfit', sans-serif; font-size: 1.05rem; margin-top: 0; margin-bottom: 0.4rem; display: flex; align-items: center; gap: 8px;">
            ⚠️ FDA Regulatory & Medical Practice Disclaimer
        </h4>
        <p style="margin: 0; font-size: 0.85rem; line-height: 1.5; color: #E5E5EA;">
            This application is a <b>clinical diagnostic simulation</b> demonstrating visual deep learning overlays (Grad-CAM) 
            combined with vector-based literature search structures (RAG). It is not certified for FDA diagnostic diagnostic workflows, 
            radiological interpretations, or prescription formulation. All medical classifications reflect numerical simulations of 
            the <b>EfficientNet-B4</b> backbone and selected Large Language Models, and are not a replacement for a board-certified 
            radiological assessment or physician screening.
        </p>
    </div>
    """,
    unsafe_allow_html=True
)

# =========================================================
# Technology Stack & Footer Grid
# =========================================================

st.markdown(
    """
    <div class="premium-footer">
        <div style="display: grid; grid-template-columns: repeat(auto-fit, minmax(200px, 1fr)); gap: 30px;">
            <div>
                <h3>📐 Imaging Backbone</h3>
                <ul style="list-style: none; padding: 0; line-height: 2;">
                    <li>EfficientNet-B4 Classifier</li>
                    <li>Gaussian Gradient Activation</li>
                    <li>Symmetric Overlay Alpha Blending</li>
                    <li>Resolution Standardizer (512px)</li>
                </ul>
            </div>
            <div>
                <h3>🤖 Decision Support Chain</h3>
                <ul style="list-style: none; padding: 0; line-height: 2;">
                    <li>Llama 3 Instruct Pipeline</li>
                    <li>FAISS Semantic Vector db</li>
                    <li>Token-Match Cosine Matcher</li>
                    <li>RAG Citation Generator</li>
                </ul>
            </div>
            <div>
                <h3>🛡️ Security & Integrity</h3>
                <ul style="list-style: none; padding: 0; line-height: 2;">
                    <li>Zero-Trust Client Processing</li>
                    <li>Local Vector Storage Nodes</li>
                    <li>Session State Isolation</li>
                    <li>Developer: Raghav Sarwan</li>
                </ul>
            </div>
        </div>
        <hr style="margin: 2rem 0 !important;">
        <div style="text-align: center; font-size: 0.8rem; color: #636366;">
            <p><b>LAB X: Multimodal Diagnostic Assistant</b></p>
            <p>Combines visual neural inference (Grad-CAM) with clinical semantic literature indices (FAISS/RAG).</p>
            <p style="margin-top: 0.7rem; font-size: 0.9rem; font-weight: 600; letter-spacing: 0.08em; text-transform: uppercase; background: linear-gradient(90deg, #ffffff 0%, #d7d7dd 35%, #6ea8ff 70%, #b17cff 100%); -webkit-background-clip: text; -webkit-text-fill-color: transparent; text-shadow: 0 0 16px rgba(110, 168, 255, 0.25);">Developed by RAGHAV SARWAN and ANIKA</p>
            <p style="margin-top: 0.8rem;">© 2026 LAB X Systems. All designs, assets, and frameworks reserved.</p>
        </div>
    </div>
    """,
    unsafe_allow_html=True
)
