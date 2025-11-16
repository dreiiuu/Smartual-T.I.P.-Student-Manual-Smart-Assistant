# ============================================================================
# SMART STUDENT ASSISTANT - T.I.P. Student Manual Q&A System
# A Streamlit application using semantic search and in-context classification
# Author: AI Assistant | Date: November 2025
# ============================================================================

import zipfile
import os
import json
import streamlit as st
import pandas as pd
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import time
from datetime import datetime

# -----------------------------------------------------------------------------
# MODEL EXTRACTION FUNCTION
# -----------------------------------------------------------------------------
def ensure_model_extracted():
zip_path = "smartual_model.zip"
extract_path = "smartual_model"


if os.path.exists(extract_path):
return extract_path


if not os.path.exists(zip_path):
st.error("Model zip file not found! Please upload smartual_model.zip")
return None


with zipfile.ZipFile(zip_path, 'r') as zip_ref:
zip_ref.extractall(extract_path)


return extract_path

# ============================================================================
# CONFIGURATION & COLOR PALETTE
# ============================================================================

MANUAL_DATA_FILE = "manual_data.json"
SECTION_EXAMPLES_FILE = "section_examples.json"
FEEDBACK_PATH = "feedback_log.csv"
SCHOOL_LOGO = "TIP_LOGO.jpg"
CHUNK_SIZE = 300
MODEL_PATH = ensure_model_extracted()

# COLOR PALETTE - Balanced Yellow
PRIMARY = "#FFA000"      # Perfect Amber Balance
SECONDARY = "#5D4037"    # Rich Brown
ACCENT = "#FFF8E1"       # Soft Yellow
BACKGROUND = "#FFFFFF"    # Clean White
TEXT = "#37474F"         # Readable Dark Gray
SUCCESS = "#4CAF50"      # Positive Green
WARNING = "#FF6D00"      # Attention Orange

# ============================================================================
# CORE FUNCTIONS
# ============================================================================

@st.cache_data
def load_manual_from_json():
    """Load the pre-structured T.I.P. Student Manual data from JSON file."""
    if not os.path.exists(MANUAL_DATA_FILE):
        st.error(f"Manual data file '{MANUAL_DATA_FILE}' not found!")
        return {}, []
    
    with open(MANUAL_DATA_FILE, 'r', encoding='utf-8') as f:
        manual_sections = json.load(f)
    
    # Create chunks from each section
    chunks = []
    for section_name, section_text in manual_sections.items():
        sentences = [s.strip() for s in section_text.split('. ') if s.strip()]
        
        current_chunk = []
        current_word_count = 0
        
        for sentence in sentences:
            words = sentence.split()
            if current_word_count + len(words) <= CHUNK_SIZE:
                current_chunk.append(sentence)
                current_word_count += len(words)
            else:
                if current_chunk:
                    chunk_text = '. '.join(current_chunk) + '.'
                    chunks.append({
                        "section": section_name,
                        "chunk_text": chunk_text,
                        "section_text": section_text,
                    })
                current_chunk = [sentence]
                current_word_count = len(words)
        
        if current_chunk:
            chunk_text = '. '.join(current_chunk) + '.'
            chunks.append({
                "section": section_name,
                "chunk_text": chunk_text,
                "section_text": section_text,
            })
    
    all_sections = list(manual_sections.keys())
    return chunks, all_sections

@st.cache_data
def load_section_examples():
    """Load example questions for in-context classification."""
    if not os.path.exists(SECTION_EXAMPLES_FILE):
        st.warning(f"Section examples file '{SECTION_EXAMPLES_FILE}' not found!")
        return {}
    
    with open(SECTION_EXAMPLES_FILE, 'r', encoding='utf-8') as f:
        examples = json.load(f)
    return examples

@st.cache_resource
def load_model():
    """Load the sentence transformer model."""
    return SentenceTransformer(MODEL_PATH)

@st.cache_resource
def build_index(_chunks, _model):
    """Create FAISS index for all chunks and save embeddings."""
    texts = [chunk["chunk_text"] for chunk in _chunks]
    chunk_embeddings = np.array(_model.encode(texts, show_progress_bar=False)).astype("float32")
    
    index = faiss.IndexFlatL2(chunk_embeddings.shape[1])
    index.add(chunk_embeddings)
    
    return index, chunk_embeddings

def classify_question(question, model, section_examples):
    """Use in-context examples to classify question's section by similarity."""
    question_embed = model.encode([question])[0]
    best_section = None
    best_score = -1
    
    for section, examples in section_examples.items():
        example_embeds = model.encode(examples, show_progress_bar=False)
        sim = cosine_similarity([question_embed], example_embeds).mean()
        if sim > best_score:
            best_section = section
            best_score = sim
    
    return best_section, float(best_score)

def retrieve_chunks(question, model, chunks, index, chunk_embeddings, top_k=3):
    """Retrieve the top K most similar chunks using FAISS."""
    question_embed = np.array(model.encode([question], show_progress_bar=False)).astype("float32")
    _, I = index.search(question_embed, top_k)
    
    top_chunks = [chunks[i] for i in I[0]]
    similarities = cosine_similarity(question_embed, chunk_embeddings[I[0]]).flatten()
    
    return top_chunks, similarities

def generate_answer(question, top_chunk, model):
    """Extract 2-3 most relevant sentences from top chunk."""
    sentences = [s.strip() for s in top_chunk["chunk_text"].split('. ') if s.strip() and len(s.strip()) > 10]
    
    if not sentences:
        return top_chunk["chunk_text"][:200] + "...", 0.5
    
    sent_embeds = model.encode(sentences, show_progress_bar=False)
    q_embed = model.encode([question], show_progress_bar=False)
    
    sims = cosine_similarity(q_embed, sent_embeds).flatten()
    top_idx = sims.argsort()[-3:][::-1]
    key_sentences = [sentences[i] for i in top_idx]
    
    answer = '. '.join(key_sentences) + '.'
    confidence = float(sims[top_idx[0]]) if len(top_idx) > 0 else 0.5
    
    return answer, confidence

def save_feedback(question, answer, section, confidence, helpful):
    """Append user feedback to a CSV file."""
    feedback = {
        "timestamp": pd.Timestamp.now(),
        "question": question,
        "answer": answer,
        "section": section,
        "confidence": round(confidence, 3),
        "helpful": helpful,
    }
    df = pd.DataFrame([feedback])
    
    if not os.path.exists(FEEDBACK_PATH):
        df.to_csv(FEEDBACK_PATH, index=False)
    else:
        df.to_csv(FEEDBACK_PATH, mode='a', header=False, index=False)

def count_sections_from_feedback():
    """Count section frequency for analytics."""
    if not os.path.exists(FEEDBACK_PATH):
        return {}
    try:
        df = pd.read_csv(FEEDBACK_PATH)
        return df["section"].value_counts().to_dict()
    except:
        return {}

# ============================================================================
# STREAMLIT UI - MODERN DESIGN WITH CENTRALIZED COLORS
# ============================================================================

def setup_css():
    """Setup CSS with centralized color palette"""
    st.markdown(f"""
    <style>
    /* Main styles */
    .main-container {{
        background-color: {BACKGROUND};
        color: {TEXT};
    }}
    
    .header-section {{
        background: linear-gradient(135deg, {PRIMARY} 0%, {WARNING} 100%);
        padding: 3rem 2rem;
        border-radius: 20px;
        margin-bottom: 2rem;
        box-shadow: 0 10px 30px rgba(0,0,0,0.1);
        text-align: center;
    }}
    
    .main-title {{
        font-size: 3rem;
        font-weight: 800;
        color: white;
        margin: 0;
        text-shadow: 2px 2px 4px rgba(0,0,0,0.2);
    }}
    
    .subtitle {{
        color: rgba(255,255,255,0.9);
        font-size: 1.3rem;
        margin: 1rem 0 0 0;
        font-weight: 400;
    }}
    
    /* Cards */
    .answer-card {{
        background: {BACKGROUND};
        padding: 2.5rem;
        border-radius: 20px;
        margin: 2rem 0;
        box-shadow: 0 8px 25px rgba(0,0,0,0.1);
        border: 1px solid #e0e0e0;
        transition: transform 0.3s ease, box-shadow 0.3s ease;
    }}
    
    .answer-card:hover {{
        transform: translateY(-5px);
        box-shadow: 0 15px 35px rgba(0,0,0,0.15);
    }}
    
    .metric-card {{
        background: linear-gradient(135deg, {PRIMARY} 0%, {WARNING} 100%);
        padding: 2rem 1rem;
        border-radius: 15px;
        text-align: center;
        color: white;
        box-shadow: 0 5px 15px rgba(0,0,0,0.1);
        transition: transform 0.2s ease;
    }}
    
    .metric-card:hover {{
        transform: scale(1.05);
    }}
    
    /* Buttons */
    .stButton>button {{
        background: linear-gradient(135deg, {PRIMARY} 0%, {WARNING} 100%);
        color: {BACKGROUND};
        border: none;
        border-radius: 12px;
        padding: 0.8rem 2rem;
        font-weight: 700;
        font-size: 1rem;
        box-shadow: 0 4px 12px {PRIMARY}33;
        transition: all 0.3s ease;
    }}
    
    .stButton>button:hover {{
        background: linear-gradient(135deg, {WARNING} 0%, #FF8C00 100%);
        color: {BACKGROUND};
        box-shadow: 0 6px 20px {WARNING}66;
        transform: translateY(-2px);
    }}
    
    .secondary-button {{
        background: linear-gradient(135deg, {SECONDARY} 0%, #616161 100%) !important;
        color: {BACKGROUND} !important;
    }}
    
    /* Sidebar */
    .sidebar .sidebar-content {{
        background: linear-gradient(180deg, #f8f9fa 0%, #e9ecef 100%);
        border-right: 1px solid #dee2e6;
    }}
    
    /* Input fields */
    .stTextInput>div>div>input {{
        border: 2px solid #e0e0e0;
        border-radius: 12px;
        padding: 1rem;
        font-size: 1rem;
        background-color: {BACKGROUND};
        color: {TEXT};
    }}
    
    .stTextInput>div>div>input:focus {{
        border-color: {PRIMARY};
        box-shadow: 0 0 0 2px {PRIMARY}33;
    }}
    
    /* Expander */
    .stExpander {{
        border: 1px solid #e0e0e0;
        border-radius: 12px;
        background-color: {BACKGROUND};
    }}
    
    /* Progress bar */
    .stProgress > div > div > div {{
        background: linear-gradient(135deg, {PRIMARY} 0%, {WARNING} 100%);
    }}
    
    /* Sample question buttons */
    .sample-question {{
        background: {BACKGROUND};
        border: 2px solid {PRIMARY};
        border-radius: 10px;
        padding: 1rem;
        margin: 0.5rem 0;
        color: {TEXT};
        font-weight: 600;
        transition: all 0.3s ease;
        width: 100%;
        text-align: left;
    }}
    
    .sample-question:hover {{
        background: {PRIMARY};
        color: {BACKGROUND};
        transform: translateX(5px);
    }}
    
    /* Feedback buttons */
    .feedback-btn {{
        margin: 0.5rem;
        border-radius: 10px;
        padding: 0.8rem 1.5rem;
        font-weight: 600;
        transition: all 0.3s ease;
    }}
    
    /* Custom sections */
    .welcome-section {{
        background: linear-gradient(135deg, {PRIMARY} 0%, {WARNING} 100%);
        color: {BACKGROUND};
        padding: 3rem;
        border-radius: 20px;
        text-align: center;
        margin: 2rem 0;
    }}
    
    .search-section {{
        background: {BACKGROUND};
        padding: 2rem;
        border-radius: 15px;
        box-shadow: 0 5px 15px rgba(0,0,0,0.08);
        margin: 2rem 0;
        border: 1px solid #e0e0e0;
    }}
    
    /* Answer highlight */
    .answer-highlight {{
        background: {ACCENT}33;
        border-left: 4px solid {PRIMARY};
        padding: 1.5rem;
        border-radius: 10px;
        margin: 1rem 0;
    }}
    
    /* Section colors */
    .section-badge {{
        background: {PRIMARY};
        color: {BACKGROUND};
        padding: 0.3rem 0.8rem;
        border-radius: 20px;
        font-size: 0.9rem;
        font-weight: 600;
    }}
    </style>
    """, unsafe_allow_html=True)

def main():
    st.set_page_config(
        page_title="Smart Student Assistant - T.I.P.",
        page_icon="🎓",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    
    # Setup CSS with centralized colors
    setup_css()
    
    # ========================================================================
    # LOAD RESOURCES
    # ========================================================================
    
    model = load_model()
    chunks, all_sections = load_manual_from_json()
    section_examples = load_section_examples()
    
    if not chunks:
        st.error("⚠️ Failed to load manual data. Please ensure 'manual_data.json' exists.")
        return
    
    index, chunk_embeds = build_index(chunks, model)
    
    # ========================================================================
    # SIDEBAR - MODERN DESIGN
    # ========================================================================
    
    with st.sidebar:
        # School Logo Section
        logo_file = "TIP LOGO.jpg"
        if os.path.exists(logo_file):
            st.image(logo_file, use_container_width=True)
        else:
            st.markdown(f"""
            <div style='text-align: center; padding: 2rem; background: linear-gradient(135deg, {PRIMARY} 0%, {WARNING} 100%); 
                        border-radius: 15px; margin-bottom: 2rem; color: white;'>
                <h2 style='margin: 0; font-size: 2rem;'>🎓</h2>
                <h3 style='margin: 0.5rem 0;'>T.I.P.</h3>
                <p style='margin: 0; font-weight: bold;'>Smart Assistant</p>
            </div>
            """, unsafe_allow_html=True)
        
        st.markdown("### 🚀 Quick Guide")
        st.markdown("""
        **How to use:**
        1. Type your question
        2. Click **Ask** or use sample questions
        3. Get instant answers from the T.I.P. Manual
        4. Rate the answer quality
        
        **Available Sections:**
        """)
        
        for section in all_sections:
            st.markdown(f"• {section}")
        
        st.divider()
        
        # Statistics
        col1, col2 = st.columns(2)
        with col1:
            st.metric("📚 Total Chunks", len(chunks))
        with col2:
            st.metric("📑 Sections", len(all_sections))
    
    # ========================================================================
    # MAIN CONTENT - REACT-STYLE COMPONENTS
    # ========================================================================
    
    # Initialize session state for React-like navigation
    if 'current_answer' not in st.session_state:
        st.session_state.current_answer = None
    if 'current_question' not in st.session_state:
        st.session_state.current_question = ""
    
    # HEADER SECTION
    st.markdown(f"""
    <div class="header-section">
        <div class="main-title">Smartual Student Assistant</div>
        <div class="subtitle">🏫 Technological Institute of the Philippines</div>
    </div>
    """, unsafe_allow_html=True)
    
    # HOME PAGE (React-style conditional rendering)
    if st.session_state.current_answer is None:
        render_home_page(model, chunks, index, chunk_embeds, section_examples, all_sections)
    else:
        render_results_page()
    
    # FOOTER
    st.markdown("---")
    st.markdown(f"""
    <div style="text-align: center; color: {SECONDARY}; padding: 2rem 0 1rem 0;">
        <p style="margin: 0.2rem; font-size: 0.9rem;">© 2025 Smart Student Assistant | T.I.P. Q&A System</p>
        <p style="margin: 0.2rem; font-size: 0.8rem;">Powered by AI • Built with Streamlit</p>
    </div>
    """, unsafe_allow_html=True)

def render_home_page(model, chunks, index, chunk_embeds, section_examples, all_sections):
    """Render the home page component (React-style)"""
    
    # Welcome Section with School Logo
    col1, col2 = st.columns([1, 2])
    with col1:
        st.image("TIP LOGO.jpg", width=150)
    with col2:
        st.markdown(f"""
        <div style='padding: 1rem;'>
            <h2 style='margin-bottom: 1rem; color: {TEXT};'>👋 Welcome to Your Smart Assistant!</h2>
            <p style='font-size: 1.2rem; margin-bottom: 0; color: {SECONDARY};'>
            Get instant answers to your questions about the T.I.P. Student Manual
            </p>
        </div>
        """, unsafe_allow_html=True)
    
    # Search Section
    st.markdown(f"""
    <div class="search-section">
        <h3 style='color: {TEXT}; margin-bottom: 1.5rem;'>💬 Ask Your Question</h3>
    </div>
    """, unsafe_allow_html=True)
    
    # Input Area
    col1, col2 = st.columns([3, 1])
    
    with col1:
        question = st.text_input(
            "Enter your question:",
            placeholder="e.g., How can I apply for a scholarship? What are the grading policies?",
            label_visibility="collapsed",
            key="main_question_input"
        )
    
    with col2:
        ask_pressed = st.button("🚀 **ASK**", type="primary", use_container_width=True)
    
    # Sample Questions
    st.markdown("### 💡 Sample Questions")
    sample_cols = st.columns(2)
    
    samples = [
        "What are the admission requirements for T.I.P.?",
        "How is the final grade computed in courses?",
        "What scholarships are available for students?",
        "What is the policy on academic probation?",
        "How many absences are allowed per semester?",
        "What services does the T.I.P. library offer?",
        "How can I request for official documents?",
        "What are the guidelines for thesis writing?"
    ]
    
    for i, sample in enumerate(samples):
        with sample_cols[i % 2]:
            if st.button(f"📌 {sample}", key=f"sample_{i}", use_container_width=True):
                st.session_state.current_question = sample
                process_question(sample, model, chunks, index, chunk_embeds, section_examples)
                st.rerun()
    
    # Manual ask processing
    if ask_pressed and question.strip():
        st.session_state.current_question = question
        process_question(question, model, chunks, index, chunk_embeds, section_examples)
        st.rerun()
    elif ask_pressed:
        st.warning("⚠️ Please enter a question first!")

def render_results_page():
    """Render the results page component (React-style)"""
    
    # Back button
    col1, col2 = st.columns([1, 4])
    with col1:
        if st.button("⬅️ Back", use_container_width=True):
            st.session_state.current_answer = None
            st.rerun()
    
    # Display Results
    answer_data = st.session_state.current_answer
    
    st.markdown(f"""
    <div class="answer-card">
        <h3 style='color: {TEXT}; margin-top: 0; border-bottom: 2px solid {PRIMARY}; padding-bottom: 1rem;'>
            💡 Answer
        </h3>
        <div class="answer-highlight">
            {answer_data['answer']}
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    # Metrics
    st.markdown("### 📊 Answer Details")
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown(f"""
        <div class="metric-card">
            <div style='font-size: 2rem;'>📂</div>
            <h4 style='margin: 0.5rem 0;'>Section</h4>
            <p style='margin: 0; font-size: 1.1rem;'><strong>{answer_data['section']}</strong></p>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        confidence_color = SUCCESS if answer_data['confidence'] > 0.6 else WARNING
        st.markdown(f"""
        <div class="metric-card">
            <div style='font-size: 2rem; color: {confidence_color};'>💯</div>
            <h4 style='margin: 0.5rem 0;'>Confidence</h4>
            <p style='margin: 0; font-size: 1.1rem;'><strong>{answer_data['confidence']:.2%}</strong></p>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        st.markdown(f"""
        <div class="metric-card">
            <div style='font-size: 2rem;'>📄</div>
            <h4 style='margin: 0.5rem 0;'>Source</h4>
            <p style='margin: 0; font-size: 1.1rem;'><strong>T.I.P. Manual</strong></p>
        </div>
        """, unsafe_allow_html=True)
    
    # Source Details
    with st.expander("🔍 View Source Information", expanded=False):
        for i, (chunk, score) in enumerate(zip(answer_data['top_chunks'], answer_data['similarities']), 1):
            st.markdown(f"""
            **Source {i}** (Relevance: `{score:.2%}`) - **Section:** *{chunk['section']}*
            
            {chunk['chunk_text']}
            """)
            if i < len(answer_data['top_chunks']):
                st.divider()
    
    # Feedback Section
    st.markdown("---")
    st.markdown("### 📣 Rate This Answer")
    
    feedback_col1, feedback_col2 = st.columns(2)
    
    with feedback_col1:
        if st.button("👍 Helpful Answer", use_container_width=True, type="primary"):
            save_feedback(
                st.session_state.current_question,
                answer_data['answer'],
                answer_data['section'],
                answer_data['confidence'],
                True
            )
            st.success("🎉 Thank you for your feedback!")
            st.balloons()
    
    with feedback_col2:
        if st.button("👎 Needs Improvement", use_container_width=True):
            save_feedback(
                st.session_state.current_question,
                answer_data['answer'],
                answer_data['section'],
                answer_data['confidence'],
                False
            )
            st.info("📝 Thanks for helping us improve!")

def process_question(question, model, chunks, index, chunk_embeds, section_examples):
    """Process question and store results in session state"""
    with st.spinner("🔍 Searching through the Student Manual..."):
        # Classify question to section
        pred_section, section_conf = classify_question(question, model, section_examples)
        
        # Retrieve top chunks
        top_chunks, similarities = retrieve_chunks(question, model, chunks, index, chunk_embeds, top_k=3)
        
        # Generate answer from best chunk
        best_chunk = top_chunks[0]
        answer, confidence = generate_answer(question, best_chunk, model)
        
        # Store in session state
        st.session_state.current_answer = {
            'answer': answer,
            'section': pred_section,
            'confidence': confidence,
            'top_chunks': top_chunks,
            'similarities': similarities
        }

# ============================================================================
# RUN APP
# ============================================================================

if __name__ == "__main__":
    main()
