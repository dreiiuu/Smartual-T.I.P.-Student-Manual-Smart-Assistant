# ============================================================================
# SMART STUDENT ASSISTANT - T.I.P. Student Manual Q&A System
# A Streamlit application using semantic search and in-context classification
# Author: AI Assistant | Date: November 2025
# ============================================================================

import os
import json
import streamlit as st
import pandas as pd
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

# ============================================================================
# CONFIGURATION
# ============================================================================

MANUAL_DATA_FILE = "manual_data.json"
SECTION_EXAMPLES_FILE = "section_examples.json"
FEEDBACK_PATH = "feedback_log.csv"
CHUNK_SIZE = 300  # words per chunk
MODEL_PATH = "D:\smartual_model" 
# The model is on the local host of one of the member, will be uploading the model in the GitHub. Thanks!

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
        # Split into sentences for better chunking
        sentences = [s.strip() for s in section_text.split('. ') if s.strip()]
        
        # Group sentences into chunks
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
        
        # Add remaining chunk
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
    
    # Create FAISS index
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
    # Split into sentences
    sentences = [s.strip() for s in top_chunk["chunk_text"].split('. ') if s.strip() and len(s.strip()) > 10]
    
    if not sentences:
        return top_chunk["chunk_text"][:200] + "...", 0.5
    
    # Encode sentences and question
    sent_embeds = model.encode(sentences, show_progress_bar=False)
    q_embed = model.encode([question], show_progress_bar=False)
    
    # Compute similarities
    sims = cosine_similarity(q_embed, sent_embeds).flatten()
    
    # Pick top 2-3 sentences by similarity
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
# STREAMLIT UI
# ============================================================================

def main():
    st.set_page_config(
        page_title="Smartual Student Assistant - T.I.P.",
        page_icon="🎓",
        layout="wide"
    )
    
    # Custom CSS
    st.markdown("""
    <style>
    .main-title {
        font-size: 2.5rem;
        font-weight: 700;
        color: #1f4788;
        text-align: center;
        margin-bottom: 1rem;
    }
    .subtitle {
        text-align: center;
        color: #666;
        margin-bottom: 2rem;
    }
    .answer-card {
        padding: 1.5rem;
        border-radius: 10px;
        margin: 1rem 0;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }
    .high-confidence {
        background-color: #E3FCEF;
        border-left: 5px solid #00A86B;
    }
    .low-confidence {
        background-color: #FFF3E0;
        border-left: 5px solid #FF9800;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 8px;
        text-align: center;
    }
    </style>
    """, unsafe_allow_html=True)
    
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
    # SIDEBAR
    # ========================================================================
    
    with st.sidebar:
        st.image("https://via.placeholder.com/150x50/1f4788/FFFFFF?text=T.I.P.", use_container_width=True)
        
        st.markdown("### 🤖 Smartual Student Assistant")
        st.markdown("""
        Ask any question about the **T.I.P. Student Manual** and get instant answers!
        
        **Features:**
        - 🔍 Semantic search across manual
        - 🧠 In-context classification
        - 💯 Confidence scoring
        - 📊 Real-time feedback tracking
        - 🔒 100% local (no internet required)
        """)
        
        st.divider()
        
        st.markdown("**📚 Manual Sections:**")
        for i, section in enumerate(all_sections, 1):
            st.markdown(f"{i}. {section}")
        
        st.divider()
        
        st.markdown(f"""
        **🔧 System Info:**
        - Model: `{MODEL_PATH}`
        - Total Chunks: `{len(chunks)}`
        - Sections: `{len(all_sections)}`
        """)
        
        # Optional analytics
        sec_counts = count_sections_from_feedback()
        if sec_counts:
            st.divider()
            st.markdown("**📈 Most Asked Sections**")
            st.bar_chart(sec_counts)
    
    # ========================================================================
    # MAIN PANEL
    # ========================================================================
    
    st.markdown('<div class="main-title">🎓 Smartual Student Assistant</div>', unsafe_allow_html=True)
    st.markdown('<div class="subtitle">Technological Institute of the Philippines (T.I.P.) - Student Manual Q&A System</div>', unsafe_allow_html=True)
    
    st.markdown("---")
    
    # Input area
    col1, col2 = st.columns([4, 1])
    
    with col1:
        question = st.text_input(
            "❓ Ask your question:",
            placeholder="e.g., How can I apply for a scholarship?",
            label_visibility="collapsed"
        )
    
    with col2:
        ask_pressed = st.button("🔍 Ask", type="primary", use_container_width=True)
    
    # Sample questions
    with st.expander("💡 Sample Questions"):
        sample_cols = st.columns(3)
        samples = [
            "What are the admission requirements?",
            "How is the final grade computed?",
            "What scholarships are available?",
            "What is academic probation?",
            "How many absences are allowed?",
            "What services does the library offer?"
        ]
        for i, sample in enumerate(samples):
            with sample_cols[i % 3]:
                if st.button(sample, key=f"sample_{i}"):
                    question = sample
                    ask_pressed = True
    
    st.markdown("---")
    
    # ========================================================================
    # PROCESS QUERY
    # ========================================================================
    
    if ask_pressed and question.strip():
        with st.spinner("🔍 Searching through the Student Manual..."):
            # Step 1: Classify question to section
            pred_section, section_conf = classify_question(question, model, section_examples)
            
            # Step 2: Retrieve top chunks
            top_chunks, sim_scores = retrieve_chunks(question, model, chunks, index, chunk_embeds, top_k=3)
            
            # Step 3: Generate answer from best chunk
            best_chunk = top_chunks[0]
            answer, confidence = generate_answer(question, best_chunk, model)
            
            # Determine card color based on confidence
            card_class = "high-confidence" if confidence > 0.6 else "low-confidence"
            confidence_emoji = "✅" if confidence > 0.6 else "⚠️"
        
        # ========================================================================
        # DISPLAY RESULTS
        # ========================================================================
        
        st.markdown(f"""
        <div class="answer-card {card_class}">
            <h3>{confidence_emoji} Answer</h3>
            <p style="font-size: 1.1rem; line-height: 1.6;">{answer}</p>
        </div>
        """, unsafe_allow_html=True)
        
        # Metadata
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.markdown(f"""
            <div class="metric-card">
                <h4>📂 Section</h4>
                <p><strong>{pred_section}</strong></p>
            </div>
            """, unsafe_allow_html=True)
        
        with col2:
            st.markdown(f"""
            <div class="metric-card">
                <h4>💯 Confidence</h4>
                <p><strong>{confidence:.2%}</strong></p>
            </div>
            """, unsafe_allow_html=True)
        
        with col3:
            st.markdown(f"""
            <div class="metric-card">
                <h4>📄 Source</h4>
                <p><strong>T.I.P. Manual 2025</strong></p>
            </div>
            """, unsafe_allow_html=True)
        
        # Show matching snippets
        with st.expander("📋 View Matching Snippets from Manual"):
            for i, (chunk, score) in enumerate(zip(top_chunks, sim_scores), 1):
                st.markdown(f"""
                **Snippet {i}** (Similarity: {score:.2%}) - Section: *{chunk['section']}*
                
                {chunk['chunk_text'][:500]}...
                """)
                st.divider()
        
        # ========================================================================
        # FEEDBACK SECTION
        # ========================================================================
        
        st.markdown("---")
        st.markdown("### 📣 Was this answer helpful?")
        
        feedback_col1, feedback_col2, feedback_col3 = st.columns([1, 1, 3])
        
        with feedback_col1:
            if st.button("👍 Helpful", use_container_width=True):
                save_feedback(question, answer, pred_section, confidence, True)
                st.success("✅ Thank you for your feedback!")
                st.balloons()
        
        with feedback_col2:
            if st.button("👎 Not Helpful", use_container_width=True):
                save_feedback(question, answer, pred_section, confidence, False)
                st.info("💬 Thanks! We'll improve our responses.")
    
    elif ask_pressed and not question.strip():
        st.warning("⚠️ Please enter a question to get started!")
    
    else:
        # Welcome message
        st.info("""
        👋 **Welcome to the Smart Student Assistant!**
        
        This system helps you find information from the T.I.P. Student Manual quickly and accurately.
        Simply type your question in the search box above or click on one of the sample questions to get started.
        
        All processing happens **locally** on your machine - no internet connection required!
        """)
    
    # Footer
    st.markdown("---")
    st.markdown("""
    <div style="text-align: center; color: #666; padding: 1rem;">
        <p>© 2025 Smart Student Assistant | Powered by Semantic Search & In-Context Classification</p>
        <p><em>Built with Streamlit • sentence-transformers • FAISS</em></p>
    </div>
    """, unsafe_allow_html=True)

# ============================================================================
# RUN APP
# ============================================================================

if __name__ == "__main__":
    main()
