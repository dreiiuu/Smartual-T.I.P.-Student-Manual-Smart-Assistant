## 🎯 Overview

# Smartual-T.I.P.-Student-Manual-Smart-Assistant
A web application that helps students at the Technological Institute of the Philippines (T.I.P.) find information from their Student Manual using semantic search and in-context classification.

### ✨ Key Features

- 🔍 **Semantic Search**: Uses sentence-transformers to find relevant information based on meaning, not just keywords
- 🧠 **In-Context Classification**: Classifies questions into manual sections using example-based similarity
- 💯 **Confidence Scoring**: Provides transparency about answer reliability
- 📊 **Real-time Feedback**: Tracks user satisfaction and generates analytics
- 🔒 **Deployed in Streamlit Platform**: Used Streamlit Platform to deploy the Web Application
- 📚 **Real Data**: Trained on the actual TIP-Manual-2025.pdf

---

## 🏗️ Architecture

### Technology Stack
- **Frontend & Backend**: Streamlit
- **Embedding Model**: sentence-transformers (`all-MiniLM-L6-v2`)
- **Vector Search**: FAISS (Facebook AI Similarity Search)
- **Similarity Computation**: scikit-learn cosine similarity
- **Data Format**: JSON (structured manual sections)

### System Components

1. **Data Layer** (`manual_data.json`, `section_examples.json`)
   - Structured student manual content organized into 14 major sections
   - Example questions for each section (used for in-context classification)

2. **Embedding & Retrieval Layer**
   - Pre-trained sentence-transformers model for semantic embeddings
   - FAISS index for efficient similarity search
   - Top-K retrieval (default K=3)

3. **Classification Layer**
   - Example-based section classifier
   - Computes average similarity between user question and section examples

4. **Answer Generation Layer**
   - Extracts 2-3 most relevant sentences from top chunk
   - Sentence-level similarity ranking

5. **Feedback & Analytics Layer**
   - CSV-based feedback logging only for trial and testing

---

## 📦 Installation & Setup

### Prerequisites
- Python 3.10 or higher
- pip package manager

### Step 1: Install Dependencies

bash
```
pip install -r requirements.txt
```

This will install:
- `streamlit` - Web framework
- `sentence-transformers` - Embedding model
- `faiss-cpu` - Vector similarity search
- `scikit-learn` - Cosine similarity computation
- `numpy`, `pandas` - Data processing

### Step 2: Prepare Data Files

Ensure these files are in the same directory as `app.py`:

1. **`manual_data.json`** - Structured T.I.P. Student Manual content
2. **`section_examples.json`** - Example questions for classification
3. **`app.py`** - Main Streamlit application
4. **`requirements.txt`** - Python dependencies

---

## 🚀 Running the application before deployment

### Launch Command

```bash
streamlit run app.py
```

The application will automatically:
1. Load the sentence-transformers model (first run downloads ~90MB)
2. Load and chunk the student manual data
3. Build FAISS index for semantic search
4. Launch the web interface (typically at `http://localhost:8501`)

### First-Time Setup
On first run, the application will download the `all-MiniLM-L6-v2` model from Hugging Face. This is a one-time download (~90MB) and will be cached locally.

---

## 📖 How to Use

### Basic Usage

1. **Ask a Question**: Type your question in the search box
   - Example: "How can I apply for a scholarship?"
   - Example: "What is academic probation?"

2. **View Answer**: The system displays:
   - Extracted answer from the manual
   - Section classification
   - Confidence score
   - Source reference (T.I.P Manual 2025)

3. **Explore Details**: Click "View Matching Snippets" to see:
   - Top 3 relevant chunks from the manual
   - Similarity scores for each chunk
   - Full context from each section

4. **Provide Feedback**: Click "Helpful" or "Not Helpful" buttons
   - Feedback is logged to `feedback_log.csv` for pre-deployment only

### Advanced Features

- **Sample Questions**: Click pre-defined sample questions for quick testing
- **Section Analytics**: View most-asked sections in the sidebar
- **Confidence Indicators**: 
  - ✅ Green card = High confidence
  - ⚠️ Orange card = Lower confidence

---

## 🧠 Technical Details

### How It Works

#### 1. **Question Processing**
```
User Question → Sentence Embedding → Vector Representation
```

#### 2. **In-Context Classification**
```
Question Embedding → Compare with Section Examples → Predict Section
```
- Computes cosine similarity with all example questions
- Selects section with highest average similarity

#### 3. **Semantic Retrieval**
```
Question Vector → FAISS Search → Top K Chunks
```
- Searches through all manual chunks
- Returns 3 most similar chunks

#### 4. **Answer Extraction**
```
Top Chunk → Split into Sentences → Rank by Similarity → Extract Top 2-3
```
- Uses sentence-level similarity
- Combines most relevant sentences into coherent answer

### Data Structure

**Manual Data** (`manual_data.json`):
```json
{
  "Section Name": "Full section content...",
  "Another Section": "Content..."
}
```

**Section Examples** (`section_examples.json`):
```json
{
  "Section Name": [
    "Example question 1",
    "Example question 2"
  ]
}
```

### Performance Characteristics

- **Chunk Size**: 300 words (configurable)
- **Retrieval**: Top 3 chunks
- **Model Size**: ~90MB (all-MiniLM-L6-v2)
- **Embedding Dimension**: 384
  
---

## 🎨 Customization

### Adjust Chunk Size
Edit `CHUNK_SIZE` in `app.py`:
```python
CHUNK_SIZE = 300  # Increase for longer contexts, decrease for more precise matching
```

### Change Retrieval Count
Modify `top_k` parameter in `retrieve_chunks()`:
```python
top_chunks, sim_scores = retrieve_chunks(question, model, chunks, index, chunk_embeds, top_k=3)
```

### Use Different Model
Change `MODEL_NAME` in `app.py`:
```python
MODEL_NAME = "paraphrase-MiniLM-L6-v2"  # or any sentence-transformers model
```

### Add More Sections
Update `manual_data.json` and `section_examples.json` with new sections and examples.

---

## 📊 Data Files Included

### 1. `manual_data.json` (14 sections)
- General Information
- Admissions
- Registration and Enrollment
- Grading System
- Academic Probation and Retention
- Graduation Requirements
- Scholarships and Financial Aid
- Student Conduct and Discipline
- Student Organizations
- Attendance Policy
- Student Services
- Tuition and Fees
- Disciplinary Offenses
- Educational Philosophy

### 2. `section_examples.json`
- 6 example questions per section (84 total)
- Used for in-context classification

### 3. `feedback_log.csv` (auto-generated ~ testing only)
- Logs all test-case user interactions
- Columns: timestamp, question, answer, section, confidence, helpful

---

## 🔧 Troubleshooting

### Issue: Model Download Fails
**Solution**: Ensure internet connection on first run. The model downloads from Hugging Face.

### Issue: "Manual data file not found"
**Solution**: Ensure `manual_data.json` is in the same directory as `app.py`.

### Issue: FAISS Installation Error
**Solution**: 
- Use `faiss-cpu` (included in requirements.txt)
- For GPU support, install `faiss-gpu` separately

---

## 📝 Code Organization

### Main Functions

1. **`load_manual_from_json()`** - Loads and chunks manual data
2. **`load_section_examples()`** - Loads example questions
3. **`build_index()`** - Creates FAISS index from embeddings
4. **`classify_question()`** - Predicts section using in-context learning
5. **`retrieve_chunks()`** - Performs semantic search
6. **`generate_answer()`** - Extracts relevant sentences
7. **`save_feedback()`** - Logs user feedback
8. **`main()`** - Streamlit UI and orchestration

### Caching Strategy
- `@st.cache_data` - For data loading functions
- `@st.cache_resource` - For model and index (non-serializable objects)

---

## 🎓 Educational Value

This project demonstrates:
- **NLP Techniques**: Semantic search, embeddings, similarity computation
- **HCI Principles**: Clean UI, feedback loops for testing, transparency (confidence scores)
- **Production Practices**: Caching and error handling
  
---

## 🚀 Future Enhancements

Potential improvements:
1. **Multi-language Support**: Add Filipino language interface
2. **Advanced Analytics**: Track question patterns, unanswered queries
3. **Re-ranking**: Implement cross-encoder for better answer selection
4. **Conversational**: Add chat history and follow-up questions
5. **Admin Panel**: Interface to update manual content
6. **Export**: Allow students to download answers as PDF


---

## 📄 License & Attribution

- Built for educational purposes
- Uses open-source libraries (Streamlit, sentence-transformers, FAISS)
- Based on TIP-Manual-2025.pdf (Technological Institute of the Philippines)

---

## 📄 Developers of smartual project

- Bona, Andrei Nycole So
- GitHub Account: https://github.com/ansbona
  
- Guariño, Danica
- GitHub Account: https://github.com/DanicaGuarino

- Santos, Andrei
- GitHub Account: https://github.com/dreiiuu

## 📄 Adviser of the smartual project
- Engr. Neal Barton James Matira
- GitHub Account: https://github.com/neeeal

---

**Happy Learning! 🎓**
