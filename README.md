# 🌱 AgriLM — Multimodal Retrieval-Grounded Agricultural Intelligence

AgriLM is a **multimodal visual–semantic reasoning system** designed for intelligent agricultural decision support.  
It integrates **image understanding, text reasoning, vector retrieval (FAISS), and retrieval-augmented generation (RAG)** to produce accurate, explainable, and evidence-grounded recommendations.

> 🚀 Fully local execution — no API keys required.

---

## 🔍 Overview

AgriLM processes **multimodal inputs**:
- 🌿 Crop/leaf images  
- 🧾 Farmer queries (text)  
- 📄 Agricultural documents / soil reports  

It produces:
- Disease diagnosis  
- Explanation  
- Treatment recommendations  
- Confidence score  

---

## 🧠 How the Model Works


Input (Image + Text + Document)
↓
Embedding Layer (CLIP + MiniLM)
↓
Cross-Modal Transformer (Fusion)
↓
Multi-Vector Representation (ColBERT-style)
↓
FAISS Retrieval (Top-K Knowledge)
↓
Retrieval-Augmented Generation (RAG)
↓
Structured Output + Confidence


---

## ⚙️ Core Features

- ✅ Multimodal reasoning (Image + Text + Documents)  
- ✅ Cross-modal transformer alignment  
- ✅ Multi-vector semantic representation  
- ✅ FAISS-based scalable retrieval  
- ✅ Retrieval-grounded generation (RAG)  
- ✅ Confidence estimation  
- ✅ Real-time inference  

---

## 📂 Project Structure

```bash
AgriLM/
│
├── app.py # Streamlit UI 
├── main.py # CLI interface
├── config.py # Hyperparameters
├── build_index.py # FAISS index builder
├── requirements.txt
├── knowledge_base.json # Knowledge base for RAG
│
├── models/
│ ├── __init__.py
│ ├── embeddings.py # Image + Text encoders
│ ├── fusion.py # Multimodal fusion
│ ├── cross_modal.py # Transformer
│ ├── multi_vector.py # Multi-vector representation
│ ├── retrieval.py # FAISS search
│ ├── rag.py # RAG generation
│ ├── pipeline.py # Full pipeline
│
├── faiss_index/
│ ├── embeddings.npy
│ ├── index.faiss
│ ├── metadata.pkl

```

## 📊 Datasets Used

| Type | Dataset |
|------|--------|
| Images | PlantVillage, PlantDoc |
| Soil Data | Crop Recommendation Dataset |
| Documents | Agricultural knowledge corpus |
| QA Data | Agricultural QA datasets |

---

## 🛠 Installation


pip install -r requirements.txt


### 🚀 Running the Project

1️⃣ **Build FAISS Index**

python build_index.py


2️⃣ **Run Streamlit UI (Recommended)**

streamlit run app.py


3️⃣ **Run CLI Mode**

python main.py

### 🧪 Example Query
**Input:**
"Yellow spots on tomato leaves"

## Output:
Diagnosis: Early Blight
Recommendation: Apply fungicide
Confidence: 0.91

## 📈 Evaluation Metrics
- **Diagnosis**
  - Accuracy
  - Precision
  - Recall
  - F1-score
- **Retrieval**
  - Precision@k
  - Recall@k
  - nDCG@k
- **System**
  - Response time
  - Usability score

## ⚡ Computational Efficiency
| Component | Complexity |
|---|---|
| Cross-modal Attention | O(nmd) |
| FAISS Retrieval | O(log N) |
| Multi-vector Storage | O(kd) |

## 🔬 Key Contributions
- Unified multimodal reasoning architecture
- Fine-grained multi-vector alignment
- Scalable FAISS-based retrieval
- Retrieval-grounded structured generation
- Joint optimization of alignment + retrieval + generation

## 📌 Implementation Workflow

Dataset → Preprocessing
        ↓
Embedding Generation
        ↓
FAISS Indexing
        ↓
Multimodal Fusion
        ↓
Retrieval + RAG
        ↓
Output Generation


## 🧾 Tech Stack
- PyTorch
- FAISS
- CLIP
- Sentence Transformers
- Streamlit

## 📄 License

Research and educational use.

## 🙌 Acknowledgments
- OpenAI CLIP
- FAISS (Meta AI)
- Sentence Transformers
- ColBERT
