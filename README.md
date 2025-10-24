# 🧠 KDD RAG Project

This project implements a **Retrieval-Augmented Generation (RAG)** pipeline for **KDD Kuwait**, allowing users to ask natural questions about products and career opportunities.  
It combines **web scraping**, **data cleaning**, **semantic search (FAISS)**, and a **local language model** to generate accurate, context-aware answers.

---

## 🚀 Overview

The system retrieves and answers user queries by integrating three main components:

1. **Scraping:** Automatically collects product and career data from KDD’s official websites.  
2. **Embedding + FAISS:** Converts text into vector embeddings and builds a FAISS index for semantic search.  
3. **RAG API:** Uses a lightweight local model (`microsoft/phi-3-mini-4k-instruct`) to generate natural, human-like responses.

The backend is built using **FastAPI**, offering both **REST endpoints** and an interactive **web interface**.

---

## 📂 Project Structure

```

KDD-RAG/
│
├── data/
│   ├── products_ice_cream_detailed.json
│   ├── products_juices_detailed.json
│   ├── careers.json
│   ├── clean/
│   │   ├── products_clean.json
│   │   ├── careers1_clean.json
│   │   ├── corpus.json
│   └── index/
│       ├── faiss.index
│       ├── meta.json
│       ├── conf.json
│
├── scripts/
│   ├── kdd_scraper.py              # Product scraper
│   ├── kdd_careers_scraper.py      # Careers scraper (Selenium)
│   ├── clean_data.py               # Cleans and merges raw data
│   ├── make_embeddings.py          # Generates embeddings
│   ├── build_faiss.py              # Builds FAISS index
│
├── rag_api.py                      # FastAPI app (RAG endpoints)
├── requirements.txt
├── README.md
└── .env

````

---

## ⚙️ Installation

### 1. Clone the Repository
```bash
git clone https://github.com/yourusername/KDD-RAG.git
cd KDD-RAG
````

### 2. Create a Virtual Environment

```bash
python -m venv venv
source venv/bin/activate     # On macOS/Linux
venv\Scripts\activate        # On Windows
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

---

## 💾 Running the Project

### Step 1: Scrape the Data

Run both scrapers to collect the raw product and career data:

```bash
python kdd_scraper.py
python kdd_careers_scraper.py
```

### Step 2: Clean and Merge the Data

```bash
python clean_data.py
```

### Step 3: Generate Embeddings

```bash
python make_embeddings.py
```

### Step 4: Build the FAISS Index

```bash
python build_faiss.py build
```

### Step 5: Run the FastAPI App

```bash
uvicorn rag_api:app --reload
```

Once it’s running, open your browser and go to:

```
http://127.0.0.1:8000/ui
```

---

## 📡 API Endpoints

| Endpoint       | Description                                    |
| -------------- | ---------------------------------------------- |
| `/v1/products` | Retrieve all product entries                   |
| `/v1/careers`  | Retrieve all career entries                    |
| `/v1/search`   | Perform semantic search using FAISS            |
| `/v1/ask`      | Execute full RAG pipeline (query + generation) |
| `/ui`          | Web interface                                  |
| `/docs`        | FastAPI auto-generated documentation           |

---

## 📚 Documentation

### 1. Scraping Methodology and Challenges

I developed two main scrapers — one for **products** and another for **careers**.

#### Product Scraper

* Built using **Requests** and **BeautifulSoup**.
* Extracted product details such as name, price, ingredients, category, and size.
* Cleaned and normalized the text using regular expressions.
* Saved results in JSON format for consistency and ease of processing.

#### Careers Scraper

* Combined **API requests** with **Selenium** for dynamic job pages.
* Extracted titles, departments, locations, skills, and job requirements.
* Cleaned, merged, and normalized the text data for downstream processing.

#### Key Challenges

| Challenge                     | Solution                                               |
| ----------------------------- | ------------------------------------------------------ |
| JavaScript-rendered job pages | Used Selenium headless browser for dynamic rendering.  |
| Inconsistent page structures  | Implemented multi-selector parsing and fallback logic. |
| Rate limiting and timeouts    | Added retries, randomized delays, and request headers. |
| Duplicate or missing records  | Used SHA-1 hashes to ensure uniqueness.                |
| Mixed English/Arabic content  | Normalized all text to UTF-8 encoding.                 |

---

### 2. Architecture of the RAG Pipeline

The **RAG (Retrieval-Augmented Generation)** pipeline connects data retrieval with a generative model to provide meaningful, data-backed responses.

**Components:**

1. **Data Preparation:** Merged all cleaned product and career datasets into a unified corpus (`corpus.json`).
2. **Embeddings:** Used `BAAI/bge-small-en-v1.5` from SentenceTransformers to generate 384-dimensional embeddings.
3. **Vector Indexing (FAISS):** Built an index using cosine similarity for fast, accurate retrieval.
4. **Retrieval:** Queries are embedded and compared to the FAISS index.
5. **Generation:** Retrieved context is passed to `microsoft/phi-3-mini-4k-instruct` to generate a final, coherent response.
6. **FastAPI Integration:** All stages are connected via REST endpoints with a web UI.

**Pipeline Flow:**

```
Scraped Data → Cleaned Corpus → Embeddings → FAISS Index → Query → Retrieval → LLM Response
```

---

### 3. Model Selection Criteria

| Component           | Model                              | Reason                                                             |
| ------------------- | ---------------------------------- | ------------------------------------------------------------------ |
| **Embedding Model** | `BAAI/bge-small-en-v1.5`           | Strong semantic accuracy and efficiency for local use.             |
| **Vector Database** | FAISS                              | Fast and scalable similarity search with GPU support.              |
| **Language Model**  | `microsoft/phi-3-mini-4k-instruct` | Lightweight, instruction-tuned model for fluent, relevant answers. |
| **(Optional)**      | `BAAI/bge-m3`                      | Supports multilingual embeddings (Arabic + English).               |

---

## 🧩 Key Features

* Full pipeline from **data scraping → cleaning → RAG inference**
* Semantic search powered by FAISS
* Local deployment without cloud dependency
* FastAPI-based backend with a modern UI
* Clear code structure and full documentation

---

## 🧠 Future Enhancements

* Add multilingual embedding models for Arabic support.
* Deploy to cloud or Hugging Face Spaces for public access.
* Implement feedback-based answer refinement.

---

## ✍️ Author

**Bushra Dajam**
AI Engineer | Data & NLP Enthusiast
University of Jeddah – Artificial Intelligence Major
📧 Email: [[your-email@example.com](mailto:your-email@example.com)]
🌐 GitHub: [github.com/yourusername]
💼 LinkedIn: [linkedin.com/in/bushradajam]

