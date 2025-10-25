# KDD RAG Project


A Retrieval-Augmented Generation **(RAG)** system that provides natural-language answers about KDD (Kuwait Danish Dairy Company) products and career opportunities.  
Built with FastAPI, this project integrates web scraping, semantic search (FAISS), and LLMs to deliver grounded, domain-specific responses.

---

## 🍦 Overview

The KDD RAG Pipeline transforms KDD’s product and career data into an interactive knowledge system.  
Users can ask questions like:

> “What ice cream flavors are available under 1 KWD?”  
> “Are there any open Python skills positions?”

The system retrieves KDD data, ranks the most relevant information, and generates clear, factual answers — all locally and without cloud dependencies.

---

## 📂 Project Structure

```

KDD PROJECT/
│
├── data/
│   ├── clean/
│   │   ├── careers1_clean.json
│   │   ├── products_clean.json
│   ├── embed/                   
│   ├── index/
│   │   ├── faiss.index
│   │   ├── conf.json
│   │   ├── meta.json
│   ├── careers.json
│   ├── corpus.json
│   ├── products_ice_cream_detailed.json
│   ├── products_juices_detailed.json
│
├── static/
│   ├── kdd_logo.png
│   ├── style.css
│   ├── tools.js
│   ├── ui.js
│
├── templates/
│   ├── tools.html
│   ├── ui.html
│
├── build_faiss.py
├── Careers_scraper.py
├── Products_Scraping.py
├── clean_and_build_corpus.py
├── make_embeddings.py
├── rag_api.py
├── requirements.txt
├── requirements_all.txt
└── README.md

````

---

## ⚙️ Installation

### 1. Clone the Repository
```bash
git clone https://github.com/Buushra1dm/RAG-MODEL-KDD.git
cd KDD-RAG
````

### 2. Create and Activate a Virtual Environment

```bash
python -m venv venv
source venv/bin/activate     # macOS/Linux
venv\Scripts\activate        # Windows
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

---

## 💾 Running the Project

### Step 1: Scrape Product and Career Data

```bash
python Products_Scraping.py
python Careers_scraper.py
```

### Step 2: Clean and Build the Corpus

```bash
python clean_and_build_corpus.py
```

### Step 3: Generate Embeddings

```bash
python make_embeddings.py
```

### Step 4: Build and Test the FAISS Index

To build the index:

```bash
python build_faiss.py build
```

To perform a quick search test:

```bash
python build_faiss.py search "your query here"
```

### Step 5: Run the FastAPI Server

```bash
uvicorn rag_api:app --reload
```

or 

```bash
python rag_api.py
```

Then open your browser at:

```
http://127.0.0.1:8000/ui
```

---

## 📡 API Endpoints

| Endpoint       | Description                             |
| -------------- | --------------------------------------- |
| `/v1/products` | Retrieve all KDD products               |
| `/v1/careers`  | Retrieve all career entries             |
| `/v1/search`   | Perform FAISS-based search              |
| `/v1/ask`      | Full RAG query (retrieval + generation) |
| `/ui`          | Chat-style web interface                |

---


##  Architecture of the RAG Pipeline

The **Retrieval-Augmented Generation (RAG)** pipeline integrates semantic retrieval, reranking, and contextual generation to provide domain-grounded answers about KDD products and career opportunities.

### Pipeline Overview

1. **Data Preparation:**
   Scrapes and cleans product and career information from KDD’s official websites, then merges both datasets into a unified structured corpus.

2. **Embeddings:**
   Uses `BAAI/bge-small-en-v1.5` to generate dense semantic embeddings that capture meaning beyond simple keywords.

3. **FAISS Indexing:**
   Stores the embeddings inside a FAISS `IndexFlatIP` structure for efficient cosine-similarity search across thousands of records.

4. **Retrieval:**
   When a user submits a query, it is embedded and compared against the FAISS index to identify semantically similar documents.

5. **Reranking:**
   The top retrieved results are passed to `BAAI/bge-reranker-large`, which cross-encodes and re-evaluates query–document relevance for maximum precision.

6. **Generation:**
   The refined context is used as input for `Qwen/Qwen2.5-1.5B-Instruct`, which generates a clear, concise, and fact-based answer derived only from verified KDD data.

7. **FastAPI Layer:**
   All operations are served through RESTful endpoints with a user-friendly web interface.

---

### Flow Diagram

```
Scraped Data → Clean Corpus → Embeddings → FAISS Index → Retrieve → Rerank → Generate Answer → Display in UI
```

<img width="1024" height="1024" alt="rag" src="https://github.com/user-attachments/assets/ca04ddef-253c-4350-a64f-626be3efc151" />


---

## ⚙️ Tech Stack

| Component | Technology |
|------------|-------------|
| Backend | FastAPI |
| Web Scraping | Requests, BeautifulSoup, Selenium |
| Embeddings | `BAAI/bge-small-en-v1.5` |
| Reranker | `BAAI/bge-reranker-large` |
| Generator | `Qwen/Qwen2.5-1.5B-Instruct` |
| Vector Database | FAISS |
| Language | Python 3.11 |
| Frontend | HTML, CSS, JavaScript |

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











# 🧠 KDD RAG Project

This project implements a **Retrieval-Augmented Generation (RAG)** pipeline for **KDD Kuwait**, enabling users to ask natural questions about KDD products and career opportunities.  
It integrates **web scraping**, **data cleaning**, **semantic search (FAISS)**, and a **local language model** to generate clear and contextually accurate answers.

---

## 🚀 Overview

The system retrieves and answers queries through three key stages:

1. **Scraping:** Collects product and career data from KDD’s official sources.  
2. **Embedding + FAISS:** Converts data into vector representations for semantic search.  
3. **RAG Pipeline:** Uses a lightweight model (`microsoft/phi-3-mini-4k-instruct`) to produce final, human-like responses.

The backend runs on **FastAPI**, featuring both REST endpoints and an interactive browser-based UI.

---


## 📚 Documentation

### 1. Scraping Methodology and Challenges

#### Product Scraping

* Implemented using **Requests** and **BeautifulSoup**.
* Extracts product name, size, flavor, ingredients, and price.
* Handles multiple product categories (ice cream, juices, dairy).
* Saves structured results to JSON for consistency.

#### Careers Scraping

* Uses **Requests** and **Selenium** to handle dynamically rendered job pages.
* Extracts fields such as title, department, location, and required skills.
* Handles multiple layouts and localized (Arabic/English) text.
* Cleans and merges data for use in embeddings.

#### Key Challenges

| Challenge                  | Solution                                             |
| -------------------------- | ---------------------------------------------------- |
| Dynamic JavaScript content | Used Selenium headless Chrome to render and extract. |
| Inconsistent structure     | Implemented multi-selector logic with fallbacks.     |
| Timeouts / rate limits     | Added retry logic and random request delays.         |
| Duplicates                 | Removed using SHA-1 hashing on job titles and URLs.  |
| Encoding issues            | Normalized all data to UTF-8 before processing.      |

---

### 2. Architecture of the RAG Pipeline

The **Retrieval-Augmented Generation (RAG)** system integrates both retrieval and generative reasoning for context-aware answers.

#### Pipeline Overview

1. **Data Preparation:** Cleans and merges scraped JSON files into a single corpus (`corpus.json`).
2. **Embeddings:** Uses `BAAI/bge-small-en-v1.5` to generate dense semantic embeddings.
3. **FAISS Indexing:** Stores embeddings in a FAISS index for cosine similarity search.
4. **Retrieval:** The user query is embedded and matched against the FAISS index.
5. **Generation:** The retrieved context is passed to `microsoft/phi-3-mini-4k-instruct` to generate a clear response.
6. **FastAPI Layer:** All components are served through `/v1/ask` and `/ui`.

#### Flow Diagram

```
Scraped Data → Clean Corpus → Embeddings → FAISS Index → Query → Retrieve → Generate Answer
```

---

### 3. Model Selection Criteria

| Component           | Model                              | Reason                                                  |
| ------------------- | ---------------------------------- | ------------------------------------------------------- |
| **Embedding Model** | `BAAI/bge-small-en-v1.5`           | Accurate and efficient for English semantic retrieval.  |
| **Vector Indexing** | FAISS                              | Fast, scalable, and supports local GPU acceleration.    |
| **Language Model**  | `microsoft/phi-3-mini-4k-instruct` | Small, fast, and instruction-tuned for reasoning tasks. |
| **Alternative**     | `BAAI/bge-m3`                      | Optional multilingual embeddings for Arabic support.    |

---

## 🧩 Key Features

* Full pipeline: Scraping → Cleaning → Embedding → FAISS → RAG.
* Semantic search powered by FAISS.
* FastAPI backend with modern UI.
* Lightweight, local-first design (no external APIs required).
* Modular codebase with clear comments and structure.

---

## 🧠 Future Enhancements

* Add multilingual embedding support for Arabic queries.
* Integrate result scoring for improved ranking.
* Deploy on cloud (AWS, Render, or Hugging Face Spaces).
* Add user feedback loops for improving model accuracy.

---

## ✍️ Author

**Bushra Dajam**
AI Engineer | Data & NLP Enthusiast
University of Jeddah – Artificial Intelligence Major
📧 Email: [[your-email@example.com](mailto:your-email@example.com)]
🌐 GitHub: [github.com/yourusername]
💼 LinkedIn: [linkedin.com/in/bushradajam]

---

## 🪪 License

This project is licensed under the **MIT License** — feel free to use, modify, and distribute with attribution.

```
