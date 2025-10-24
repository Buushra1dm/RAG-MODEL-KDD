from pathlib import Path
import json
import numpy as np
from typing import Tuple

# Input files generated from make_embeddings.py
EMB_DIR   = Path("data/embed")
EMB_PATH  = EMB_DIR / "embeddings.npy"
META_PATH = EMB_DIR / "meta.json"
CONF_PATH = EMB_DIR / "conf.json"

# Output files for the FAISS index
IDX_DIR   = Path("data/index")
IDX_DIR.mkdir(parents=True, exist_ok=True)
FAISS_IDX = IDX_DIR / "faiss.index"
FAISS_META= IDX_DIR / "meta.json"
FAISS_CONF= IDX_DIR / "conf.json"


# ----------------------------
# Load embeddings and metadata
# ----------------------------
def _load_embed_pack():
    """Load embeddings, metadata, and optional configuration."""
    if not EMB_PATH.exists() or not META_PATH.exists():
        raise FileNotFoundError("Run make_embeddings.py first to create embeddings.")
    
    emb  = np.load(EMB_PATH)
    meta = json.loads(META_PATH.read_text(encoding="utf-8"))
    conf = json.loads(CONF_PATH.read_text(encoding="utf-8")) if CONF_PATH.exists() else {}

    # Basic validation
    if emb.shape[0] != len(meta):
        raise ValueError(f"Embedding/meta size mismatch: emb={emb.shape[0]} vs meta={len(meta)}")
    
    return emb, meta, conf


# ----------------------------
# Main: Build and save FAISS index
# ----------------------------
def save_faiss_index():
    """Build and save a FAISS index (cosine similarity with normalized vectors)."""
    import faiss

    emb, meta, conf = _load_embed_pack()
    dim = emb.shape[1]
    print(f"Building FAISS index (dim={dim}, size={emb.shape[0]}) ...")

    # FlatIP (inner product) with L2-normalized vectors ≈ cosine similarity
    index = faiss.IndexFlatIP(dim)
    index.add(emb)
    faiss.write_index(index, str(FAISS_IDX))

    # Save meta and config
    FAISS_META.write_text(json.dumps(meta, ensure_ascii=False), encoding="utf-8")

    search_conf = {
        "search_space": "cosine",
        "from_conf": conf,   # Includes model_name, normalization flag, device, etc.
        "built_with": {"index": "IndexFlatIP"}
    }
    FAISS_CONF.write_text(json.dumps(search_conf, ensure_ascii=False), encoding="utf-8")

    print(f"FAISS index saved to: {FAISS_IDX}")
    print(f"Meta saved to:       {FAISS_META}")
    print(f"Config saved to:     {FAISS_CONF}")


# ----------------------------
# Helpers for searching
# ----------------------------
def _pick_device() -> str:
    """Choose CUDA if available, otherwise CPU."""
    try:
        import torch
        if torch.cuda.is_available():
            return "cuda"
    except Exception:
        pass
    return "cpu"


def _prep_query_text(query: str, model_name: str) -> str:
    """
    Prepare a query text according to model conventions.
    E5 models use 'query: ...'
    BGE models use their own prefix for retrieval tasks.
    """
    mn = (model_name or "").lower()
    if "e5" in mn:
        return f"query: {query}"
    if "bge" in mn:
        return f"Represent this sentence for searching relevant passages: {query}"
    return query


# ----------------------------
# CLI Search using FAISS
# ----------------------------
def cli_search(query: str, k: int = 5):
    """
    Search the FAISS index for the given query and print results in JSON format.
    """
    import faiss
    from sentence_transformers import SentenceTransformer

    # Ensure index and metadata exist
    if not (FAISS_IDX.exists() and FAISS_META.exists() and FAISS_CONF.exists()):
        raise FileNotFoundError("Index not found. Run: python build_faiss.py build")

    # Load metadata and configuration
    meta = json.loads(FAISS_META.read_text(encoding="utf-8"))
    conf = json.loads(FAISS_CONF.read_text(encoding="utf-8")) or {}
    from_conf = conf.get("from_conf") or {}
    model_name = from_conf.get("model_name") or "BAAI/bge-small-en-v1.5"
    was_normalized = bool(from_conf.get("normalized", True))

    # Load FAISS index
    index = faiss.read_index(str(FAISS_IDX))

    # Load model
    device = _pick_device()
    model = SentenceTransformer(model_name, device=device)

    # Encode query
    qtext = _prep_query_text(query, model_name)
    q = model.encode([qtext], convert_to_numpy=True).astype("float32")

    # Normalize query if index embeddings were normalized
    if was_normalized:
        q /= (np.linalg.norm(q, axis=1, keepdims=True) + 1e-12)

    k = max(1, min(k, len(meta)))  # Do not exceed index size
    D, I = index.search(q, k)
    scores = D[0]
    idxs = I[0]

    # Collect top results
    results = []
    for score, i in zip(scores, idxs):
        if 0 <= i < len(meta):
            m = meta[i]
            results.append({
                "score": float(score),
                "id": m.get("id"),
                "type": m.get("type"),
                "title": m.get("title"),
                "url": m.get("url"),
                "snippet": (m.get("text") or "")[:300]
            })

    # Print formatted JSON results
    print(json.dumps({
        "model": model_name,
        "normalized": was_normalized,
        "device": device,
        "query": query,
        "results": results
    }, ensure_ascii=False, indent=2))

# run
if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser("Build & test FAISS index")
    sub = parser.add_subparsers(dest="cmd", required=True)

    #  build index
    sub.add_parser("build", help="Build FAISS index from embeddings")

    #  search index
    search_p = sub.add_parser("search", help="Search using FAISS index")
    search_p.add_argument("query", help="Search query text")
    search_p.add_argument("--k", type=int, default=5, help="Number of results to return")

    args = parser.parse_args()

    if args.cmd == "build":
        save_faiss_index()
    elif args.cmd == "search":
        cli_search(args.query, args.k)
