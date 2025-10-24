from pathlib import Path
import os, json, hashlib
from typing import List, Dict, Any, Tuple
import numpy as np
import re

CORPUS = Path("data/corpus.json")

# Output paths for embeddings
EMB_DIR   = Path("data/embed")
EMB_PATH  = EMB_DIR / "embeddings.npy"
META_PATH = EMB_DIR / "meta.json"
CONF_PATH = EMB_DIR / "conf.json"

# Default model:
# - English only: "BAAI/bge-small-en-v1.5"
MODEL_NAME = os.getenv("EMB_MODEL", "BAAI/bge-small-en-v1.5")

WS = re.compile(r"\s+")

# -------------------------------------------------
# Text normalization
# -------------------------------------------------
def normalize_text(x: str) -> str:
    """Trim whitespace and collapse multiple spaces."""
    return WS.sub(" ", (x or "").strip())


# -------------------------------------------------
# Load corpus
# -------------------------------------------------
def load_corpus() -> List[Dict[str, Any]]:
    """Load the corpus.json file."""
    if not CORPUS.exists():
        raise FileNotFoundError(f"Missing corpus: {CORPUS}")
    return json.loads(CORPUS.read_text(encoding="utf-8"))


# -------------------------------------------------
# Extract useful text fields from metadata
# -------------------------------------------------
def _from_meta(meta: Dict[str, Any], keys: List[str]) -> List[str]:
    """Return a list of string fields from meta by keys."""
    out = []
    for k in keys:
        v = meta.get(k)
        if isinstance(v, list):
            out.append(" ".join(map(str, v)))
        elif v is not None:
            out.append(str(v))
    return out


# -------------------------------------------------
# Build a metadata list for embedding
# -------------------------------------------------
def build_meta(corpus: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Build a list of metadata entries from corpus.json.

    Each entry in corpus.json has this general structure:
      {
        "type": "product" | "career",
        "title": "...",
        "url": "...",
        "category"/"department"/"location": "...",
        "text": "...",          # Base text to embed
        "meta": {...}           # Original raw fields
      }
    """
    meta = []
    for i, d in enumerate(corpus):
        title = d.get("title") or ""
        url   = d.get("url") or ""
        base_text = d.get("text") or ""
        doc_type = d.get("type") or ""
        m = d.get("meta") or {}

        extra_parts = []

        # -------- Products --------
        if doc_type == "product":
            extra_parts += _from_meta(m, [
                "desc", "flavor", "category", "brand",
                "sku", "product_code", "options_text"
            ])
            # Add size or quantity details if available
            if m.get("volume_ml"):
                extra_parts.append(f"{m['volume_ml']} ml")
            if m.get("weight_g"):
                extra_parts.append(f"{m['weight_g']} g")
            if m.get("pack_units"):
                extra_parts.append(f"{m['pack_units']} pcs")

        # -------- Careers --------
        if doc_type == "career":
            extra_parts += _from_meta(m, [
                "embedding_text",
                "department_name", "location",
                "skills_required", "duties", "requirements"
            ])
            # Add experience information
            miny, maxy = m.get("min_years"), m.get("max_years")
            if miny is not None and maxy is not None and maxy != miny:
                extra_parts.append(f"experience {miny}-{maxy} years")
            elif miny is not None and maxy is None:
                extra_parts.append(f"experience {miny}+ years")
            elif miny is not None:
                extra_parts.append(f"experience {miny} years")

        # Add high-level fields (common across types)
        for k in ["category", "department", "location", "flavor"]:
            if d.get(k):
                extra_parts.append(str(d[k]))

        text = normalize_text(" ".join([base_text] + extra_parts))

        # Generate a simple stable ID
        hid = hashlib.sha1(f"{title}|{url}|{i}".encode()).hexdigest()[:16]

        meta.append({
            "id": hid,
            "type": doc_type,
            "title": title,
            "url": url,
            "text": text,
            "extra": {
                "top": {k: v for k, v in d.items() if k not in ("text",)},
                "meta": m
            }
        })
    return meta


# -------------------------------------------------
# Pick device and batch size
# -------------------------------------------------
def pick_device_and_batch() -> Tuple[str, int]:
    """Return ('cuda' or 'cpu', batch_size) depending on device availability."""
    try:
        import torch
        if torch.cuda.is_available():
            return "cuda", 128  
    except Exception:
        pass
    return "cpu", 64  


# -------------------------------------------------
# embedding generation
# -------------------------------------------------
def main():
    """Generate embeddings and save them to disk."""
    from sentence_transformers import SentenceTransformer

    EMB_DIR.mkdir(parents=True, exist_ok=True)
    corpus = load_corpus()
    meta   = build_meta(corpus)

    texts = [m["text"] for m in meta]
    if not any(t.strip() for t in texts):
        raise RuntimeError("Corpus texts are empty after normalization.")

    device, batch_size = pick_device_and_batch()
    print(f"Encoding {len(texts)} documents with {MODEL_NAME} on {device} (batch={batch_size}) ...")

    model = SentenceTransformer(MODEL_NAME, device=device)

    # The SentenceTransformer automatically handles truncation by tokenizer.max_len
    emb = model.encode(
        texts,
        batch_size=batch_size,
        show_progress_bar=True,
        convert_to_numpy=True,
        normalize_embeddings=False 
    ).astype("float32")

    # L2-normalize so that inner product = cosine similarity
    norms = np.linalg.norm(emb, axis=1, keepdims=True) + 1e-12
    emb = emb / norms

    # Save outputs
    np.save(EMB_PATH, emb)
    META_PATH.write_text(json.dumps(meta, ensure_ascii=False), encoding="utf-8")
    CONF_PATH.write_text(json.dumps({
        "model_name": MODEL_NAME,
        "normalized": True,
        "device": device,
        "batch_size": batch_size
    }, ensure_ascii=False), encoding="utf-8")

    print(f"Embeddings saved to: {EMB_PATH}")
    print(f"Metadata saved to:   {META_PATH}")
    print(f"Config saved to:     {CONF_PATH}")
    print("Shape:", emb.shape)


# -------------------------------------------------
# run
# -------------------------------------------------
if __name__ == "__main__":
    main()
