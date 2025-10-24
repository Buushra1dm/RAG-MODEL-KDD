from __future__ import annotations
from pathlib import Path
from typing import List, Optional, Any, Dict, Literal, Tuple
from functools import lru_cache
import json
import math
import os
import re
import numpy as np
import torch
from fastapi import FastAPI, Query, HTTPException, Depends, Body
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, RedirectResponse
from pydantic import BaseModel, Field
from pydantic_settings import BaseSettings
from fastapi import Request
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from transformers import StoppingCriteria, StoppingCriteriaList

# ---------------------------------------------------------------------
# Settings / configuration
# ---------------------------------------------------------------------
try:
    from pydantic_settings import SettingsConfigDict
    _HAS_PYD_SETTINGS_V2 = True
except Exception:
    _HAS_PYD_SETTINGS_V2 = False

class Settings(BaseSettings):
    """
    Centralized runtime configuration.
    """
    data_dir: str = "data/clean"
    products_file: str = "products_clean.json"
    careers_file: str = "careers1_clean.json"

    faiss_index_dir: str = "data/index"  
    enable_semantic: bool = True

    # Pydantic v1/v2 compatibility for reading .env with prefix KDD_
    if _HAS_PYD_SETTINGS_V2:
        model_config = SettingsConfigDict(env_file=".env", env_prefix="KDD_")
    else:
        class Config:
            env_file = ".env"
            env_prefix = "KDD_"

@lru_cache()
def get_settings() -> Settings:
    """Load settings once per process (cached)."""
    return Settings()

# ---------------------------------------------------------------------
# LLM : generation toggles and defaults
# ---------------------------------------------------------------------
USE_HF_LLM = True  

HF_MODEL = os.getenv("HF_MODEL", "Qwen/Qwen2.5-1.5B-Instruct")
HF_4BIT    = os.getenv("HF_4BIT", "false").lower() in {"1","true","yes"}
WARMUP_HF = os.getenv("WARMUP_HF", "false").lower() in {"1","true","yes"}

# Text-generation default hyperparameters (balanced for speed/quality)
GEN_MAX_TOKENS_DEFAULT = int(os.getenv("GEN_MAX_TOKENS", "140"))  
GEN_TEMPERATURE        = float(os.getenv("GEN_TEMPERATURE", "0.1"))
GEN_TOP_P              = float(os.getenv("GEN_TOP_P", "0.9"))
GEN_REP_PENALTY        = float(os.getenv("GEN_REP_PENALTY", "1.05"))
GEN_NO_REPEAT_NGRAM    = int(os.getenv("GEN_NO_REPEAT_NGRAM", "3"))

# A single global pipeline instance to avoid reloading model/tokenizer
_hf_pipe = None

class StopOnSubstrings(StoppingCriteria):
    """
    Custom stopping criteria that halts generation when any given substring
    appears in the decoded output (e.g., 'END_OF_ANSWER').
    """
    def __init__(self, tokenizer, stop_strings):
        self.tok = tokenizer
        self.stop_strings = stop_strings

    def __call__(self, input_ids, scores, **kwargs):
        # Only check the last sequence in batch (index 0 for single input).
        text = self.tok.decode(input_ids[0], skip_special_tokens=True)
        return any(s in text for s in self.stop_strings)

def get_hf_pipeline():
    """
    Initialize a Hugging Face text-generation pipeline pinned fully on GPU if available.
    - Avoids device_map='auto' to prevent unwanted CPU offload.
    - Chooses dtype: bf16 > fp16 > fp32 depending on CUDA support.
    - Ensures pad/eos token IDs are set to avoid generation errors.
    """
    global _hf_pipe
    if _hf_pipe is not None:
        return _hf_pipe

    import os, torch
    from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline

    has_cuda = torch.cuda.is_available()
    bf16_ok = has_cuda and torch.cuda.is_bf16_supported()
    dtype = torch.bfloat16 if bf16_ok else (torch.float16 if has_cuda else torch.float32)

    try:
        torch.set_float32_matmul_precision("high")
    except Exception:
        pass

    hf_token = os.getenv("HF_TOKEN") or None

    tok = AutoTokenizer.from_pretrained(HF_MODEL, use_fast=True, token=hf_token)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token or tok.pad_token

    # Load model without device_map to control placement explicitly
    mdl = AutoModelForCausalLM.from_pretrained(
        HF_MODEL,
        token=hf_token,
        dtype=dtype,                
        low_cpu_mem_usage=True,
    )

    if has_cuda:
        mdl.to("cuda")       # Pin model on cuda:0 

    mdl.eval()

    # Make sure generation config has valid pad/eos IDs
    try:
        if getattr(mdl.generation_config, "pad_token_id", None) is None and tok.pad_token_id is not None:
            mdl.generation_config.pad_token_id = tok.pad_token_id
        if getattr(mdl.generation_config, "eos_token_id", None) is None and tok.eos_token_id is not None:
            mdl.generation_config.eos_token_id = tok.eos_token_id
    except Exception:
        pass

    dev = next(mdl.parameters()).device
    print(f">>> HF model loaded on device: {dev} | dtype={dtype}")

    _hf_pipe = pipeline("text-generation", model=mdl, tokenizer=tok, device=0 if has_cuda else -1)
    return _hf_pipe


# ---------------------------------------------------------------------
# API response models
# ---------------------------------------------------------------------
class ProductOut(BaseModel):
    """Normalized product fields exposed by the API."""
    name: Optional[str] = None
    brand: Optional[str] = None
    category: Optional[str] = None
    flavor: Optional[str] = None
    volume_ml: Optional[int] = None
    weight_g: Optional[int] = None  
    pack_units: Optional[int] = None
    product_url: Optional[str] = None
    sku: Optional[str] = None
    availability: Optional[str] = None
    price_now_kwd: Optional[float] = None
    options_text: Optional[str] = None
    price_notes: Optional[str] = None
    reviews_count: int = 0
    rating_stars: Optional[int] = None


class CareerOut(BaseModel):
    """Normalized career fields exposed by the API."""
    title: Optional[str] = None
    url: Optional[str] = None
    location: Optional[str] = None
    department_code: Optional[str] = None
    department_name: Optional[str] = None
    job_type: Optional[str] = None
    min_years: Optional[int] = None
    max_years: Optional[int] = None
    skills_required: Optional[List[str]] = None

class PageMeta(BaseModel):
    """Pagination metadata used by list endpoints."""
    page: int
    per_page: int
    total: int
    total_pages: int

class PagedProducts(BaseModel):
    """Paginated wrapper for products."""
    data: List[ProductOut]
    meta: PageMeta

class PagedCareers(BaseModel):
    """Paginated wrapper for careers."""
    data: List[CareerOut]
    meta: PageMeta

class SearchHit(BaseModel):
    """
    Generic search hit returned by FAISS retrieval.
    type: 'product' or 'career'
    meta.text usually carries the indexed passage content.
    """
    score: float
    type: Literal["product","career"]
    title: Optional[str] = None
    url: Optional[str] = None
    snippet: Optional[str] = None
    meta: Dict[str, Any] = Field(default_factory=dict)

class SearchResponse(BaseModel):
    """Envelope for the /v1/search results."""
    query: str
    count: int
    results: List[SearchHit]

# ---------------------------------------------------------------------
# URL normalization helpers
# ---------------------------------------------------------------------
URL_ABS_BASE = os.getenv("KDD_BASE_URL", "https://www.kdd.com.kw")
URL_RE = re.compile(r"^(?:https?:)?//", re.I)

def absolutize_url(u: Optional[str]) -> Optional[str]:
    """
    Convert relative URLs to absolute using URL_ABS_BASE.
    Accepts http(s) and protocol-relative URLs as-is.
    """
    if not u:
        return None
    u = u.strip()
    if URL_RE.match(u):                # http://, https://, أو //example.com
        return u if u.lower().startswith("http") else f"https:{u}"
    if u.startswith("/"):             
        return URL_ABS_BASE.rstrip("/") + u
    return u                

# ---------------------------------------------------------------------
# JSON data loading helpers
# ---------------------------------------------------------------------
def _load_json(path: Path) -> Any:
    """Read and parse a UTF-8 JSON file or raise if missing."""    
    if not path.exists():
        raise FileNotFoundError(f"Missing file: {path}")
    return json.loads(path.read_text(encoding="utf-8"))

def load_products(settings: Settings) -> List[Dict[str, Any]]:
    """Load products dataset and enforce presence of common keys."""
    p = Path(settings.data_dir) / settings.products_file
    items: List[Dict[str, Any]] = _load_json(p)
    for it in items:
        it.setdefault("brand", None)
        it.setdefault("category", None)
    return items

def load_careers(settings: Settings) -> List[Dict[str, Any]]:
    """Load careers dataset."""
    p = Path(settings.data_dir) / settings.careers_file
    items: List[Dict[str, Any]] = _load_json(p)
    return items

# ---------------------------------------------------------------------
# List/Filter/Sort/Paginate utilities
# ---------------------------------------------------------------------
def paginate(items: List[Dict[str, Any]], page: int, per_page: int) -> Tuple[List[Dict[str, Any]], PageMeta]:
    """Slice items for the requested page and compute pagination metadata."""
    total = len(items)
    total_pages = max(1, math.ceil(total / per_page)) if per_page else 1
    if page < 1: page = 1
    if per_page < 1: per_page = 10
    start = (page - 1) * per_page
    end = start + per_page
    return items[start:end], PageMeta(page=page, per_page=per_page, total=total, total_pages=total_pages)

def like(text: Optional[str], needle: str) -> bool:
    """Case-insensitive substring match helper."""
    return bool(text) and (needle.lower() in text.lower())

def sort_items(items: List[Dict[str, Any]], sort: Optional[str]) -> List[Dict[str, Any]]:
    """
    Sort by a given field name; prefix '-' to sort descending.
    Safely falls back to unsorted if key missing or incomparable.
    """
    if not sort:
        return items
    key = sort.lstrip("-")
    reverse = sort.startswith("-")
    try:
        return sorted(items, key=lambda x: (x.get(key) is None, x.get(key)), reverse=reverse)
    except Exception:
        return items

# ---------------------------------------------------------------------
# Careers text helpers for strict filtering and scoring
# ---------------------------------------------------------------------
def _safe_join(xs):
    """Join list of strings safely; otherwise return empty string."""
    return " ".join(xs) if isinstance(xs, list) else (xs or "")

def career_text_blob(it: Dict[str, Any]) -> str:
    """Concatenate key textual fields of a career item for keyword scoring."""
    parts = [
        it.get("title") or "",
        it.get("department_name") or "",
        it.get("job_type") or "",
        _safe_join(it.get("skills_required")),
        _safe_join(it.get("requirements")),
        _safe_join(it.get("duties")),
        it.get("location") or "",
        it.get("embedding_text") or "",
    ]
    return " ".join(map(lambda s: s.lower(), parts))

def contains_any(text: str, needles: List[str]) -> bool:
    """True if any of the needles appear in the text (case-insensitive)."""
    tl = text.lower()
    return any(n.lower() in tl for n in needles)

def score_keywords(text: str, needles: List[str]) -> int:
    """Count total keyword occurrences for relevance scoring."""
    tl = text.lower()
    return sum(tl.count(n.lower()) for n in needles)


# ---------------------------------------------------------------------
# Product hydration from hits (regex extraction + JSON lookup)
# ---------------------------------------------------------------------
def _norm_name(s: Optional[str]) -> str:
    """Normalize names for dictionary keys (lowercase, single spaces)."""
    return re.sub(r"\s+", " ", (s or "").strip().lower())

def build_product_lookup(settings: Settings) -> Dict[str, Dict[str, Any]]:
    """
    Build indices for product lookup by URL, name, and SKU.
    Used to enrich search hits with canonical product data.
    """
    items = load_products(settings)
    by_url: Dict[str, Dict[str, Any]] = {}
    by_name: Dict[str, Dict[str, Any]] = {}
    by_sku: Dict[str, Dict[str, Any]] = {}

    for it in items:
        url = (it.get("product_url") or "").strip()
        name = _norm_name(it.get("name"))
        sku  = (it.get("sku") or it.get("product_code") or "").strip().upper()

        if url:
            by_url[url] = it
        if name:
            by_name[name] = it
        if sku:
            by_sku[sku] = it

    return {"by_url": by_url, "by_name": by_name, "by_sku": by_sku}

# Regex patterns to extract fields from free text (fallback)
_PRICE_RE   = re.compile(r"(\d+(?:\.\d+)?)\s*(?:kwd|kd)\b", re.I)
_ML_RE      = re.compile(r"(\d+(?:\.\d+)?)\s*ml\b", re.I)
_G_RE       = re.compile(r"(\d+(?:\.\d+)?)\s*g\b", re.I)
_AVAIL_RE   = re.compile(r"\b(in[_\s-]?stock|out[_\s-]?of[_\s-]?stock|pre\s*order|preorder|available|unavailable)\b", re.I)
_FLAVOR_RE  = re.compile(r"\b(mango|vanilla|strawberry|chocolate|banana|orange|grape|lemon|pineapple)\b", re.I)

def hydrate_product_hit(h: SearchHit, lookup: Dict[str, Dict[str, Any]]) -> Dict[str, Any]:
    """
    Merge a semantic hit with canonical product data and extract missing fields
    using regex from the hit's text/snippet.
    """
    meta = h.meta or {}
    txt  = f"{meta.get('text','') or ''} {(h.snippet or '') or ''}"

    out = {
        "name": h.title,
        "brand": None, "category": None, "flavor": None,
        "volume_ml": None, "weight_g": None, "pack_units": None,
        "sku": None, "availability": None, "price_now_kwd": None,
        "rating_stars": None, "reviews_count": None, "product_url": h.url, "_ref": None,
        "options_text": None, "price_notes": None,
    }

    # 1) Copy any structured meta first (if present)
    for k in ["name","brand","category","flavor","volume_ml","weight_g","pack_units",
              "sku","availability","price_now_kwd","rating_stars","reviews_count",
              "product_url","options_text","price_notes"]:
        if k in meta and meta[k] not in (None, "", []):
            out[k] = meta[k]

    # If meta has a more specific product_url, prefer it
    if meta.get("product_url"):
        out["product_url"] = absolutize_url(meta["product_url"])

    # 2) Resolve against canonical JSON by SKU, then URL, then normalized name
    src = None
    sku_key = (out.get("sku") or meta.get("product_code") or "").strip().upper()
    if not src and sku_key and sku_key in lookup.get("by_sku", {}):
        src = lookup["by_sku"][sku_key]
    if not src and out.get("product_url") and out["product_url"] in lookup.get("by_url", {}):
        src = lookup["by_url"][out["product_url"]]
    if not src:
        nm = _norm_name(out.get("name") or h.title)
        src = lookup.get("by_name", {}).get(nm)

    if src:
        for k in ["name","brand","category","flavor","volume_ml","weight_g","pack_units",
                  "sku","availability","price_now_kwd","rating_stars","reviews_count",
                  "product_url","options_text","price_notes"]:
            if k in src and src[k] not in (None, "", []):
                out[k] = src[k]

    # 3) Fill any gaps from text using regex (best-effort)
    if out["price_now_kwd"] is None:
        m = _PRICE_RE.search(txt)
        if m:
            try: out["price_now_kwd"] = float(m.group(1))
            except: pass

    if out["volume_ml"] is None:
        m = _ML_RE.search(txt)
        if m:
            try: out["volume_ml"] = int(float(m.group(1)))
            except: pass

    if out["weight_g"] is None:
        m = _G_RE.search(txt)
        if m:
            try: out["weight_g"] = int(float(m.group(1)))
            except: pass

    if out["availability"] is None:
        m = _AVAIL_RE.search(txt)
        if m:
            val = m.group(1).lower().replace(" ", "_").replace("-", "_")
            if val == "available": val = "in_stock"
            if val in ("pre_order","preorder"): val = "preorder"
            if val == "unavailable": val = "out_of_stock"
            out["availability"] = val

    if out["flavor"] is None:
        m = _FLAVOR_RE.search(txt)
        if m:
            out["flavor"] = m.group(1).capitalize()

    return out


# ---------------------------------------------------------------------
# FAISS semantic index wrapper
# ---------------------------------------------------------------------
class SemanticIndex:
    """
    Thin wrapper around a FAISS index + metadata. Also encapsulates
    an embedding model (SentenceTransformers) for queries and passages.
    """
    def __init__(self, idx_dir: Path):
        self.idx_dir = idx_dir
        self.index_path = idx_dir / "faiss.index"
        self.meta_path  = idx_dir / "meta.json"
        self.conf_path  = idx_dir / "conf.json"
        self._index = None
        self._meta: List[Dict[str, Any]] = []
        self._embedder_name = None
        self._embedder = None
        self._enc_cache: Dict[str, Any] = {}

    def available(self) -> bool:
        """True if both index and meta exist on disk."""
        return self.index_path.exists() and self.meta_path.exists()

    def _load(self):
        """Lazy-load FAISS index, metadata, and config (embedder name)."""
        import faiss  # type: ignore
        if self._index is None:
            self._index = faiss.read_index(str(self.index_path))
        if not self._meta:
            self._meta = json.loads(self.meta_path.read_text(encoding="utf-8"))
        if self._embedder_name is None:
            if self.conf_path.exists():
                try:
                    conf = json.loads(self.conf_path.read_text(encoding="utf-8"))
                    base = conf.get("from_conf") or {}
                    self._embedder_name = base.get("model_name") or "intfloat/multilingual-e5-base"
                except Exception:
                    self._embedder_name = "intfloat/multilingual-e5-base"
            else:
                self._embedder_name = "intfloat/multilingual-e5-base"

    def _encode(self, text: str, is_query: bool = False):
        """
        Encode text using SentenceTransformer. Applies model-specific
        query prefixes (e5/bge) and returns L2-normalized vectors.
        """
        import numpy as np  
        from sentence_transformers import SentenceTransformer  

        # cache only for passages (not for queries)
        if text in self._enc_cache and not is_query:
            return self._enc_cache[text]

        if self._embedder is None:
            use_cuda = torch.cuda.is_available()
            bf16_ok = use_cuda and torch.cuda.is_bf16_supported()
            dtype = torch.bfloat16 if bf16_ok else (torch.float16 if use_cuda else torch.float32)
            device = "cuda" if use_cuda else "cpu"

            self._embedder = SentenceTransformer(
                self._embedder_name,
                device=device,
                model_kwargs={
                    "torch_dtype": dtype,
                    "low_cpu_mem_usage": False,    
                    "attn_implementation": "eager" #
                }
            )

        t = text
        mn = (self._embedder_name or "").lower()
        if is_query and "e5" in mn:
            t = f"query: {text}"
        elif is_query and "bge" in mn:
            t = f"Represent this sentence for searching relevant passages: {text}"

        v = self._embedder.encode([t]).astype("float32")
        v /= (np.linalg.norm(v, axis=1, keepdims=True) + 1e-12)

        if not is_query:
            self._enc_cache[text] = v
        return v

    def search(self, query: str, k: int = 5) -> List[SearchHit]:
        """Search the index for the top-k hits for the given query string."""
        import faiss  
        self._load()
        q = self._encode(query, is_query=True)
        D, I = self._index.search(q, k)
        scores = D[0]; idxs = I[0]
        out: List[SearchHit] = []
        for s, i in zip(scores, idxs):
            if 0 <= i < len(self._meta):
                m = self._meta[i]
                full_meta = {"text": m.get("text") or ""}
                if isinstance(m.get("extra"), dict):
                    full_meta.update(m.get("extra") or {})

                out.append(SearchHit(
                    score=float(s),
                    type=m.get("type"),
                    title=m.get("title"),
                    url=m.get("url"),
                    snippet=(m.get("text") or "")[:800],
                    meta=full_meta,
                ))
        return out

_semantic: Optional[SemanticIndex] = None

def get_semantic(settings: Settings) -> Optional[SemanticIndex]:
    """Singleton accessor for the FAISS semantic index."""
    global _semantic
    if _semantic is None:
        idx_dir = Path(settings.faiss_index_dir)
        _semantic = SemanticIndex(idx_dir)
    return _semantic

# ---------------------------------------------------------------------
# FastAPI application and middleware
# ---------------------------------------------------------------------
app = FastAPI(
    title="KDD Products & Careers Helper",
    version="1.3",
    openapi_tags=[
        {"name": "products", "description": "Query products"},
        {"name": "careers", "description": "Query careers"},
        {"name": "search", "description": "Semantic search )"},
        {"name": "ask", "description": "RAG Q&A via HF "},
        {"name": "ui", "description": "Minimal web UI"},
    ],
)

app.mount("/static", StaticFiles(directory="static"), name="static")
templates = Jinja2Templates(directory="templates")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], allow_credentials=True, allow_methods=["*"], allow_headers=["*"],
)

@app.on_event("startup")
def _warmup():
    try:
        if USE_HF_LLM and WARMUP_HF:
            get_hf_pipeline()
    except Exception:
        pass


# Root: redirect to UI for convenience.
@app.get("/", include_in_schema=False)
def root_redirect():
    return RedirectResponse(url="/ui", status_code=307)

# ---------------------------------------------------------------------
# /v1/products — Filterable product listing
# ---------------------------------------------------------------------
@app.get("/v1/products", response_model=PagedProducts, tags=["products"])
def list_products(
    q: Optional[str] = Query(None, description="Full-text filter on name/brand/flavor"),
    category: Optional[str] = Query(None),
    brand: Optional[str] = Query(None),
    availability: Optional[Literal["in_stock","out_of_stock","preorder","unknown"]] = Query(None),
    price_min: Optional[float] = Query(None, ge=0),
    price_max: Optional[float] = Query(None, ge=0),
    sort: Optional[str] = Query(None, description="sort by field, prefix '-' for desc (e.g. -price_now_kwd)"),
    page: int = Query(1, ge=1),
    per_page: int = Query(20, ge=1, le=200),
    settings: Settings = Depends(get_settings),
):
    """
    Return a paginated subset of products with simple text filters and sorting.
    """
    items = load_products(settings)

    if q:
        needle = q.lower()
        items = [it for it in items if any([
            like(it.get("name"), needle),
            like(it.get("brand"), needle),
            like(it.get("flavor"), needle),
            like(it.get("options_text"), needle),
        ])]
    if category:
        items = [it for it in items if (it.get("category") or "").lower() == category.lower()]
    if brand:
        items = [it for it in items if (it.get("brand") or "").lower() == brand.lower()]
    if availability:
        items = [it for it in items if it.get("availability") == availability]
    if price_min is not None:
        items = [it for it in items if (it.get("price_now_kwd") is not None and it["price_now_kwd"] >= price_min)]
    if price_max is not None:
        items = [it for it in items if (it.get("price_now_kwd") is not None and it["price_now_kwd"] <= price_max)]

    # Sort & paginate
    items = sort_items(items, sort)
    page_items, meta = paginate(items, page, per_page)

    # Map to stable response schema
    mapped = [{
        "name": it.get("name"),
        "brand": it.get("brand"),
        "category": it.get("category"),
        "flavor": it.get("flavor"),
        "volume_ml": it.get("volume_ml"),
        "weight_g": it.get("weight_g"),         
        "pack_units": it.get("pack_units"),
        "product_url": it.get("product_url"),
        "sku": it.get("sku"),
        "availability": it.get("availability"),
        "price_now_kwd": it.get("price_now_kwd"),
        "options_text": it.get("options_text"),
        "price_notes": it.get("price_notes"),
        "reviews_count": it.get("reviews_count") or 0,
        "rating_stars": it.get("rating_stars"),
    } for it in page_items]

    return {"data": mapped, "meta": meta.dict()}

# ---------------------------------------------------------------------
# /v1/careers — Filterable career listing
# ---------------------------------------------------------------------

@app.get("/v1/careers", response_model=PagedCareers, tags=["careers"])
def list_careers(
    q: Optional[str] = Query(None, description="Full-text filter on title/department/skills"),
    department: Optional[str] = Query(None),
    location: Optional[str] = Query(None),
    min_years: Optional[int] = Query(None, ge=0),
    max_years: Optional[int] = Query(None, ge=0),
    sort: Optional[str] = Query(None, description="sort by field, prefix '-' for desc (e.g. min_years)"),
    page: int = Query(1, ge=1),
    per_page: int = Query(20, ge=1, le=200),
    settings: Settings = Depends(get_settings),
):
    items = load_careers(settings)

    def item_matches(it: Dict[str, Any], text: str) -> bool:
        return any([
            like(it.get("title"), text),
            like(it.get("department_name"), text),
            like(" ".join(it.get("skills_required") or []), text),
            like(" ".join(it.get("duties") or []), text),
            like(" ".join(it.get("requirements") or []), text),
        ])

    if q:
        needle = q.lower()
        items = [it for it in items if item_matches(it, needle)]

    if department:
        items = [it for it in items if (it.get("department_name") or "").lower() == department.lower()]

    if location:
        items = [it for it in items if (it.get("location") or "").lower() == location.lower()]

    if min_years is not None:
        items = [it for it in items if it.get("min_years") is not None and it["min_years"] >= min_years]

    if max_years is not None:
        items = [it for it in items if it.get("max_years") is not None and it["max_years"] <= max_years]

    items = sort_items(items, sort)
    page_items, meta = paginate(items, page, per_page)

    mapped = [{
        "title": it.get("title"),
        "url": absolutize_url(it.get("url")),
        "location": it.get("location"),
        "department_code": it.get("department_code"),
        "department_name": it.get("department_name"),
        "job_type": it.get("job_type"),
        "min_years": it.get("min_years"),
        "max_years": it.get("max_years"),
        "skills_required": it.get("skills_required"),
    } for it in page_items]

    return {"data": mapped, "meta": meta.dict()}

# ---------------------------------------------------------------------
# /v1/search — Semantic search 
# ---------------------------------------------------------------------

@app.get("/v1/search", response_model=SearchResponse, tags=["search"])
def semantic_search(
    q: str = Query(..., min_length=1),
    k: int = Query(5, ge=1, le=50),
    settings: Settings = Depends(get_settings),
):
    #Perform semantic search using FAISS
    if not settings.enable_semantic:
        raise HTTPException(status_code=404, detail="Semantic search disabled.")

    sem = get_semantic(settings)
    if not sem or not sem.available():
        raise HTTPException(status_code=503, detail="Semantic index not available. Build FAISS first.")

    # base FAISS results
    vec_hits = sem.search(q, k=k)
    fused = vec_hits[:k]

    # keyword boost for job queries that mention Python 
    ql = q.lower()
    is_job_query = any(w in ql for w in ("job","jobs","career","position","hiring","role","vacancy","وظيف","وظائف"))
    if is_job_query and ("python" in ql or "بايثون" in ql):
        items = load_careers(settings)
        kw = ["python", "بايثون"]
        kw_candidates: List[Tuple[float, SearchHit]] = []
        max_kw = 0
        for it in items:
            blob = career_text_blob(it)
            cnt = score_keywords(blob, kw)
            if cnt > 0:
                max_kw = max(max_kw, cnt)
                kw_candidates.append((cnt, it))

        if kw_candidates and max_kw > 0:
            normalized: List[SearchHit] = []
            for cnt, it in kw_candidates:
                norm_score = cnt / max_kw  # in (0..1]
                normalized.append(SearchHit(
                    score=float(norm_score),
                    type="career",
                    title=it.get("title"),
                    url=it.get("url"),
                    snippet=(m := (it.get("skills_required") or [])) and (", ".join(m)) or (it.get("department_name") or "")[:300],
                    meta={"location": it.get("location"), "dept": it.get("department_name"), "text": career_text_blob(it)},
                ))

            # Weighted merge of vector hits and keyword hits            
            alpha = 0.75
            vec_map = { (h.title or ""): float(getattr(h, 'score', 0.0) or 0.0) for h in vec_hits }

            merged: Dict[str, SearchHit] = {}
            for h in vec_hits:
                merged[(h.title or "")] = h

            for kh in normalized:
                t = (kh.title or "")
                vec_score = vec_map.get(t, 0.0)
                new_score = alpha * vec_score + (1.0 - alpha) * float(getattr(kh, 'score', 0.0) or 0.0)
                if t in merged:
                    mh = merged[t]
                    mh.score = new_score
                    if isinstance(kh.meta, dict) and kh.meta.get('text'):
                        mh.meta = {**(mh.meta or {}), **kh.meta}
                    merged[t] = mh
                else:
                    kh.score = new_score
                    merged[t] = kh

            fused = sorted(merged.values(), key=lambda x: float(getattr(x,'score',0.0) or 0.0), reverse=True)[:k]

    return {"query": q, "count": len(fused), "results": fused}

# --------------------------
# RAG helpers: domain inference, prompt building, reranking, hydration
# --------------------------
def infer_domain(q: str) -> Optional[str]:
    """Heuristically decide if the query is about careers or products."""
    ql = q.lower()
    if any(k in ql for k in ("job","jobs","career","position","hiring","role","vacancy","وظيف","وظائف")):
        return "career"
    if any(k in ql for k in (
        "product","products","item","items",
        "juice","milk","ice cream","flavor","flavors",
        "price","sku","available","availability",
        "منتج","منتجات","عصير","حليب","آيس","نكهة","نكهات","سعر","متوفر","التوفر"
    )):
        return "product"
    return None

def extract_skill(q: str) -> Optional[str]:
    """Extract a skill token from the user query (e.g., 'require Python')."""
    m = re.search(r"(require|need|with|using)\s+([A-Za-z+#.\-\s]{2,})", q, re.I)
    if m:
        return m.group(2).split()[0].strip()
    m2 = re.search(r"\b(jobs?|roles?).*\b([A-Za-z+#.\-]{2,})\s*\?*$", q, re.I)
    return (m2.group(2).strip() if m2 else None)


WS = re.compile(r"\s+")

# --------------------------
#    Build a concise, English-only instruction prompt for the LLM.
#    - Outputs numbered short paragraphs (1., 2., 3., …).
#    - Uses 'facts_json' as verified truth; passages only as reference.
#    - Forbids links/ratings/markdown; skips missing fields.
# --------------------------

def build_prompt(
    user_query: str,
    passages: List["SearchHit"],
    domain: Literal["career", "product"],
    facts_json: str
) -> str:


    def _clean_ctx(s: str) -> str:
        # Strip any previous 'chatty' speaker tags that may mislead the model.
        return re.split(r"\b(?:Human|Assistant|System)\s*:\s*", s)[0].strip()

    #Compose readable context snippets
    ctx_lines = []
    for i, p in enumerate(passages, start=1):
        meta_text = (p.meta or {}).get("text") or (p.snippet or "")
        title = WS.sub(" ", (p.title or "").strip())[:160]
        snippet = WS.sub(" ", _clean_ctx(meta_text))[:700]
        ctx_lines.append(f"[{i}] Title: {title}\nSnippet: {snippet}")
    ctx = "\n\n".join(ctx_lines)

    if domain == "product":
        guidance = (
            "You are a precise product analyst.\n"
            "Using ONLY the verified information, write numbered short paragraphs (1., 2., 3., …).\n"
            "Each paragraph must describe ONE product with: name, brand, category, flavor/variant (if any), "
            "size/weight/volume, availability (in stock / out of stock / preorder), and price in KWD."
        )
    else:
        guidance = (
            "You are a concise career assistant.\n"
            "Using ONLY the verified information, write numbered short paragraphs (1., 2., 3., …).\n"
            "Each paragraph must describe ONE job with: job title, department, location, job type, "
            "experience range (min–max years), and 2–3 key skills."
        )

    rules = (
        "Rules:\n"
        "- English only.\n"
        "- Do NOT include URLs, ratings, review counts, IDs, or markdown symbols.\n"
        "- Do NOT repeat the question or add comments.\n"
        "- No bullet symbols or brackets. If a field is missing, skip it.\n"
        "- Separate each numbered paragraph with ONE blank line."
    )

    prompt = f"""
{guidance}
{rules}

User Question:
{user_query}

Facts (verified database data):
{facts_json}

Context Passages (for reference only):
{ctx}

Example:
1. Vanilla Ice Cream — in stock, 1 L, 1.00 KWD.

2. Chocolate Mix — preorder, 500 ml, 0.80 KWD.

Write your answer now in the exact style above and
end your answer with exactly: END_OF_ANSWER
""".strip()

    return prompt


# ---------- Rerank & formatting helpers ----------
# Token pattern to extract alphanumeric and common tech symbols from queries
TOKEN_RE = re.compile(r"[A-Za-z0-9#+\-_/]+")

def extract_terms(q: str) -> List[str]:
    """
    Extract lightweight keywords from a user query.
    - Lowercases everything
    - Drops common stopwords
    - Keeps tokens length > 1
    """
    terms = [t.lower() for t in TOKEN_RE.findall(q)]
    stop = {
        "what","which","give","more","about","jobs","job","require","requires",
        "requiring","position","positions","role","roles","with","using","and",
        "the","a","an","in","at","of","for","please","me","details","detail"
    }
    return [t for t in terms if t not in stop and len(t) > 1]

def kw_overlap_score(text: str, terms: List[str]) -> int:
    """
    Raw overlap score: counts how many times each term appears in the text.
    Simple bag-of-words count used for lightweight keyword boosting.
    """
    tl = text.lower()
    return sum(tl.count(t) for t in terms)

def normalized_kw_score(text: str, terms: List[str]) -> float:
    """
    Log-scaled keyword score normalized to [0..1].
    - Prevents exploding counts when a term repeats a lot.
    - ~1.0 around ~19 total occurrences due to log1p/raw scaling.
    """
    tl = (text or "").lower()
    if not terms:
        return 0.0
    raw = sum(tl.count(t.lower()) for t in terms if t)
    if raw <= 0:
        return 0.0
    return min(1.0, math.log1p(raw) / 3.0)  

def _safe_json_from_text(txt: str) -> Optional[List[Dict[str, Any]]]:
    """Extract a JSON array of objects from free text using regex."""
    m = re.search(r"\[\s*{.*}\s*\]", txt, flags=re.S)
    if not m:
        return None
    try:
        arr = json.loads(m.group(0))
        if isinstance(arr, list):
            return arr
    except Exception:
        return None
    return None


def _extract_refs(txt: str) -> List[int]:
    """
    Extract numeric reference IDs from a tail pattern like: 'Refs: [1, 3, 5]'.
    Used when the LLM explicitly cites passages by index.
    """
    m = re.search(r"Refs:\s*\[(.*?)\]\s*$", txt, flags=re.I)
    if not m:
        return []
    return [int(x) for x in re.findall(r"\d+", m.group(1))]


def rerank_hits_for_careers(q: str, hits: List[SearchHit], settings: Settings) -> List[SearchHit]:
    """
    Hybrid rerank for CAREER hits:
    - Combines FAISS vector score + keyword overlap score + location hint.
    - De-duplicates by title while preserving the best-scored instance.
    """
    terms = extract_terms(q)
    # Location hints commonly appearing in KDD career text
    loc_terms = [t for t in terms if t in {"farwaniya","ahmadi","kuwait","sabhan","hawally"}]
    # Build quick blob map for detailed keyword checks
    items = load_careers(settings)
    blob_by_title = {(it.get("title") or "").strip().lower(): career_text_blob(it) for it in items}

    scored = []
    for h in hits:
        base = float(getattr(h, "score", 0.0) or 0.0)
        title_key = (h.title or "").strip().lower()
        blob = blob_by_title.get(title_key, f"{(h.title or '')} {(h.snippet or '')}")
        kw = kw_overlap_score(blob, terms)
        # Small location bump if query mentions a location term found in blob
        loc_boost = 2 if (loc_terms and any(l in blob for l in loc_terms)) else 0
        combined = base + 0.02 * kw + 0.03 * loc_boost
        scored.append((combined, h))

    # Sort by combined score and de-duplicate by normalized title
    scored.sort(key=lambda x: x[0], reverse=True)
    seen = set(); out = []
    for _, h in scored:
        t = (h.title or "").strip().lower()
        if t and t not in seen:
            seen.add(t); out.append(h)
    return out


# ---------- Skill helpers (strict filter) ----------
def _norm(x: Optional[str]) -> str:
    """Safe lowercase/trim helper."""
    return (x or "").strip().lower()

def _has_skill_in_text(text: str, skill: str) -> bool:
    """Exact/substring match for a skill inside a free-text blob."""
    return _norm(skill) in _norm(text)

def _has_skill_in_career(it: Dict[str, Any], skill: str) -> bool:
    """Check if a career item explicitly mentions the skill."""
    skill = _norm(skill)
    if not skill:
        return False
    skills = it.get("skills_required") or []
    if isinstance(skills, list) and any(_norm(s) == skill or skill in _norm(s) for s in skills):
        return True
    blob = career_text_blob(it)
    return _has_skill_in_text(blob, skill)

def _career_to_hit(it: Dict[str, Any], score: float = 0.0) -> SearchHit:
    """Convert a career dict to a SearchHit."""
    return SearchHit(
        score=float(score),
        type="career",
        title=it.get("title"),
        url=it.get("url"),
        snippet=", ".join((it.get("skills_required") or [])[:8]) or (it.get("department_name") or ""),
        meta={"location": it.get("location"), "dept": it.get("department_name"), "text": career_text_blob(it)},
    )

def _filter_hits_by_skill(
    hits: List[SearchHit], settings: Settings, skill: Optional[str]
) -> List[SearchHit]:
    """Keep only hits whose underlying career explicitly mentions the skill."""
    if not skill:
        return hits
    try:
        careers = load_careers(settings)
        by_title = {(c.get("title") or "").strip().lower(): c for c in careers}
    except Exception:
        return hits
    out: List[SearchHit] = []
    for h in hits:
        it = by_title.get(_norm(h.title))
        if it and _has_skill_in_career(it, skill):
            out.append(h)
    return out


# ===== Dynamic field detection for products (no fixed synonyms) =====
_EMB_F = None   # Cached embedding function
_E_DIM = None   # Cached embedding dimensionality


def _get_embedder():
    """
    Lazy-load SentenceTransformer for field-name embeddings only.
    """
    global _EMB_F, _E_DIM
    if _EMB_F is None:
        from sentence_transformers import SentenceTransformer
        import torch, numpy as np

        use_cuda = torch.cuda.is_available()
        bf16_ok = use_cuda and torch.cuda.is_bf16_supported()
        dtype = torch.bfloat16 if bf16_ok else (torch.float16 if use_cuda else torch.float32)
        device = "cuda" if use_cuda else "cpu"

        _m = SentenceTransformer(
            "intfloat/multilingual-e5-base",
            device=device,
            model_kwargs={
                "torch_dtype": dtype,
                "low_cpu_mem_usage": False,
                "attn_implementation": "eager",
            },
        )

        def _emb(texts: List[str]) -> np.ndarray:
            v = _m.encode(
                texts,
                convert_to_numpy=True,
                show_progress_bar=False,
                normalize_embeddings=True,
            )
            return v.astype("float32")

        _EMB_F = _emb
        _E_DIM = int(_emb(["a"]).shape[1])
    return _EMB_F



def _norm_field_name(x: str) -> str:
    """
    Normalize field labels:
    - Lowercase, replace separators, compress whitespace
    - Apply light canonicalization for common tokens like price/ml/g/kwd
    """
    x = (x or "").replace("_", " ").lower()
    x = re.sub(r"[-/]", " ", x)
    x = re.sub(r"\s+", " ", x).strip()
    x = x.replace(" price now", " price").replace(" price_now", " price")
    x = x.replace(" ml", " ml").replace(" g", " g").replace(" kwd", " kwd")
    return x

def build_products_schema(settings: Settings) -> Dict[str, Any]:
    """
    Build a dynamic schema from product JSON keys and pre-compute their embeddings.
    Enables semantic matching between user phrasing and field names.
    """
    data = load_products(settings)
    if not data:
        return {"names": [], "norm": [], "E": np.zeros((0, 1), dtype="float32")}
    name_set = set()
    for it in data:
        for k in it.keys():
            name_set.add(k)
    names = sorted(name_set)
    norm = [_norm_field_name(n) for n in names]

    emb = _get_embedder()
    E = emb(norm)  # (n, d)

    return {"names": names, "norm": norm, "E": E}

_PRODUCTS_SCHEMA_CACHE: Optional[Dict[str, Any]] = None
def get_products_schema(settings: Settings) -> Dict[str, Any]:
    """
    Singleton-style cache for product schema (names + embeddings).
    Avoids recomputing across requests.
    """
    global _PRODUCTS_SCHEMA_CACHE
    if _PRODUCTS_SCHEMA_CACHE is None:
        _PRODUCTS_SCHEMA_CACHE = build_products_schema(settings)
    return _PRODUCTS_SCHEMA_CACHE

def detect_fields_from_query(q: str, schema: Dict[str, Any], topk: int = 2, thresh: float = 0.35) -> List[Tuple[str, float]]:
    """
    Select closest product-field names to the user query using cosine similarity.
    - Returns (field_name, similarity) for top-k above a threshold.
    - If nothing is confidently close, returns [] (no forced synonyms).
    """
    if not schema or not schema.get("names"):
        return []
    emb = _get_embedder()
    qn = re.sub(r"\s+", " ", q.strip().lower())
    qv = emb([qn])[0:1]           # (1, d)
    sims = (qv @ schema["E"].T)[0]  # (n,)
    idxs = np.argsort(-sims)[:topk]
    out: List[Tuple[str, float]] = []
    for i in idxs:
        s = float(sims[i])
        if s >= thresh:
            out.append((schema["names"][i], s))
    return out

def expand_query_with_fields(q: str, fields: List[Tuple[str, float]]) -> str:
    """
    Expand a product query with semantically-close field names (names only).
    This helps FAISS retrieval; the RAG step still produces the final answer.
    """
    if not fields:
        return q
    add = " ".join(_norm_field_name(f) for f, _ in fields)
    return f"{q} {add}".strip()

# Cross-Encoder reranker 
from typing import Iterable
_CE = None
def get_reranker_ce():
    """
    Lazy-load a CrossEncoder (bge-reranker-base) for pairwise query-doc scoring.
    Uses GPU if available; otherwise falls back to CPU.
    """
    global _CE
    if _CE is None:
        from sentence_transformers import CrossEncoder
        import torch
        device = "cuda" if torch.cuda.is_available() else "cpu"
        _CE = CrossEncoder("BAAI/bge-reranker-base", device=device)
    return _CE


def rerank_with_cross_encoder(query: str, hits: List[SearchHit]) -> List[SearchHit]:
    """
    Rerank a small set of hits with a cross-encoder.
    - Safer to call on already-pruned lists (<= ~50) for performance.
    """
    if not hits:
        return hits
    try:
        ce = get_reranker_ce()
        pairs = []
        for h in hits:
            txt = ""
            try:
                txt = f"{h.title or ''} {(h.meta or {}).get('text','') or h.snippet or ''}"
            except Exception:
                txt = f"{h.title or ''} {h.snippet or ''}"
            pairs.append((query, txt))
        scores = ce.predict(pairs)  # numpy array
        scored = sorted(zip(scores, hits), key=lambda x: float(x[0]), reverse=True)
        return [h for _, h in scored]
    except Exception:
        return hits

# ===== Career hydration (match by exact title) =====
def build_career_lookup(settings: Settings) -> Dict[str, Dict[str, Any]]:
    """
    Build a fast title->career map for reliable hydration.
    """
    items = load_careers(settings)
    by_title = {_norm_name(it.get("title")): it for it in items if it.get("title")}
    return by_title

def hydrate_career_hit(h: SearchHit, by_title: Dict[str, Dict[str, Any]]) -> Dict[str, Any]:
    """
    Enrich a SearchHit (career) with structured fields from the canonical JSON.
    - Exact-title match first, then best-effort regex parsing for years.
    """
    src = by_title.get(_norm_name(h.title))
    out = {
        "title": h.title, "url": absolutize_url(h.url), "location": None, "department_name": None, "job_type": None,
        "min_years": None, "max_years": None, "skills_required": [], "duties": [], "requirements": [], "_ref": None
    }
    if src:
        for k in ["title","url","location","department_name","job_type","min_years","max_years",
                  "skills_required","duties","requirements"]:
            if k in src and src[k] not in (None, "", []):
                out[k] = src[k]
    txt = f"{(h.meta or {}).get('text','') or ''} {(h.snippet or '') or ''}"
    m = re.search(r"(\d+)\s*[-–]\s*(\d+)\s*(?:yrs|years)", txt, re.I)
    if m and out["min_years"] is None and out["max_years"] is None:
        out["min_years"] = int(m.group(1)); out["max_years"] = int(m.group(2))
    m = re.search(r"(\d+)\s*\+\s*(?:yrs|years)", txt, re.I)
    if m and out["min_years"] is None:
        out["min_years"] = int(m.group(1))
    return out

def hits_to_facts_json(domain: str, hits: List[SearchHit], settings: Settings, k: int) -> str:
    """
    Convert top-k hits into a compact JSON array for the LLM prompt.
    - Product path: hydrate with product JSON and regex extras.
    - Career path: hydrate via exact-title match and parsing.
    - Falls back to bare {title/url} if anything goes wrong.
    """
    try:
        if domain == "product":
            lookup = build_product_lookup(settings)  
            items = [hydrate_product_hit(h, lookup) for h in hits[:max(k, 3)]]
        else:
            by_title = build_career_lookup(settings)
            items = [hydrate_career_hit(h, by_title) for h in hits[:max(k, 3)]]
        return json.dumps(items, ensure_ascii=False)
    except Exception:
        bare = []
        for h in hits[:max(k, 3)]:
            if domain == "product":
                bare.append({"name": h.title, "product_url": h.url})
            else:
                bare.append({"title": h.title, "url": h.url})
        return json.dumps(bare, ensure_ascii=False)


@app.post("/v1/ask", tags=["ask"])
def ask_endpoint(
    payload: Dict[str, Any] = Body(
        ...,
        example={"q": "which jobs require python?", "k": 3, "max_tokens": 140}
    ),
    settings: Settings = Depends(get_settings),
):
    """
    RAG endpoint:
    1) Infer domain (product/career). If outside scope => short message.
    2) (Product) Optionally expand query with semantically close field names.
    3) Retrieve with FAISS, filter by domain, strict-skill filter (careers).
    4) Score fusion (vector + keywords + location), de-duplicate.
    5) Optional rerank with CrossEncoder.
    6) Build 'facts_json' + prompt; call HF local model with stop token.
    7) Clean & normalize output to numbered short paragraphs.
    8) Return answer + source chips (title/url/score).
    """
    # -------- helper: fallback lines if LLM returns nothing --------
    def _fallback_lines_from_facts(facts_json: str, domain: str, k: int) -> list[str]:
        try:
            arr = json.loads(facts_json)
        except Exception:
            return []
        lines: list[str] = []
        if domain == "product":
            for i, it in enumerate(arr[:max(k, 3)], 1):
                name = (it.get("name") or "").strip()
                brand = (it.get("brand") or "")
                cat   = (it.get("category") or "")
                flv   = (it.get("flavor") or "")
                size  = (f"{it['volume_ml']} ml" if it.get("volume_ml") else
                         f"{it['weight_g']} g" if it.get("weight_g") else "")
                avail = (it.get("availability") or "")
                price = (f"{it['price_now_kwd']} KWD" if it.get("price_now_kwd") is not None else "")
                parts = [name, brand, cat, flv, size, avail, price]
                body  = ", ".join([p for p in parts if p])
                if body:
                    lines.append(f"{i}. {body}")
        else:
            for i, it in enumerate(arr[:max(k, 3)], 1):
                title = (it.get("title") or "").strip()
                dep   = (it.get("department_name") or "")
                loc   = (it.get("location") or "")
                jt    = (it.get("job_type") or "")
                yrs   = []
                if it.get("min_years") is not None: yrs.append(str(it["min_years"]))
                if it.get("max_years") is not None: yrs.append(str(it["max_years"]))
                yrs_s = "–".join(yrs) + " yrs" if yrs else ""
                skills = ", ".join((it.get("skills_required") or [])[:3])
                parts = [title, dep, loc, jt, yrs_s, skills]
                body  = ", ".join([p for p in parts if p])
                if body:
                    lines.append(f"{i}. {body}")
        return lines

    # -------- welcome / readiness --------
    q_raw = payload.get("q")
    if not q_raw or not str(q_raw).strip():
        welcome = (
            "👋 Welcome to the KDD Smart Assistant!\n"
            "Ask me anything about KDD products or careers, and I’ll answer from verified company data.\n\n"
            "Try examples like:\n"
            "- What products are available right now?\n"
            "- Which jobs require Python?\n"
            "- Ice cream flavors under 1 KWD?\n"
        )
        return {"query": "", "answer": welcome, "sources": []}

    # Ensure the semantic index is ready
    sem = get_semantic(settings)
    if not sem or not sem.available():
        raise HTTPException(status_code=503, detail="Semantic index not available. Build FAISS first.")

    # -------- inputs --------
    q = str(q_raw).strip()
    k = int(payload.get("k") or 3)
    max_tokens = max(48, min(int(payload.get("max_tokens") or GEN_MAX_TOKENS_DEFAULT), 512))
    ql = q.lower()
    domain = infer_domain(q)

    # -------- dynamic field expansion (products only) --------
    expanded_q = q
    if domain == "product":
        # Try to guess relevant product fields and add them to the query text
        try:
            p_schema = get_products_schema(settings)
            guessed_fields = detect_fields_from_query(q, p_schema, topk=2, thresh=0.35)
            expanded_q = expand_query_with_fields(q, guessed_fields)
        except Exception:
            expanded_q = q

    # -------- FAISS retrieval --------
    raw_hits = sem.search(expanded_q, k=max(k * 2, 8))

    # -------- domain filtering --------
    if domain:
        typed = [h for h in raw_hits if getattr(h, "type", None) == domain]
        raw_hits = typed or raw_hits
    else:
        return {
            "query": q,
            "answer": "This question is outside the KDD domain. I can only answer about KDD products and careers.",
            "sources": []
        }

    # -------- strict skill filter (careers) --------
    if domain == "career":
        # Extract named skill heuristically; fallback to a small whitelist
        skill = extract_skill(q)
        if not skill:
            for kw in ["python", "بايثون", "java", "c#", "c++", "sql", "etl", "pandas", "spark", "excel", "power bi"]:
                if kw in ql:
                    skill = kw
                    break
        raw_hits = _filter_hits_by_skill(raw_hits, settings, skill)

    # -------- scoring fusion (FAISS + keywords + location) --------
    SKILL_TERMS = [
        "python","بايثون","java","sql","etl","excel","power bi","pandas","spark","data","ml","machine learning",
        "agile","scrum","leadership","safety","nebosh","iosh","osha"
    ]
    present_terms = [t for t in SKILL_TERMS if re.search(rf"\b{re.escape(t)}\b", ql)]
    LOC_TERMS = ["farwaniya","ahmadi","kuwait","sabhan","hawally"]

    # Fallback text map for keyword scoring when meta.text is empty
    try:
        careers_all = load_careers(settings)
        by_title_blob = {(it.get("title") or "").strip().lower(): career_text_blob(it) for it in careers_all}
    except Exception:
        by_title_blob = {}

    fused: List[SearchHit] = []
    terms_for_scoring = extract_terms(q)
    for h in raw_hits:
        title_key = (h.title or "").strip().lower()
        meta_text = (h.meta or {}).get("text", "") or by_title_blob.get(title_key, "")
        base = float(getattr(h, "score", 0.0) or 0.0)
        kw_terms = list(set((present_terms or []) + terms_for_scoring))
        kw = normalized_kw_score(f"{h.title or ''} {h.snippet or ''} {meta_text}", kw_terms)
        loc_boost = 0.05 if any(l in (meta_text.lower()) for l in LOC_TERMS) else 0.0
        new_score = 0.70 * base + 0.30 * kw + loc_boost
        fused.append(
            SearchHit(
                score=new_score, type=h.type, title=h.title, url=h.url,
                snippet=(h.snippet or "")[:800], meta=h.meta or {}
            )
        )

    # De-duplicate by normalized title (keep highest score), then cross-encode
    uniq: Dict[str, SearchHit] = {}
    for h in fused:
        key = (h.title or "").strip().lower()
        if key and (key not in uniq or h.score > uniq[key].score):
            uniq[key] = h
    hits = sorted(uniq.values(), key=lambda x: x.score, reverse=True)[:max(k, 3)]
    hits = rerank_with_cross_encoder(expanded_q, hits)[:max(k, 3)]

    if not hits:
        return {"query": q, "answer": "I don't know based on the given information.", "sources": []}

    # -------- build prompt and call LLM --------
    facts_json = hits_to_facts_json(domain, hits, settings, k)
    prompt = build_prompt(q, hits, domain, facts_json)

    try:
        pipe = get_hf_pipeline()
        tok = pipe.tokenizer
        mdl = pipe.model

        inputs = tok(prompt, return_tensors="pt")
        if torch.cuda.is_available():
            inputs = {k: v.to("cuda") for k, v in inputs.items()}

        # Stop the generation exactly at END_OF_ANSWER 
        stop_list = StoppingCriteriaList([StopOnSubstrings(tok, ["END_OF_ANSWER"])])
        gen_ids = mdl.generate(
            **inputs,
            max_new_tokens=max_tokens,
            temperature=0.4,              
            do_sample=True,               
            top_p=0.9,
            repetition_penalty=GEN_REP_PENALTY,
            no_repeat_ngram_size=GEN_NO_REPEAT_NGRAM,
            stopping_criteria=stop_list,
            eos_token_id=getattr(tok, "eos_token_id", None),
            pad_token_id=getattr(tok, "pad_token_id", getattr(tok, "eos_token_id", None)),
        )
        raw = tok.decode(gen_ids[0], skip_special_tokens=True)
        if prompt in raw:
            raw = raw.split(prompt, 1)[1].strip()
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Hugging Face error: {e}")

    # -------- clean output & format numbered lines --------
    # Drop anything after the explicit stop token
    raw = raw.split("END_OF_ANSWER", 1)[0]
    if not raw.strip():
        raw = "No matching items found."

    # Remove artifacts (role prefixes, links, stray tokens, extra spaces)
    text = raw
    text = re.sub(r"(?im)^\s*(Human|Assistant|System)\s*:.*$", "", text)
    text = re.sub(r"ENDOF\w*\s*.*", "", text)
    text = re.sub(r"https?://\S+", "", text)
    text = re.sub(r"\(\s?\d\.\d{2,3}\s?\)", "", text)
    text = re.sub(r"[ \t]+", " ", text).strip()

    # Split into paragraphs on blank lines
    paras = [p.strip() for p in re.split(r"\n\s*\n+", text) if p.strip()]

    # Normalize to "1. ..." numbered lines; ignore empty/symbol-only lines
    items: list[str] = []
    for p in paras:
        m = re.match(r"^\s*(\d+)\s*[.)\-:]\s*(.+)$", p, flags=re.S)
        body = (m.group(2).strip() if m else p.strip())
        body = re.sub(r"(?i)\b(?:assistant|human|system)\s*:\s*", "", body)
        body = re.sub(r"\s+", " ", body).strip()
        if not body or re.fullmatch(r"[.\-–—*•،؛\s]+", body):
            continue
        if m:
            items.append(f"{m.group(1)}. {body}")
        else:
            items.append(body)
    # If the LLM produced nothing useful, fall back to facts-based lines
    if not items:
        items = _fallback_lines_from_facts(facts_json, domain, k)

    answer_text = "\n\n".join(items[:max(k, 3)]).strip() or "I don't know based on the given information."

    # -------- sources (keep chips if UI uses them) --------
    used_ids = _extract_refs(raw)
    DEFAULT_CAREER_URL = "https://career.kddc.com"
    sources = []
    for h in hits:
        link = (h.url or DEFAULT_CAREER_URL) if h.type == "career" else h.url
        sources.append({
            "title": h.title,
            "url": absolutize_url(link) if link else None,
            "score": round(float(h.score or 0.0), 3)
        })
    
    # If the LLM explicitly referenced [i] indices, reorder the sources accordingly
    if used_ids:
        ordered = []
        for i in used_ids:
            if 1 <= i <= len(hits):
                h = hits[i - 1]
                link = (h.url or DEFAULT_CAREER_URL) if h.type == "career" else h.url
                ordered.append({
                    "i": i,
                    "title": h.title,
                    "url": absolutize_url(link) if link else None,
                    "score": float(h.score or 0.0)
                })
        if ordered:
            sources = ordered

    return {"query": q, "answer": answer_text, "sources": sources}
# --------------------------
# Simple Web UI at /ui
# --------------------------
@app.get("/ui", response_class=HTMLResponse, tags=["ui"])
def chat_ui(request: Request):
    """Render the main chat UI """
    return templates.TemplateResponse("ui.html", {"request": request})

@app.get("/ui/tools", response_class=HTMLResponse)
def chat_tools(request: Request):
    """Render the tools page """
    return templates.TemplateResponse("tools.html", {"request": request})

# --------------------------
# Run
# --------------------------
if __name__ == "__main__":
    # Local dev entrypoint (reload for hot dev)
    import uvicorn, pathlib
    module_name = pathlib.Path(__file__).stem
    uvicorn.run(f"{module_name}:app", host="127.0.0.1", port=8000, reload=True)