from __future__ import annotations
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
import json, re, hashlib
from urllib.parse import urlsplit, urlunsplit

URL_ABS_BASE = "https://www.kdd.com.kw"

_URL_RE = re.compile(r"^(?:https?:)?//", re.I)

def absolutize_url(u: Optional[str]) -> Optional[str]:
    """Return an absolute HTTP(S) URL for site-relative or protocol-relative inputs."""
    if not u:
        return None
    u = u.strip()
    # Already absolute (http://, https://) or protocol-relative ("//")
    if _URL_RE.match(u):
        return u if u.lower().startswith("http") else f"https:{u}"
    # Site-relative path
    if u.startswith("/"):
        return URL_ABS_BASE.rstrip("/") + u
    # Leave other strings unchanged
    return u


# ========= Input / Output paths =========
PRODUCT_INPUTS = [
    Path('data/products_ice_cream_detailed.json'),
    Path('data/products_juices_detailed.json'),
]
CAREER_INPUTS = [Path('data/careers.json')]

OUT_DIR = Path('data/clean')
OUT_DIR.mkdir(parents=True, exist_ok=True)

PROD_JSON  = OUT_DIR / 'products_clean.json'
PROD_JSONL = OUT_DIR / 'products_clean.jsonl'
CARE_JSON  = OUT_DIR / 'careers1_clean.json'
CARE_JSONL = OUT_DIR / 'careers1_clean.jsonl'

CORPUS_JSON = Path('data/corpus.json')

print('Products in :', [str(p) for p in PRODUCT_INPUTS])
print('Careers  in :', [str(p) for p in CAREER_INPUTS])
print('Outputs     :', PROD_JSON, PROD_JSONL, CARE_JSON, CARE_JSONL, CORPUS_JSON)


# ========= General helpers =========
WS_RE = re.compile(r'\s+')

def clean_text(x: Optional[str]) -> str:
    """Strip tags, normalize punctuation/dashes, collapse whitespace."""
    if x is None:
        return ''
    x = re.sub(r'<[^>]+>', ' ', str(x))  # very light HTML removal
    x = x.replace('–', '-').replace('—', '-').replace('’', "'")
    x = WS_RE.sub(' ', x).strip()
    return x

def clean_list_str(xs: Optional[List[Any]]) -> List[str]:
    """Normalize a list of arbitrary values into a unique list of cleaned strings."""
    if not xs:
        return []
    out: List[str] = []
    seen = set()
    for v in xs:
        t = clean_text(str(v))
        if t and t.lower() not in seen:
            seen.add(t.lower())
            out.append(t)
    return out

def id_hash(*parts: Any, n: int = 16) -> str:
    """Stable short hash ID from arbitrary parts."""
    s = '|'.join(clean_text(str(p)) for p in parts if p)
    return hashlib.sha1(s.encode()).hexdigest()[:n]

def write_json_and_jsonl(rows: List[Dict[str, Any]], json_path: Path, jsonl_path: Path):
    """Write a list of dicts to both JSON and JSONL files."""
    json_path.parent.mkdir(parents=True, exist_ok=True)
    with json_path.open('w', encoding='utf-8') as f:
        json.dump(rows, f, ensure_ascii=False, indent=2)
    with jsonl_path.open('w', encoding='utf-8') as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + '\n')

def to_int_safe(x: Any) -> Optional[int]:
    """Convert value to int when possible; otherwise return None."""
    try:
        if x is None:
            return None
        return int(float(str(x)))
    except Exception:
        return None


# ========= Price / size parsing helpers =========
NUM_RE  = re.compile(r'([0-9]+(?:[.,][0-9]{1,3})?)')
VOL_IN_NAME_RE = re.compile(r'(\d+(?:\.\d+)?)\s*(ml|l|ltr|liter|litre)\b', re.I)
PCS_IN_NAME_RE = re.compile(r'(\d+)\s*(pcs|pieces|piece|pc)\b', re.I)
CARTON_QTY_IN_LABEL = re.compile(r'(\d+)\s*(Pieces|Piece|pcs|pc)\b', re.I)

# Weight in grams + "Pack of N"
WEIGHT_RE = re.compile(r'(\d+(?:\.\d+)?)\s*(g|gram|grams)\b', re.I)
PACK_OF_RE = re.compile(r'Pack\s+of\s+(One|Two|Three|Four|Five|Six|Seven|Eight|Nine|Ten|\d+)', re.I)
WORD2NUM = {"one":1,"two":2,"three":3,"four":4,"five":5,"six":6,"seven":7,"eight":8,"nine":9,"ten":10}

def norm_currency(cur: Optional[str]) -> Optional[str]:
    """Normalize currency codes to 'KWD' or 'USD' when possible."""
    if not cur:
        return None
    c = cur.strip().upper()
    if c in {'KD','KWD','د.ك','ك.د'}:
        return 'KWD'
    if c in {'USD','$'}:
        return 'USD'
    return c

def to_kwd(val: Optional[float], cur: Optional[str]) -> Optional[float]:
    """
    Convert a value to KWD only when the currency is already KWD or unknown.
    Returns None for other currencies to avoid implicit conversion guesses.
    """
    if val is None:
        return None
    c = norm_currency(cur)
    if c in (None, 'KWD'):
        return float(val)
    return None

def https_no_query(u: str) -> str:
    """Force https scheme and strip query/fragment from the URL."""
    if not u:
        return ''
    sp = urlsplit(u)
    return urlunsplit(('https', sp.netloc, sp.path, '', ''))

def availability_to_enum(x: str) -> str:
    """Map various availability phrases into a small enum."""
    t = clean_text(x).lower()
    if not t:
        return 'unknown'
    if any(k in t for k in ['in stock','available','متوفر']):
        return 'in_stock'
    if any(k in t for k in ['out of stock','غير متوفر','نفذ']):
        return 'out_of_stock'
    if any(k in t for k in ['pre-order','preorder','طلب مسبق']):
        return 'preorder'
    return 'unknown'

def extract_volume_ml(name: str, fallback_blob: str = '') -> Optional[int]:
    """Extract volume in ml from name or fallback text; supports ml/l/ltr/liter/litre."""
    for source in (name, fallback_blob):
        if not source:
            continue
        m = VOL_IN_NAME_RE.search(source)
        if m:
            qty = float(m.group(1))
            unit = m.group(2).lower()
            if unit in ('l','ltr','liter','litre'):
                return int(round(qty * 1000))
            return int(round(qty))
    return None

def extract_weight_g(name: str, fallback_blob: str = '') -> Optional[int]:
    """Extract weight in grams from name or fallback text."""
    for source in (name, fallback_blob):
        if not source:
            continue
        m = WEIGHT_RE.search(source)
        if m:
            qty = float(m.group(1))
            return int(round(qty))
    return None

def extract_pack_units_from_name(name: str) -> Optional[int]:
    """Extract 'pieces' count from the product name if present."""
    if not name:
        return None
    m = PCS_IN_NAME_RE.search(name.lower())
    if not m:
        return None
    try:
        return int(m.group(1))
    except Exception:
        return None

def parse_option_label_price(label: str):
    """
    Parse a price and currency from an option label.
    Returns (value, currency) where currency is normalized if possible.
    """
    t = label.strip()
    m = re.search(r'\(([^)]+)\)\s*$', t)
    candidate = m.group(1) if m else t
    cur = None
    for c in ['KD','KWD','د.ك','ك.د','USD','$']:
        if c in candidate:
            cur = c
            break
    mnum = NUM_RE.search(candidate) or NUM_RE.search(t)
    val = float(mnum.group(1).replace(',', '.')) if mnum else None
    return val, norm_currency(cur)

def options_to_text(options: List[Dict[str, Any]], currency_fallback: Optional[str]):
    """
    Convert product options JSON into a compact pipe-delimited string.
    Also detects a 'Carton' option and tries to infer per-piece pricing notes.
    """
    if not options:
        return '', None, None
    rows = []
    carton_total = None
    carton_qty = None
    for o in options:
        label = clean_text(o.get('label'))
        qty   = o.get('quantity')
        unit  = o.get('unit')

        p_val, p_cur = parse_option_label_price(label)
        if p_cur is None:
            p_cur = norm_currency(o.get('currency') or currency_fallback)

        # Carton detection
        if re.search(r'\bCarton\b', label, re.I):
            carton_total = p_val
            mqty = CARTON_QTY_IN_LABEL.search(label)
            if mqty:
                try:
                    carton_qty = int(mqty.group(1))
                except Exception:
                    pass

        # Piece default when quantity is not provided
        if re.search(r'\bPiece\b', label, re.I) and qty is None:
            qty, unit = 1, 'piece'

        # "Pack of N" textual number handling
        mp = PACK_OF_RE.search(label)
        if mp and qty is None:
            word = mp.group(1).lower()
            qty = WORD2NUM.get(word, int(word) if word.isdigit() else None)
            unit = unit or 'pieces'

        left = label.split('(')[0].strip()
        bits = [left]
        if qty and unit:
            bits.append(f'{qty} {unit}')
        if p_val is not None:
            bits.append(f'{p_cur or "KWD"} {p_val}')
        rows.append(' | '.join([b for b in bits if b]))

    return ' || '.join(rows), carton_total, carton_qty


# ========= Products cleaning =========
def infer_product_category_from_breadcrumbs(d: Dict[str, Any]) -> Optional[str]:
    """Try to infer category (ice cream / juices) from breadcrumb head."""
    b = d.get('breadcrumbs') or []
    if not b:
        return None
    head = clean_text(b[0]).lower()
    if 'ice' in head:
        return 'ice cream'
    if 'juice' in head:
        return 'juices'
    return head or None

def clean_product_record(d: Dict[str, Any], category_hint: Optional[str]) -> Dict[str, Any]:
    """Normalize a single product JSON record into a compact, consistent schema."""
    name = clean_text(d.get('name'))
    brand = clean_text(d.get('brand')) or None
    product_url = https_no_query(clean_text(d.get('product_url')))
    product_code = clean_text(d.get('product_code')) or None
    sku = clean_text(d.get('sku') or product_code) or None

    cp = clean_text(d.get('category_page'))

    # Category resolution: filename hint > page label > breadcrumbs > None
    if category_hint:
        category = category_hint.lower()
    elif 'Ice-Cream' in cp or 'Ice Cream' in cp:
        category = 'ice cream'
    elif 'Juices' in cp:
        category = 'juices'
    else:
        category = infer_product_category_from_breadcrumbs(d) or None

    price_now = d.get('price_now')
    currency  = norm_currency(d.get('currency'))
    price_now_kwd = to_kwd(price_now, currency)
    availability = availability_to_enum(d.get('availability',''))

    reviews_count = d.get('reviews_count')
    rating_stars = d.get('rating_stars')
    # If there are zero/no reviews, drop the rating to avoid misleading signals
    if reviews_count in (0,'0',None):
        rating_stars = None

    # Build a descriptive blob from various fields
    blob = ' '.join([
        clean_text(d.get('description','')),
        clean_text(d.get('ingredients','')),
        clean_text(d.get('pack_info','')),
        clean_text(d.get('weight_or_volume','')),
    ])

    volume_ml = extract_volume_ml(name, blob)
    weight_g  = extract_weight_g(d.get('weight_or_volume',''), blob)
    pack_units_from_name = extract_pack_units_from_name(name)
    flavor = (d.get('flavor') or '').strip().lower() or None
    options_text, carton_total, carton_qty = options_to_text(d.get('options') or [], currency_fallback=currency)
    price_notes = None
    if carton_total and carton_qty:
        try:
            price_notes = f'carton_per_piece≈{carton_total/carton_qty:.3f} KWD'
        except Exception:
            pass

    dedup_key = product_url or id_hash(name, d.get('pack_info'), brand)
    return {
        'dedup_key': dedup_key,
        'name': name or None,
        'brand': brand,
        'category': category,
        'flavor': flavor,
        'volume_ml': volume_ml,
        'weight_g': weight_g,
        'pack_units': pack_units_from_name,
        'product_url': product_url or None,
        'product_code': product_code,
        'sku': sku,
        'availability': availability,
        'price_now_kwd': price_now_kwd,
        'options_text': options_text or None,
        'price_notes': price_notes,
        'reviews_count': int(reviews_count) if isinstance(reviews_count,(int,float,str)) and str(reviews_count).isdigit() else 0,
        'rating_stars': to_int_safe(rating_stars),
        'desc': blob or None,  # retained for downstream embedding
    }

def infer_product_category_from_filename(p: Path) -> Optional[str]:
    """Infer category from file name for a coarse hint."""
    n = p.name.lower()
    if 'juice' in n:
        return 'juices'
    if 'ice' in n:
        return 'ice cream'
    return None

def load_products(inputs: List[Path]):
    """Load product JSONs from a list of files, returning tuples (record, category_hint)."""
    rows = []
    for p in inputs:
        if not p.exists():
            print('Missing:', p)
            continue
        try:
            data = json.loads(Path(p).read_text(encoding='utf-8'))
        except Exception as e:
            print('Failed to load', p, e)
            data = []
        cat_hint = infer_product_category_from_filename(p)
        for d in data:
            if isinstance(d, dict):
                rows.append((d, cat_hint))
    return rows

def run_products_cleaning(inputs: List[Path]):
    """Clean, normalize, deduplicate products and write outputs."""
    raw = load_products(inputs)
    cleaned = []
    for d, cat_hint in raw:
        try:
            cleaned.append(clean_product_record(d, cat_hint))
        except Exception as e:
            print("product clean failed:", (d.get('product_url') or d.get('name')), e)

    # First pass: dedupe by product_url (keep first)
    seen_url = set()
    uniq_by_url = []
    for r in cleaned:
        u = r.get('product_url')
        if u and u not in seen_url:
            seen_url.add(u)
            uniq_by_url.append(r)
        elif not u:
            uniq_by_url.append(r)

    # Second pass: dedupe by dedup_key
    seen_key = set()
    final_rows = []
    for r in uniq_by_url:
        k = r.get('dedup_key')
        if not k or k not in seen_key:
            if k:
                seen_key.add(k)
            final_rows.append(r)

    write_json_and_jsonl(final_rows, PROD_JSON, PROD_JSONL)
    print(f'Products: cleaned {len(final_rows)} → {PROD_JSON}')
    return final_rows


# ========= Careers cleaning =========
YEAR_RANGE_RE  = re.compile(r'(\d+)\s*(?:-|–|—|to)\s*(\d+)\s*years?', re.I)
YEAR_PLUS_RE   = re.compile(r'(\d+)\s*\+\s*years?', re.I)
YEAR_SINGLE_RE = re.compile(r'(\d+)\s*years?', re.I)

def parse_years_careers(d: Dict[str, Any]) -> Tuple[Optional[int], Optional[int]]:
    """
    Extract min/max years of experience from various fields.
    Supported forms: "5-7 years", "5–7 years", "5+ years", "5 years".
    """
    candidates: List[str] = []
    if d.get('years'):
        candidates.append(str(d.get('years')))
    for k in ('details','Job_duties_include_but_are_not_limited_to','Required_Professional_Skills'):
        arr = d.get(k)
        if isinstance(arr, list):
            for line in arr:
                candidates.append(clean_text(line))

    for c in candidates:
        m = YEAR_RANGE_RE.search(c)
        if m:
            a, b = int(m.group(1)), int(m.group(2))
            return (min(a,b), max(a,b))

    for c in candidates:
        m = YEAR_PLUS_RE.search(c)
        if m:
            val = int(m.group(1))
            return (val, None)

    for c in candidates:
        m = YEAR_SINGLE_RE.search(c)
        if m:
            val = int(m.group(1))
            return (val, val)

    return None, None

def split_department(dep: Optional[str]) -> Tuple[Optional[str], Optional[str]]:
    """
    Split a department field like 'IT-123 — Information Technology'
    into (code, name) when possible. Supports various separators.
    """
    if not dep:
        return None, None
    t = clean_text(dep)
    m = re.match(r'^\s*([\w/]+)\s*[-–—:]\s*(.+)$', t)
    if m:
        return m.group(1), clean_text(m.group(2))
    return None, t or None

def build_embedding_text_career(rec: Dict[str, Any]) -> str:
    """Construct a compact, descriptive text used for embedding a career record."""
    bits = []
    if rec.get('title'):
        bits.append(f"Title: {rec['title']}")
    if rec.get('department_name'):
        bits.append(f"Department: {rec['department_name']}")
    if rec.get('location'):
        bits.append(f"Location: {rec['location']}")
    if rec.get('min_years') is not None:
        if rec.get('max_years') is not None and rec['max_years'] != rec['min_years']:
            bits.append(f"Experience: {rec['min_years']}-{rec['max_years']} years")
        elif rec.get('max_years') is None:
            bits.append(f"Experience: {rec['min_years']}+ years")
        else:
            bits.append(f"Experience: {rec['min_years']} years")
    skills = rec.get('skills_required') or []
    if skills:
        bits.append('Skills: ' + '; '.join(skills))
    if rec.get('duties'):
        bits.append('Duties: ' + ' '.join(rec['duties']))
    if rec.get('requirements'):
        bits.append('Requirements: ' + ' '.join(rec['requirements']))
    return ' | '.join(bits)

def clean_career_record(d: Dict[str, Any]) -> Dict[str, Any]:
    """Normalize a single career record into a consistent schema."""
    rid = clean_text(d.get('id')) or None
    title = clean_text(d.get('title')) or None
    url = clean_text(d.get('url')) or None
    location = clean_text(d.get('location')) or None
    department = clean_text(d.get('department')) or None
    job_type = clean_text(d.get('job_type')) or None
    dep_code, dep_name = split_department(department)

    skills = clean_list_str(d.get('skills_required'))

    duties: List[str] = []
    for k in ('Job_duties_include_but_are_not_limited_to','details','duties','responsibilities'):
        duties.extend(clean_list_str(d.get(k)))

    requirements = clean_list_str(d.get('Required_Professional_Skills'))
    if not requirements:
        # If explicit requirements are missing, infer some from duties-like lines
        inferred = [ln for ln in duties if re.search(r'\b(Bachelor|Master|Minimum|Proven|Certification|Fluency|experience)\b', ln, re.I)]
        requirements = clean_list_str(inferred)

    min_years, max_years = parse_years_careers(d)
    dedup = id_hash(title, location, dep_name)

    rec = {
        'dedup_key': dedup,
        'id_src': rid,
        'title': title,
        'url': url or None,
        'location': location or None,
        'department_code': dep_code,
        'department_name': dep_name,
        'job_type': job_type or None,
        'min_years': min_years,
        'max_years': max_years,
        'skills_required': skills or None,
        'duties': duties or None,
        'requirements': requirements or None,
    }
    rec['embedding_text'] = build_embedding_text_career(rec)
    return rec

def load_careers(inputs: List[Path]):
    """Load all careers from the given JSON files (lists of objects)."""
    rows = []
    for p in inputs:
        if not p.exists():
            print('Missing:', p)
            continue
        try:
            data = json.loads(Path(p).read_text(encoding='utf-8'))
        except Exception as e:
            print('Failed to load', p, e)
            data = []
        if isinstance(data, list):
            rows.extend(data)
    return rows

def run_careers_cleaning(inputs: List[Path]):
    """Clean, normalize, deduplicate careers and write outputs."""
    raw = load_careers(inputs)
    cleaned = []
    for d in raw:
        try:
            cleaned.append(clean_career_record(d))
        except Exception as e:
            print("career clean failed:", d.get('title'), e)

    kept: Dict[str, Dict[str, Any]] = {}
    for r in cleaned:
        k = r['dedup_key']
        if k not in kept:
            kept[k] = r
        else:
            # Prefer entries that carry a URL if the first one did not
            if (r.get('url') and not kept[k].get('url')):
                kept[k] = r

    final_rows = list(kept.values())
    write_json_and_jsonl(final_rows, CARE_JSON, CARE_JSONL)
    print(f'Careers: cleaned {len(final_rows)} → {CARE_JSON}')
    return final_rows


# ========= Build unified corpus.json for embedding/search =========
def build_corpus(products_file: Path = PROD_JSON, careers_file: Path = CARE_JSON, out_path: Path = CORPUS_JSON):
    """
    Merge cleaned products and careers into a single corpus file
    suitable for downstream embedding and semantic search.
    """
    out_path.parent.mkdir(parents=True, exist_ok=True)
    corpus: List[Dict[str, Any]] = []

    # ----- Products -----
    products = []
    if products_file.exists():
        try:
            products = json.loads(products_file.read_text(encoding='utf-8'))
        except Exception as e:
            print('Failed to load products:', products_file, e)
            products = []

    for p in products:
        # Normalize category
        cat = (p.get("category") or "").strip().lower()
        if "ice" in cat:
            cat = "ice cream"
        elif "juice" in cat:
            cat = "juices"

        text = re.sub(r"\s+", " ", " | ".join([
            str(p.get("name", "")),
            str(cat),
            (f"{p.get('price_now_kwd')} KWD" if p.get("price_now_kwd") is not None else ""),
            str(p.get("brand", "")),
            str(p.get("flavor", "")),
            (f"{p.get('volume_ml')} ml" if p.get("volume_ml") else ""),
            (f"{p.get('weight_g')} g" if p.get("weight_g") else ""),
            (f"{p.get('pack_units')} pcs" if p.get("pack_units") else ""),
            str(p.get("options_text") or ""),
            str(p.get("desc") or ""),
        ])).strip()

        doc = {
            "type": "product",
            "title": p.get("name"),
            "url": p.get("product_url"),
            "category": cat,
            "text": text,
            "meta": p
        }
        corpus.append(doc)

    # ----- Careers -----
    careers = []
    if careers_file.exists():
        try:
            careers = json.loads(careers_file.read_text(encoding='utf-8'))
        except Exception as e:
            print('Failed to load careers:', careers_file, e)
            careers = []

    for c in careers:
        base_text = c.get("embedding_text") or ""
        skills = "; ".join(c.get("skills_required") or [])
        extra  = " ".join((c.get("duties") or []) + (c.get("requirements") or []))
        text = re.sub(r"\s+", " ", " | ".join([
            base_text,
            f"Skills: {skills}",
            extra,
        ])).strip()

        doc = {
            "type": "career",
            "title": c.get("title"),
            "url": absolutize_url(c.get("url")),
            "department": c.get("department_name"),
            "location": c.get("location"),
            "text": text,
            "meta": c
        }
        corpus.append(doc)

    with out_path.open("w", encoding="utf-8") as out:
        json.dump(corpus, out, ensure_ascii=False, indent=2)

    print(f"Corpus built: {len(corpus)} docs → {out_path}")
    return corpus


# run 
if __name__ == "__main__":
    products_final = run_products_cleaning(PRODUCT_INPUTS)
    careers_final  = run_careers_cleaning(CAREER_INPUTS)
    print("Counts:", len(products_final), len(careers_final))

    build_corpus(PROD_JSON, CARE_JSON, CORPUS_JSON)

    try:
        corpus = json.loads(CORPUS_JSON.read_text(encoding='utf-8'))
        for r in corpus[:3]:
            print(json.dumps(r, ensure_ascii=False, indent=2))
            print('-'*60)
    except Exception:
        pass
