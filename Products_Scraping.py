import os, time, re, hashlib, json
from urllib.parse import urljoin, urlencode, urlsplit, urlunsplit, parse_qsl
from typing import List, Dict, Optional, Tuple
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
from bs4 import BeautifulSoup
from tqdm import tqdm

# Category entry points
ICE_CREAM_URL = "https://eshop.kddc.com/en/Ice-Cream"
JUICES_URL    = "https://eshop.kddc.com/en/Juices"

# Output folder for raw scraped JSON
DATA_DIR = "data"
os.makedirs(DATA_DIR, exist_ok=True)

# Conservative headers and timeouts
HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)",
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
    "Accept-Language": "en-US,en;q=0.9",
}
CONNECT_TIMEOUT = 10
READ_TIMEOUT = 60
TIMEOUT = (CONNECT_TIMEOUT, READ_TIMEOUT)

# Throttle between requests to be polite
SLEEP = 0.8

# Requests session with retries and exponential backoff
_session = requests.Session()
retries = Retry(
    total=6,
    connect=3,
    read=3,
    backoff_factor=0.8,
    status_forcelist=(429, 500, 502, 503, 504),
    allowed_methods=("GET", "HEAD"),
)
_session.mount("https://", HTTPAdapter(max_retries=retries))
_session.mount("http://", HTTPAdapter(max_retries=retries))


# ---------------- Helpers ----------------
def clean(x: Optional[str]) -> str:
    """Collapse whitespace and trim."""
    if not x:
        return ""
    return re.sub(r"\s+", " ", x).strip()

def hid(*parts) -> str:
    """Stable short hash for IDs."""
    return hashlib.sha1("|".join(str(p) for p in parts).encode()).hexdigest()[:12]

def with_limit(url: str, limit: int = 100):
    """Add/override ?limit= param on category pages."""
    sp = urlsplit(url)
    qs = dict(parse_qsl(sp.query))
    qs["limit"] = str(limit)
    return urlunsplit((sp.scheme, sp.netloc, sp.path, urlencode(qs), sp.fragment))

def get(url: str) -> requests.Response:
    """GET with retrying session + standard headers."""
    r = _session.get(url, headers=HEADERS, timeout=TIMEOUT, stream=False)
    r.raise_for_status()
    return r

def soup(url: str) -> BeautifulSoup:
    """Fetch a URL and parse with lxml parser."""
    r = get(url)
    time.sleep(SLEEP)
    return BeautifulSoup(r.text, "lxml")

def find_next_page(s: BeautifulSoup, base_url: str) -> Optional[str]:
    """Find the next pagination link if present."""
    nxt = s.select_one("a[rel=next]") or s.select_one("ul.pagination li.active + li a")
    if nxt and nxt.get("href"):
        return urljoin(base_url, nxt["href"])
    return None

def parse_price(txt: str) -> Tuple[Optional[float], Optional[str]]:
    """Extract numeric value and currency symbol/code from a price label."""
    t = clean(txt)
    if not t:
        return None, None
    currencies = ["KD", "KWD", "د.ك", "ك.د", "USD", "$"]
    cur = next((c for c in currencies if c in t), None)
    m = re.search(r"([0-9]+(?:[.,][0-9]{1,3})?)", t)
    val = float(m.group(1).replace(",", ".")) if m else None
    return val, cur

def kv_from_list(ul) -> Dict[str, str]:
    """Convert <ul><li>Key: Value</li></ul> structures into a dict."""
    out = {}
    if not ul:
        return out
    for li in ul.select("li"):
        text = clean(li.get_text(" "))
        if ":" in text:
            k, v = text.split(":", 1)
            out[clean(k)] = clean(v)
    return out

def kv_from_table(table) -> Dict[str, str]:
    """Convert two-column tables into key/value dict."""
    out = {}
    if not table:
        return out
    for tr in table.select("tr"):
        tds = tr.find_all(["td", "th"])
        if len(tds) >= 2:
            k = clean(tds[0].get_text(" "))
            v = clean(tds[1].get_text(" "))
            if k:
                out[k] = v
    return out

def _contains_fallback(soup_obj: BeautifulSoup, needle: str):
    """
    Fallback for selectors like p:contains('Ingredients').
    BeautifulSoup does not support :contains, so we search text manually.
    """
    needle_l = needle.lower()
    for tag in soup_obj.find_all(True):
        txt = tag.get_text(" ", strip=True)
        if txt and needle_l in txt.lower():
            return tag
    return None

def first_text(detail_soup: BeautifulSoup, selectors: List[str]) -> str:
    """Return the first non-empty text found by trying multiple selectors."""
    for sel in selectors:
        if ":contains(" in sel:
            # Extract the text inside contains('...') if present and do a manual search
            m = re.search(r":contains\(['\"](.+?)['\"]\)", sel)
            if m:
                el = _contains_fallback(detail_soup, m.group(1))
                if el:
                    txt = clean(el.get_text(" "))
                    if txt:
                        return txt
            continue
        el = detail_soup.select_one(sel)
        if el:
            txt = clean(el.get_text(" "))
            if txt:
                return txt
    return ""


# ---------------- Product detail scraper ----------------
def scrape_product_detail(product_url: str) -> Dict:
    """
    Scrape a product detail page and return a normalized dict.
    Only fields used by the cleaning/ETL pipeline are included.
    """
    d = {
        "product_url": product_url,
        "price_now": None,
        "price_old": None,
        "currency": None,
        "availability": "",
        "product_code": "",
        "brand": "",
        "sku": "",
        "weight_or_volume": "",
        "pack_size": "",
        "flavor": "",
        "description": "",
        "ingredients": "",
        "breadcrumbs": [],
        "reviews_count": None,
        "rating_stars": None,
        "options": [],
    }

    s = soup(product_url)

    # Prices (handle either "price-new", "price-normal" or "h2.price")
    price_new_el = s.select_one("span.price-new") or s.select_one("span.price-normal") or s.select_one("h2.price")
    price_old_el = s.select_one("span.price-old")
    if price_new_el:
        d["price_now"], d["currency"] = parse_price(price_new_el.get_text())
    if price_old_el:
        d["price_old"], _ = parse_price(price_old_el.get_text())

    # Meta blocks often contain availability and codes
    meta_blocks = s.select("ul.list-unstyled li, div.product-info li, div.product-info .row")
    for el in meta_blocks:
        txt = clean(el.get_text(" "))
        if re.search(r"\bAvailability\b", txt, re.I):
            m = re.search(r"Availability\s*:\s*(.+)", txt, re.I)
            d["availability"] = clean(m.group(1)) if m else clean(txt.replace("Availability", ""))
        if re.search(r"\b(Product\s*Code|Code|SKU)\b", txt, re.I):
            m = re.search(r"(?:Product\s*Code|Code|SKU)\s*:\s*(.+)", txt, re.I)
            val = clean(m.group(1)) if m else ""
            d["product_code"] = d["product_code"] or val
            d["sku"] = d["sku"] or val

    # Alternative availability badges if not found above
    if not d["availability"]:
        d["availability"] = first_text(s, [
            "span.label-success", ".product-stock span", ".stock span",
            "span.stock-status", "div#stock", "span.available"
        ])

    # Ratings and review counts (best-effort)
    rating_text = first_text(s, [".rating", "div.rating", "div.review"])
    m_rev = re.search(r"(\d+)\s+Reviews", rating_text, re.I)
    d["reviews_count"] = int(m_rev.group(1)) if m_rev else None
    filled = s.select(".fa.fa-star, .fa-solid.fa-star, .icon-star")
    d["rating_stars"] = len(filled) or None

    # Breadcrumbs for category hints
    d["breadcrumbs"] = [clean(a.get_text()) for a in s.select("ul.breadcrumb li a, nav.breadcrumb a") if clean(a.get_text())]

    # Rich description (try several containers)
    for sel in ["#tab-description", ".tab-content #tab-description",
                "div.product-description", "div#description", "div#tab-description"]:
        el = s.select_one(sel)
        if el:
            d["description"] = clean(el.get_text(" "))
            if d["description"]:
                break

    # Ingredients section (varies)
    d["ingredients"] = first_text(s, ["div.ingredients", "p:contains('Ingredients')", "li:contains('Ingredients')"])

    # Best-effort extraction of weight/volume and pack size from page text
    # To improve recall, include table/list key-value text if present (not stored)
    specs_text = []
    for sel in ["#tab-specification .table", "table.table-bordered", "table#specs", "div.specification table"]:
        specs_text.append(json.dumps(kv_from_table(s.select_one(sel)), ensure_ascii=False))
    for sel in ["#tab-specification .list-unstyled", "ul.list-unstyled.specs", "div.specification ul.list-unstyled"]:
        specs_text.append(json.dumps(kv_from_list(s.select_one(sel)), ensure_ascii=False))

    blob = " ".join([d["description"], d["ingredients"], " ".join(specs_text)]).lower()
    m = re.search(r"(\d+(?:\.\d+)?)\s*(ml|l|g|kg)", blob)
    if m:
        d["weight_or_volume"] = m.group(0)
    m2 = re.search(r"(\d+)\s*[x×]\s*(\d+(?:\.\d+)?)\s*(ml|l|g|kg)", blob)
    if m2:
        d["pack_size"] = m2.group(0)

    # Naive flavor extraction by keyword
    for key in ["vanilla","chocolate","strawberry","mango","orange","apple","banana","caramel","coffee","pistachio"]:
        if key in blob:
            d["flavor"] = key
            break

    # Options (selects and radio/checkbox with inline price/qty hints)
    for sel in ["div.options select[name^='option']", "div.product-options select[name^='option']",
                "select#input-option", "select[name^='option']"]:
        for sel_el in s.select(sel):
            for opt in sel_el.select("option"):
                text = clean(opt.get_text())
                if not text or text.lower().startswith(("choose", "select")):
                    continue
                price_val, price_cur = parse_price(text)
                qty = None
                unit = None
                mqty = re.search(r"(\d+)\s*(Pieces|Piece|pcs|pc)", text, re.I)
                if mqty:
                    qty = int(mqty.group(1))
                    unit = "pieces"
                d["options"].append({
                    "type": "select",
                    "label": text,
                    "price": price_val if price_val is not None else d["price_now"],
                    "currency": price_cur or d["currency"],
                    "quantity": qty,
                    "unit": unit
                })

    for wrap in s.select("div.options, div.product-options, #product"):
        for lab in wrap.select("label"):
            text = clean(lab.get_text(" "))
            if not text:
                continue
            price_val, price_cur = parse_price(text)
            mqty = re.search(r"(\d+)\s*(Pieces|Piece|pcs|pc)", text, re.I)
            qty = int(mqty.group(1)) if mqty else None
            unit = "pieces" if mqty else None
            key = ("radio/checkbox", text, price_val, price_cur, qty, unit)
            already = any(o for o in d["options"] if (o.get("_k") == key))
            if not already:
                d["options"].append({
                    "_k": key,
                    "type": "choice",
                    "label": text,
                    "price": price_val if price_val is not None else d["price_now"],
                    "currency": price_cur or d["currency"],
                    "quantity": qty,
                    "unit": unit
                })
    # Remove internal marker used to deduplicate radio/checkbox labels
    for o in d["options"]:
        o.pop("_k", None)

    return d


# ---------------- Category scraper ----------------
def scrape_category(category_url: str) -> List[Dict]:
    """
    Iterate through a category with pagination, scrape cards and then per-item details.
    Returns a list of normalized product dicts ready for cleaning.
    """
    url = with_limit(category_url, 100)
    items: List[Dict] = []
    page = 1

    while url:
        try:
            s = soup(url)
        except requests.exceptions.ReadTimeout:
            print(f"[WARN] Timeout on {url}, retrying...")
            time.sleep(3)
            s = soup(url)
        except Exception as e:
            print(f"[ERROR] Skipping {url}: {e}")
            break

        cards = s.select("div.product-layout")
        for c in tqdm(cards, desc=f"Products page {page}"):
            try:
                a = c.select_one("h4 a")
                name = clean(a.get_text()) if a else ""
                product_url = urljoin(url, a["href"]) if a and a.get("href") else ""

                price_el = c.select_one("span.price-new") or c.select_one("span.price-normal")
                price = clean(price_el.get_text()) if price_el else ""

                pack = clean(c.select_one("div.caption p").get_text() if c.select_one("div.caption p") else "")

                base = {
                    "id": hid(name, price, pack, product_url),
                    "name": name,
                    "price_label": price,
                    "pack_info": pack,
                    "category_page": category_url,
                    "product_url": product_url
                }

                details = {}
                if product_url:
                    try:
                        details = scrape_product_detail(product_url)
                    except Exception as de:
                        print(f"[WARN] detail error for {product_url}: {de}")

                merged = {**base, **details}
                items.append(merged)
            except Exception:
                # Skip malformed cards rather than aborting the whole page
                continue

        next_url = find_next_page(s, url)
        url = next_url
        page += 1
    return items


def main():
    # Ice Cream
    print("\nScraping Ice Cream products...")
    ice = scrape_category(ICE_CREAM_URL)
    with open(os.path.join(DATA_DIR, "products_ice_cream_detailed.json"), "w", encoding="utf-8") as f:
        json.dump(ice, f, ensure_ascii=False, indent=2)
    print(f"Saved {len(ice)} ice-cream products (detailed).")

    # Juices
    print("\nScraping Juices products...")
    juices = scrape_category(JUICES_URL)
    with open(os.path.join(DATA_DIR, "products_juices_detailed.json"), "w", encoding="utf-8") as f:
        json.dump(juices, f, ensure_ascii=False, indent=2)
    print(f"Saved {len(juices)} juices products (detailed).")

    print("\nAll scraping complete. JSON files saved in /data.")


if __name__ == "__main__":
    main()
