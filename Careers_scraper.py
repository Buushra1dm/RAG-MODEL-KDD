import os, re, json, requests
from typing import List, Dict, Any, Optional
from urllib.parse import quote, urlparse

from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from webdriver_manager.chrome import ChromeDriverManager

EMPLOYER_API = "https://career.kddc.com/api/career/employers/2930/jobs?pageSize=1000&page=1"
BASE = "https://career.kddc.com"
OUT_DIR = "data"
os.makedirs(OUT_DIR, exist_ok=True)

UA = "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120 Safari/537.36"
HEADERS_JSON = {"User-Agent": UA, "Accept": "application/json, text/plain, */*"}


# -------- Utilities --------
def clean(s: Optional[str]) -> str:
    """Collapse whitespace and trim."""
    return re.sub(r"\s+", " ", s).strip() if s else ""


def slugify_for_path(title: str) -> str:
    """
    Create a filesystem-safe slug:
    - spaces -> underscore
    - keep dashes '-' and parentheses '()'
    - keep Arabic letters
    - drop other punctuation
    """
    if not title:
        return ""
    s = re.sub(r"\s+", "_", title.strip())
    s = re.sub(r"[^A-Za-z0-9\u0621-\u064A\-\(\)_]", "", s)
    s = re.sub(r"_+", "_", s).strip("_")
    return s


def get_json(url: str) -> Any:
    """GET JSON with minimal headers and errors raised on non-2xx."""
    r = requests.get(url, headers=HEADERS_JSON, timeout=30)
    r.raise_for_status()
    return r.json()


# -------- Job list (robust) --------
def fetch_job_list() -> List[Dict[str, Any]]:
    """
    Return a list of job dicts.
    Handles multiple possible JSON shapes: {content:[...]}, {jobs:[...]}, {data:[...]}, or a bare list.
    Ignores non-dict items defensively.
    """
    data = get_json(EMPLOYER_API)

    candidates: List[Any] = []
    if isinstance(data, dict):
        for v in data.values():
            if isinstance(v, list):
                candidates.extend(v)
    elif isinstance(data, list):
        candidates = data

    items: List[Dict[str, Any]] = [x for x in candidates if isinstance(x, dict)]

    jobs: List[Dict[str, Any]] = []
    for j in items:
        jobs.append({
            "id": str(j.get("id") or j.get("jobId") or ""),
            "title": j.get("title") or j.get("jobTitle") or "",
            "url": (j.get("url") or j.get("jobUrl") or j.get("applyUrl") or "") or "",
            "location": j.get("location") or j.get("city") or "",
            "department": j.get("department") or "",
            "job_type": j.get("jobType") or j.get("employmentType") or "",
            "years": j.get("experience") or j.get("exp") or "",
        })

    return jobs


def canonical_job_url(job: Dict[str, Any]) -> str:
    """
    Build a canonical, stable job URL the site accepts:
    /jobs/job_<id>_<slug>, URL-encoded for safety.
    """
    jid = job.get("id") or ""
    title = job.get("title") or ""
    if not jid:
        return job.get("url") or ""
    slug = slugify_for_path(title)
    path = f"/jobs/job_{jid}_{slug}"
    return BASE + quote(path, safe="/_()-")


# -------- Selenium setup --------
def launch(headless: bool = True):
    """Launch a Chrome WebDriver with sensible flags for scraping."""
    opts = Options()
    if headless:
        opts.add_argument("--headless=new")
    opts.add_argument("--no-sandbox")
    opts.add_argument("--disable-dev-shm-usage")
    opts.add_argument("--window-size=1440,2400")
    opts.add_argument("--lang=en-US")
    opts.add_argument("user-agent=" + UA)
    # Reduce automation fingerprints
    opts.add_experimental_option("excludeSwitches", ["enable-automation"])
    opts.add_experimental_option("useAutomationExtension", False)

    drv = webdriver.Chrome(service=Service(ChromeDriverManager().install()), options=opts)
    # Hide webdriver flag where supported
    try:
        drv.execute_script("Object.defineProperty(navigator, 'webdriver', {get: () => undefined})")
    except Exception:
        pass
    return drv


def wait_any_of(driver, timeout: int, locators: List[tuple]):
    """
    Wait until any of the provided locators is present.
    Uses EC.any_of if available (Selenium >= 4.8); otherwise merges CSS selectors or falls back to the first condition.
    """
    conditions = [EC.presence_of_element_located(loc) for loc in locators]
    try:
        any_of = getattr(EC, "any_of")  # Selenium >= 4.8
        WebDriverWait(driver, timeout).until(any_of(*conditions))
    except AttributeError:
        css_only = [sel for by, sel in locators if by == By.CSS_SELECTOR]
        if css_only:
            merged_css = ", ".join(css_only)
            WebDriverWait(driver, timeout).until(
                EC.presence_of_element_located((By.CSS_SELECTOR, merged_css))
            )
        else:
            WebDriverWait(driver, timeout).until(conditions[0])


# -------- Detail page scraper --------
def scrape_detail(driver, url: str) -> Dict[str, Any]:
    """
    Return a flat dict with:
      - skills_required: list[str]
      - dynamic keys for each heading section (derived from strong/b/h2/h3)
      - fallback 'details' list if no sections could be parsed
    """
    result: Dict[str, Any] = {"skills_required": []}

    def norm_txt(t: str) -> str:
        return re.sub(r"\s+", " ", t or "").strip()

    def unique_key(base: str, existing: Dict[str, Any]) -> str:
        """Create a unique dict key from a heading text."""
        k = slugify_for_path((base or "").rstrip(":："))
        if not k or k in existing:
            i = 2
            kk = k or "section"
            while kk in existing:
                kk = f"{k or 'section'}_{i}"
                i += 1
            k = kk
        return k

    try:
        driver.get(url)

        # Wait for any plausible content
        wait_any_of(
            driver,
            30,
            [
                (By.CSS_SELECTOR, "h1"),
                (By.XPATH, "//*[contains(normalize-space(.),'Skills Required')]"),
                (By.XPATH, "//ul/li"),
                (By.XPATH, "//ol/li"),
            ],
        )

        # Skills tags (best-effort)
        skills = []
        try:
            skills_header = driver.find_elements(
                By.XPATH,
                "//*[self::h2 or self::h3 or self::h4 or self::p or self::strong or self::b]"
                "[contains(translate(normalize-space(.),' :',''), 'SKILLS REQUIRED')]"
            )
            if skills_header:
                block = skills_header[0].find_element(By.XPATH, "following-sibling::*[1]")
                skill_nodes = block.find_elements(
                    By.XPATH,
                    ".//*[self::span or self::a or self::div]"
                    "[contains(@class,'tag') or contains(@class,'badge') or "
                    " contains(@class,'chip') or contains(@class,'pill') or "
                    " contains(@class,'ant-tag') or string-length(normalize-space(.))>0]"
                )
                skills = [norm_txt(e.text) for e in skill_nodes if norm_txt(e.text)]
            if not skills:
                skill_nodes = driver.find_elements(
                    By.XPATH,
                    "//*[self::span or self::a or self::div]"
                    "[contains(@class,'ant-tag') or contains(@class,'tag') or "
                    " contains(@class,'badge') or contains(@class,'chip') or contains(@class,'pill')]"
                )
                skills = [norm_txt(e.text) for e in skill_nodes if norm_txt(e.text)]
        except Exception:
            pass

        # de-duplicate while preserving order
        seen = set()
        skills_unique = []
        for s in skills:
            if s not in seen:
                seen.add(s)
                skills_unique.append(s)
        result["skills_required"] = skills_unique

        # Headings (h2/h3/strong/b) -> section keys
        heading_nodes = []
        heading_nodes += driver.find_elements(By.XPATH, "//h2[normalize-space()] | //h3[normalize-space()]")
        heading_nodes += driver.find_elements(By.XPATH, "//strong[normalize-space()] | //b[normalize-space()]")

        # remove duplicates while preserving DOM order
        seen_ids = set()
        ordered_heads = []
        for node in heading_nodes:
            _id = node._id if hasattr(node, "_id") else id(node)
            if _id not in seen_ids:
                seen_ids.add(_id)
                ordered_heads.append(node)

        for h in ordered_heads:
            heading_text = norm_txt(h.text)
            if not heading_text:
                continue

            tag = h.tag_name.lower()
            container = h
            if tag in ("strong", "b"):
                # Use nearest paragraph/div as the container if strong/b is nested
                try:
                    container = h.find_element(By.XPATH, "ancestor::p[1]")
                except Exception:
                    try:
                        container = h.find_element(By.XPATH, "ancestor::div[1]")
                    except Exception:
                        container = h

            # Collect following siblings until a new heading-like node appears
            items = []
            sib = None
            try:
                sib = container.find_element(By.XPATH, "following-sibling::*[1]")
            except Exception:
                pass

            while sib:
                st = sib.tag_name.lower()
                is_new_heading = False
                if st in ("h2", "h3"):
                    is_new_heading = True
                else:
                    try:
                        has_bold = sib.find_elements(By.XPATH, ".//strong[normalize-space()] | .//b[normalize-space()]")
                        if has_bold:
                            is_new_heading = True
                    except Exception:
                        pass

                if is_new_heading:
                    break

                if st in ("ul", "ol"):
                    for li in sib.find_elements(By.XPATH, ".//li"):
                        t = norm_txt(li.text)
                        if t:
                            items.append(t)
                elif st in ("p", "div", "span"):
                    t = norm_txt(sib.text)
                    if t and len(t) > 2:
                        items.append(t)

                nxt = None
                try:
                    nxt = sib.find_element(By.XPATH, "following-sibling::*[1]")
                except Exception:
                    pass
                sib = nxt

            if items:
                key = unique_key(heading_text, result)
                result[key] = items

        # Fallback: populate 'details' from bullets if no sections were detected
        if len([k for k in result.keys() if k != "skills_required"]) == 0:
            bullets = driver.find_elements(By.XPATH, "//ul/li | //ol/li")
            items = [norm_txt(b.text) for b in bullets if norm_txt(b.text)]
            if items:
                result["details"] = items

        return result

    except Exception:
        # Silent failure with empty minimal structure
        return result


# -------- Main --------
def main():
    print("Fetching job list ...")
    jobs = fetch_job_list()
    print(f"{len(jobs)} jobs found")

    driver = launch(headless=True)
    results = []
    try:
        for j in jobs:
            if not isinstance(j, dict):
                continue
            detail_url = canonical_job_url(j)
            parsed = urlparse(detail_url)
            detail_url = f"{parsed.scheme}://{parsed.netloc}{quote(parsed.path, safe='/_()-')}"
            detail = scrape_detail(driver, detail_url)
            merged = {**j, **detail}
            results.append(merged)
    finally:
        try:
            driver.quit()
        except Exception:
            pass

    out_path = os.path.join(OUT_DIR, "careers.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    print(f"Saved {len(results)} jobs -> {out_path}")


if __name__ == "__main__":
    main()
