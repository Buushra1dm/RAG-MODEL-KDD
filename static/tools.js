const DEFAULT_CAREER_URL = "https://career.kddc.com";

/* -----------------------------
   Helpers (loading & rendering)
   ----------------------------- */

/**
 * Toggle a loading state inside a result box and (optionally) disable a button.
 * Shows a spinner while loading.
 */
function setLoading(box, btn, loading = true) {
  if (loading) {
    if (btn) btn.disabled = true;
    box.innerHTML = '<div class="empty"><span class="spinner"></span> Loading…</div>';
  } else {
    if (btn) btn.disabled = false;
  }
}

/**
 * Render a neutral "empty" placeholder message in a box.
 */
function emptyState(box, text = 'No results found') {
  box.innerHTML = '<div class="empty">' + text + '</div>';
}

/**
 * Render an error message in a box.
 */
function errorState(box, text = 'Something went wrong') {
  box.innerHTML = '<div class="error">❌ ' + text + '</div>';
}

/**
 * Normalize response shapes across different endpoints.
 * Supports common keys: data/items/results/rows.
 */
function extractRows(data) {
  return data?.data || data?.items || data?.results || data?.rows || [];
}

/* -----------------------------
   Card renderers (HTML strings)
   ----------------------------- */

/**
 * Render one product as a card.
 */
function renderProductCard(it) {
  const price = (it?.price_now_kwd != null) ? `${it.price_now_kwd} KWD` : '—';
  const url = (it?.product_url)
    ? `<div class="chips"><span class="chip"><a href="${it.product_url}" target="_blank" rel="noopener">View Product</a></span></div>`
    : '';
  return `
    <div class="card">
      <div class="title">${it?.name || '-'}</div>
      <div class="meta">${it?.brand || ''} • ${it?.category || ''} • ${it?.flavor || ''}</div>
      <div class="price">Price: <strong>${price}</strong></div>
      ${url}
    </div>`;
}

/**
 * Render one career as a card.
 */
function renderCareerCard(it) {
  const yrs = (it?.min_years != null ? it.min_years : '—') + (it?.max_years != null ? `–${it.max_years}` : '');
  const link = (it?.url && String(it.url).trim()) ? it.url : DEFAULT_CAREER_URL;
  return `
    <div class="card">
      <div class="title">${it?.title || '-'}</div>
      <div class="meta">${it?.department_name || ''} • ${it?.location || ''}</div>
      <div class="meta" style="margin-top:4px">Experience: ${yrs || '—'} years</div>
      <div class="chips" style="margin-top:8px">
        <span class="chip"><a href="${link}" target="_blank" rel="noopener">View Details</a></span>
      </div>
    </div>`;
}

/**
 * Render one semantic search hit as a card (product or career).
 */
function renderSemanticCard(r) {
  const href = (r?.url && String(r.url).trim()) ? r.url : (r?.type === 'career' ? DEFAULT_CAREER_URL : '#');
  const score = (r?.score != null) ? Number(r.score).toFixed(3) : '—';
  return `
    <div class="card">
      <div class="title">${r?.title || '-'}</div>
      <div class="meta">${r?.type || ''} • score: ${score}</div>
      <div class="meta" style="margin-top:6px">${(r?.snippet || '').slice(0, 220)}</div>
      <div class="chips"><span class="chip"><a href="${href}" target="_blank" rel="noopener">Open Source</a></span></div>
    </div>`;
}

/* -----------------------------
   API calls and page wiring
   ----------------------------- */

/**
 * Load products via GET /v1/products using current filters.
 * Renders a list of product cards or an empty/error state.
 */
async function loadProducts() {
  const box = document.getElementById('p_res');
  const btn = document.getElementById('p_btn');
  setLoading(box, btn, true);
  try {
    const q = document.getElementById('p_q').value.trim();
    const cat = document.getElementById('p_cat').value.trim();
    const pmin = document.getElementById('p_price_min').value.trim();
    const pmax = document.getElementById('p_price_max').value.trim();

    const url = new URL('/v1/products', location.origin);
    if (q) url.searchParams.set('q', q);
    if (cat) url.searchParams.set('category', cat);
    if (pmin) url.searchParams.set('price_min', pmin);
    if (pmax) url.searchParams.set('price_max', pmax);

    const res = await fetch(url, { headers: { 'Accept': 'application/json' } });
    const text = await res.text();
    if (!res.ok) throw new Error(`HTTP ${res.status} - ${text}`);

    let data;
    try { data = JSON.parse(text); } catch { throw new Error('Invalid JSON: ' + text.slice(0, 200)); }

    const rows = extractRows(data);
    if (!rows.length) { emptyState(box); return; }
    box.innerHTML = rows.map(renderProductCard).join('');
  } catch (e) {
    console.error('[Products] error:', e);
    errorState(box, e.message || e);
  } finally {
    setLoading(box, btn, false);
  }
}

/**
 * Load careers via GET /v1/careers using current filters.
 * Renders a list of career cards or an empty/error state.
 */
async function loadCareers() {
  const box = document.getElementById('c_res');
  const btn = document.getElementById('c_btn');
  setLoading(box, btn, true);
  try {
    const q = document.getElementById('c_q').value.trim();
    const dep = document.getElementById('c_dep').value.trim();
    const loc = document.getElementById('c_loc').value.trim();

    const url = new URL('/v1/careers', location.origin);
    if (q) url.searchParams.set('q', q);
    if (dep) url.searchParams.set('department', dep);
    if (loc) url.searchParams.set('location', loc);

    const res = await fetch(url, { headers: { 'Accept': 'application/json' } });
    const text = await res.text();
    if (!res.ok) throw new Error(`HTTP ${res.status} - ${text}`);

    let data;
    try { data = JSON.parse(text); } catch { throw new Error('Invalid JSON: ' + text.slice(0, 200)); }

    const rows = extractRows(data);
    if (!rows.length) { emptyState(box); return; }
    box.innerHTML = rows.map(renderCareerCard).join('');
  } catch (e) {
    console.error('[Careers] error:', e);
    errorState(box, e.message || e);
  } finally {
    setLoading(box, btn, false);
  }
}

/**
 * Execute semantic search via GET /v1/search?q=&k=
 * Renders mixed product/career hits as cards.
 */
async function doSemantic() {
  const box = document.getElementById('s_res');
  const btn = document.getElementById('s_btn');
  setLoading(box, btn, true);
  try {
    const q = document.getElementById('s_q').value.trim();
    const k = document.getElementById('s_k').value;
    if (!q) { emptyState(box, 'Please enter a question'); return; }

    const url = new URL('/v1/search', location.origin);
    url.searchParams.set('q', q);
    url.searchParams.set('k', k);

    const res = await fetch(url, { headers: { 'Accept': 'application/json' } });
    const text = await res.text();
    if (!res.ok) throw new Error(`HTTP ${res.status} - ${text}`);

    let data;
    try { data = JSON.parse(text); } catch { throw new Error('Invalid JSON: ' + text.slice(0, 200)); }

    const rows = extractRows(data);
    if (!rows.length) { emptyState(box); return; }
    box.innerHTML = rows.map(renderSemanticCard).join('');
  } catch (e) {
    console.error('[Semantic] error:', e);
    errorState(box, e.message || e);
  } finally {
    setLoading(box, btn, false);
  }
}

/* Make functions available to inline onclick handlers in HTML */
window.loadProducts = loadProducts;
window.loadCareers  = loadCareers;
window.doSemantic   = doSemantic;

/* Fallback wiring if you remove inline onclick later */
window.addEventListener('DOMContentLoaded', () => {
  document.getElementById('p_btn')?.addEventListener('click', loadProducts);
  document.getElementById('c_btn')?.addEventListener('click', loadCareers);
  document.getElementById('s_btn')?.addEventListener('click', doSemantic);
});
