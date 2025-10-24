const DEFAULT_CAREER_URL = "https://career.kddc.com";

/**
 * Fill the main query input with a sample text and focus it.
 */
function fill(text){
  const el = document.getElementById('a_q');
  if(!el) return;
  el.value = text;
  el.focus();
}

/**
 * Append a chat bubble to the chat area.
 * @param {string} html - bubble inner HTML (already formatted/escaped as needed)
 * @param {'user'|'assistant'} who - bubble side/style
 * @returns {HTMLElement} the created bubble element
 */
function addMsg(html, who='assistant'){
  // who: 'user' | 'assistant'
  const wrap = document.createElement('div');
  wrap.className = 'msg ' + (who === 'user' ? 'user' : 'assistant');
  wrap.innerHTML = html;
  const chat = document.getElementById('chat');
  if (chat) {
    chat.appendChild(wrap);
    // keep the latest message in view
    chat.scrollTop = chat.scrollHeight;
  }
  return wrap;
}

/**
 * Render a single "source" as a clickable chip.
 * Falls back to DEFAULT_CAREER_URL when URL is missing.
 * NOTE: Assumes title/url from server are trusted. If not, escape them first.
 */
function toChip(source){
  const score = (source.score!=null) ? ` (${Number(source.score).toFixed(3)})` : '';
  const title = source.title || 'Source';
  const href = (source.url && String(source.url).trim()) ? source.url : DEFAULT_CAREER_URL;
  return `<span class="chip"><a href="${href}" target="_blank" rel="noopener">${title}</a>${score}</span>`;
}

/**
 * Render a list of sources as a chips container.
 */
function renderSources(list){
  if(!list || !list.length) return '';
  const chips = list.map(toChip).join(' ');
  return `<div class="sources">${chips}</div>`;
}

let lastQ = '';   // store last user query to support ArrowUp recall

/**
 * Main "Ask" handler:
 *  - Reads input
 *  - Posts to /v1/ask
 *  - Renders assistant reply + sources
 *  - Handles loading/error states
 */
async function ask(){
  const input = document.getElementById('a_q');
  const btn   = document.getElementById('ask_btn');
  const kSel  = document.getElementById('a_k');

  if(!input || !btn){ return; }

  const k = kSel ? Number(kSel.value) : 3;
  const q = (input.value || '').trim();
  if(!q) return;

  // Show user bubble and reset input
  lastQ = q;
  addMsg(q, 'user');
  input.value = '';
  btn.disabled = true;

  // Show a lightweight "thinking..." bubble
  const thinking = addMsg('<span class="spinner"></span> thinking…', 'assistant');

  try{
    // Send both q and query for compatibility with different API versions
    const payload = { q, query: q, k, max_tokens: 512 };

    const res = await fetch('/v1/ask', {
      method: 'POST',
      headers: {'Content-Type':'application/json'},
      body: JSON.stringify(payload)
    });

    let data;
    // Try JSON first; if it fails, surface raw text with status
    try {
      data = await res.json();
    } catch {
      const txt = await res.text();
      throw new Error(`HTTP ${res.status}: ${txt}`);
    }

    if(!res.ok){
      // Surface server errors in the chat
      throw new Error(data?.detail || JSON.stringify(data));
    }

    // Server contract: { answer: string, sources: [...] }
    const answer  = (data.answer || '—').trim();
    const sources = renderSources(data.sources || data.context || []);

    // Minimal XSS-safe rendering for the answer body:
    //  - Escape & and < (most common vectors)
    //  - Convert newlines to <br>
    // NOTE: Titles/URLs in sources are inserted as-is; ensure backend sends safe strings.
    const safeAnswer = answer
      .replace(/&/g, '&amp;')   // must escape first
      .replace(/</g, '&lt;')    // prevent HTML injection
      .replace(/\n/g, '<br>');  // preserve line breaks

    thinking.innerHTML = `<div class="text">${safeAnswer}</div>${sources}`;

  }catch(err){
    // Replace the "thinking" bubble with an error message
    if(thinking) thinking.innerHTML = 'Error: ' + (err?.message || err);
  }finally{
    // Re-enable the button and keep view scrolled to bottom
    btn.disabled = false;
    const chat = document.getElementById('chat');
    if (chat) chat.scrollTop = chat.scrollHeight;
  }
}

/**
 * Wire up events after DOM is ready:
 *  - Click on Ask button
 *  - Keyboard shortcuts for the input:
 *      Enter       => submit
 *      ArrowUp     => recall last query
 */
window.addEventListener('DOMContentLoaded', () => {
  const qInput = document.getElementById('a_q');
  const btn    = document.getElementById('ask_btn');

  // Avoid double-binding on hot reloads
  if(btn && !btn._bound){ btn.addEventListener('click', ask); btn._bound = true; }

  if(qInput && !qInput._bound){
    qInput.addEventListener('keydown', (e) => {
      // Recall last query with ArrowUp (if no modifiers)
      if(e.key === 'ArrowUp' && !e.shiftKey && !e.ctrlKey && !e.metaKey){
        if(lastQ){
          e.preventDefault();
          qInput.value = lastQ;
          qInput.setSelectionRange(lastQ.length,lastQ.length);
        }
      }
      // Submit on Enter (unless Shift is pressed)
      if(e.key === 'Enter' && !e.shiftKey){
        e.preventDefault();
        ask();
      }
    });
    qInput._bound = true;
  }
});
