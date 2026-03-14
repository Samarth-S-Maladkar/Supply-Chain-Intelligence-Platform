/**
 * epoch/frontend/app.js
 * ----------------------
 * Shared frontend utilities. Imported by producer.html and consumer.html.
 * All page-specific logic lives in each HTML file's <script> block.
 */

/** Format a number with locale commas */
function fmtNumber(n, decimals = 0) {
  return Number(n).toLocaleString(undefined, {
    minimumFractionDigits: decimals,
    maximumFractionDigits: decimals,
  });
}

/** Format currency */
function fmtCurrency(n) {
  return '$' + fmtNumber(n, 0);
}

/** Show a toast notification */
function showToast(msg, type = '') {
  const t = document.getElementById('toast');
  if (!t) return;
  t.textContent = msg;
  t.className   = `show ${type}`;
  setTimeout(() => (t.className = ''), 3500);
}

/** Build a FormData with the given file under key "file" */
function buildFormData(file) {
  const fd = new FormData();
  fd.append('file', file);
  return fd;
}

/** Generic fetch wrapper with error handling */
async function apiFetch(url, options = {}) {
  const res = await fetch(url, options);
  if (!res.ok) {
    const err = await res.json().catch(() => ({ error: res.statusText }));
    throw new Error(err.error || `HTTP ${res.status}`);
  }
  return res.json();
}

/** Upload CSV to an endpoint and return parsed JSON */
async function uploadCSV(endpoint, file) {
  return apiFetch(endpoint, {
    method: 'POST',
    body: buildFormData(file),
  });
}
