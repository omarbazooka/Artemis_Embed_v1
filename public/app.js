const $ = (id) => document.getElementById(id);

function toast(message, error = false) {
  const el = $("toast");
  el.textContent = message;
  el.classList.toggle("error", error);
  el.classList.add("show");
  clearTimeout(window.__toastTimer);
  window.__toastTimer = setTimeout(() => el.classList.remove("show"), 3500);
}

async function api(path, options = {}) {
  const response = await fetch(path, options);
  let body;
  try { body = await response.json(); } catch { body = { detail: await response.text() }; }
  if (!response.ok) {
    const detail = typeof body.detail === "string" ? body.detail : JSON.stringify(body.detail || body);
    throw new Error(detail || `Request failed (${response.status})`);
  }
  return body;
}

function setBusy(button, busy, text) {
  if (!button.dataset.label) button.dataset.label = button.textContent;
  button.disabled = busy;
  button.textContent = busy ? text : button.dataset.label;
}

function renderResults(container, results, contentKey = "text") {
  container.innerHTML = "";
  if (!results || !results.length) {
    container.innerHTML = '<div class="result-card"><p>No results.</p></div>';
    return;
  }
  results.forEach((item, index) => {
    const card = document.createElement("article");
    card.className = "result-card";
    const score = Number(item.score ?? item.similarity ?? 0);
    const rank = item.rank ?? index + 1;
    const text = item[contentKey] ?? item.content ?? "";
    card.innerHTML = `<header><strong>#${rank}</strong><span>cosine ${score.toFixed(4)}</span></header><p></p>`;
    card.querySelector("p").textContent = text;
    container.appendChild(card);
  });
}

document.querySelectorAll(".tab").forEach((button) => {
  button.addEventListener("click", () => {
    document.querySelectorAll(".tab").forEach((b) => b.classList.remove("active"));
    document.querySelectorAll(".tab-panel").forEach((p) => p.classList.remove("active"));
    button.classList.add("active");
    $(`tab-${button.dataset.tab}`).classList.add("active");
  });
});

$("compare-btn").addEventListener("click", async () => {
  const button = $("compare-btn");
  setBusy(button, true, "Embedding…");
  try {
    const result = await api("/api/similarity", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        text_a: $("text-a").value,
        text_b: $("text-b").value,
        dimension: Number($("similarity-dimension").value),
      }),
    });
    $("similarity-score").textContent = Number(result.similarity).toFixed(4);
    $("similarity-meta").textContent = `${result.dimension}D normalized vectors`;
    $("similarity-result").classList.remove("hidden");
  } catch (error) { toast(error.message, true); }
  finally { setBusy(button, false); }
});

$("search-btn").addEventListener("click", async () => {
  const button = $("search-btn");
  const documents = $("search-docs").value.split(/\n+/).map((x) => x.trim()).filter(Boolean);
  setBusy(button, true, "Searching…");
  try {
    const result = await api("/api/search", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        query: $("search-query").value,
        documents,
        dimension: Number($("search-dimension").value),
        top_k: Number($("search-topk").value),
      }),
    });
    renderResults($("search-results"), result.results, "text");
  } catch (error) { toast(error.message, true); }
  finally { setBusy(button, false); }
});

async function checkStatus() {
  try {
    const info = await api("/api/model");
    const dot = $("documents-dot");
    if (info.documents_enabled) {
      dot.classList.add("ok");
      $("documents-status").textContent = `Supabase + Artemis ready · ${info.document_dimension}D index`;
    } else {
      dot.classList.add("bad");
      $("documents-status").textContent = "Document retrieval needs Artemis remote inference + Supabase configuration";
    }
  } catch {
    $("documents-dot").classList.add("bad");
    $("documents-status").textContent = "Service configuration unavailable";
  }
}
checkStatus();

$("document-file").addEventListener("change", () => {
  const file = $("document-file").files[0];
  $("upload-info").textContent = file ? `${file.name} · ${(file.size / 1024).toFixed(1)} KB` : "";
});

$("upload-btn").addEventListener("click", async () => {
  const file = $("document-file").files[0];
  if (!file) return toast("Choose a PDF, TXT or MD file first.", true);
  const button = $("upload-btn");
  const form = new FormData();
  form.append("file", file);
  setBusy(button, true, "Chunking & indexing…");
  try {
    const result = await api("/api/documents/upload", { method: "POST", body: form });
    $("document-id").value = result.document_id;
    $("upload-info").textContent = `${result.name} · ${result.chunk_count} chunks · ${result.dimension}D · ready`;
    toast("Document indexed successfully.");
  } catch (error) { toast(error.message, true); }
  finally { setBusy(button, false); }
});

$("document-query-btn").addEventListener("click", async () => {
  const button = $("document-query-btn");
  setBusy(button, true, "Retrieving…");
  try {
    const documentId = $("document-id").value.trim();
    const result = await api("/api/documents/query", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        document_id: documentId || null,
        query: $("document-query").value,
        top_k: Number($("document-topk").value),
      }),
    });
    renderResults($("document-results"), result.results, "content");
  } catch (error) { toast(error.message, true); }
  finally { setBusy(button, false); }
});
