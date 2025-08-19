import streamlit as st
from pathlib import Path
import json
import faiss
from sentence_transformers import SentenceTransformer
import requests
import base64
import os
from typing import Any, Dict, List, Optional
from utils.components import app_header, hide_default_pages_nav
from utils.auth import require_auth, logout_button

st.set_page_config(page_title="Policy Navigator  • Admin", page_icon="❓", layout="wide")
hide_default_pages_nav()
require_auth()

with st.sidebar:
    st.image("assets/shelflytics_logo_transparent_white.png")

    st.page_link("pages/1_Home.py", label="🏠 Home")
    logout_button()

    st.divider()
    st.markdown("**U.S.**")
    st.page_link("pages/2_SKUs.py", label="📦 SKUs")
    st.page_link("pages/3_Outlets.py", label="🏬 Outlets")
    st.page_link("pages/4_SKU_Recommender.py", label="🎁 SKU Recommender")
    st.page_link("pages/6_Routes.py", label="🗺️ Route Optimiser")
    st.page_link("pages/7_Merchandisers.py", label="🧑‍🤝‍🧑 Merchandisers")

    st.divider()
    st.markdown("**China**")
    st.page_link("pages/chatbot_page.py", label="💬 Chatbot")
    st.page_link("pages/predict_page.py", label="📈 Predict Item Performance")

    st.divider()
    st.page_link("pages/sku_detection.py", label="👁️ Detector") 
    st.page_link("pages/policy_faq.py", label="❓ Policy FAQ")
    st.page_link("pages/5_Settings.py", label="⚙️ Settings")


def _make_serializable(obj: Any) -> Any:
    if isinstance(obj, dict):
        return { _make_serializable(k): _make_serializable(v) for k, v in obj.items() }
    if isinstance(obj, (list, tuple)):
        return [ _make_serializable(v) for v in obj ]
    if obj is None or isinstance(obj, (str, bool, int, float)):
        return obj
    try:
        if hasattr(obj, "item") and callable(getattr(obj, "item")):
            return obj.item()
    except Exception:
        pass
    try:
        return str(obj)
    except Exception:
        return None


@st.cache_resource
def load_resources(index_dir: str):
    idx_path = Path(index_dir) / "faiss.index"
    meta_path = Path(index_dir) / "metadata.json"
    if not idx_path.exists() or not meta_path.exists():
        raise FileNotFoundError(f"Index or metadata not found in {index_dir}")
    index = faiss.read_index(str(idx_path))
    with open(meta_path, "r", encoding="utf8") as f:
        metadata = json.load(f)
    embed_model = SentenceTransformer("all-MiniLM-L6-v2")
    return index, metadata, embed_model


@st.cache_data
def read_pdf_bytes(path: str) -> Optional[bytes]:
    p = Path(path)
    if not p.exists():
        return None
    try:
        return p.read_bytes()
    except Exception:
        return None


def build_prompt(snippets: List[str], question: str) -> str:
    system = "You are a helpful internal policy assistant. Use only the provided context to answer. If answer not present, say 'I don't know - consult the policies.'"
    context = "\n\n".join([f"[{i+1}] {s}" for i, s in enumerate(snippets)])
    user_message = f"Context:\n{context}\n\nQuestion: {question}\n\n. Answer concisely."
    return f"{system}\n\n{user_message}"


def call_generative_api(policy_api_key: str, ai_model: str, prompt: str, temperature: float = 0.0, max_tokens: int = 512) -> str:
    url = f"https://generativelanguage.googleapis.com/v1beta/models/{ai_model}:generateContent?key={policy_api_key}"
    payload = {
        "contents": [{"parts": [{"text": prompt}]}],
        "generationConfig": {"temperature": temperature, "maxOutputTokens": max_tokens}
    }
    resp = requests.post(url, json=payload, timeout=30)
    resp.raise_for_status()
    data = resp.json()
    # Parse common response shape
    if "candidates" in data and data["candidates"]:
        c = data["candidates"][0]
        if "content" in c and "parts" in c["content"]:
            return c["content"]["parts"][0].get("text", "").strip()
    # Fallback safe parsing
    if "output" in data:
        if isinstance(data["output"], list):
            parts = []
            for o in data["output"]:
                if isinstance(o, dict):
                    parts.append(o.get("content", ""))
            return " ".join(parts).strip()
    return "Could not parse a valid answer from the model's response."


def sanitize_source_to_pdf(pdfs_dir: str, source: str) -> Optional[Path]:
    # Use basename to avoid directory traversal via metadata
    filename = os.path.basename(source or "")
    if not filename:
        return None
    candidate = Path(pdfs_dir) / filename
    try:
        # Ensure the candidate is inside the pdfs_dir
        if Path(candidate).resolve().is_relative_to(Path(pdfs_dir).resolve()):
            return candidate
    except Exception:
        # older Python may not have is_relative_to
        try:
            if str(candidate.resolve()).startswith(str(Path(pdfs_dir).resolve())):
                return candidate
        except Exception:
            return None
    return None


def relevance_label(score: Any) -> str:
    """Map a numeric similarity score to a human-friendly relevance label.

    - >= 0.75 -> high relevance
    - >= 0.50 -> medium relevance
    - otherwise -> low relevance
    """
    try:
        s = float(score)
    except Exception:
        return "unknown relevance"
    if s >= 0.75:
        return "high relevance"
    if s >= 0.5:
        return "medium relevance"
    return "low relevance"


def main():
    app_header("Policy Navigator")
    st.write("Ask internal policy questions. Answers use only the provided policy documents and cite sources.")

    # Read config from secrets
    policy = st.secrets.get("policy", {})
    policy_api_key = policy.get("policy_api_key", "AIzaSyAmiVFuDdfSsNdztrx1tC4dEGhmJBQhLm8")
    ai_model = policy.get("ai_model", "gemini-2.0-flash")
    index_dir = policy.get("index_dir", "docs/index")
    pdfs_dir = policy.get("pdfs_dir", "docs/pdfs")

    if not policy_api_key or not ai_model:
        st.error("Policy API key or AI model not configured in .streamlit/secrets.toml under [policy]")
        st.stop()

    try:
        index, metadata, embed_model = load_resources(index_dir)
    except Exception as e:
        st.error(f"Failed to load resources: {e}")
        st.stop()

    question = st.text_area("Question", height=140)
    ask = st.button("Ask")

    if ask:
        q = question.strip()
        if not q:
            st.warning("Enter a question first.")
            return

        with st.spinner("Retrieving relevant policy snippets..."):
            try:
                qvec = embed_model.encode([q], convert_to_numpy=True).astype("float32")
            except Exception as e:
                st.error(f"Embedding error: {e}")
                return
            faiss.normalize_L2(qvec)
            k = 3
            D, I = index.search(qvec, k)
            results: List[Dict[str, Any]] = []
            snippets: List[str] = []
            for score, idx in zip(D[0], I[0]):
                if idx < 0:
                    continue
                m = metadata[int(idx)]
                text = m.get("text", "")
                if len(text) > 800:
                    preview = text[:800] + "..."
                else:
                    preview = text
                snippets.append(f"[{m.get('source')}] {preview}")
                results.append({
                    "source": m.get("source"),
                    "chunk_id": int(idx),
                    "score": float(score),
                    "text": preview,
                })

        prompt = build_prompt(snippets, q)

        with st.spinner("Generating answer from model..."):
            try:
                answer = call_generative_api(policy_api_key, ai_model, prompt, temperature=0.0, max_tokens=512)
            except requests.exceptions.RequestException as e:
                st.error(f"API request failed: {e}")
                return
            except Exception as e:
                st.error(f"LLM error: {e}")
                return

        st.subheader("Answer")
        st.write(answer)

        st.subheader("Cited sources")

        # Group results by source file so multiple chunks from the same PDF are shown together
        grouped: Dict[str, List[Dict[str, Any]]] = {}
        for r in results:
            src = r.get("source") or "Unknown"
            file_name = Path(str(src)).stem.strip("[]")
            grouped.setdefault(file_name, []).append(r)

        # sort groups by highest chunk score descending
        groups_sorted = sorted(
            grouped.items(),
            key=lambda kv: max((float(x.get("score", 0)) for x in kv[1])),
            reverse=True,
        )

        for file_name, chunks in groups_sorted:
            # compute aggregate relevance from the best chunk
            scores = [float(c.get("score", 0)) for c in chunks]
            best_score = max(scores) if scores else 0.0
            rel = relevance_label(best_score)

            if isinstance(rel, str) and rel.startswith("high"):
                emoji = "🟢"
            elif isinstance(rel, str) and rel.startswith("medium"):
                emoji = "🟡"
            elif isinstance(rel, str) and rel.startswith("low"):
                emoji = "🔴"
            else:
                emoji = "⚪️"

            header = f"{file_name} ({len(chunks)} chunk{'s' if len(chunks) != 1 else ''}) - {emoji} {rel}"
            with st.expander(header, expanded=False):
                st.caption(f"Aggregated from {len(chunks)} snippet(s). Best similarity: {best_score:.3f}")

                # show each chunk with its own mini-header so users can distinguish them
                for c in sorted(chunks, key=lambda x: float(x.get("score", 0)), reverse=True):
                    cid = c.get("chunk_id")
                    score = float(c.get("score", 0))
                    st.markdown(f"**Chunk {cid} — Score: {score:.3f}**")
                    st.write(c.get("text", ""))

                    try:
                        pct = max(0, min(100, int(score * 100)))
                        st.progress(pct)
                        st.caption(f"Similarity score: {score:.3f} ({pct}%)")
                    except Exception:
                        st.caption("Similarity score: unknown")

                # offer the PDF download once per file
                pdf_path = sanitize_source_to_pdf(pdfs_dir, f"{file_name}.pdf")
                if pdf_path and pdf_path.exists():
                    pdf_bytes = read_pdf_bytes(str(pdf_path))
                    if pdf_bytes:
                        st.download_button(
                            label=f"Download {Path(pdf_path).name}",
                            data=pdf_bytes,
                            file_name=Path(pdf_path).name,
                            mime="application/pdf",
                        )
                    else:
                        st.info("PDF exists but could not be read")
                else:
                    st.info("PDF not found for this source")


if __name__ == "__main__":
    main()
