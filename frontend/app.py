"""
frontend/app.py
===============
Stateless Streamlit UI — two tabs:
  1. 📥 Ingest     — upload **multiple** documents into Qdrant
  2. 🔍 RAG Query  — agentic multi-strategy Q&A with live iteration trace + image display

Run:
    streamlit run frontend/app.py

Env:
    API_BASE_URL  (default: http://localhost:8000)
"""
from __future__ import annotations

import json
import os
import time
from typing import Any, Dict, List

import requests
import streamlit as st

API_BASE = os.getenv("API_BASE_URL", "http://localhost:8000")

st.set_page_config(
    page_title="M-RAG", page_icon="🧠",
    layout="wide", initial_sidebar_state="expanded",
)

# ─────────────────────────────────────────────────────────────────────────────
# Data helpers (cached) — unchanged
# ─────────────────────────────────────────────────────────────────────────────
@st.cache_data(ttl=10)
def _ingest_defaults() -> Dict[str, Any]:
    try: return requests.get(f"{API_BASE}/config/defaults", timeout=5).json()
    except: return {}

@st.cache_data(ttl=10)
def _collections() -> List[str]:
    try: return requests.get(f"{API_BASE}/collections", timeout=5).json().get("collections", [])
    except: return []

@st.cache_data(ttl=30)
def _strategies() -> List[str]:
    try: return requests.get(f"{API_BASE}/rag/strategies", timeout=5).json().get("strategies", [])
    except: return ["query_expansion","query_rewriting","query_decomposition",
                    "step_back_prompting","hyde","multi_query","sub_query"]

def _ok() -> bool:
    try: return requests.get(f"{API_BASE}/health", timeout=3).status_code == 200
    except: return False

# ─────────────────────────────────────────────────────────────────────────────
# Sidebar — unchanged
# ─────────────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.title("🧠 M-RAG")
    healthy = _ok()
    if healthy:
        st.success(f"API • {API_BASE}", icon="✅")
    else:
        st.error("API unreachable", icon="🔴")
    if not healthy:
        st.code("uvicorn api.main:app --reload --port 8000")
    st.divider()
    st.caption("Multimodal Agentic RAG\nLangGraph · LangChain · Groq · Qdrant")

# ─────────────────────────────────────────────────────────────────────────────
# Tabs
# ─────────────────────────────────────────────────────────────────────────────
tab_ingest, tab_rag = st.tabs(["📥 Ingest Documents", "🔍 Agentic RAG Query"])

# ══════════════════════════════════════════════════════════════════════════════
# TAB 1 — INGESTION (MULTI-FILE) — unchanged
# ══════════════════════════════════════════════════════════════════════════════
with tab_ingest:
    st.header("📥 Document Ingestion")
    D = _ingest_defaults()
    cfg_col, up_col = st.columns([2, 3])

    with cfg_col:
        st.subheader("⚙️ Pipeline Config")

        describe_imgs = st.toggle("Describe images (Vision LLM)", value=D.get("describe_images", True))
        img_prompt = st.text_area("Image prompt", value=D.get("image_description_prompt","Describe this image in rich detail."), height=60, disabled=not describe_imgs)

        st.divider()
        chunk_method = st.selectbox("Chunking", ["recursive","hierarchical","semantic"],
            index=["recursive","hierarchical","semantic"].index(D.get("chunking_method","recursive")))
        chunk_overlap = st.slider("Overlap (tokens)", 0, 500, D.get("chunk_overlap",100), 10)

        cs = pcs = ccs = sb = None
        if chunk_method == "recursive":
            cs = st.slider("Chunk size", 100, 8000, D.get("chunk_size",1000), 100)
        elif chunk_method == "hierarchical":
            pcs = st.slider("Parent size", 500, 16000, D.get("parent_chunk_size",2000), 100)
            ccs = st.slider("Child size",  100,  4000, D.get("child_chunk_size",800),   50)
        else:
            sb  = st.slider("Breakpoint", 0.0, 1.0, D.get("semantic_breakpoint",0.8), 0.01)

        st.divider()
        emb_type = st.selectbox("Embedder type", ["text","multimodal","image"],
            index=["text","multimodal","image"].index(D.get("embedding_type","text")))
        _opts = {"text":["all-MiniLM-L6-v2","all-mpnet-base-v2"],"multimodal":["clip-ViT-B-32"],"image":["clip-ViT-B-32"]}
        emb_model = st.selectbox("Model", _opts[emb_type])
        custom_emb = st.text_input("Custom model", placeholder="sentence-transformers/…", key="ingest_custom_emb")
        if custom_emb.strip(): emb_model = custom_emb.strip()

        st.divider()
        llm_m    = st.text_input("LLM", value=D.get("llm_model","openai/gpt-oss-120b"), key="ingest_llm")
        vis_m    = st.text_input("Vision LLM", value=D.get("vision_model","meta-llama/llama-4-scout-17b-16e-instruct"), disabled=not describe_imgs, key="ingest_vis")
        cols_c   = _collections()
        coll_n   = st.text_input("Collection", value=D.get("collection_name","mrag_default"), key="ingest_coll")
        if cols_c: st.caption(f"Existing: {', '.join(cols_c)}")

    with up_col:
        st.subheader("📄 Upload Documents")
        uploaded_files = st.file_uploader(
            "Select files to ingest", 
            type=["pdf","txt","md","png","jpg","jpeg"],
            accept_multiple_files=True,
            help="You can select as many PDFs, images, text files, etc. as you want"
        )
        
        if uploaded_files:
            st.info(f"**{len(uploaded_files)} file(s) selected**")
            for f in uploaded_files:
                st.info(f"**{f.name}** • {f.size/1024:.1f} KB")

        ingest_cfg: Dict[str, Any] = dict(
            describe_images=describe_imgs, image_description_prompt=img_prompt,
            chunking_method=chunk_method, chunk_overlap=chunk_overlap,
            embedding_type=emb_type, embedding_model=emb_model,
            llm_model=llm_m, vision_model=vis_m, collection_name=coll_n,
        )
        if cs  is not None: ingest_cfg["chunk_size"]        = cs
        if pcs is not None: ingest_cfg["parent_chunk_size"] = pcs
        if ccs is not None: ingest_cfg["child_chunk_size"]   = ccs
        if sb  is not None: ingest_cfg["semantic_breakpoint"] = sb

        with st.expander("📋 Config (applied to all files)", expanded=False):
            st.json(ingest_cfg)

        go_ingest = st.button(
            "🚀 Ingest All Files", 
            disabled=not(uploaded_files and healthy), 
            type="primary", 
            use_container_width=True
        )

        if go_ingest and uploaded_files:
            st.divider()
            for upload in uploaded_files:
                st.subheader(f"🚀 Processing: **{upload.name}**")
                
                with st.spinner("Submitting…"):
                    try:
                        r = requests.post(
                            f"{API_BASE}/ingest",
                            files={"file": (upload.name, upload.getvalue(), upload.type)},
                            data={"config_json": json.dumps(ingest_cfg)},
                            timeout=45
                        )
                        r.raise_for_status()
                        jid = r.json()["job_id"]
                    except Exception as exc:
                        st.error(f"Submit failed for {upload.name}: {exc}")
                        continue
                
                st.success(f"Job • `{jid}` queued for **{upload.name}**")
                
                bar = st.progress(0, "Queued…")
                stat_ph = st.empty()
                met_ph = st.empty()
                PROG = {"queued":5, "running":50, "success":100, "failed":100}

                for _ in range(300):
                    time.sleep(2)
                    try:
                        d = requests.get(f"{API_BASE}/ingest/{jid}", timeout=10).json()
                    except:
                        continue
                        
                    s = d.get("status", "?")
                    bar.progress(PROG.get(s, 50)/100, s)
                    
                    if s == "success":
                        stat_ph.success(f"✅ Done • `{d.get('document_id')}`")
                        met = d.get("metrics") or {}
                        if met:
                            mc = st.columns(min(len(met), 4))
                            for i, (k, v) in enumerate(met.items()):
                                with mc[i % 4]:
                                    st.metric(
                                        k.replace("_", " ").title(),
                                        f"{v:.2f}s" if isinstance(v, float) and "time" in k else str(v)
                                    )
                        st.success(f"✅ **{upload.name}** successfully ingested!")
                        break
                        
                    elif s == "failed":
                        stat_ph.error(f"❌ {d.get('error')}")
                        break
                    else:
                        stat_ph.info(f"⏳ {s}…")
                
                st.divider()

# ══════════════════════════════════════════════════════════════════════════════
# TAB 2 — AGENTIC RAG (with real image display)
# ══════════════════════════════════════════════════════════════════════════════
with tab_rag:
    st.header("🔍 Agentic RAG Query")
    st.caption(
        "The agent loops recursively—applying query strategies, retrieving, re-ranking—"
        "until confidence ≥ threshold or the iteration limit is reached. "
        "Each LLM call is ≤ ~700 tokens."
    )

    all_strats = _strategies()
    avail_cols = _collections()

    left, right = st.columns([2, 3])

    with left:
        st.subheader("⚙️ Agent Config")
        rag_coll = st.selectbox("Collection", avail_cols or ["mrag_default"], key="rag_coll")

        st.markdown("**Strategy**")
        strat_mode = st.radio("Mode", ["auto (planner picks)", "fixed"], horizontal=True, label_visibility="collapsed", key="strat_mode")

        if strat_mode == "fixed":
            fixed_s = st.selectbox("Fixed strategy", all_strats, key="fixed_strat")
            chosen_s = fixed_s
            enabled_s = all_strats
        else:
            chosen_s = "auto"
            st.caption("Enable strategies for the planner:")
            enabled_s = []
            sc = st.columns(2)
            for i, s in enumerate(all_strats):
                with sc[i % 2]:
                    if st.checkbox(s.replace("_"," ").title(), value=True, key=f"en_{s}"):
                        enabled_s.append(s)
            if not enabled_s:
                enabled_s = all_strats

        with st.expander("ℹ️ Strategy descriptions"):
            _desc = {
                "query_expansion":     "Adds synonyms & domain terms",
                "query_rewriting":     "Rephrases for dense vector search",
                "query_decomposition": "Splits into independent sub-questions",
                "step_back_prompting": "Abstracts to background-level question",
                "hyde":                "Generates hypothetical answer as query",
                "multi_query":         "Creates diverse phrasings of the question",
                "sub_query":           "Identifies still-missing atomic facts",
            }
            for n,d in _desc.items():
                st.markdown(f"**{n.replace('_',' ').title()}** — {d}")

        st.divider()
        st.markdown("**Retrieval & Re-ranking**")
        top_k        = st.slider("Top-K", 1, 20, 5, key="rag_topk")
        score_thr    = st.slider("Score threshold", 0.0, 1.0, 0.25, 0.05, key="rag_thr")
        rerank_mode  = st.radio("Re-rank", ["llm","score"], horizontal=True, key="rag_rerank")
        rerank_top_n = st.slider("Chunks after rerank", 1, 10, 4, key="rag_topn")

        st.divider()
        st.markdown("**Agent Loop**")
        max_iter  = st.slider("Max iterations", 1, 10, 5, key="rag_maxiter")
        conf_thr  = st.slider("Confidence threshold", 0.0, 1.0, 0.75, 0.05, key="rag_conf",
                              help="Agent stops early when confidence ≥ this")

        st.divider()
        st.markdown("**LLM (per-call budget)**")
        rag_llm   = st.text_input("Groq model", "openai/gpt-oss-120b", key="rag_llm")
        max_tok   = st.slider("Max tokens / call", 64, 2048, 512, 64, key="rag_maxtok",
                              help="Keep small to minimize cost per call")
        rag_temp  = st.slider("Temperature", 0.0, 1.0, 0.0, 0.05, key="rag_temp")
        inc_src   = st.checkbox("Include sources", True, key="rag_src")

        rag_cfg = {
            "collection_name": rag_coll,
            "strategy": chosen_s,
            "enabled_strategies": enabled_s,
            "top_k": top_k,
            "score_threshold": score_thr,
            "rerank_mode": rerank_mode,
            "rerank_top_n": rerank_top_n,
            "max_iterations": max_iter,
            "confidence_threshold": conf_thr,
            "llm_model": rag_llm,
            "temperature": rag_temp,
            "max_tokens_per_call": max_tok,
            "include_sources": inc_src,
            "embedding_type": emb_type,
            "embedding_model": emb_model,
        }
        with st.expander("📋 Config JSON", expanded=False):
            st.json(rag_cfg)

    with right:
        st.subheader("💬 Ask")
        query_txt = st.text_area(
            "Question", placeholder="e.g. What are the main conclusions about transformer scaling?",
            height=110, key="rag_query",
        )

        if max_iter and max_tok:
            est_tokens = (max_iter * (300+350+250+600) + 700) * (max_tok / 512)
            st.caption(
                f"💰 Estimated token budget: ~{int(est_tokens):,} tokens total "
                f"({max_iter} iter × ~1 500 t/iter + synthesis)"
            )

        go_rag = st.button("🔍 Run Agent", disabled=not(query_txt.strip() and healthy),
                           type="primary", use_container_width=True, key="go_rag")

        if go_rag and query_txt.strip():
            with st.spinner("Submitting…"):
                try:
                    r = requests.post(f"{API_BASE}/rag/query",
                        json={"query": query_txt, "config": rag_cfg}, timeout=30)
                    r.raise_for_status()
                    rjid = r.json()["job_id"]
                except Exception as exc:
                    st.error(f"Submit failed: {exc}"); st.stop()

            st.success(f"Job • `{rjid}`")

            rag_prog   = st.progress(0, "Starting…")
            probe_ph   = st.empty()
            rag_meta   = st.empty()
            rag_trace  = st.empty()
            rag_answer = st.empty()
            rag_srcs   = st.empty()

            DONE = {"success","failed","max_iterations","no_store"}
            last_shown   = 0
            probe_shown  = False

            for tick in range(600):
                time.sleep(2)
                try:
                    rd = requests.get(f"{API_BASE}/rag/query/{rjid}", timeout=10).json()
                except: continue

                rs         = rd.get("status","running")
                iters      = rd.get("iterations_used") or 0
                conf       = rd.get("answer_confidence") or 0.0
                notepad    = rd.get("notepad") or []
                answer     = rd.get("final_answer") or ""
                sources    = rd.get("answer_sources") or []
                probe_diag = rd.get("probe_diagnostics","")

                if probe_diag and not probe_shown:
                    probe_shown = True
                    if rs in ("failed","no_store") or "does not exist" in probe_diag or "0 points" in probe_diag:
                        probe_ph.error(f"🔴 **Store probe:** {probe_diag}")
                    else:
                        probe_ph.success(f"🟢 **Store probe:** {probe_diag}")

                if rs in DONE:
                    rag_prog.progress(1.0, f"Done — {rs}")
                else:
                    rag_prog.progress(min(0.9, iters/max_iter * 0.85 + 0.05),
                                      f"Iter {iters}/{max_iter}  •  confidence {conf:.0%}")

                icon = "✅" if rs=="success" else ("⚠️" if rs=="max_iterations" else ("❌" if rs in("failed","no_store") else "⏳"))
                rag_meta.markdown(
                    f"{icon} **Status:** `{rs}` &nbsp;|&nbsp; "
                    f"**Iterations:** {iters}/{max_iter} &nbsp;|&nbsp; "
                    f"**Confidence:** {conf:.0%}"
                )

                if notepad and len(notepad) > last_shown:
                    last_shown = len(notepad)
                    with rag_trace.container():
                        st.markdown("---")
                        st.markdown("#### 📓 Agent Reasoning Trace")
                        for entry in notepad:
                            with st.expander(
                                f"**Iter {entry['iteration']}** · "
                                f"{entry['strategy'].replace('_',' ').title()} · "
                                f"{entry['chunks_found']} chunks retrieved",
                                expanded=(entry['iteration'] == len(notepad)),
                            ):
                                qs = entry.get("queries_used", [])
                                if qs:
                                    st.markdown("**Queries sent to Qdrant:**")
                                    for q in qs:
                                        st.code(q, language=None)
                                st.info(f"**Key finding:** {entry['key_findings']}")

                # ── FINAL ANSWER + IMAGE DISPLAY ─────────────────────────────────────
                if rs in DONE and answer:
                    with rag_answer.container():
                        st.markdown("---")
                        if rs == "max_iterations":
                            st.warning(f"⚠️ Hit iteration limit ({max_iter}). Best available answer:")
                        elif rs in ("failed","no_store"):
                            st.error(f"Pipeline error: {rd.get('error') or answer}")
                        else:
                            st.success(f"Answer generated in {iters} iteration(s) with {conf:.0%} confidence")
                        st.markdown("### 💡 Final Answer")
                        st.markdown(answer)

                    # ── IMPROVED SOURCES (now shows full text!) ─────────────────
                    if sources:
                        with rag_srcs.container():
                            st.markdown("---")
                            st.markdown("### 📚 Sources")
                            st.caption("Retrieved chunks and **images** used by the agent")

                            # Temporary debug (remove after you confirm it works)
                            with st.expander("🔍 Debug: Raw sources from backend", expanded=False):
                                st.json(sources)

                            for i, src in enumerate(sources, 1):
                                if isinstance(src, str) and src.lower().endswith(('.png','.jpg','.jpeg','.webp','.gif')):
                                    # IMAGE
                                    st.markdown(f"**Source {i}** — 🖼️ Image")
                                    img_url = src if src.startswith(('http://','https://')) else f"{API_BASE.rstrip('/')}/{src.lstrip('/')}"
                                    try:
                                        st.image(img_url, use_column_width=True)
                                        st.caption(f"`{src}`")
                                    except Exception:
                                        st.warning(f"Could not load image (check path):")
                                        st.code(src)

                                else:
                                    # TEXT CHUNK (handles string, int, dict — now shows real text)
                                    title = f"**Source {i}** — Text chunk"
                                    content = ""

                                    if isinstance(src, dict):
                                        content = (
                                            src.get("text") or
                                            src.get("page_content") or
                                            src.get("content") or
                                            src.get("chunk") or
                                            str(src)
                                        )
                                        score = src.get("score") or src.get("relevance")
                                        if score:
                                            title += f" [score: {score:.3f}]"
                                    else:
                                        content = str(src)

                                    with st.expander(title, expanded=True):
                                        st.markdown(content)

                    break