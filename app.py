import streamlit as st
import json
import sys
import os

sys.path.insert(0, os.path.dirname(__file__))
import rank

st.set_page_config(page_title="Redrob Ranker — VectorVerse", layout="wide")
st.title("Redrob Candidate Ranker")
st.caption("Senior AI Engineer · VectorVerse · Hack2Skill")

st.markdown("""
**How it works:** Rule-based scorer across 6 dimensions —
skill match (40%), title/career (22%), experience (14%),
education (8%), location (7%), behavioral signals (9%).
Runs fully offline, no LLM calls.
""")

# ── Load candidates ────────────────────────────────────────────────────────────

@st.cache_data
def load_sample():
    path = os.path.join(os.path.dirname(__file__), "data", "sample_candidates.json")
    with open(path, encoding="utf-8") as f:
        return json.load(f)

uploaded = st.file_uploader("Upload your own candidates (.json array or .jsonl)", type=["json", "jsonl"])

if uploaded:
    raw = uploaded.read().decode("utf-8").strip()
    if raw.startswith("["):
        candidates = json.loads(raw)
    else:
        candidates = [json.loads(l) for l in raw.splitlines() if l.strip()]
    st.success(f"Loaded {len(candidates):,} candidates from upload.")
else:
    candidates = load_sample()
    st.info(f"Using built-in sample ({len(candidates)} candidates). Upload a file above to use your own.")

# ── Run ranker ─────────────────────────────────────────────────────────────────

top_n = st.slider("Show top N results", 5, min(100, len(candidates)), 10)

if st.button("Run Ranker", type="primary"):
    with st.spinner("Scoring candidates..."):
        results = []
        for c in candidates:
            r = rank.score_candidate(c)
            if r is not None:
                results.append(r)

        results.sort(key=lambda x: (-x[0], x[1]))
        top = results[:top_n]

    st.success(f"Scored {len(results)} candidates.")

    rows = []
    for i, (score, cid, reason) in enumerate(top, 1):
        rows.append({"Rank": i, "Candidate ID": cid, "Score": round(score, 4), "Reasoning": reason})

    st.dataframe(rows, use_container_width=True)
