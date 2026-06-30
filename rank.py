#!/usr/bin/env python3
"""
rank.py — Redrob Hackathon: Senior AI Engineer ranker
Usage:  python rank.py --candidates data/candidates.jsonl --out submission.csv
Rules:  CPU only · no network · <5 min · 16 GB RAM · top-100 only
"""

import json
import csv
import argparse
import re
import os
from datetime import date
from concurrent.futures import ProcessPoolExecutor

# ── Job Description: what the recruiter is filtering for ─────────────────────

# Hard skill keywords from JD  (O(1) lookup via set)
CORE_SKILLS = {
    # retrieval / embeddings
    "embedding", "embeddings", "sentence-transformer", "sentence transformers",
    "bge", "e5", "gte", "openai embeddings",
    "faiss", "pinecone", "weaviate", "qdrant", "milvus", "chroma", "chromadb",
    "opensearch", "elasticsearch",
    "rag", "retrieval", "retrieval-augmented", "vector database", "vector db",
    "vector search", "hybrid search", "semantic search",
    # NLP / LLMs
    "nlp", "natural language processing",
    "llm", "llms", "large language model", "large language models",
    "transformers", "hugging face", "huggingface",
    "bert", "gpt", "fine-tuning", "fine tuning", "finetuning", "lora", "qlora",
    "text classification", "named entity recognition", "ner",
    # evaluation
    "ndcg", "mrr", "map", "mean average precision",
    "ranking", "ranking system", "information retrieval", "reranking",
    "a/b testing", "offline evaluation",
    # core ML / DL
    "pytorch", "tensorflow", "keras", "jax",
    "deep learning", "neural network", "neural networks",
    "machine learning", "ml",
    # Python ecosystem
    "python", "numpy", "pandas", "scikit-learn", "sklearn",
    # MLOps / infra
    "mlops", "ml pipeline", "model serving", "triton", "onnx",
    "docker", "kubernetes", "k8s",
    "recommendation system", "recsys", "collaborative filtering",
}

# Secondary / nice-to-have skills
SECONDARY_SKILLS = {
    "data science", "data engineering", "spark", "pyspark", "kafka", "airflow",
    "aws", "gcp", "azure", "cloud",
    "sql", "redis", "dbt",
    "statistics", "probability",
    "java", "scala", "go", "rust", "c++",
    "computer vision", "cv", "image classification", "object detection",
    "speech recognition", "tts", "asr",
}

# JD sweet-spot experience
EXP_MIN, EXP_IDEAL_LO, EXP_IDEAL_HI, EXP_MAX = 2, 5, 9, 20

# Preferred India cities (JD says Pune / Noida, open to Tier-1)
INDIA_CITIES = {
    "pune", "noida", "bangalore", "bengaluru", "mumbai",
    "hyderabad", "delhi", "gurgaon", "gurugram", "chennai",
    "kolkata", "ahmedabad",
}

# AI / ML job titles (recruiter's "relevant title" filter)
AI_TITLES = {
    "ai engineer", "ml engineer", "machine learning engineer",
    "senior ai engineer", "senior ml engineer",
    "applied scientist", "applied ml engineer",
    "nlp engineer", "deep learning engineer", "research engineer",
    "data scientist", "computer vision engineer",
    "junior ml engineer", "senior machine learning engineer",
    "ai researcher", "ml researcher",
}

TODAY = date.today()

# Compile keyword sets into regexes — one C-level scan per string vs Python loop over 50 keywords
_CORE_RE = re.compile("|".join(re.escape(k) for k in sorted(CORE_SKILLS, key=len, reverse=True)))
_SEC_RE  = re.compile("|".join(re.escape(k) for k in sorted(SECONDARY_SKILLS, key=len, reverse=True)))
_TITLE_RE = re.compile("|".join(re.escape(k) for k in sorted(AI_TITLES, key=len, reverse=True)))
_CAREER_KW_RE = re.compile(r"embedding|vector|rag|nlp|llm|retrieval|pytorch")



# ── Scoring components ────────────────────────────────────────────────────────

def score_skills(skills, assessment_scores: dict):
    """Returns (score, core_skill_names). Caller reuses core_skill_names for reasoning."""
    core_hits = 0.0
    sec_hits = 0.0
    core_names = []

    for s in skills:
        name = s.get("name", "").lower()
        prof = s.get("proficiency", "beginner")
        dur = s.get("duration_months", 0)
        endr = s.get("endorsements", 0)
        assess = assessment_scores.get(s.get("name", ""), -1)

        pw = {"beginner": 0.4, "intermediate": 0.65, "advanced": 0.85, "expert": 1.0}.get(prof, 0.4)
        trust = 1.0 + (0.1 if dur >= 12 else 0) + (0.1 if endr >= 10 else 0)
        if assess >= 70:
            trust += 0.1
        w = pw * trust

        if _CORE_RE.search(name):
            core_hits += w
            core_names.append(s.get("name", ""))
        elif _SEC_RE.search(name):
            sec_hits += w * 0.3

    return min(1.0, core_hits / 8.0 + min(0.15, sec_hits / 10.0)), core_names


def score_experience(yoe: float) -> float:
    """Linear ramp into the JD sweet spot, gentle taper outside."""
    if yoe < EXP_MIN:
        return 0.2
    if yoe < EXP_IDEAL_LO:
        return 0.2 + 0.8 * (yoe - EXP_MIN) / (EXP_IDEAL_LO - EXP_MIN)
    if yoe <= EXP_IDEAL_HI:
        return 1.0
    if yoe <= EXP_MAX:
        return 1.0 - 0.4 * (yoe - EXP_IDEAL_HI) / (EXP_MAX - EXP_IDEAL_HI)
    return 0.4


def score_title(current_title: str, career: list) -> float:
    t = current_title.lower()
    base = 1.0 if _TITLE_RE.search(t) else (
        0.55 if any(x in t for x in ("software", "backend", "data", "platform", "infra")) else 0.2
    )

    ai_months = 0
    for job in career:
        if _TITLE_RE.search(job.get("title", "").lower()) or \
           _CAREER_KW_RE.search(job.get("description", "").lower()):
            ai_months += job.get("duration_months", 0)

    career_bonus = min(0.3, ai_months / 36)
    return min(1.0, base + career_bonus * (1 - base))


def score_location(profile: dict, willing_to_relocate: bool) -> float:
    """India + right city = max; willing to relocate = partial credit."""
    country = profile.get("country", "").lower()
    city = profile.get("location", "").lower()

    if country == "india":
        return 1.0 if any(c in city for c in INDIA_CITIES) else 0.8
    return 0.45 if willing_to_relocate else 0.15


def score_education(education: list) -> float:
    """Tier + CS/ML field bonus."""
    if not education:
        return 0.3
    best = 0.0
    for edu in education:
        tier_w = {"tier_1": 1.0, "tier_2": 0.75, "tier_3": 0.5,
                  "tier_4": 0.3, "unknown": 0.4}.get(edu.get("tier", "unknown"), 0.4)
        field = edu.get("field_of_study", "").lower()
        field_w = 1.0 if any(x in field for x in
                              ("computer", "software", "data", "ai ", "machine",
                               "electrical", "math", "stat", "information")) else 0.55
        best = max(best, tier_w * field_w)
    return best


def score_behavioral(sig: dict) -> float:
    """
    Recruiter signals: is this person reachable, active, serious?
    Mirrors what a recruiter sees on the portal activity panel.
    """
    # Not open to work → heavy penalty (still in pool but scored low)
    if not sig.get("open_to_work_flag", False):
        return 0.1

    parts = []

    # Response rate  (will they reply to InMail?)
    parts.append(sig.get("recruiter_response_rate", 0.0))

    # Interview completion  (will they show up?)
    parts.append(sig.get("interview_completion_rate", 0.0))

    # GitHub activity  (technical signal)
    gh = sig.get("github_activity_score", -1)
    parts.append(gh / 100 if gh >= 0 else 0.25)

    # Profile completeness
    parts.append(sig.get("profile_completeness_score", 0) / 100)

    # Recency: penalise if inactive > 6 months
    try:
        days_ago = (TODAY - date.fromisoformat(sig.get("last_active_date", "2020-01-01"))).days
    except ValueError:
        days_ago = 999
    parts.append(max(0.0, 1.0 - days_ago / 180))

    # Notice period: shorter = easier hire  (0d→1.0, 90d→0.0)
    notice = sig.get("notice_period_days", 90)
    parts.append(max(0.0, 1.0 - notice / 90) * 0.6)  # lower weight

    return sum(parts) / len(parts)


def is_honeypot(skills: list, career: list, yoe: float, sig: dict) -> bool:
    """
    Detect subtly impossible profiles.
    Very conservative — only flag clear contradictions.
    """
    flags = 0

    # Expert skills with zero usage months — can't be expert without using it
    zero_expert = sum(1 for s in skills
                      if s.get("proficiency") == "expert" and s.get("duration_months", 0) == 0)
    if zero_expert >= 4:
        flags += 1

    # Career months total massively exceeds claimed experience
    total_career_months = sum(j.get("duration_months", 0) for j in career)
    if total_career_months > yoe * 12 * 1.6 + 12:
        flags += 1

    # All three rates perfect simultaneously — statistically impossible
    rr = sig.get("recruiter_response_rate", 0)
    ic = sig.get("interview_completion_rate", 0)
    oa = sig.get("offer_acceptance_rate", -1)
    if rr == 1.0 and ic == 1.0 and oa == 1.0:
        flags += 1

    # Many skills, zero career — keyword stuffer
    if len(skills) >= 12 and len(career) == 0:
        flags += 1

    return flags >= 2  # need 2+ contradictions — conservative


def build_reasoning(profile: dict, sig: dict, core_skill_names: list) -> str:
    """1–2 sentences from real data. No hallucination."""
    title = profile.get("current_title", "Unknown")
    yoe = profile.get("years_of_experience", 0)
    city = profile.get("location", "")
    country = profile.get("country", "")
    rr = sig.get("recruiter_response_rate", 0)
    gh = sig.get("github_activity_score", -1)
    notice = sig.get("notice_period_days", 90)

    top_skills = ", ".join(core_skill_names[:4]) if core_skill_names else "general skills"
    s1 = f"{title} with {yoe:.1f} yrs exp; core AI skills: {top_skills}; {city}, {country}."

    signals = []
    if rr >= 0.7:
        signals.append(f"response rate {rr:.0%}")
    elif rr < 0.25:
        signals.append(f"low response rate ({rr:.0%})")
    if gh > 60:
        signals.append(f"GitHub {gh:.0f}/100")
    elif gh < 0:
        signals.append("no GitHub")
    if notice <= 15:
        signals.append(f"immediate joiner")
    elif notice >= 90:
        signals.append(f"notice {notice}d")

    s2 = ("; ".join(signals) + ".") if signals else ""
    return (s1 + " " + s2).strip()[:300]


# ── Main scorer ───────────────────────────────────────────────────────────────

def score_candidate(c: dict):
    """Returns (score, candidate_id, reasoning) or None if honeypot."""
    profile = c.get("profile", {})
    sig = c.get("redrob_signals", {})
    skills = c.get("skills", [])
    career = c.get("career_history", [])
    education = c.get("education", [])
    yoe = profile.get("years_of_experience", 0)

    if is_honeypot(skills, career, yoe, sig):
        return None

    assess = sig.get("skill_assessment_scores", {})

    sk, core_names = score_skills(skills, assess)  # core_names reused — not recomputed
    tc  = score_title(profile.get("current_title", ""), career)
    ex  = score_experience(yoe)
    ed  = score_education(education)
    loc = score_location(profile, sig.get("willing_to_relocate", False))
    beh = score_behavioral(sig)

    wm_bonus = 0.03 if sig.get("preferred_work_mode", "") in ("hybrid", "flexible", "remote") else 0.0

    score = (
        0.40 * sk  +
        0.22 * tc  +
        0.14 * ex  +
        0.08 * ed  +
        0.07 * loc +
        0.09 * beh +
        wm_bonus
    )

    if not sig.get("open_to_work_flag", False):
        score *= 0.55

    reasoning = build_reasoning(profile, sig, core_names)
    return round(score, 6), c["candidate_id"], reasoning


# ── Worker (must be module-level for Windows spawn to pickle it) ─────────────

def _process_chunk(lines):
    """Parse + score a list of raw byte lines. Called in worker processes."""
    out = []
    for line in lines:
        line = line.strip()
        if not line:
            continue
        try:
            c = json.loads(line)
        except (ValueError, json.JSONDecodeError):
            continue
        r = score_candidate(c)
        if r is not None:
            out.append(r)
    return out


# ── Entry point ───────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--candidates", default="data/candidates.jsonl")
    parser.add_argument("--out", default="submission.csv")
    args = parser.parse_args()

    print(f"Scanning {args.candidates} ...")

    results = []

    with open(args.candidates, "rb") as f:
        first_byte = f.read(1)
        f.seek(0)
        is_array = first_byte == b"["

        if is_array:
            # Sample file (50 items) — single process is fine
            for c in json.load(f):
                r = score_candidate(c)
                if r is not None:
                    results.append(r)
        else:
            # JSONL — bulk read + parallel scoring across all CPU cores
            raw_lines = f.read().split(b"\n")
            ncpu = os.cpu_count() or 4
            step = (len(raw_lines) + ncpu - 1) // ncpu
            chunks = [raw_lines[i : i + step] for i in range(0, len(raw_lines), step)]

            with ProcessPoolExecutor(max_workers=ncpu) as pool:
                for chunk_results in pool.map(_process_chunk, chunks):
                    results.extend(chunk_results)

    print(f"  {len(results):,} scored across {os.cpu_count()} cores")

    # Sort by score desc, tie-break candidate_id asc (spec requirement)
    # This guarantees scores are non-increasing — no post-processing needed
    results.sort(key=lambda x: (-x[0], x[1]))
    top100 = results[:100]

    with open(args.out, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["candidate_id", "rank", "score", "reasoning"])
        for rank, (score, cid, reason) in enumerate(top100, 1):
            w.writerow([cid, rank, f"{score:.6f}", reason])

    print(f"Written: {args.out}")
    print("\nTop 5:")
    for i, (sc, cid, reason) in enumerate(top100[:5], 1):
        print(f"  {i}. {cid}  {sc:.4f}  {reason[:90]}")


if __name__ == "__main__":
    main()
