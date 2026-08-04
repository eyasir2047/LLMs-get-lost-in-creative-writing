"""
Story Annotation Tool — Streamlit app
======================================
- Password-gated login (credentials stored in .streamlit/secrets.toml)
- Loads stories from results_qwen.jsonl + results_gemma4.jsonl
- Hides model / condition from annotators
- Stores every rating in SQLite (ratings.db)
- Lets any logged-in annotator download the full CSV
"""
import json
import random
from pathlib import Path

import pandas as pd
import streamlit as st

from db import get_all_ratings, get_progress, get_user_ratings, init_db, save_rating

# ── Config ─────────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Story Annotation Tool",
    page_icon="📖",
    layout="wide",
    initial_sidebar_state="expanded",
)

HERE = Path(__file__).parent
PROJECT_ROOT = HERE.parent  # ../  (Sisters Program Synthetica/)

DATA_FILES = [
    PROJECT_ROOT / "results_qwen.jsonl",
    PROJECT_ROOT / "results_gemma4.jsonl",
]

CRITERIA = [
    ("relevance",  "Relevance",  "Does the story match its title/topic?",
     "1 = Nothing to do with the topic  •  5 = Matches perfectly"),
    ("coherence",  "Coherence",  "Does the story make sense from start to end?",
     "1 = Completely incoherent  •  5 = Perfectly coherent"),
    ("empathy",    "Empathy",    "Did you connect with the characters' feelings?",
     "1 = No emotion comes through  •  5 = Fully emotionally involved"),
    ("surprise",   "Surprise",   "Was the ending surprising?",
     "1 = Obvious from the start  •  5 = Surprised, but clues were there"),
    ("engagement", "Engagement", "Did you enjoy reading / want to keep reading?",
     "1 = Boring  •  5 = Very engaging, wished there was more"),
    ("complexity", "Complexity", "How rich and detailed is the story?",
     "1 = Very simple  •  5 = Very developed (rich characters, plot, setting)"),
]

CRITERIA_KEYS = [c[0] for c in CRITERIA]

# ── CSS ────────────────────────────────────────────────────────────────────────
st.markdown(
    """
<style>
/* ---------- Global ---------- */
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&family=Lora:ital,wght@0,400;0,600;1,400&display=swap');

html, body, [class*="css"] {
    font-family: 'Inter', sans-serif;
}

/* ---------- Background ---------- */
.stApp {
    background: linear-gradient(135deg, #0f0f1a 0%, #1a1a2e 50%, #16213e 100%);
    min-height: 100vh;
}

/* ---------- Sidebar ---------- */
[data-testid="stSidebar"] {
    background: rgba(15, 15, 30, 0.95) !important;
    border-right: 1px solid rgba(139, 92, 246, 0.2);
}

/* ---------- Login card ---------- */
.login-card {
    background: rgba(255,255,255,0.04);
    border: 1px solid rgba(139,92,246,0.25);
    border-radius: 20px;
    padding: 48px 40px;
    max-width: 420px;
    margin: 80px auto 0 auto;
    backdrop-filter: blur(20px);
    box-shadow: 0 25px 50px rgba(0,0,0,0.5);
}

.login-logo {
    font-size: 3rem;
    text-align: center;
    margin-bottom: 8px;
}

.login-title {
    font-family: 'Lora', serif;
    font-size: 1.6rem;
    font-weight: 600;
    text-align: center;
    color: #e2d9f3;
    margin-bottom: 4px;
}

.login-sub {
    text-align: center;
    color: #8b8da0;
    font-size: 0.9rem;
    margin-bottom: 28px;
}

/* ---------- Story card ---------- */
.story-card {
    background: rgba(255,255,255,0.03);
    border: 1px solid rgba(139,92,246,0.15);
    border-radius: 16px;
    padding: 28px 32px;
    backdrop-filter: blur(10px);
    margin-bottom: 20px;
}

.story-title {
    font-family: 'Lora', serif;
    font-size: 1.5rem;
    font-weight: 600;
    color: #e2d9f3;
    margin-bottom: 16px;
    line-height: 1.4;
}

.story-body {
    font-family: 'Lora', serif;
    font-size: 0.97rem;
    line-height: 1.85;
    color: #b8bcd0;
    max-height: 420px;
    overflow-y: auto;
    padding-right: 8px;
    white-space: pre-wrap;
}

.story-body::-webkit-scrollbar { width: 6px; }
.story-body::-webkit-scrollbar-track { background: transparent; }
.story-body::-webkit-scrollbar-thumb {
    background: rgba(139,92,246,0.4);
    border-radius: 3px;
}

/* ---------- Rating section ---------- */
.rating-card {
    background: rgba(255,255,255,0.03);
    border: 1px solid rgba(139,92,246,0.15);
    border-radius: 16px;
    padding: 24px 28px;
    margin-bottom: 16px;
    backdrop-filter: blur(10px);
}

.crit-label {
    font-weight: 600;
    color: #c4b5fd;
    font-size: 0.95rem;
    margin-bottom: 2px;
}

.crit-desc {
    color: #9ca3af;
    font-size: 0.8rem;
    margin-bottom: 2px;
}

.crit-hint {
    color: #6b7280;
    font-size: 0.75rem;
    font-style: italic;
    margin-bottom: 8px;
}

/* ---------- Progress ---------- */
.prog-label {
    font-size: 0.8rem;
    color: #8b8da0;
    text-transform: uppercase;
    letter-spacing: 0.08em;
    margin-bottom: 4px;
}

.prog-count {
    font-size: 1.4rem;
    font-weight: 700;
    color: #c4b5fd;
}

/* ---------- Badge (rated) ---------- */
.badge-rated {
    display: inline-block;
    background: rgba(52,211,153,0.15);
    border: 1px solid rgba(52,211,153,0.4);
    color: #34d399;
    border-radius: 999px;
    padding: 2px 12px;
    font-size: 0.78rem;
    font-weight: 600;
    letter-spacing: 0.05em;
}

.badge-unrated {
    display: inline-block;
    background: rgba(251,191,36,0.12);
    border: 1px solid rgba(251,191,36,0.3);
    color: #fbbf24;
    border-radius: 999px;
    padding: 2px 12px;
    font-size: 0.78rem;
    font-weight: 600;
    letter-spacing: 0.05em;
}

/* ---------- Nav counter ---------- */
.nav-counter {
    text-align: center;
    color: #6b7280;
    font-size: 0.88rem;
    padding-top: 6px;
}

/* ---------- Streamlit tweaks ---------- */
div[data-testid="stVerticalBlock"] > div { gap: 0.6rem; }

.stButton > button {
    border-radius: 10px;
    font-weight: 500;
}

/* Primary submit button */
.stButton > button[kind="primary"] {
    background: linear-gradient(135deg, #7c3aed, #a855f7) !important;
    border: none !important;
    color: white !important;
    font-weight: 600 !important;
    padding: 12px !important;
    letter-spacing: 0.03em;
}

.stButton > button[kind="primary"]:hover {
    background: linear-gradient(135deg, #6d28d9, #9333ea) !important;
    box-shadow: 0 4px 20px rgba(139,92,246,0.4);
    transform: translateY(-1px);
    transition: all 0.2s ease;
}

/* Slider color */
[data-testid="stSlider"] [data-baseweb="slider"] [role="slider"] {
    background-color: #8b5cf6 !important;
}

/* Inputs */
.stTextInput input, .stTextInput input:focus {
    background: rgba(255,255,255,0.06) !important;
    border: 1px solid rgba(139,92,246,0.3) !important;
    border-radius: 10px !important;
    color: #e2d9f3 !important;
}

/* Divider */
hr { border-color: rgba(139,92,246,0.15) !important; }
</style>
""",
    unsafe_allow_html=True,
)


# ── Data loading ───────────────────────────────────────────────────────────────
@st.cache_data
def load_stories() -> list[dict]:
    stories = []
    for path in DATA_FILES:
        if path.exists():
            with open(path, encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    r = json.loads(line)
                    uid = (
                        f'{r["domain"]}_{r["item_id"]}_{r["condition"]}_'
                        f'{r["model"].replace("/", "-").replace(":", "-")}'
                    )
                    stories.append(
                        {
                            "story_uid": uid,
                            "title": r["title"],
                            "story_text": r["story"],
                            # metadata — stored in DB but never shown in UI
                            "domain": r["domain"],
                            "item_id": str(r["item_id"]),
                            "condition": r["condition"],
                            "model": r["model"],
                        }
                    )
        else:
            st.warning(f"Data file not found: {path}")
    return stories


def shuffled_for_user(username: str, stories: list[dict]) -> list[dict]:
    """Each annotator gets the same, consistent shuffle based on their username."""
    rng = random.Random(hash(username) & 0xFFFFFFFF)
    shuffled = stories[:]
    rng.shuffle(shuffled)
    return shuffled


# ── Auth ───────────────────────────────────────────────────────────────────────
def check_credentials(username: str, password: str) -> bool:
    try:
        return st.secrets["users"].get(username) == password
    except Exception:
        return False


# ── Login page ─────────────────────────────────────────────────────────────────
def login_page():
    st.markdown(
        """
        <div class="login-card">
          <div class="login-logo">📖</div>
          <div class="login-title">Story Annotation Tool</div>
          <div class="login-sub">Sign in with your annotator credentials to begin.</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # Float the form over the card with columns trick
    _, mid, _ = st.columns([1, 1.4, 1])
    with mid:
        st.markdown("<br>", unsafe_allow_html=True)
        with st.form("login_form", clear_on_submit=False):
            username = st.text_input("Username", placeholder="e.g. annotator1")
            password = st.text_input("Password", type="password", placeholder="••••••••")
            submitted = st.form_submit_button("Sign In →", use_container_width=True, type="primary")

        if submitted:
            if check_credentials(username, password):
                st.session_state.logged_in = True
                st.session_state.username = username
                st.session_state.story_idx = 0
                st.rerun()
            else:
                st.error("❌  Invalid username or password. Please try again.")


# ── Annotation page ────────────────────────────────────────────────────────────
def annotation_page():
    username: str = st.session_state.username
    stories = load_stories()

    if not stories:
        st.error("No story data files found. Please check the project directory.")
        return

    shuffled = shuffled_for_user(username, stories)
    total = len(shuffled)

    # Fetch DB state for this user
    rated: dict = get_user_ratings(username)
    n_rated = sum(1 for s in shuffled if s["story_uid"] in rated)

    # ── Sidebar ──────────────────────────────────────────────────────────────
    with st.sidebar:
        st.markdown(f"### 👤 &nbsp;{username}")
        st.markdown("---")

        pct = n_rated / total if total > 0 else 0
        st.markdown('<p class="prog-label">Your Progress</p>', unsafe_allow_html=True)
        st.progress(pct)
        st.markdown(
            f'<p class="prog-count">{n_rated} <span style="color:#6b7280;font-size:1rem;font-weight:400">/ {total} rated</span></p>',
            unsafe_allow_html=True,
        )

        st.markdown("---")

        # Jump to next unrated
        if st.button("⏭  Jump to next unrated", use_container_width=True):
            for i, s in enumerate(shuffled):
                if s["story_uid"] not in rated:
                    st.session_state.story_idx = i
                    break
            st.rerun()

        st.markdown("---")

        # Download CSV (any logged-in user can pull the full export)
        all_data = get_all_ratings()
        if all_data:
            df = pd.DataFrame(all_data)
            st.download_button(
                "⬇️  Download All Ratings (CSV)",
                data=df.to_csv(index=False).encode("utf-8"),
                file_name="ratings_export.csv",
                mime="text/csv",
                use_container_width=True,
            )

            # Rater progress summary
            prog = get_progress()
            if len(prog) > 1:
                st.markdown("---")
                st.markdown("**All rater progress**")
                for u, n in sorted(prog.items()):
                    bar = "█" * int(n / total * 10) + "░" * (10 - int(n / total * 10))
                    st.markdown(
                        f"<small style='color:#8b8da0'>{u}</small><br>"
                        f"<small style='color:#c4b5fd;font-family:monospace'>{bar} {n}/{total}</small>",
                        unsafe_allow_html=True,
                    )
        else:
            st.markdown("<small style='color:#6b7280'>No ratings yet.</small>", unsafe_allow_html=True)

        st.markdown("---")
        if st.button("🚪  Log Out", use_container_width=True):
            for key in ["logged_in", "username", "story_idx"]:
                st.session_state.pop(key, None)
            st.rerun()

    # ── Ensure story index is valid ───────────────────────────────────────────
    if "story_idx" not in st.session_state:
        st.session_state.story_idx = 0
    idx = max(0, min(st.session_state.story_idx, total - 1))
    story = shuffled[idx]
    uid = story["story_uid"]
    is_rated = uid in rated
    existing_scores = rated.get(uid, {})

    # ── Top navigation ────────────────────────────────────────────────────────
    col_prev, col_counter, col_next = st.columns([1, 5, 1])
    with col_prev:
        if st.button("◀  Prev", disabled=(idx == 0), use_container_width=True):
            st.session_state.story_idx = idx - 1
            st.rerun()
    with col_counter:
        status_badge = (
            '<span class="badge-rated">✓ Rated</span>'
            if is_rated
            else '<span class="badge-unrated">Unrated</span>'
        )
        st.markdown(
            f'<p class="nav-counter">Story <strong style="color:#e2d9f3">{idx + 1}</strong> of {total} &nbsp;·&nbsp; {status_badge}</p>',
            unsafe_allow_html=True,
        )
    with col_next:
        if st.button("Next  ▶", disabled=(idx == total - 1), use_container_width=True):
            st.session_state.story_idx = idx + 1
            st.rerun()

    st.markdown("<br>", unsafe_allow_html=True)

    # ── Story display ─────────────────────────────────────────────────────────
    st.markdown(
        f"""
        <div class="story-card">
          <div class="story-title">{story["title"]}</div>
          <div class="story-body">{story["story_text"].replace("<", "&lt;").replace(">", "&gt;")}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # ── Rating form ───────────────────────────────────────────────────────────
    st.markdown('<div class="rating-card">', unsafe_allow_html=True)
    st.markdown(
        "#### Rate this story &nbsp;<small style='color:#6b7280;font-weight:400'>— all six dimensions, 1 (worst) to 5 (best)</small>",
        unsafe_allow_html=True,
    )

    with st.form(key=f"rating_{uid}"):
        cols = st.columns(3)
        scores = {}
        for i, (key, label, desc, hint) in enumerate(CRITERIA):
            with cols[i % 3]:
                default_val = existing_scores.get(key, 3)
                st.markdown(
                    f'<p class="crit-label">{label}</p>'
                    f'<p class="crit-desc">{desc}</p>'
                    f'<p class="crit-hint">{hint}</p>',
                    unsafe_allow_html=True,
                )
                scores[key] = st.slider(
                    label,
                    min_value=1,
                    max_value=5,
                    value=default_val,
                    label_visibility="collapsed",
                    key=f"sl_{uid}_{key}",
                )

        btn_label = "✅  Update Rating" if is_rated else "✅  Submit Rating"
        submitted = st.form_submit_button(btn_label, type="primary", use_container_width=True)

        if submitted:
            save_rating(username, story, scores)
            st.success("Rating saved!")
            # auto-advance to next story
            if idx < total - 1:
                st.session_state.story_idx = idx + 1
            st.rerun()

    st.markdown("</div>", unsafe_allow_html=True)


# ── Entry point ────────────────────────────────────────────────────────────────
def main():
    init_db()

    if "logged_in" not in st.session_state:
        st.session_state.logged_in = False

    if not st.session_state.logged_in:
        login_page()
    else:
        annotation_page()


if __name__ == "__main__":
    main()
