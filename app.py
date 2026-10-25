import html
import json
import os
import re
import time
import uuid
from datetime import datetime
from pathlib import Path

import streamlit as st
import streamlit.components.v1 as components
from dotenv import load_dotenv

from src.config import CHAT_MODELS, GENDERS, COMMON_SYMPTOMS, DURATIONS, LANGUAGES
from src.cache_manager import setup_cache
from src.chains import get_medical_chain, stream_narrative
from src.utils import parse_llm_json

load_dotenv()

# ---------------------------------------------------------------------------
# Page configuration
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="MediGuide AI | Symptom Assessment",
    page_icon="🩺",
    layout="wide",
    initial_sidebar_state="expanded",
)


RETIRED_PREFIXES = ("gemini-2.5", "gemini-2.0", "gemini-1.5", "gemini-pro", "gemini-1.0")
CURRENT_MODELS = [
    "gemini-3.8-flash",
    "gemini-3.6-flash",
    "gemini-3.5-flash-lite",
    "gemini-3.1-flash-lite",
    "gemini-3.1-pro-preview",
]
MODEL_OPTIONS = [m for m in CHAT_MODELS if not str(m).startswith(RETIRED_PREFIXES)]
for _m in CURRENT_MODELS:
    if _m not in MODEL_OPTIONS:
        MODEL_OPTIONS.append(_m)
# Put the most reliable model first
MODEL_OPTIONS.sort(key=lambda m: (m != "gemini-3.8-flash", m))


def H(markup: str) -> str:
    """Strip indentation/blank lines so Markdown never treats HTML as a code block."""
    return "\n".join(line.strip() for line in markup.splitlines() if line.strip())


def esc(value) -> str:
    return html.escape(str(value), quote=True)


def default_api_key() -> str:
    """Streamlit secrets first (deployment), then .env / environment (local)."""
    try:
        if "GOOGLE_API_KEY" in st.secrets:
            return str(st.secrets["GOOGLE_API_KEY"]).strip()
    except Exception:
        pass
    return os.getenv("GOOGLE_API_KEY", "")


def invoke_with_retry(chain, inputs, tries=3):
    """Retry briefly when Gemini is overloaded (503) or rate-limited (429)."""
    for attempt in range(tries):
        try:
            return chain.invoke(inputs)
        except Exception as error:
            text = str(error)
            busy = any(
                code in text
                for code in ("503", "UNAVAILABLE", "429", "RESOURCE_EXHAUSTED", "overloaded")
            )
            daily_limit = "PerDay" in text or "per day" in text.lower()
            if busy and not daily_limit and attempt < tries - 1:
                time.sleep(2 * (attempt + 1))
                continue
            raise


def run_with_fallback(selected_model, inputs):
    """Run the chain; if a model is retired (404), try the next current model."""
    candidates = [selected_model] + [m for m in MODEL_OPTIONS if m != selected_model]
    last_error = None
    for candidate in candidates:
        try:
            chain, llm = get_medical_chain(candidate)
            raw = invoke_with_retry(chain, inputs)
            return raw, llm, candidate
        except Exception as error:
            last_error = error
            text = str(error)
            if any(k in text for k in ("404", "NOT_FOUND", "no longer available",
                                       "429", "RESOURCE_EXHAUSTED", "quota")):
                continue  # retired model or quota used up -> try the next model
            raise
    raise last_error or RuntimeError("No Gemini model is available right now.")


def explain_error(error):
    text = str(error)
    if "429" in text or "RESOURCE_EXHAUSTED" in text or "quota" in text.lower():
        return (
            "Daily free-tier quota used up on every available model. "
            "Try again tomorrow, create a new API key in a different Google AI Studio project, "
            "or enable billing on your project."
        )
    if "503" in text or "UNAVAILABLE" in text:
        return "Google's servers are busy right now. Please try again in a minute."
    return text



# ---------------------------------------------------------------------------
# Session memory (past assessments) + report generation
# ---------------------------------------------------------------------------
if "history" not in st.session_state:
    st.session_state.history = []
if "active_id" not in st.session_state:
    st.session_state.active_id = None


def get_record(record_id):
    for rec in st.session_state.history:
        if rec["id"] == record_id:
            return rec
    return None


def open_record(record_id):
    st.session_state.active_id = record_id


def new_assessment():
    st.session_state.active_id = None


def clear_history():
    st.session_state.history = []
    st.session_state.active_id = None


def md_to_html(text):
    """Tiny, safe Markdown -> HTML (bold, bullets, headings) for the report."""
    out, in_list = [], False
    for line in esc(text or "").splitlines():
        line = re.sub(r"\*\*(.+?)\*\*", r"<b>\1</b>", line.strip())
        if line.startswith(("* ", "- ", "• ")):
            if not in_list:
                out.append("<ul>")
                in_list = True
            out.append(f"<li dir='auto'>{line[2:]}</li>")
            continue
        if in_list:
            out.append("</ul>")
            in_list = False
        if line.startswith("#"):
            out.append(f"<h4 dir='auto'>{line.lstrip('#').strip()}</h4>")
        elif line:
            out.append(f"<p dir='auto'>{line}</p>")
    if in_list:
        out.append("</ul>")
    return "".join(out)


REPORT_CSS = """
*{box-sizing:border-box;} body{font-family:'Segoe UI','Nunito Sans',Arial,sans-serif; color:#1b2b3c; margin:0; background:#eef2f7;
-webkit-print-color-adjust:exact; print-color-adjust:exact;}
.page{max-width:820px; margin:24px auto; background:#fff; padding:0 0 28px; box-shadow:0 8px 30px rgba(0,0,0,.12); border-radius:6px; overflow:hidden;}
.head{background:linear-gradient(110deg,#06223f,#0b5cab); color:#fff; padding:26px 34px;}
.head h1{margin:0; font-size:26px;} .head p{margin:6px 0 0; opacity:.85; font-size:13px;}
.body{padding:8px 34px 0;}
h2{font-size:16px; color:#06223f; border-bottom:2px solid #0b5cab; padding-bottom:5px; margin:26px 0 10px; text-transform:uppercase; letter-spacing:.05em;}
h4{margin:10px 0 4px; color:#0b5cab;}
table{width:100%; border-collapse:collapse; font-size:14px;} td{padding:7px 10px; border:1px solid #dbe3ec; vertical-align:top;}
td.k{width:32%; background:#f3f6fa; font-weight:700;}
.badge{display:inline-block; padding:8px 18px; border-radius:999px; color:#fff; font-weight:800; letter-spacing:.04em;}
.cond{border:1px solid #dbe3ec; border-left:5px solid #0b5cab; border-radius:8px; padding:8px 14px; margin:8px 0; page-break-inside:avoid;}
.cond b{color:#06223f;} p,li{font-size:14px; line-height:1.6;} ul{padding-left:20px; margin:6px 0;}
.flags li{color:#8f1d14;}
.disc{margin:26px 34px 0; padding:12px 16px; background:#fff4f3; border:1px solid #f2c7c2; border-radius:8px; font-size:12.5px; color:#6b1a14;}
.foot{text-align:center; color:#7a8797; font-size:12px; margin-top:14px;}
.bar{max-width:820px; margin:18px auto 0; text-align:right;}
.bar button{background:#0b5cab; color:#fff; border:0; border-radius:999px; padding:10px 22px; font-weight:700; cursor:pointer; font-size:14px;}
@media print{ body{background:#fff;} .page{box-shadow:none; margin:0; max-width:none;} .noprint{display:none !important;} }
"""


def build_report_html(rec):
    inp, parsed = rec["inputs"], rec["parsed"]
    urgency = str(parsed.get("urgency_level", "LOW")).upper()
    _, _, accent, _, label, message = URGENCY.get(urgency, URGENCY["LOW"])

    rows = [
        ("Age", inp["age"]), ("Gender", inp["gender"]), ("Symptoms", inp["symptoms"]),
        ("Duration", inp["duration"]), ("Severity (1-10)", inp["severity"]),
        ("Existing conditions", inp["existing_conditions"]),
        ("Current medications", inp["current_medications"]),
        ("Additional notes", inp["notes"]), ("Answer language", inp["language"]),
    ]
    table = "".join(f"<tr><td class='k'>{esc(k)}</td><td dir='auto'>{esc(v)}</td></tr>" for k, v in rows)

    conds = ""
    for c in parsed.get("possible_conditions", []) or []:
        name = c.get("name", "Condition") if isinstance(c, dict) else str(c)
        reason = c.get("reason", "") if isinstance(c, dict) else ""
        conds += f"<div class='cond'><b dir='auto'>{esc(name)}</b><p dir='auto'>{esc(reason)}</p></div>"
    conds = conds or "<p>No specific conditions listed.</p>"

    steps = parsed.get("recommended_next_steps", "")
    steps_html = ("<ul>" + "".join(f"<li dir='auto'>{esc(x)}</li>" for x in as_list(steps)) + "</ul>"
                  if isinstance(steps, (list, tuple)) else md_to_html(str(steps)))
    questions = "".join(f"<li dir='auto'>{esc(q)}</li>" for q in as_list(parsed.get("questions_for_doctor"))) or "<li>None listed.</li>"
    flags = "".join(f"<li dir='auto'>{esc(w)}</li>" for w in as_list(parsed.get("warning_signs"))) or "<li>None listed.</li>"
    narrative = (f"<h2>Detailed Guidance</h2>{md_to_html(rec['narrative'])}" if rec.get("narrative") else "")
    stamp = rec["ts"].strftime("%d %B %Y, %I:%M %p")

    return f"""<!DOCTYPE html><html><head><meta charset="utf-8"><title>MediGuide AI Report - {stamp}</title>
<style>{REPORT_CSS}</style></head><body>
<div class="bar noprint"><button onclick="window.print()">🖨️ Print / Save as PDF</button></div>
<div class="page">
<div class="head"><h1>MediGuide AI — Symptom Assessment Report</h1>
<p>Generated {esc(stamp)} · Model: {esc(rec['model'])} · Report ID: {esc(rec['id'][:8].upper())}</p></div>
<div class="body">
<h2>Urgency</h2>
<span class="badge" style="background:{accent}">{esc(label.upper())}</span>
<p dir="auto">{esc(message)}</p>
<h2>Patient Information</h2><table>{table}</table>
<h2>Summary</h2><p dir="auto">{esc(parsed.get("summary", ""))}</p>
<h2>Possible Conditions (Educational Only)</h2>{conds}
<h2>Recommended Next Steps</h2>{steps_html}
<h2>Questions for Your Doctor</h2><ul>{questions}</ul>
<h2>Warning Signs (Red Flags)</h2><ul class="flags">{flags}</ul>
{narrative}
</div>
<div class="disc"><b>Medical disclaimer:</b> MediGuide AI is an educational AI prototype. This report is NOT a diagnosis and is not a
replacement for a licensed healthcare professional or emergency services. If you think you may be having a medical emergency,
call your local emergency number immediately.</div>
<div class="foot">MediGuide AI · Educational prototype</div>
</div></body></html>"""


def save_report_to_disk(rec):
    folder = Path("reports")
    folder.mkdir(exist_ok=True)
    path = folder / f"MediGuide_Report_{rec['ts']:%Y%m%d_%H%M%S}_{rec['id'][:6]}.html"
    path.write_text(build_report_html(rec), encoding="utf-8")
    return str(path)


def mark_report_saved(record_id):
    """Runs when the Download button is clicked: store a copy in the system."""
    rec = get_record(record_id)
    if not rec:
        return
    try:
        rec["saved_path"] = save_report_to_disk(rec)
    except Exception:
        rec["saved_path"] = "session history"


def print_button_html(report_html):
    payload = json.dumps(report_html).replace("</", "<\\/")
    return f"""
<style>
html,body{{margin:0; background:transparent;}}
button{{width:100%; background:#0f1015; color:#fff; border:2px solid #3b8de0; border-radius:999px; padding:12px 18px;
font:800 15px 'Nunito Sans','Segoe UI',Arial,sans-serif; cursor:pointer; transition:all .25s;}}
button:hover{{background:#3b8de0; transform:translateY(-3px); box-shadow:0 10px 24px rgba(59,141,224,.4);}}
</style>
<button id="p">🖨️ Print Report</button>
<script>
const REPORT = {payload};
document.getElementById('p').onclick = function(){{
  const w = window.open('', '_blank');
  if(!w){{ alert('Please allow pop-ups for this site to print the report.'); return; }}
  w.document.open(); w.document.write(REPORT); w.document.close();
  w.focus(); setTimeout(function(){{ w.print(); }}, 600);
}};
</script>"""


def report_actions(rec):
    report_html = build_report_html(rec)
    st.markdown(H("""
    <div class="section" style="margin-top:2rem;">
      <div class="section-title" style="font-size:1.5rem;">📄 Your report</div>
      <div class="rule"></div>
      <div class="section-sub">Print it, or download a copy. Every assessment is also kept in Past Assessments.</div>
    </div>
    """), unsafe_allow_html=True)

    c1, c2, c3 = st.columns([1, 1, 2], gap="medium")
    with c1:
        try:
            components.html(print_button_html(report_html), height=58)
        except Exception:
            st.markdown('<div class="note">Use the download button, then print the file.</div>', unsafe_allow_html=True)
    with c2:
        st.download_button(
            "⬇️ Download Report",
            data=report_html.encode("utf-8"),
            file_name=f"MediGuide_Report_{rec['ts']:%Y%m%d_%H%M}.html",
            mime="text/html",
            key=f"dl_{rec['id']}",
            on_click=mark_report_saved,
            args=(rec["id"],),
        )
    with c3:
        if rec.get("saved_path"):
            st.markdown(
                f'<div class="note">✅ Report added to the system: <b>{esc(rec["saved_path"])}</b></div>',
                unsafe_allow_html=True,
            )


history_state = {"rendered": False}


def render_history_sidebar():
    """Sidebar list of past assessments (like a chat history)."""
    if history_state["rendered"]:
        return
    history_state["rendered"] = True
    with history_box:
        st.markdown('<div class="side-title">🕘 Past Assessments</div>', unsafe_allow_html=True)
        st.button("➕ New Assessment", key="new_chat", on_click=new_assessment)
        if not st.session_state.history:
            st.markdown('<div class="note">Your assessments will appear here during this session.</div>', unsafe_allow_html=True)
        for rec in st.session_state.history:
            urgency = str(rec["parsed"].get("urgency_level", "LOW")).upper()
            icon = URGENCY.get(urgency, URGENCY["LOW"])[3]
            title = rec["title"] if len(rec["title"]) <= 26 else rec["title"][:25] + "…"
            saved = " 📄" if rec.get("saved_path") else ""
            st.button(
                f"{icon} {title} · {rec['ts']:%I:%M %p}{saved}",
                key=f"hist_{rec['id']}",
                on_click=open_record,
                args=(rec["id"],),
                type="primary" if rec["id"] == st.session_state.active_id else "secondary",
            )
        if st.session_state.history:
            st.button("🗑️ Clear history", key="clear_hist", on_click=clear_history)
            st.caption("Kept for this browser session only.")
        st.divider()


def show_saved_record(rec):
    """Re-display a past assessment from memory (no API call)."""
    st.markdown(H(f"""
    <div class="section" id="results">
      <div class="section-title">Past assessment · {esc(rec['ts'].strftime('%d %b %Y, %I:%M %p'))}</div>
      <div class="rule"></div>
      <div class="section-sub">{esc(rec['inputs']['symptoms'])} · Model: {esc(rec['model'])}</div>
    </div>
    """), unsafe_allow_html=True)
    tabs = st.tabs(["📋 Structured Summary", "💬 Live Narrative Stream", "🔍 Raw Output"])
    with tabs[0]:
        render_structured(rec["parsed"], rec["severity"])
    with tabs[1]:
        if rec.get("narrative"):
            st.markdown(rec["narrative"])
        else:
            st.markdown('<div class="note">No narrative was generated for this assessment.</div>', unsafe_allow_html=True)
    with tabs[2]:
        st.code(rec["raw"], language="json")
    report_actions(rec)
    try:
        components.html(
            "<script>setTimeout(function(){var e=window.parent.document.getElementById('results');"
            "if(e){e.scrollIntoView({behavior:'smooth'});}},350);</script>",
            height=0,
        )
    except Exception:
        pass


def loader(title, sub):
    return H(f"""
    <div class="loader-wrap">
      <div class="loader"></div>
      <div class="loader-text">{title}<span class="loader-dots"><i></i><i></i><i></i></span></div>
      <div class="loader-sub">{sub}</div>
    </div>
    """)


# ---------------------------------------------------------------------------
# Design system (inspired by Cleveland Clinic's clean, trustworthy look)
# ---------------------------------------------------------------------------
STYLE = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Nunito+Sans:wght@400;600;700;800;900&display=swap');

:root{
  color-scheme:dark;
  --bg:#0a0a0d; --surface:#121218; --surface2:#17181f; --soft:#14151b;
  --blue:#3b8de0; --blue-d:#2a6fb8; --blue-l:rgba(59,141,224,.14);
  --navy:#ffffff; --green:#19c39a;
  --line:#272a35; --ink:#e9edf4; --mute:#98a2b3;
  --shadow:0 8px 28px rgba(0,0,0,.55); --shadow-h:0 16px 40px rgba(0,0,0,.7);
}
html{scroll-behavior:smooth;}
html, body, .stApp, .stMarkdown, p, label, h1, h2, h3, h4, input, textarea, button{
  font-family:'Nunito Sans','Segoe UI',system-ui,-apple-system,sans-serif;
}
.stApp{background:var(--bg) !important; color:var(--ink);}
.stApp h1, .stApp h2, .stApp h3, .stApp h4{color:#fff;}
[data-testid="stMarkdownContainer"] p, [data-testid="stMarkdownContainer"] li{color:var(--ink);}
#MainMenu, footer, [data-testid="stDecoration"], [data-testid="stAppDeployButton"], .stAppDeployButton, .stDeployButton,
[data-testid="stMainMenu"]{display:none !important;}
[data-testid="stHeader"]{background:transparent !important; z-index:1000;}

/* ---- sidebar toggle: always visible so the sidebar can be reopened ---- */
[data-testid="stSidebarCollapseButton"]{opacity:1 !important; visibility:visible !important;}
[data-testid="stSidebarCollapseButton"] button, [data-testid="stSidebarCollapseButton"] svg{color:#fff !important;}
[data-testid="stExpandSidebarButton"], [data-testid="stSidebarCollapsedControl"], [data-testid="collapsedControl"]{
  display:flex !important; visibility:visible !important; opacity:1 !important; z-index:1001;
  background:var(--surface) !important; border:1px solid var(--line) !important; border-radius:12px !important;
  box-shadow:var(--shadow); transition:all .25s;}
[data-testid="stExpandSidebarButton"]:hover, [data-testid="stSidebarCollapsedControl"]:hover, [data-testid="collapsedControl"]:hover{
  border-color:var(--green) !important; transform:translateX(3px);}
[data-testid="stExpandSidebarButton"] *, [data-testid="stSidebarCollapsedControl"] *, [data-testid="collapsedControl"] *{
  color:#fff !important; fill:#fff !important; visibility:visible !important;}

/* ---- never dim elements while the script is running ---- */
[data-stale="true"], .stale-element, [data-testid="stElementContainer"][data-stale="true"]{
  opacity:1 !important; filter:none !important; transition:none !important;}
.block-container{max-width:1180px; padding-top:1rem; padding-bottom:0;}

/* ---------- animations ---------- */
@keyframes fadeUp{from{opacity:0; transform:translateY(18px);} to{opacity:1; transform:none;}}
@keyframes pulse{0%{box-shadow:0 0 0 0 rgba(25,195,154,.55);} 70%{box-shadow:0 0 0 10px rgba(25,195,154,0);} 100%{box-shadow:0 0 0 0 rgba(25,195,154,0);}}
@keyframes spin{to{transform:rotate(360deg);}}
@keyframes bounce{0%,80%,100%{transform:translateY(0); opacity:.4;} 40%{transform:translateY(-8px); opacity:1;}}
@keyframes glow{0%,100%{box-shadow:0 0 22px rgba(59,141,224,.25);} 50%{box-shadow:0 0 38px rgba(25,195,154,.45);}}
.fade-up{animation:fadeUp .7s cubic-bezier(.2,.7,.2,1) both;}

/* ---------- utility bar + nav ---------- */
.utility{display:flex; justify-content:space-between; gap:1rem; flex-wrap:wrap;
  background:var(--soft); border:1px solid var(--line); border-radius:10px; padding:.45rem 1rem; font-size:.8rem; color:var(--mute);}
.utility b{color:#fff;}
.dot{display:inline-block; width:8px; height:8px; border-radius:50%; background:var(--green); margin-right:.45rem; animation:pulse 2s infinite;}
.nav{display:flex; align-items:center; justify-content:space-between; padding:1rem .25rem; border-bottom:1px solid var(--line);}
.brand{display:flex; align-items:center; gap:.65rem; text-decoration:none !important;}
.brand-mark{width:40px; height:40px; border-radius:12px; background:linear-gradient(135deg,var(--blue),#19c39a);
  display:flex; align-items:center; justify-content:center; box-shadow:0 4px 16px rgba(59,141,224,.45); transition:transform .3s;}
.brand:hover .brand-mark{transform:rotate(-8deg) scale(1.06);}
.brand-name{font-size:1.45rem; font-weight:900; color:#fff !important; letter-spacing:-.02em;}
.brand-name span{color:var(--green);}
.nav-links a{margin-left:1.6rem; font-weight:700; font-size:.95rem; color:var(--ink) !important; text-decoration:none !important;
  padding-bottom:4px; border-bottom:2px solid transparent; transition:all .25s;}
.nav-links a:hover{color:var(--green) !important; border-color:var(--green);}

/* ---------- hero ---------- */
.hero{position:relative; overflow:hidden; margin-top:1.1rem; border-radius:22px; padding:3.6rem 3rem 3.2rem;
  border:1px solid rgba(255,255,255,.08);
  background:radial-gradient(900px 400px at 85% -10%, rgba(25,195,154,.28), transparent 60%),
             radial-gradient(700px 400px at 0% 110%, rgba(59,141,224,.38), transparent 60%),
             linear-gradient(110deg,#07101d 0%,#0a2a54 55%,#0d4a8c 100%);
  box-shadow:var(--shadow);}
.hero::after{content:""; position:absolute; right:-80px; top:-80px; width:380px; height:380px; border-radius:50%;
  border:60px solid rgba(255,255,255,.04);}
.eyebrow{display:inline-block; background:rgba(255,255,255,.12); color:#fff; font-weight:700; font-size:.78rem;
  letter-spacing:.08em; text-transform:uppercase; padding:.35rem .8rem; border-radius:999px; margin-bottom:1rem;}
.hero-title{font-size:clamp(2rem,4.4vw,3.3rem); line-height:1.08; font-weight:900; color:#fff !important; max-width:760px; letter-spacing:-.02em;}
.hero-sub{font-size:1.12rem; color:rgba(255,255,255,.88) !important; max-width:640px; margin:1rem 0 1.8rem; line-height:1.55;}
.btn{display:inline-block; border-radius:999px; padding:.8rem 1.7rem; font-weight:800; font-size:.98rem; margin:0 .7rem .6rem 0;
  text-decoration:none !important; transition:all .25s ease;}
.btn-primary{background:#fff; color:#0a2a54 !important;}
.btn-primary:hover{background:var(--green); color:#04110d !important; transform:translateY(-3px); box-shadow:0 10px 24px rgba(25,195,154,.4);}
.btn-outline{border:2px solid rgba(255,255,255,.7); color:#fff !important;}
.btn-outline:hover{background:rgba(255,255,255,.14); transform:translateY(-3px);}

/* ---------- section headings ---------- */
.section{margin:3rem 0 1.2rem; scroll-margin-top:70px;}
.section-title{font-size:2rem; font-weight:900; color:#fff !important; letter-spacing:-.02em; margin:0;}
.section-sub{color:var(--mute) !important; margin:.35rem 0 0; font-size:1.02rem;}
.rule{height:3px; width:64px; border-radius:3px; background:linear-gradient(90deg,var(--green),var(--blue)); margin:.8rem 0 0;}

/* ---------- cards ---------- */
.grid{display:grid; gap:1.1rem; grid-template-columns:repeat(4,1fr);}
.card{background:var(--surface); border:1px solid var(--line); border-radius:18px; padding:1.4rem 1.3rem;
  transition:transform .3s ease, box-shadow .3s ease, border-color .3s ease; animation:fadeUp .7s cubic-bezier(.2,.7,.2,1) both;}
.card:hover{transform:translateY(-6px); box-shadow:var(--shadow-h); border-color:var(--blue);}
.card:nth-child(2){animation-delay:.08s;} .card:nth-child(3){animation-delay:.16s;} .card:nth-child(4){animation-delay:.24s;}
.icon{width:54px; height:54px; border-radius:50%; background:var(--blue-l); display:flex; align-items:center; justify-content:center;
  font-size:1.5rem; margin-bottom:.9rem; transition:all .3s;}
.card:hover .icon{background:var(--blue); transform:scale(1.08) rotate(-6deg);}
.card .card-title{font-size:1.12rem; font-weight:800; color:#fff !important; margin:0 0 .3rem; padding:0;}
.card p{color:var(--mute) !important; font-size:.94rem; margin:0; line-height:1.5;}

/* ---------- safety callout ---------- */
.callout{display:flex; align-items:center; gap:1.2rem; background:linear-gradient(100deg,rgba(201,55,44,.16),rgba(201,55,44,.03));
  border:1px solid rgba(255,90,77,.3); border-left:6px solid #ff5a4d; border-radius:16px; padding:1.1rem 1.4rem; margin-top:1.4rem;}
.callout-ic{font-size:1.8rem;}
.callout b{color:#ff8f86;} .callout p{margin:0; color:var(--ink) !important; font-size:.95rem; line-height:1.5;}

/* ---------- form + inputs ---------- */
[data-testid="stForm"]{background:var(--surface); border:1px solid var(--line); border-radius:22px; padding:1.8rem 1.8rem 1.2rem;
  box-shadow:var(--shadow); animation:fadeUp .7s both;}
[data-baseweb="input"], [data-baseweb="base-input"], [data-baseweb="select"] > div, [data-baseweb="textarea"]{
  background:#0f1015 !important; border-radius:12px !important; transition:box-shadow .2s, border-color .2s;}
[data-baseweb="input"], [data-baseweb="select"] > div, [data-baseweb="textarea"]{border:1px solid var(--line) !important;}
[data-baseweb="input"] [data-baseweb="base-input"], [data-baseweb="textarea"] textarea{background:transparent !important;}
input, textarea{color:#ffffff !important; -webkit-text-fill-color:#ffffff !important; caret-color:#ffffff !important;}
input::placeholder, textarea::placeholder{color:#7b8494 !important; -webkit-text-fill-color:#7b8494 !important; opacity:1 !important;}
[data-baseweb="select"] div, [data-baseweb="select"] span{color:#ffffff;}
[data-baseweb="input"]:focus-within, [data-baseweb="select"] > div:focus-within, [data-baseweb="textarea"]:focus-within{
  box-shadow:0 0 0 3px rgba(59,141,224,.3) !important; border-color:var(--blue) !important;}
span[data-baseweb="tag"]{background:var(--blue) !important; color:#fff !important; border-radius:8px !important;}
[data-testid="stSlider"] [role="slider"]{background:var(--blue) !important; box-shadow:0 0 0 4px rgba(59,141,224,.25) !important;}
[data-testid="stSlider"] [data-testid="stThumbValue"]{color:var(--blue) !important; font-weight:800;}
[data-testid="stSliderTickBarMin"], [data-testid="stSliderTickBarMax"]{color:var(--mute) !important;}
[data-baseweb="popover"] [data-baseweb="menu"], [data-baseweb="popover"] ul{background:#16171d !important;}
[data-baseweb="popover"] li{color:#eef1f6 !important;} [data-baseweb="popover"] li:hover{background:rgba(59,141,224,.2) !important;}
[data-testid="stWidgetLabel"] p, [data-testid="stWidgetLabel"] label{font-weight:700; color:#e9edf4 !important;}
[data-testid="stFormSubmitButton"] button, .stButton > button{
  background:var(--blue); color:#fff; border:none; border-radius:999px; padding:.7rem 2.2rem; font-weight:800; font-size:1rem;
  transition:all .25s ease;}
[data-testid="stFormSubmitButton"] button:hover, .stButton > button:hover{
  background:var(--green); color:#04110d; transform:translateY(-3px); box-shadow:0 10px 24px rgba(25,195,154,.4);}
[data-testid="stFormSubmitButton"] button, [data-testid="stDownloadButton"] button{
  background:var(--blue) !important; color:#fff !important; border:none !important; opacity:1 !important;}
[data-testid="stFormSubmitButton"] button:hover, [data-testid="stDownloadButton"] button:hover{
  background:var(--green) !important; color:#04110d !important;}
[data-testid="stFormSubmitButton"] button:focus:not(:hover), [data-testid="stFormSubmitButton"] button:active:not(:hover),
[data-testid="stDownloadButton"] button:focus:not(:hover), [data-testid="stDownloadButton"] button:active:not(:hover){
  background:var(--blue-d) !important; color:#fff !important; outline:none !important; box-shadow:0 0 0 3px rgba(59,141,224,.35) !important;}
[data-testid="stFormSubmitButton"] button p, [data-testid="stDownloadButton"] button p{color:inherit !important;}
[data-testid="stFormSubmitButton"] button:active{transform:translateY(0);}
.form-head{font-weight:900; color:#fff !important; font-size:1.1rem; margin:.2rem 0 .6rem;}

/* ---------- sidebar ---------- */
[data-testid="stSidebar"]{background:#0d0d11 !important; border-right:1px solid var(--line);}
[data-testid="stSidebar"] h1, [data-testid="stSidebar"] h2, [data-testid="stSidebar"] h3{color:#fff;}
.note{background:var(--blue-l); border:1px solid rgba(59,141,224,.35); color:#cfe3fa; border-radius:12px; padding:.8rem 1rem; font-size:.9rem; line-height:1.5;}

/* ---------- tabs ---------- */
button[data-baseweb="tab"]{font-weight:800; font-size:1rem; color:#aab3c2 !important; transition:color .2s, background .2s; border-radius:10px 10px 0 0;}
button[data-baseweb="tab"] p{color:inherit !important;}
button[data-baseweb="tab"]:hover{background:var(--blue-l); color:#fff !important;}
button[data-baseweb="tab"][aria-selected="true"]{color:#fff !important;}
[data-baseweb="tab-highlight"]{background:var(--green) !important; height:3px !important;}
[data-baseweb="tab-border"]{background:var(--line) !important;}

/* ---------- expanders ---------- */
[data-testid="stExpander"]{border:1px solid var(--line) !important; border-radius:14px !important; background:var(--surface);
  transition:box-shadow .25s, border-color .25s, transform .25s; animation:fadeUp .6s both;}
[data-testid="stExpander"]:hover{box-shadow:var(--shadow); border-color:var(--blue) !important; transform:translateY(-2px);}
[data-testid="stExpander"] summary p{font-weight:800; color:#fff;}
[data-testid="stExpander"] p{color:var(--ink);}

/* ---------- loader ---------- */
.loader-wrap{display:flex; flex-direction:column; align-items:center; text-align:center; padding:2.6rem 1rem; margin:.6rem 0;
  background:var(--surface); border:1px solid var(--line); border-radius:20px; animation:fadeUp .5s both;}
.loader{width:68px; height:68px; border-radius:50%; border:5px solid rgba(255,255,255,.08);
  border-top-color:var(--green); border-right-color:var(--blue); animation:spin 1s linear infinite, glow 2.4s ease-in-out infinite;}
.loader-text{margin-top:1.3rem; font-size:1.25rem; font-weight:900; color:#fff !important;}
.loader-sub{margin-top:.3rem; color:var(--mute) !important; font-size:.95rem;}
.loader-dots{display:inline-flex; gap:5px; margin-left:8px; vertical-align:middle;}
.loader-dots i{width:7px; height:7px; border-radius:50%; background:var(--green); animation:bounce 1.2s infinite ease-in-out;}
.loader-dots i:nth-child(2){animation-delay:.15s;} .loader-dots i:nth-child(3){animation-delay:.3s;}

/* ---------- results ---------- */
.urgency{display:flex; align-items:center; gap:1.1rem; border-radius:18px; padding:1.2rem 1.5rem; margin:.6rem 0 1.1rem; animation:fadeUp .6s both;}
.urgency .u-ic{font-size:2.2rem;}
.urgency .u-label{font-size:.78rem; font-weight:800; letter-spacing:.1em; text-transform:uppercase; opacity:.8;}
.urgency .u-main{font-size:1.45rem; font-weight:900; line-height:1.15;}
.urgency .u-sub{font-size:.98rem; margin-top:.15rem;}
.tiles{display:grid; grid-template-columns:repeat(4,1fr); gap:.9rem; margin-bottom:1.4rem;}
.tile{background:var(--surface); border-radius:16px; padding:1rem 1.1rem; border:1px solid var(--line); transition:transform .25s, box-shadow .25s; animation:fadeUp .6s both;}
.tile:hover{transform:translateY(-4px); box-shadow:var(--shadow);}
.tile b{display:block; font-size:1.7rem; font-weight:900; color:var(--blue);}
.tile span{font-size:.78rem; font-weight:700; letter-spacing:.08em; text-transform:uppercase; color:var(--mute);}
.panel{background:var(--surface); border:1px solid var(--line); border-radius:18px; padding:1.3rem 1.5rem; margin:.4rem 0 1.2rem; animation:fadeUp .6s both;}
.panel-title{font-size:1.25rem; font-weight:900; color:#fff !important; margin:0 0 .5rem;}
.panel p, .panel li{color:var(--ink) !important; font-size:1rem; line-height:1.65;}
.panel.next{border-left:6px solid var(--blue); background:linear-gradient(100deg,rgba(59,141,224,.16),var(--surface) 70%);}
.panel.flags{border-left:6px solid #ff5a4d; background:linear-gradient(100deg,rgba(255,90,77,.14),var(--surface) 70%);}
.panel.ask{border-left:6px solid var(--green); background:linear-gradient(100deg,rgba(25,195,154,.14),var(--surface) 70%);}
.panel ul{padding-left:1.1rem; margin:0;} .panel li{margin-bottom:.5rem;}
.sub-head{font-size:1.25rem; font-weight:900; color:#fff; margin:1.2rem 0 .6rem;}

/* ---------- footer ---------- */
.footer{margin:3.5rem -4rem 0; background:#050507; border-top:1px solid var(--line); color:#aeb8c7; padding:3rem 4rem 2rem; border-radius:24px 24px 0 0;}
.footer-grid{display:grid; grid-template-columns:1.4fr 1fr 1fr; gap:2rem;}
.footer h5{color:#fff !important; font-size:1rem; font-weight:800; margin:0 0 .8rem;}
.footer p, .footer li{color:#aeb8c7 !important; font-size:.9rem; line-height:1.6;}
.footer ul{list-style:none; padding:0; margin:0;}
.footer li{padding:.18rem 0; transition:transform .2s, color .2s;} .footer li:hover{transform:translateX(5px); color:#fff !important;}
.footer-bottom{border-top:1px solid rgba(255,255,255,.1); margin-top:2rem; padding-top:1.1rem; font-size:.8rem; color:#7f8ba0 !important;}

/* ---------- history + report buttons ---------- */
.side-title{font-size:1.05rem; font-weight:900; color:#fff; margin:.2rem 0 .6rem;}
[data-testid="stSidebar"] .stButton > button{width:100%; justify-content:flex-start; text-align:left; background:transparent; color:#dfe5ee;
  border:1px solid var(--line); border-radius:12px; padding:.5rem .8rem; font-weight:700; font-size:.86rem; box-shadow:none; margin-bottom:.15rem;}
[data-testid="stSidebar"] .stButton > button p{white-space:nowrap; overflow:hidden; text-overflow:ellipsis; margin:0;}
[data-testid="stSidebar"] .stButton > button:hover{background:var(--blue-l); border-color:var(--blue); color:#fff; transform:none; box-shadow:none;}
[data-testid="stSidebar"] .stButton > button[kind="primary"], [data-testid="stSidebar"] button[data-testid="stBaseButton-primary"]{
  background:var(--blue-l); border-color:var(--blue); color:#fff;}
.st-key-new_chat button{background:linear-gradient(135deg,var(--blue),#19c39a) !important; color:#fff !important; border:none !important;
  justify-content:center !important; text-align:center !important; font-weight:800 !important; margin-bottom:.6rem !important;}
.st-key-new_chat button:hover{transform:translateY(-2px) !important; box-shadow:0 8px 20px rgba(25,195,154,.35) !important;}
[data-testid="stDownloadButton"] button{background:var(--blue); color:#fff; border:none; border-radius:999px; padding:.7rem 1.6rem; font-weight:800;
  width:100%; transition:all .25s ease;}
[data-testid="stDownloadButton"] button:hover{background:var(--green); color:#04110d; transform:translateY(-3px); box-shadow:0 10px 24px rgba(25,195,154,.4);}

@media (max-width:900px){
  .grid, .tiles{grid-template-columns:repeat(2,1fr);}
  .footer-grid{grid-template-columns:1fr;} .hero{padding:2.4rem 1.5rem;} .nav-links{display:none;}
  .footer{margin:3rem -1rem 0; padding:2rem 1.2rem;}
}
@media (max-width:560px){ .grid, .tiles{grid-template-columns:1fr;} }
</style>
"""
st.markdown(H(STYLE), unsafe_allow_html=True)

# ---------------------------------------------------------------------------
# Sidebar (functionality unchanged)
# ---------------------------------------------------------------------------
with st.sidebar:
    history_box = st.container()
    st.header("⚙️ Settings")

    api_key_input = st.text_input(
        "Google API Key",
        type="password",
        value=default_api_key(),
        help="Stored only for this session. On deployment, use Streamlit Secrets.",
    )

    model_name = st.selectbox("Select Chat Model", MODEL_OPTIONS)

    stream_enabled = st.checkbox(
        "Generate live narrative",
        value=True,
        help="Uses 1 extra API request per assessment. Turn off to save free-tier quota.",
    )

    st.divider()
    st.subheader("🚀 Performance")
    cache_option = st.selectbox(
        "Select Cache Type",
        ["None", "In-Memory (RAM)", "SQLite (Disk)"],
    )
    setup_cache(cache_option)

    st.divider()
    st.markdown("### ℹ️ Disclaimer")
    st.markdown(
        '<div class="note">This tool provides preliminary educational guidance only. '
        'Always consult a qualified healthcare professional for medical concerns.</div>',
        unsafe_allow_html=True,
    )

if api_key_input:
    os.environ["GOOGLE_API_KEY"] = api_key_input

# ---------------------------------------------------------------------------
# Header, hero and landing sections
# ---------------------------------------------------------------------------
st.markdown(H("""
<div class="utility fade-up">
  <span><span class="dot"></span><b>Educational prototype</b> · Not a medical service</span>
  <span>In an emergency, call your local emergency number immediately</span>
</div>
<div class="nav fade-up">
  <a class="brand" href="#top" target="_self">
    <div class="brand-mark">
      <svg width="22" height="22" viewBox="0 0 24 24" fill="none"><path d="M9 3h6v6h6v6h-6v6H9v-6H3V9h6V3z" fill="#fff"/></svg>
    </div>
    <div class="brand-name">MediGuide<span> AI</span></div>
  </a>
  <div class="nav-links">
    <a href="#assessment" target="_self">Symptom Check</a>
    <a href="#how" target="_self">How It Works</a>
    <a href="#safety" target="_self">Safety</a>
  </div>
</div>
<div class="hero fade-up">
  <div class="eyebrow">AI-Powered Patient Guidance</div>
  <div class="hero-title">We're here to help you make sense of your symptoms.</div>
  <div class="hero-sub">Share how you're feeling and get clear, safety-first guidance: how urgent it may be,
  what could be going on, and what to ask your doctor.</div>
  <a class="btn btn-primary" href="#assessment" target="_self">Start Assessment</a>
  <a class="btn btn-outline" href="#how" target="_self">How It Works</a>
</div>
<div class="section" id="how">
  <div class="section-title">Care guidance, clearly explained</div>
  <div class="rule"></div>
  <div class="section-sub">Everything you need to prepare for a conversation with your healthcare provider.</div>
</div>
<div class="grid">
  <div class="card"><div class="icon">📋</div><div class="card-title">Symptom Review</div>
    <p>Tell us your age, symptoms, duration and existing conditions.</p></div>
  <div class="card"><div class="icon">🚦</div><div class="card-title">Urgency Level</div>
    <p>See whether to rest at home, book a visit, or seek urgent help.</p></div>
  <div class="card"><div class="icon">❓</div><div class="card-title">Questions for Your Doctor</div>
    <p>Walk into your appointment prepared with the right questions.</p></div>
  <div class="card"><div class="icon">🚩</div><div class="card-title">Red-Flag Warning Signs</div>
    <p>Know exactly which symptoms mean you should get help right away.</p></div>
</div>
<div class="callout" id="safety">
  <div class="callout-ic">🚨</div>
  <p><b>Important medical &amp; safety notice:</b> MediGuide AI is an educational AI prototype only. It is NOT a
  replacement for a licensed doctor, professional diagnosis, or emergency services. Never use this for actual
  medical treatment or life-threatening situations.</p>
</div>
<div class="section" id="assessment">
  <div class="section-title">Start your assessment</div>
  <div class="rule"></div>
  <div class="section-sub">Tell us about the patient and the symptoms. The more detail, the better the guidance.</div>
</div>
"""), unsafe_allow_html=True)

# ---------------------------------------------------------------------------
# Patient form (fields unchanged)
# ---------------------------------------------------------------------------
with st.form("symptom_form"):
    col1, col2 = st.columns(2, gap="large")

    with col1:
        st.markdown('<div class="form-head">👤 Patient & Symptoms</div>', unsafe_allow_html=True)
        age = st.text_input("Patient Age", placeholder="e.g., 30")
        gender = st.selectbox("Gender", GENDERS)
        selected_symptoms = st.multiselect("Select Symptoms", COMMON_SYMPTOMS)
        custom_symptoms = st.text_input("Additional Symptoms (Optional)", placeholder="e.g., mild shivering")
        duration = st.selectbox("Duration of Symptoms", DURATIONS)

    with col2:
        st.markdown('<div class="form-head">🩺 Medical Background</div>', unsafe_allow_html=True)
        severity = st.slider("Symptom Severity (1 = Mild, 10 = Severe)", 1, 10, 3)
        existing_conditions = st.text_area("Existing Medical Conditions", placeholder="e.g., Asthma, Diabetes")
        current_medicines = st.text_area("Current Medications", placeholder="e.g., Inhaler as needed")
        notes = st.text_area("Additional Notes", placeholder="Any other relevant details...")
        language = st.selectbox("Answer Language", LANGUAGES)

    submitted = st.form_submit_button("🔍 Run Assessment")

# ---------------------------------------------------------------------------
# Results
# ---------------------------------------------------------------------------
URGENCY = {
    "EMERGENCY": ("#ff9a92", "rgba(201,55,44,.15)", "#ff5a4d", "🚨", "Emergency", "Seek immediate emergency medical help!"),
    "HIGH": ("#ffb27a", "rgba(224,112,31,.15)", "#ff8a3d", "⚠️", "High", "Consult a healthcare professional promptly."),
    "MEDIUM": ("#ffd666", "rgba(224,168,0,.14)", "#f2b705", "⚡", "Medium", "Schedule an appointment with a doctor soon."),
    "LOW": ("#5ee0b8", "rgba(25,195,154,.15)", "#19c39a", "✅", "Low", "Monitor symptoms at home and rest."),
}


def as_list(value):
    if not value:
        return []
    if isinstance(value, (list, tuple)):
        return [str(v) for v in value]
    return [str(value)]


def bullets(items, marker=""):
    return "".join(f"<li>{marker}{esc(i)}</li>" for i in items) or "<li>Nothing listed.</li>"


def render_structured(parsed_data, severity_value):
    urgency = str(parsed_data.get("urgency_level", "LOW")).upper()
    fg, bg, accent, icon, label, message = URGENCY.get(urgency, URGENCY["LOW"])
    conditions = parsed_data.get("possible_conditions", []) or []
    warnings = as_list(parsed_data.get("warning_signs", []))
    questions = as_list(parsed_data.get("questions_for_doctor", []))

    st.markdown(H(f"""
    <div class="urgency" style="background:{bg}; border:1px solid {accent}55; border-left:8px solid {accent}; color:{fg};">
      <div class="u-ic">{icon}</div>
      <div>
        <div class="u-label">Urgency level</div>
        <div class="u-main">{esc(label)}</div>
        <div class="u-sub">{esc(message)}</div>
      </div>
    </div>
    <div class="tiles">
      <div class="tile"><b>{esc(label)}</b><span>Urgency</span></div>
      <div class="tile"><b>{esc(severity_value)}/10</b><span>Reported severity</span></div>
      <div class="tile"><b>{len(conditions)}</b><span>Possible conditions</span></div>
      <div class="tile"><b>{len(warnings)}</b><span>Warning signs</span></div>
    </div>
    <div class="panel">
      <div class="panel-title">📌 Summary</div>
      <p>{esc(parsed_data.get("summary", "No summary provided."))}</p>
    </div>
    <div class="sub-head">🩺 Possible Conditions (For Education Only)</div>
    """), unsafe_allow_html=True)

    if conditions:
        for cond in conditions:
            name = cond.get("name", "Condition") if isinstance(cond, dict) else str(cond)
            reason = cond.get("reason", "") if isinstance(cond, dict) else ""
            with st.expander(f"📌 {name}"):
                st.write(reason)
    else:
        st.write("No specific conditions listed.")

    next_steps = parsed_data.get("recommended_next_steps", "")
    if isinstance(next_steps, (list, tuple)):
        next_html = f"<ul>{bullets(as_list(next_steps))}</ul>"
    else:
        next_html = f"<p>{esc(next_steps)}</p>"

    st.markdown(H(f"""
    <div class="panel next" style="margin-top:1.2rem;">
      <div class="panel-title">👣 Recommended Next Steps</div>
      {next_html}
    </div>
    """), unsafe_allow_html=True)

    col_q, col_w = st.columns(2, gap="medium")
    with col_q:
        st.markdown(H(f"""
        <div class="panel ask">
          <div class="panel-title">❓ Questions for Your Doctor</div>
          <ul>{bullets(questions)}</ul>
        </div>
        """), unsafe_allow_html=True)
    with col_w:
        st.markdown(H(f"""
        <div class="panel flags">
          <div class="panel-title">⚠️ Warning Signs (Red Flags)</div>
          <ul>{bullets(warnings, "🔴 ")}</ul>
        </div>
        """), unsafe_allow_html=True)


if submitted:
    if not api_key_input:
        st.error("Please enter your Google API Key in the sidebar to proceed.")
        render_history_sidebar()
        st.stop()

    if not selected_symptoms and not custom_symptoms:
        st.warning("⚠️ Please select or enter at least one symptom before submitting.")
        render_history_sidebar()
        st.stop()

    # Combine selected and custom symptoms
    all_symptoms_list = selected_symptoms.copy()
    if custom_symptoms:
        all_symptoms_list.append(custom_symptoms)
    symptoms_str = ", ".join(all_symptoms_list)

    inputs = {
        "age": age if age else "Not specified",
        "gender": gender,
        "symptoms": symptoms_str,
        "duration": duration,
        "severity": str(severity),
        "existing_conditions": existing_conditions if existing_conditions else "None reported",
        "current_medications": current_medicines if current_medicines else "None reported",
        "notes": notes if notes else "None",
        "language": language,
    }

    st.markdown(H("""
    <div class="section" id="results">
      <div class="section-title">Your assessment results</div>
      <div class="rule"></div>
      <div class="section-sub">Educational guidance based on the information you provided.</div>
    </div>
    """), unsafe_allow_html=True)

    raw_response = None
    llm = None
    parsed_ok = None
    narrative_text = ""
    used_model_final = model_name
    tabs = st.tabs(["📋 Structured Summary", "💬 Live Narrative Stream", "🔍 Raw Output"])

    with tabs[0]:
        slot = st.empty()
        slot.markdown(
            loader("Analyzing your symptoms", "Generating your guidance. This usually takes a few seconds."),
            unsafe_allow_html=True,
        )
        try:
            raw_response, llm, used_model = run_with_fallback(model_name, inputs)
            parsed_data = parse_llm_json(raw_response)
            slot.empty()
            if used_model != model_name:
                st.markdown(
                    f'<div class="note">⚠️ <b>{esc(model_name)}</b> is no longer available, '
                    f'so <b>{esc(used_model)}</b> was used instead.</div>',
                    unsafe_allow_html=True,
                )

            if "error" in parsed_data:
                st.error(parsed_data["error"])
                st.text(parsed_data.get("raw_output", ""))
            else:
                parsed_ok = parsed_data
                used_model_final = used_model
                render_structured(parsed_data, severity)

        except Exception as e:
            slot.empty()
            st.error(f"An error occurred during execution: {explain_error(e)}")

    with tabs[1]:
        st.markdown('<div class="sub-head">✍️ Live Streaming Guidance</div>', unsafe_allow_html=True)
        slot2 = st.empty()
        slot2.markdown(
            loader("Loading results", "Preparing your live narrative."),
            unsafe_allow_html=True,
        )

        def narrative_stream():
            first = True
            for chunk in stream_narrative(llm, inputs):
                if first:
                    slot2.empty()
                    first = False
                yield chunk

        if not stream_enabled:
            slot2.empty()
            st.markdown(
                '<div class="note">Live narrative is turned off in the sidebar (saves API quota).</div>',
                unsafe_allow_html=True,
            )
        elif llm is None:
            slot2.empty()
            st.markdown(
                '<div class="note">The narrative is unavailable because the assessment did not complete.</div>',
                unsafe_allow_html=True,
            )
        else:
            try:
                narrative_text = st.write_stream(narrative_stream()) or ""
            except Exception as e:
                slot2.empty()
                st.error(f"Streaming failed: {explain_error(e)}")
        slot2.empty()

    with tabs[2]:
        st.markdown('<div class="sub-head">🔍 Raw Model Output Debugger</div>', unsafe_allow_html=True)
        if raw_response is not None:
            st.code(raw_response, language="json")
        else:
            st.markdown('<div class="note">Submit the form to view raw outputs.</div>', unsafe_allow_html=True)

    # ---- save to session memory + show report actions ----
    if parsed_ok is not None:
        record = {
            "id": uuid.uuid4().hex,
            "ts": datetime.now(),
            "title": symptoms_str or "Assessment",
            "inputs": inputs,
            "severity": severity,
            "model": used_model_final,
            "parsed": parsed_ok,
            "raw": raw_response,
            "narrative": narrative_text,
            "saved_path": None,
        }
        st.session_state.history.insert(0, record)
        st.session_state.active_id = record["id"]
        render_history_sidebar()
        report_actions(record)
    else:
        render_history_sidebar()

elif st.session_state.active_id and get_record(st.session_state.active_id):
    render_history_sidebar()
    show_saved_record(get_record(st.session_state.active_id))

else:
    render_history_sidebar()

# ---------------------------------------------------------------------------
# Footer
# ---------------------------------------------------------------------------
st.markdown(H("""
<div class="footer">
  <div class="footer-grid">
    <div>
      <h5>About MediGuide AI</h5>
      <p>An educational Streamlit project that turns symptom information into safety-first,
      easy-to-understand guidance using LangChain and Google Gemini.</p>
    </div>
    <div>
      <h5>Safety First</h5>
      <ul>
        <li>Never a confirmed diagnosis</li>
        <li>Not for emergencies</li>
        <li>Always consult a clinician</li>
        <li>Your inputs are not stored</li>
      </ul>
    </div>
    <div>
      <h5>Built With</h5>
      <ul>
        <li>Streamlit</li>
        <li>LangChain (LCEL)</li>
        <li>Google Gemini</li>
        <li>In-memory &amp; SQLite caching</li>
      </ul>
    </div>
  </div>
  <div class="footer-bottom">
    MediGuide AI is an educational prototype and does not provide medical advice.
    If you think you may be having a medical emergency, call your local emergency number now.
  </div>
</div>
"""), unsafe_allow_html=True)