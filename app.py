"""MedShortage AI - Streamlit dashboard for predicting medicine shortages before they occur.
Run:  streamlit run app.py
"""
from __future__ import annotations

import difflib
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

import pipeline as pl
import inspect

WIDE = {"width": "stretch"} if "width" in inspect.signature(st.plotly_chart).parameters else {"use_container_width": True}

HERE = Path(__file__).parent
DATA_DIR, MODEL_DIR = HERE / "data", HERE / "models"
REAL_CSV, DEMO_CSV = DATA_DIR / "shortages_raw.csv", DATA_DIR / "DEMO_sample_shortages.csv"

# Validated categorical slots (blue, orange, aqua) + reserved status colours (always shown with icon + label)
BLUE, ORANGE, AQUA, GREY = "#2a78d6", "#eb6834", "#1baf7a", "#8a8a85"   # categorical slots (multi-series charts)
PRIMARY, PRIMARY_LIGHT = "#1E8A5A", "#BFE3CF"                            # brand green for single-series charts
SEQ = ["#E8F5EE", "#C4E6D2", "#97D2B2", "#66BB8F", "#3A9F70", "#1F7F55", "#0F5B38"]
RISK = {"High": ("#d03b3b", "▲"), "Elevated": ("#fab219", "●"), "Low": ("#0ca30c", "▼")}
INK, INK2 = "#0C1F16", "#58695F"

st.set_page_config(page_title="MedShortage AI", page_icon="💊", layout="wide", initial_sidebar_state="collapsed")
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Instrument+Serif:ital@0;1&family=DM+Sans:wght@400;500;600;700&display=swap');
:root{--ink:#0C1F16;--ink2:#58695F;--line:#E3E9E4;--canvas:#E9EEEA;--card:#F3F6F3;--green:#2E9E5B;--green-d:#1E8A5A;--mint:#A7E8C0;--band:420px;}
.stApp{font-family:'DM Sans',system-ui,-apple-system,'Segoe UI',sans-serif;background:var(--canvas);color:var(--ink);}
.stApp h1,.stApp h2,.stApp h3,.stApp h4,.stApp h5,.stApp p,.stApp label,.stApp button,.stApp input,.stApp textarea,.stApp li,.stApp td,.stApp th,.stApp summary{font-family:'DM Sans',system-ui,sans-serif;}
.stApp h2,.stApp h3{font-family:'Instrument Serif',Georgia,serif !important;font-weight:400 !important;letter-spacing:-.01em;color:var(--ink);}
.stApp h2{font-size:1.9rem !important;padding-top:.3rem;}
.stApp h3{font-size:1.5rem !important;}
.stApp h4,.stApp h5{font-size:1rem;font-weight:650;color:var(--ink);}
[data-testid="stToolbar"],[data-testid="stDecoration"],[data-testid="stStatusWidget"],header[data-testid="stHeader"],footer,#MainMenu,
[data-testid="stSidebar"],[data-testid="stSidebarCollapsedControl"],[data-testid="stExpandSidebarButton"]{display:none !important;}
/* dark hero band behind the top of every page */
[data-testid="stMainBlockContainer"]{max-width:100% !important;padding:10px 10px 28px !important;position:relative;}
[data-testid="stMainBlockContainer"]::before{content:"";position:absolute;top:10px;left:10px;right:10px;height:var(--band);border-radius:34px;z-index:0;
 background:radial-gradient(55% 90% at 84% 22%,rgba(52,209,127,.50),transparent 62%),radial-gradient(45% 70% at 6% 100%,rgba(30,138,90,.40),transparent 66%),
 linear-gradient(180deg,#06130D 0%,#0A2418 100%);}
[data-testid="stMainBlockContainer"]>div{position:relative;z-index:1;}
/* top navigation */
.st-key-topnav{padding:8px 14px 0;}
.st-key-topnav [data-testid="stRadioGroup"]{flex-direction:row !important;flex-wrap:nowrap;gap:2px;width:fit-content;margin:0 auto;padding:4px;border-radius:999px;background:rgba(255,255,255,.08);border:1px solid rgba(255,255,255,.14);backdrop-filter:blur(8px);}
.st-key-topnav [data-testid="stRadioGroup"]>div{width:auto;}
.st-key-topnav [data-testid="stRadioOption"]{padding:8px 17px;border-radius:999px;margin:0;}
.st-key-topnav [data-testid="stRadioOption"]>div>div:first-child{display:none;}
.st-key-topnav [data-testid="stRadioOption"] p{color:rgba(255,255,255,.74) !important;font-size:.9rem;margin:0;white-space:nowrap;}
.st-key-topnav [data-testid="stRadioOption"]:hover p{color:#fff !important;}
.st-key-topnav [data-testid="stRadioOption"][data-selected="true"]{background:#fff;}
.st-key-topnav [data-testid="stRadioOption"][data-selected="true"] p{color:#0C1F16 !important;font-weight:600;}
.brand{display:flex;align-items:center;gap:11px;}
.brand b{display:block;color:#fff;font-size:1.12rem;font-weight:700;letter-spacing:-.01em;line-height:1.15;}
.brand span{display:block;font-size:.74rem;color:rgba(255,255,255,.62);}
.chip{display:flex;align-items:center;gap:10px;justify-content:flex-end;margin-left:auto;width:fit-content;background:rgba(255,255,255,.08);border:1px solid rgba(255,255,255,.16);border-radius:999px;padding:7px 16px 7px 12px;}
.chip .d{width:9px;height:9px;border-radius:50%;flex:none;}
.chip b{display:block;font-size:.82rem;font-weight:600;line-height:1.2;color:#fff;}
.chip span{display:block;font-size:.72rem;color:rgba(255,255,255,.64);line-height:1.25;}
/* hero */
.hero{position:relative;padding:46px 30px 0;min-height:var(--hero,230px);}
.hero h1{font-family:'Instrument Serif',Georgia,serif !important;font-weight:400 !important;font-size:clamp(2.4rem,4.6vw,4rem) !important;line-height:1.03 !important;letter-spacing:-.02em;color:#F3F8F4 !important;max-width:640px;margin:0 !important;padding:0 !important;}
.hero h1 em{font-style:italic;color:var(--mint);}
.hero p{color:rgba(243,248,244,.74) !important;font-size:1.04rem;max-width:48ch;margin:16px 0 0;line-height:1.55;}
.hero .acts{display:flex;gap:12px;margin-top:26px;flex-wrap:wrap;}
.hero a.cta{background:var(--green);color:#fff !important;border-radius:999px;padding:12px 26px;font-weight:600;font-size:.95rem;text-decoration:none !important;box-shadow:0 10px 26px -10px rgba(46,158,91,.9);}
.hero a.cta:hover{background:#27874D;}
.hero a.ghost{color:#fff !important;border:1px solid rgba(255,255,255,.34);border-radius:999px;padding:11px 24px;font-weight:500;font-size:.95rem;text-decoration:none !important;}
.hero a.ghost:hover{background:rgba(255,255,255,.08);}
.hero img.art{position:absolute;right:6px;top:-6px;pointer-events:none;}
.hero img.art.big{height:420px;top:4px;}
.hero img.art.small{height:225px;right:34px;top:-12px;}
.hero.sm h1{font-size:clamp(2.1rem,3.6vw,3.1rem) !important;}
.hero.sm{padding-top:40px;}
/* the white sheet that rises over the hero */
.st-key-sheet{background:#fff;border-radius:32px;padding:30px 28px 38px;margin-top:-86px;position:relative;z-index:2;box-shadow:0 34px 70px -38px rgba(5,28,17,.55);}
.sec{margin:34px 0 14px;}
.sec h2{font-family:'Instrument Serif',Georgia,serif !important;font-weight:400 !important;font-size:2.15rem !important;line-height:1.1;letter-spacing:-.015em;margin:0 !important;padding:0 !important;color:var(--ink);}
.sec h2 em{font-style:italic;color:var(--green-d);}
.sec p{color:var(--ink2);margin:6px 0 0;max-width:62ch;font-size:.97rem;}
/* cards */
.kpi{min-height:128px;background:var(--card);border-radius:22px;padding:18px 20px;}
.kpi .l{font-size:.86rem;color:var(--ink2);padding-right:52px;}
.kpi .v{font-family:'Instrument Serif',Georgia,serif;font-size:2.9rem;font-weight:400;letter-spacing:-.02em;color:var(--ink);line-height:1.1;margin-top:4px;}
.kpi .s{font-size:.8rem;color:var(--ink2);margin-top:2px;}
.kpi.arc{position:relative;}
.kpi.arc svg{position:absolute;right:14px;bottom:14px;width:108px;height:auto;}
.kpi.arc .s{max-width:56%;}
[data-testid="stPlotlyChart"]{background:#fff;border:1px solid #DFE8E1;border-radius:24px;padding:0 0 3px;overflow:hidden;box-shadow:0 22px 38px -30px rgba(8,38,24,.6);}
[data-testid="stDataFrame"]{border:1px solid var(--line);border-radius:18px;overflow:hidden;}
[data-testid="stAlert"]{border-radius:16px;}
.stButton>button,.stDownloadButton>button{border-radius:999px;border:1px solid #C9D6CE;font-weight:500;background:#fff;padding:.45rem 1.2rem;}
.stButton>button:hover,.stDownloadButton>button:hover{border-color:var(--green);color:var(--green-d);}
.stButton>button[kind="primary"]{background:var(--green);border-color:var(--green);color:#fff;box-shadow:0 8px 20px -10px rgba(46,158,91,.9);}
.stButton>button[kind="primary"]:hover{background:#27874D;border-color:#27874D;color:#fff;}
[data-baseweb="input"],[data-baseweb="select"]>div{border-radius:14px !important;}
[data-baseweb="input"]{background:#F3F6F3 !important;border:1px solid #D9E3DC !important;}
[data-baseweb="base-input"]{background:transparent !important;}
button[data-baseweb="tab"]{font-weight:500;font-size:.95rem;}
[data-baseweb="tab-highlight"]{background:var(--green) !important;}
[data-testid="stExpander"]{border:none !important;}
[data-testid="stExpander"] details{background:var(--card);border:none !important;border-radius:18px;}
[data-testid="stExpander"] summary{padding:14px 18px;font-weight:600;}
/* risk label */
.badge{display:inline-block;padding:4px 14px;border-radius:999px;font-weight:600;font-size:.88rem;color:#fff;white-space:nowrap;}
.rx{background:var(--card);border-radius:24px;padding:22px 24px 20px;}
.rx-top{display:flex;justify-content:space-between;align-items:center;gap:10px;}
.rx-k{color:var(--ink2);font-size:.9rem;}
.rx-num{font-family:'Instrument Serif',Georgia,serif;font-size:4.6rem;font-weight:400;letter-spacing:-.03em;line-height:1;color:var(--ink);margin:12px 0 4px;}
.rx-sub{color:var(--ink2);font-size:.92rem;margin-bottom:22px;}
.sw{position:relative;padding-top:34px;}
.sb{height:14px;border-radius:7px;}
.mk{position:absolute;top:0;transform:translateX(-50%);text-align:center;}
.mk em{display:block;font-style:normal;font-size:.72rem;font-weight:600;color:var(--ink);white-space:nowrap;height:16px;}
.mk u{display:block;width:4px;height:32px;background:var(--ink);border-radius:2px;margin:0 auto;border:2px solid #fff;box-sizing:content-box;margin-top:-2px;}
.lg{display:flex;flex-wrap:wrap;gap:6px 16px;margin-top:14px;font-size:.78rem;color:var(--ink2);}
.lg span::before{content:"";display:inline-block;width:9px;height:9px;border-radius:50%;margin-right:6px;background:var(--c);}
/* watch list */
.wn{background:var(--card);border-radius:24px;padding:20px 22px 10px;}
.wn h4{margin:0;padding:0;font-size:1.05rem;}
.wn .wsub{color:var(--ink2);font-size:.84rem;margin:2px 0 8px;}
.wr{display:grid;grid-template-columns:minmax(0,1.5fr) minmax(0,1fr) 62px;gap:12px;align-items:center;padding:10px 0;border-top:1px solid #E1E8E3;}
.wr .n{font-weight:600;font-size:.92rem;white-space:nowrap;overflow:hidden;text-overflow:ellipsis;}
.wr .c{color:var(--ink2);font-size:.76rem;white-space:nowrap;overflow:hidden;text-overflow:ellipsis;}
.wr .t{height:8px;background:#DFE7E1;border-radius:4px;overflow:hidden;}
.wr .t i{display:block;height:100%;border-radius:4px;}
.wr .p{text-align:right;font-weight:600;font-size:.92rem;}
.wr .p small{display:block;font-weight:500;font-size:.7rem;color:var(--ink2);}
.dh h2{font-size:2.6rem !important;line-height:1.05;margin:0 0 8px !important;padding:0 !important;}
.tags{display:flex;flex-wrap:wrap;gap:6px;margin-bottom:16px;}
.tags span{background:#E4ECE6;color:#34493C;border-radius:999px;padding:4px 12px;font-size:.8rem;}
.demo{background:#FFF6E6;border:1px solid #F2CE8A;border-radius:16px;padding:11px 16px;color:#6E4300;margin-bottom:16px;font-size:.9rem;}
.callout{background:var(--card);border-left:4px solid var(--green);border-radius:14px;padding:13px 18px;margin:6px 0 14px;font-size:.93rem;}
.win{display:flex;flex-wrap:wrap;background:var(--card);border-radius:22px;margin:6px 0 16px;overflow:hidden;}
.win>div{padding:16px 22px;}
.win .w1{flex:1.6;min-width:220px;background:#0B2A1D;}
.win .w1 small{color:rgba(255,255,255,.65);}
.win .w1 b{color:#fff;font-family:'Instrument Serif',Georgia,serif;font-weight:400;font-size:1.7rem;}
.win .w2{flex:1;min-width:110px;}
.win small{display:block;color:var(--ink2);font-size:.78rem;}
.win .w2 b{font-family:'Instrument Serif',Georgia,serif;font-weight:400;font-size:2rem;letter-spacing:-.01em;}
/* image card */
.imgcard{position:relative;border-radius:28px;overflow:hidden;min-height:390px;background:#0A2418;}
.imgcard>img{position:absolute;inset:0;width:100%;height:100%;object-fit:cover;}
.imgcard .ic-tags{position:absolute;top:16px;left:16px;right:16px;display:flex;justify-content:space-between;gap:8px;}
.imgcard .ic-tags span{background:rgba(255,255,255,.16);backdrop-filter:blur(8px);border:1px solid rgba(255,255,255,.25);color:#fff;border-radius:999px;padding:5px 13px;font-size:.78rem;}
.imgcard .ic-box{position:absolute;left:16px;right:16px;bottom:16px;background:rgba(10,24,17,.5);backdrop-filter:blur(14px);border:1px solid rgba(255,255,255,.2);border-radius:20px;padding:14px 18px 16px;color:#fff;}
.ic-box small{display:block;color:rgba(255,255,255,.7);font-size:.76rem;}
.ic-box .nm{font-family:'Instrument Serif',Georgia,serif;font-size:1.6rem;line-height:1.15;margin:2px 0 10px;}
.ic-box .gb{position:relative;height:10px;border-radius:5px;background:linear-gradient(90deg,#35B26E,#F2C94C 55%,#E5534B);}
.ic-box .gb i{position:absolute;top:-4px;width:18px;height:18px;margin-left:-9px;border-radius:50%;background:#fff;border:3px solid #0C1F16;}
.ic-box .pc{display:flex;justify-content:space-between;align-items:baseline;margin-top:10px;font-size:.78rem;color:rgba(255,255,255,.75);}
.ic-box .pc b{font-size:1.15rem;color:#fff;}
/* footer */
.ft{margin-top:26px;border-radius:34px;padding:42px 46px 30px;color:rgba(224,238,229,.78);display:grid;grid-template-columns:1.5fr 1fr 1fr 1.2fr;gap:30px;
 background:radial-gradient(50% 120% at 85% 120%,rgba(52,209,127,.30),transparent 62%),radial-gradient(40% 90% at 0% 0%,rgba(30,138,90,.28),transparent 60%),linear-gradient(180deg,#0A2418,#06130D);}
.ft b{display:block;color:#fff;font-size:.9rem;margin-bottom:8px;font-weight:600;}
.ft p{margin:0 0 4px;font-size:.84rem;line-height:1.55;color:rgba(224,238,229,.74) !important;}
.ft .fine{grid-column:1/-1;border-top:1px solid rgba(255,255,255,.12);padding-top:16px;font-size:.78rem;color:rgba(224,238,229,.55);}
.demo-pill{display:inline-block;background:#F5C77E;color:#4A2E00;border-radius:999px;padding:2px 10px;font-weight:600;font-size:.76rem;}
.kpi{position:relative;}
.kpi .ic{position:absolute;top:16px;right:16px;width:42px;height:42px;border-radius:14px;background:#DCEFE3;display:grid;place-items:center;}
.kpi .ic svg{width:22px;height:22px;stroke:#1E8A5A;fill:none;stroke-width:1.8;stroke-linecap:round;stroke-linejoin:round;}
.band{position:relative;overflow:hidden;border-radius:30px;min-height:218px;padding:36px 42px;margin:26px 0 8px;
 background:radial-gradient(50% 120% at 88% 40%,rgba(52,209,127,.34),transparent 64%),radial-gradient(40% 90% at 0% 0%,rgba(30,138,90,.30),transparent 62%),linear-gradient(120deg,#06130D,#0B2C1E);}
.band .bt{position:relative;z-index:1;max-width:46%;}
.band h3{font-family:'Instrument Serif',Georgia,serif !important;font-weight:400 !important;font-size:2.2rem !important;line-height:1.08;color:#F3F8F4 !important;margin:0 0 10px !important;padding:0 !important;}
.band h3 em{font-style:italic;color:var(--mint);}
.band p{color:rgba(243,248,244,.76) !important;font-size:.99rem;line-height:1.55;margin:0;max-width:44ch;}
.band img{position:absolute;right:14px;top:50%;transform:translateY(-50%);height:128%;max-width:60%;object-fit:contain;pointer-events:none;}
@media (max-width:900px){.band img{display:none;}.band .bt{max-width:100%;}}
@media (max-width:900px){.hero img.art{display:none;}.ft{grid-template-columns:1fr;}.wr{grid-template-columns:minmax(0,1fr) 60px;}.wr .t{display:none;}.rx-num{font-size:3.4rem;}.st-key-sheet{padding:20px 16px 28px;}}
</style>""", unsafe_allow_html=True)



# ----------------------------------------------------------------------------------------------- helpers
import html as _html

CHIP = ""          # set by the router: data-source chip shown at the top right of every page


def esc(x) -> str:
    return _html.escape(str(x), quote=True)


ICONS = {
    "records": '<path d="M6 3h9l4 4v14H6z"/><path d="M15 3v4h4M9 12h7M9 16h7"/>',
    "pill": '<g transform="rotate(-45 12 12)"><rect x="2.5" y="7.5" width="19" height="9" rx="4.5"/><path d="M12 7.5v9"/></g>',
    "gauge": '<path d="M4 17a8 8 0 1 1 16 0"/><path d="M12 17l4-5"/>',
    "alert": '<path d="M12 3.5l9 16H3z"/><path d="M12 10v4.5M12 17.2v.1"/>',
    "target": '<circle cx="12" cy="12" r="8"/><circle cx="12" cy="12" r="3.5"/>',
    "avg": '<path d="M4 19V9M10 19V5M16 19v-7M21 19H3"/>',
    "shield": '<path d="M12 3l8 3v6c0 4.5-3.2 7.8-8 9-4.8-1.2-8-4.5-8-9V6z"/><path d="M8.5 12l2.5 2.5L15.5 10"/>',
    "history": '<path d="M4 12a8 8 0 1 0 3-6.2"/><path d="M4 4v4h4M12 8v4l3 2"/>',
    "calendar": '<rect x="4" y="5" width="16" height="15" rx="3"/><path d="M4 10h16M9 3v4M15 3v4"/>',
    "factory": '<path d="M3 20V10l6 4V10l6 4V6h6v14z"/>',
    "list": '<path d="M8 6h12M8 12h12M8 18h12M4 6h.1M4 12h.1M4 18h.1"/>',
    "trend": '<path d="M3 17l6-6 4 4 8-9"/><path d="M15 6h6v6"/>',
}


def icon_html(name):
    return (f'<div class="ic"><svg viewBox="0 0 24 24" aria-hidden="true">{ICONS[name]}</svg></div>' if name in ICONS else "")


def kpi(col, label, value, sub="", icon=None):
    col.markdown(f'<div class="kpi">{icon_html(icon)}<div class="l">{label}</div><div class="v">{value}</div><div class="s">{sub}</div></div>',
                 unsafe_allow_html=True)


def page_header(*_args, **_kwargs):
    """Kept for compatibility: the hero is now drawn once by the router."""
    return None


import base64
import math
from urllib.parse import quote

ASSETS = HERE / "assets"


@st.cache_data(show_spinner=False)
def img_uri(name: str) -> str:
    p = ASSETS / name
    if not p.exists():
        return ""
    mime = {".svg": "image/svg+xml", ".png": "image/png", ".jpg": "image/jpeg", ".jpeg": "image/jpeg", ".webp": "image/webp"}[p.suffix.lower()]
    return f"data:{mime};base64," + base64.b64encode(p.read_bytes()).decode()


st.markdown(f"""<style>
.st-key-sheet{{background-color:#fff;background-image:url("{img_uri('pattern.svg')}"),radial-gradient(640px 340px at 100% 0%,rgba(46,158,91,.10),transparent 70%),radial-gradient(780px 440px at 0% 100%,rgba(46,158,91,.09),transparent 70%);background-size:360px 360px,auto,auto;}}
</style>""", unsafe_allow_html=True)


def card_image_uri() -> str:
    """Use your own photo if you drop assets/photo.jpg (or .png/.webp) in the project; otherwise the built-in illustration."""
    for ext in ("jpg", "jpeg", "png", "webp"):
        if (ASSETS / f"photo.{ext}").exists():
            return img_uri(f"photo.{ext}")
    return img_uri("card_capsules.svg")


ART_FOR = {"Risk checker": "hero_risk.svg", "Watchlist": "hero_watch.svg", "Forecast": "hero_forecast.svg",
           "Models": "hero_models.svg", "Data": "hero_data.svg", "Setup": "hero_setup.svg"}


def hero_html(title, sub, big=False, ctas=None, art_name=None):
    art = img_uri("hero_art.svg" if big else (art_name or "hero_small.svg"))
    acts = "".join(f'<a class="{c}" href="?page={quote(p)}" target="_self">{esc(t)}</a>' for t, p, c in (ctas or []))
    return (f'<style>:root{{--band:{"560px" if big else "340px"};--hero:{"450px" if big else "272px"};}}</style>'
            f'<div class="hero{"" if big else " sm"}"><img class="art {"big" if big else "small"}" src="{art}" alt="">'
            f'<h1>{title}</h1><p>{esc(sub)}</p>' + (f'<div class="acts">{acts}</div>' if acts else "") + '</div>')


def band_html(art, title, text):
    return (f'<div class="band"><img src="{img_uri(art)}" alt=""><div class="bt"><h3>{title}</h3><p>{text}</p></div></div>')


def section(title, sub=""):
    st.markdown(f'<div class="sec"><h2>{title}</h2>' + (f"<p>{sub}</p>" if sub else "") + "</div>", unsafe_allow_html=True)


def kpi_arc(col, label, frac, value, sub="", icon=None):
    f = max(0.0, min(float(frac), 1.0))
    L = math.pi * 90
    ang = math.pi * (1 - f)
    x, y = 100 + 90 * math.cos(ang), 100 - 90 * math.sin(ang)
    svg = ('<svg viewBox="0 0 200 112" aria-hidden="true"><defs><linearGradient id="ag" x1="0" x2="1"><stop offset="0" stop-color="#35B26E"/>'
           '<stop offset=".6" stop-color="#A8D45A"/><stop offset="1" stop-color="#F2C94C"/></linearGradient></defs>'
           '<path d="M10 100 A90 90 0 0 1 190 100" fill="none" stroke="#DCE5DE" stroke-width="14" stroke-linecap="round"/>'
           f'<path d="M10 100 A90 90 0 0 1 190 100" fill="none" stroke="url(#ag)" stroke-width="14" stroke-linecap="round" stroke-dasharray="{f*L:.1f} {L:.1f}"/>'
           f'<circle cx="{x:.1f}" cy="{y:.1f}" r="9" fill="#fff" stroke="#1E8A5A" stroke-width="4"/></svg>')
    col.markdown(f'<div class="kpi arc">{icon_html(icon)}<div class="l">{label}</div><div class="v">{value}</div><div class="s">{sub}</div>{svg}</div>', unsafe_allow_html=True)


def imgcard_html(drug, category, p, horizon):
    top = min(1.0, max(p * 1.35, 0.25))
    pos = min(max(p / top, 0.03), 0.97) * 100
    return (f'<div class="imgcard"><img src="{card_image_uri()}" alt="">'
            f'<div class="ic-tags"><span>Highest risk now</span><span>{esc(category)}</span></div>'
            f'<div class="ic-box"><small>Chance of a new shortage in {horizon} months</small><div class="nm">{esc(str(drug).title())}</div>'
            f'<div class="gb"><i style="left:{pos:.1f}%"></i></div><div class="pc"><span>Low</span><b>{p*100:.0f}%</b><span>High</span></div></div></div>')


def footer_html(A):
    if A:
        m, ew = A["report"]["meta"], A["report"]["early_warning"]
        demo = '<span class="demo-pill">Demo data, not real</span>' if m.get("is_demo") else ""
        data = (f'<p>openFDA Drug Shortages</p><p>{m["n_clean"]:,} records through {esc(m["last_posting"])}</p>'
                f'<p>Source: {esc(m["data_source"])}</p><p>{demo}</p>')
        fine = f'Models trained {esc(m["trained_at"].replace("T", " "))}. Early-warning horizon: {ew["horizon"]} months.'
    else:
        data = "<p>openFDA Drug Shortages</p><p>Not loaded yet. Open Setup to download it.</p>"
        fine = "No models trained yet."
    return (f'<div class="ft"><div><div class="brand"><div><b>MedShortage</b><span>Shortage early warning</span></div></div>'
            '<p style="margin-top:14px">An early-warning system that learns from the FDA\'s public shortage records.</p></div>'
            f'<div><b>Data</b>{data}</div>'
            '<div><b>Models</b><p>Medicine shortage forecast (3-12 months)</p><p>Monthly volume (context)</p><p>Duration estimate</p><p>Medicine segments</p></div>'
            '<div><b>Please note</b><p>Scores are estimates from FDA posting history, not live stock levels. They support planning and are not medical advice.</p></div>'
            f'<div class="fine">{fine}</div></div>')


FONT = "DM Sans, system-ui, sans-serif"
AXIS, GRID = "#CFDAD2", "#E9EFEA"
PCFG = {"displayModeBar": False, "responsive": True}


def show(slot, fig):
    """Render a Plotly figure without the hover toolbar, and with our own styling (not Streamlit's theme)."""
    try:
        slot.plotly_chart(fig, config=PCFG, theme=None, **WIDE)
    except TypeError:
        slot.plotly_chart(fig, config=PCFG, **WIDE)


def fig_style(fig, height=360, legend=True, sub=None, legend_pos="bottom"):
    t = fig.layout.title.text
    if t:
        t = f"<b>{t}</b>" + (f"<br><span style='font-size:12.5px;font-weight:400;color:{INK2}'>{sub}</span>" if sub else "")
    fig.update_layout(height=height, margin=dict(l=18, r=30, t=(86 if sub else 62) if t else 24, b=18), paper_bgcolor="#fff", plot_bgcolor="#fff",
                      font=dict(family=FONT, color=INK2, size=12.5), hovermode="closest", bargap=0.4, bargroupgap=0.08,
                      legend=(dict(orientation="v", x=1.02, y=1, xanchor="left", yanchor="top", bgcolor="rgba(0,0,0,0)", font=dict(size=12, color=INK2)) if legend_pos == "right"
                              else dict(orientation="h", y=1.02, x=1, xanchor="right", yanchor="bottom", bgcolor="rgba(0,0,0,0)", font=dict(size=12.5, color=INK2)) if legend_pos == "top"
                              else dict(orientation="h", y=-0.22, x=0, bgcolor="rgba(0,0,0,0)", font=dict(size=12.5, color=INK2))) if legend else None,
                      showlegend=legend, title=dict(text=t, x=0, xref="container", xanchor="left", y=1, yref="container", yanchor="top", pad=dict(l=24, t=20), font=dict(size=16, color=INK, family=FONT)),
                      hoverlabel=dict(bgcolor="#fff", bordercolor="#BFD0C5", font=dict(family=FONT, color=INK, size=13)))
    fig.update_xaxes(showgrid=False, linecolor=AXIS, ticks="outside", tickcolor=AXIS, tickfont=dict(color=INK2), automargin=True,
                     title_font=dict(size=12.5, color=INK2))
    fig.update_yaxes(gridcolor=GRID, gridwidth=1, zeroline=False, showline=False, tickfont=dict(color=INK2), automargin=True,
                     title_font=dict(size=12.5, color=INK2))
    try:
        fig.update_traces(marker_cornerradius=5, selector=dict(type="bar"))
    except Exception:  # noqa: BLE001 - older Plotly versions have no rounded bars
        pass
    return fig


def hbar_fig(labels, values, title, height, sub=None, color=None, fmt="{:.0f}", hover="%{y}<br>%{x:,.1f}<extra></extra>", unit=""):
    """Single-series horizontal bars, thin and rounded, value at the tip, largest at the top."""
    labels, values = list(labels), list(values)
    f = go.Figure(go.Bar(y=labels, x=values, orientation="h", marker_color=color or PRIMARY, text=[fmt.format(v) + unit for v in values],
                         textposition="outside", cliponaxis=False, textfont=dict(size=12, color=INK2), hovertemplate=hover))
    f.update_yaxes(categoryorder="array", categoryarray=labels, showgrid=False)
    f.update_xaxes(showgrid=True, gridcolor=GRID, range=[0, max(values) * 1.16 if values else 1], showline=False, ticks="")
    f.update_layout(title=title)
    return fig_style(f, height, False, sub)


def tier_bar_fig(df, namecol, probcol, levelcol, title, height=460, sub=None, n=15):
    """Top-n medicines as horizontal bars. Colour = risk tier, always shown with icon and label in the legend."""
    t = df.head(n)
    order = t[namecol].astype(str).str.title().tolist()[::-1]
    f = go.Figure()
    for lvl in ("High", "Elevated", "Low"):
        s_ = t[t[levelcol] == lvl]
        if s_.empty:
            continue
        f.add_bar(y=s_[namecol].astype(str).str.title(), x=s_[probcol] * 100, orientation="h", name=f"{RISK[lvl][1]} {lvl}",
                  marker_color=RISK[lvl][0], text=[f"{v * 100:.0f}%" for v in s_[probcol]], textposition="outside", cliponaxis=False,
                  textfont=dict(size=12, color=INK2), hovertemplate="%{y}<br>%{x:.1f}% chance<br>" + lvl + " risk<extra></extra>")
    f.update_layout(barmode="overlay", title=title)
    f.update_yaxes(categoryorder="array", categoryarray=order, showgrid=False)
    f.update_xaxes(range=[0, float(t[probcol].max()) * 100 * 1.16], showgrid=True, gridcolor=GRID, showline=False, ticks="")
    return fig_style(f, height, True, sub)


def tier_bounds(thr, base):
    """Elevated starts at half the decision threshold, but never below the average medicine's risk."""
    return min(max(thr * 0.5, base), thr), thr


def risk_scale_html(p, thr, base, level, vs, horizon):
    """The signature element: one gradient scale from low to high, with a marker for this medicine."""
    top = float(min(1.0, max(thr * 3, p * 1.25, 0.2)))
    pos = lambda v: max(0.0, min(v / top, 1.0)) * 100   # noqa: E731
    e_lo, _ = tier_bounds(thr, base)
    z1, z2 = pos(e_lo), pos(thr)
    mid = (z1 + z2) / 2
    grad = (f"linear-gradient(90deg,#35B26E 0%,#35B26E {z1*0.55:.1f}%,#F2C94C {mid:.1f}%,"
            f"#E5534B {min(z2 + 4, 99):.1f}%,#D33F3F 100%)")
    col, icon = RISK[level]
    mk = min(max(pos(p), 5), 95)
    txt = "#3B2A00" if level == "Elevated" else "#fff"
    return (f'<div class="rx"><div class="rx-top"><span class="rx-k">Chance of a new shortage in the next {horizon} months</span>'
            f'<span class="badge" style="background:{col};color:{txt}">{icon} {level} risk</span></div>'
            f'<div class="rx-num">{p*100:.1f}%</div>'
            f'<div class="rx-sub">{vs:.1f}× the average medicine ({base*100:.1f}%)</div>'
            f'<div class="sw"><div class="mk" style="left:{mk:.1f}%"><em>This medicine</em><u></u></div>'
            f'<div class="sb" style="background:{grad}"></div></div>'
            f'<div class="lg"><span style="--c:#35B26E">Low: under {e_lo*100:.1f}%</span>'
            + (f'<span style="--c:#F2C94C">Elevated: {e_lo*100:.1f}–{thr*100:.1f}%</span>' if thr - e_lo > 1e-9 else "")
            + f'<span style="--c:#D33F3F">High: over {thr*100:.1f}%</span></div></div>')


def watch_rows_html(df, horizon):
    top = max(float(df.risk_probability.max()), 0.01)
    rows = []
    for r in df.itertuples():
        col, icon = RISK[r.risk_level]
        rows.append(f'<div class="wr"><div><div class="n">{esc(str(r.drug).title())}</div><div class="c">{esc(r.category)}</div></div>'
                    f'<div class="t"><i style="width:{max(r.risk_probability / top * 100, 3):.0f}%;background:{col}"></i></div>'
                    f'<div class="p">{r.risk_probability*100:.0f}%<small>{icon} {r.risk_level}</small></div></div>')
    return (f'<div class="wn"><h4>Watch now</h4><div class="wsub">Most likely to get a new FDA shortage in the next {horizon} months</div>'
            + "".join(rows) + '</div>')


def winner_html(T):
    b = next(r for r in T["models"] if r["model"] == T["best"])
    t = b["test"]
    cells = [("Recall", pct(t["recall"])), ("Precision", pct(t["precision"])), ("F2 score", num(t["f2"])), ("PR-AUC", num(t["pr_auc"]))]
    return ('<div class="win"><div class="w1"><small>Selected model</small><b>' + esc(T["best"]) + '</b></div>'
            + "".join(f'<div class="w2"><small>{k}</small><b>{v}</b></div>' for k, v in cells) + '</div>')


FEATURE_LABELS = {
    "c_total": "Total past postings", "c3": "Postings, last 3 mo", "c6": "Postings, last 6 mo", "c12": "Postings, last 12 mo",
    "c24": "Postings, last 24 mo", "months_since_last": "Months since last posting", "age_months": "Months since first posting",
    "episodes": "Distinct months with a posting", "n_companies": "Manufacturers involved", "cat_c3": "Category postings, last 3 mo",
    "cat_c12": "Category postings, last 12 mo", "glob_c3": "FDA-wide postings, last 3 mo", "glob_c12": "FDA-wide postings, last 12 mo",
    "moy_sin": "Seasonality (sin)", "moy_cos": "Seasonality (cos)", "is_injection": "Injectable form",
    "company_freq": "Manufacturer frequency", "n_ndc": "Number of NDC packages", "presentation_len": "Presentation text length",
    "has_brand": "Has brand name", "has_epc": "Has pharmacologic class", "posting_year": "Posting year", "posting_month": "Posting month",
    "drug_prior_postings": "Earlier postings of this drug", "company_prior_postings": "Earlier postings by this manufacturer",
    "category_postings_prior_90d": "Category postings, prior 90 days"}


def flabel(f):
    if f in FEATURE_LABELS:
        return FEATURE_LABELS[f]
    for p, n in (("cat_", "Category: "), ("form_", "Form: "), ("route_", "Route: ")):
        if f.startswith(p):
            return n + f[len(p):]
    return f.replace("_", " ")


def pct(x, d=0):
    return "n/a" if x is None or (isinstance(x, float) and np.isnan(x)) else f"{x*100:.{d}f}%"


def num(x, d=2):
    return "n/a" if x is None or (isinstance(x, float) and np.isnan(x)) else f"{x:.{d}f}"


@st.cache_data(show_spinner=False)
def load_artifacts(stamp: float):
    rep = json.load(open(MODEL_DIR / "metrics.json"))
    art = joblib.load(MODEL_DIR / "early_warning_model.joblib")
    sc = pd.read_csv(MODEL_DIR / "risk_scores.csv")
    lo, hi = tier_bounds(art["threshold"], art["base_rate"])
    sc["risk_level"] = np.where(sc.risk_probability >= hi, "High", np.where(sc.risk_probability >= lo, "Elevated", "Low"))
    d = dict(
        report=rep,
        data=pd.read_csv(MODEL_DIR / "clean_data.csv", parse_dates=["initial_posting_date", "update_date", "discontinued_date"]),
        scores=sc,
        profiles=pd.read_csv(MODEL_DIR / "drug_profiles.csv"),
        contrib=pd.read_csv(MODEL_DIR / "risk_contributions.csv") if (MODEL_DIR / "risk_contributions.csv").exists() else None,
        fc_hist=pd.read_csv(MODEL_DIR / "forecast_history.csv", parse_dates=["month"]) if (MODEL_DIR / "forecast_history.csv").exists() else None,
        fc_future=pd.read_csv(MODEL_DIR / "forecast_future.csv", parse_dates=["month"]) if (MODEL_DIR / "forecast_future.csv").exists() else None,
        hz=pd.read_csv(MODEL_DIR / "horizon_scores.csv") if (MODEL_DIR / "horizon_scores.csv").exists() else None,
        art=art,
    )
    return d


def artifacts():
    if not pl.artifacts_ready(MODEL_DIR):
        return None
    return load_artifacts((MODEL_DIR / "metrics.json").stat().st_mtime)


def demo_banner(rep):
    if rep["meta"].get("is_demo"):
        st.markdown('<div class="demo"><b>DEMO DATA.</b> These numbers come from a generated sample used only to test the '
                    'software - they are <b>not</b> real FDA results. Load <code>shortages_raw.csv</code> or download live '
                    'data on the <b>Setup</b> page and retrain.</div>', unsafe_allow_html=True)


# ----------------------------------------------------------------------------------------------- pages
def page_setup():
    page_header("Data & training", "Choose where the FDA shortage data comes from, then train the models. Everything runs on this computer.")
    c1, c2 = st.columns(2)
    with c1:
        st.subheader("Step 1: Choose your data")
        have_real, have_demo = REAL_CSV.exists(), DEMO_CSV.exists()
        if have_real:
            st.success(f"Real data file found: `{REAL_CSV.name}` ({pd.read_csv(REAL_CSV, usecols=[0]).shape[0]:,} rows)")
        else:
            st.info("No `data/shortages_raw.csv` yet. Upload the file saved by your Phase 2 notebook, or download it live.")
        up = st.file_uploader("Upload shortages_raw.csv (from Phase 2)", type="csv")
        if up is not None:
            try:
                df = pl.load_raw(up)
                DATA_DIR.mkdir(exist_ok=True)
                df.to_csv(REAL_CSV, index=False)
                st.success(f"Saved {len(df):,} records. Now press **Train models**.")
            except pl.DataError as e:
                st.error(f"Could not use this file. {e}")
        if st.button("Download live from openFDA", help="Fetches all pages from api.fda.gov (~1,600 records, about a minute)."):
            bar = st.progress(0.0, text="Connecting to openFDA ...")
            try:
                df = pl.fetch_live(lambda f, m: bar.progress(f, text=m))
                df.to_csv(REAL_CSV, index=False)
                bar.progress(1.0, text=f"Saved {len(df):,} records.")
                st.success("Download complete. Now press **Train models**.")
            except pl.DataError as e:
                bar.empty()
                st.error(f"{e}")
        if have_demo and not have_real:
            st.caption("Only want to try the software? A clearly-labelled generated demo file is available:")
            if st.button("Use DEMO data (not real)"):
                st.session_state["use_demo"] = True
    with c2:
        st.subheader("Step 2: Train the models")
        source = REAL_CSV if REAL_CSV.exists() else (DEMO_CSV if st.session_state.get("use_demo") and DEMO_CSV.exists() else None)
        st.write(f"Data to be used: **{source.name if source else 'none selected'}**")
        horizon = st.select_slider("Early-warning horizon (months ahead)", [3, 6, 9, 12], value=6)
        st.caption("Trains and compares ~8 classifiers, 6 regressors, 7 forecasters and 3 clustering methods "
                   "(takes about 1–3 minutes).")
        if st.button("Train models", type="primary", disabled=source is None):
            bar = st.progress(0.0, text="Starting ...")
            try:
                raw = pl.load_raw(source)
                pl.run_all(raw, MODEL_DIR, source.name, horizon=horizon, progress=lambda f, m: bar.progress(min(f, 1.0), text=m))
                load_artifacts.clear()
                bar.progress(1.0, text="Training complete")
                st.success("Models trained. Use the sidebar to open the dashboard.")
                st.rerun()
            except pl.DataError as e:
                bar.empty()
                st.error(f"Training stopped: {e}")
            except Exception as e:  # noqa: BLE001
                bar.empty()
                st.error("Something unexpected happened during training. The details below can be shared for help.")
                st.exception(e)


def page_overview(A):
    rep, df = A["report"], A["data"]
    m, ew, sc = rep["meta"], rep["early_warning"], A["scores"]
    H = ew["horizon"]
    demo_banner(rep)
    c = st.columns(3)
    kpi(c[0], "Shortage records", f"{m['n_clean']:,}", f"{m['n_raw']-m['n_clean']} removed in cleaning", icon="records")
    kpi(c[1], "Medicines with a shortage history", f"{m['n_medicines']:,}", f"{int((sc.risk_level == 'High').sum())} flagged high risk", icon="pill")
    kpi_arc(c[2], "Still active", m["active_share"], pct(m["active_share"]), "Not yet resolved", icon="history")
    st.write("")
    l, r = st.columns([2, 3])
    l.markdown(watch_rows_html(sc.head(8), H), unsafe_allow_html=True)
    mon = df.set_index("initial_posting_date").resample("MS").size().rename("n").reset_index()
    mon["roll"] = mon["n"].rolling(12, min_periods=3).mean()
    f = go.Figure()
    f.add_bar(x=mon.initial_posting_date, y=mon.n, name="New postings per month", marker_color=PRIMARY_LIGHT, marker_cornerradius=3,
              hovertemplate="%{y} new postings<extra></extra>")
    f.add_scatter(x=mon.initial_posting_date, y=mon.roll, name="12-month average", mode="lines",
                  line=dict(color=PRIMARY, width=2.5, shape="spline", smoothing=0.5), fill="tozeroy", fillcolor="rgba(30,138,90,.10)",
                  hovertemplate="%{y:.1f} on average<extra></extra>")
    pk = mon.loc[mon.n.idxmax()]
    f.add_annotation(x=pk.initial_posting_date, y=pk.n, text=f"Busiest month: {int(pk.n)} ({pk.initial_posting_date:%b %Y})", showarrow=True,
                     arrowhead=0, arrowcolor=AXIS, ax=-90, ay=8, font=dict(size=12, color=INK2), bgcolor="#fff", bordercolor=AXIS, borderpad=4)
    f.update_layout(title="New shortage postings per month", hovermode="x unified", bargap=0.15)
    f.update_xaxes(rangeselector=dict(buttons=[dict(count=2, label="2y", step="year", stepmode="backward"), dict(count=5, label="5y", step="year", stepmode="backward"),
                                              dict(step="all", label="All")], x=1, xanchor="right", y=1.17, bgcolor="#F1F6F2", activecolor="#CBE6D6",
                                   font=dict(size=12, color=INK)), hoverformat="%b %Y")
    show(r, fig_style(f, 500, True, "Bars are each month; the green line smooths them into a 12-month average"))
    st.markdown(band_html("banner_shelf.svg", "Shortages arrive in <em>clusters</em>", "A medicine with recent shortages, several affected manufacturers and a busy category is far more likely to see the next one. That pattern is what the model learns."), unsafe_allow_html=True)

    section("Where the pressure is <em>building</em>", "Which therapeutic areas produce the most shortage records, and when in the year new ones tend to appear.")
    l, r = st.columns(2)
    cat = df.groupby("primary_category").agg(n=("drug", "size"), active=("is_shortage", "mean")).sort_values("n").tail(10).reset_index()
    f = hbar_fig(cat.primary_category, cat.n, "Therapeutic categories with the most records", 400, "Hover a bar to see how many of those records are still active",
                 hover="%{y}<br>%{x} records<extra></extra>")
    f.update_traces(customdata=cat.active, hovertemplate="%{y}<br>%{x} records, %{customdata:.0%} still active<extra></extra>")
    show(l, f)
    hm = df.assign(y=df.initial_posting_date.dt.year, mo=df.initial_posting_date.dt.month).groupby(["y", "mo"]).size().unstack(fill_value=0)
    f = go.Figure(go.Heatmap(z=hm.values, x=["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"][:hm.shape[1]],
                             y=hm.index.astype(str), colorscale=[[i / 6, c] for i, c in enumerate(SEQ)], xgap=2, ygap=2,
                             hovertemplate="%{y} %{x}: %{z} postings<extra></extra>",
                             colorbar=dict(title=dict(text="postings", font=dict(size=11, color=INK2)), thickness=8, len=0.75, outlinewidth=0, tickfont=dict(size=11, color=INK2))))
    f.update_yaxes(type="category", autorange="reversed")
    show(r, fig_style(f.update_layout(title="Seasonality: postings by year and month"), 400, False, "Darker squares are busier months; a column that stays dark year after year is a season"))

    section("Smarter signals, <em>earlier</em>", "What the models do with the FDA's shortage history.")
    l, r = st.columns([3, 2], gap="large")
    with l:
        with st.expander("Early detection", expanded=True):
            st.write(f"For every medicine and month, the model looks at its own posting history, how busy its category is and the time of year, "
                     f"then estimates the chance of a new FDA shortage in the next {H} months. It is tested only on months it never saw.")
        with st.expander("Risk you can explain"):
            st.write("Each score comes with the facts behind it: recent postings, months since the last one, how many manufacturers were involved. "
                     "The Models page shows which inputs matter most and how every model compared.")
        with st.expander("A forecast you can plan around"):
            st.write("The Forecast page ranks every medicine by its chance of a shortage in the next 3, 6, 9 and 12 months, so purchasing can see which ones to secure first.")
    top = sc.iloc[0]
    r.markdown(imgcard_html(top.drug, top.category, float(top.risk_probability), H), unsafe_allow_html=True)


def risk_badge(level):
    col, ic = RISK[level]
    return f'<span class="badge" style="background:{col}">{ic} {level} risk</span>'


def page_checker(A):
    rep, df, sc, prof = A["report"], A["data"], A["scores"], A["profiles"]
    ew, art = rep["early_warning"], A["art"]
    H = ew["horizon"]
    page_header("Risk checker", f"Search a medicine to see the estimated chance that the FDA lists a <b>new shortage</b> for it in the next <b>{H} months</b>.")
    demo_banner(rep)
    names = sorted(set(sc.drug) | set(df.brand_name.dropna().str.upper()))
    brand_map = df.dropna(subset=["brand_name"]).assign(b=lambda d: d.brand_name.str.upper()).drop_duplicates("b").set_index("b")["drug"].to_dict()
    q = st.text_input("Search a medicine (generic or brand name)", placeholder="e.g. cisplatin, amoxicillin, Adderall ...").strip().upper()
    if not q:
        st.markdown(band_html("banner_search.svg", "Which medicine are you <em>worried</em> about?", "Type a name above to see its chance of a new FDA shortage, or pick one from the highest-risk list below."), unsafe_allow_html=True)
        pick = st.selectbox("…or choose from the top 25 highest-risk medicines", [""] + sc.drug.head(25).tolist(), format_func=lambda x: x or "Select ...")
        if not pick:
            return
        q = pick
    exact = q if q in set(sc.drug) else brand_map.get(q)
    if exact is None:
        matches = [n for n in names if q in n][:12]
        if not matches:
            matches = difflib.get_close_matches(q, names, n=6, cutoff=0.5)
        st.error(f"“{q.title()}” is not in the FDA shortage history this model was trained on.")
        st.markdown("The model only scores medicines that have appeared in the FDA Drug Shortages database at least once - "
                    "a medicine that has never had a listed shortage has no history to learn from. "
                    "This is **not** a statement that it is safe from shortage.")
        if matches:
            exact = st.selectbox("Did you mean one of these?", [""] + [brand_map.get(m, m) for m in matches], format_func=lambda x: x or "Select ...")
        if not exact:
            return
    row = sc[sc.drug == exact].iloc[0]
    thr = art["threshold"]
    p, level = float(row.risk_probability), row.risk_level
    hist = df[df.drug == exact].sort_values("initial_posting_date")
    p_row = prof[prof.drug == exact].iloc[0]

    tags = [p_row.category, p_row.form, f"Segment: {p_row.segment}"] + ([f"Brand: {p_row.brand}"] if isinstance(p_row.brand, str) and p_row.brand else [])
    st.markdown(f'<div class="dh"><h2>{esc(exact.title())}</h2><div class="tags">' + "".join(f"<span>{esc(t)}</span>" for t in tags) + "</div></div>", unsafe_allow_html=True)
    l, r = st.columns([2, 3])
    l.markdown(risk_scale_html(p, thr, art["base_rate"], level, row.vs_baseline, H), unsafe_allow_html=True)
    with r:
        c = st.columns(4)
        kpi(c[0], "Past shortage postings", f"{int(row.c_total)}", f"{int(row.c12)} in last 12 mo", icon="history")
        kpi(c[1], "Months since last", f"{int(row.months_since_last)}", "0 = posted this month", icon="calendar")
        kpi(c[2], "Manufacturers", f"{int(row.n_companies)}", icon="factory")
        kpi(c[3], "Still active", pct(p_row.share_active), f"{int(p_row.n_records)} records", icon="gauge")
        st.write("")
        cb = A["contrib"]
        if cb is not None and exact in set(cb.drug):
            v = cb[cb.drug == exact].drop(columns="drug").iloc[0]
            v = v[v.abs() > 1e-6].sort_values(key=abs, ascending=False).head(7)[::-1]
            labs = [flabel(x) for x in v.index]
            f = go.Figure()
            for nm, colr, mask in (("Raises risk", ORANGE, v.values > 0), ("Lowers risk", BLUE, v.values <= 0)):
                if mask.any():
                    f.add_bar(y=[a for a, k in zip(labs, mask) if k], x=v.values[mask], orientation="h", name=nm, marker_color=colr,
                              text=[f"{a:+.2f}" for a in v.values[mask]], textposition="outside", cliponaxis=False, textfont=dict(size=12, color=INK2),
                              hovertemplate="%{y}<br>" + nm.lower() + " by %{x:+.2f}<extra></extra>")
            f.update_layout(barmode="overlay", title="What drives this score")
            f.update_yaxes(categoryorder="array", categoryarray=labs, showgrid=False)
            rng = float(np.abs(v.values).max()) * 1.3
            f.update_xaxes(range=[-rng, rng], zeroline=True, zerolinecolor=AXIS, showline=False, ticks="", title="← lowers the risk · raises the risk →")
            show(st, fig_style(f, 310, True, "Orange facts push this medicine above the average one; blue facts pull it below"))
        else:
            imp = pd.DataFrame(ew["importance"]).head(7)[::-1]
            show(st, hbar_fig(imp.feature.map(flabel), imp.importance, "Most influential factors (all medicines)", 300, fmt="{:.3f}",
                              hover="%{y}<br>importance %{x:.3f}<extra></extra>"))
    colmap = {"Current": ORANGE, "To Be Discontinued": "#d03b3b", "Resolved": AQUA}
    f = go.Figure()
    for s, d in hist.groupby("status"):
        f.add_scatter(x=d.initial_posting_date, y=d.company_name, mode="markers", name=s,
                      marker=dict(size=11, color=colmap.get(s, GREY), line=dict(width=2, color="#fff")),
                      hovertemplate="%{x|%d %b %Y}<br>%{y}<br>" + s + "<extra></extra>")
    ncomp = max(hist.company_name.nunique(), 1)
    f.update_yaxes(showgrid=True, gridcolor=GRID, autorange="reversed")
    f.update_xaxes(showgrid=False)
    show(st, fig_style(f.update_layout(title="Shortage history", hovermode="closest"), int(min(120 + 34 * ncomp, 420)) + 40, True,
                       "Each dot is one FDA shortage posting, by date and manufacturer"))
    peers = sc[(sc.category == row.category) & (sc.drug != exact)].head(5)[["drug", "risk_probability", "risk_level"]]
    if len(peers):
        st.markdown(f"##### Highest-risk peers in {row.category}")
        st.dataframe(peers, hide_index=True, **WIDE, column_config={
            "drug": "Medicine", "risk_probability": st.column_config.ProgressColumn("Risk", format="%.2f", min_value=0, max_value=1), "risk_level": "Level"})
    st.caption(f"Model: **{art['name']}** · {ew['n_train']:,} training observations · evaluated on {ew['n_test']:,} held-out future observations. "
               "A risk score is a statistical estimate from past FDA postings, not a guarantee.")


def page_watchlist(A):
    rep, sc = A["report"], A["scores"]
    ew = rep["early_warning"]
    page_header("Watchlist", f"Every medicine with FDA shortage history, ranked by the chance of a new shortage in the next {ew['horizon']} months.")
    demo_banner(rep)
    f1, f2, f3 = st.columns([2, 2, 1])
    cats = sorted(sc.category.dropna().unique())
    cat = f1.multiselect("Therapeutic category", cats)
    lvl = f2.multiselect("Risk level", ["High", "Elevated", "Low"], default=["High", "Elevated"])
    topn = f3.number_input("Show top", 5, 500, 50, step=5)
    v = sc.copy()
    if cat:
        v = v[v.category.isin(cat)]
    if lvl:
        v = v[v.risk_level.isin(lvl)]
    if v.empty:
        st.warning("No medicines match these filters. Try clearing the category or adding more risk levels.")
        return
    c = st.columns(4)
    kpi(c[0], "Medicines shown", f"{len(v):,}", icon="list")
    kpi(c[1], "High risk", f"{int((v.risk_level=='High').sum())}", icon="alert")
    kpi(c[2], "Elevated", f"{int((v.risk_level=='Elevated').sum())}", icon="trend")
    kpi(c[3], "Highest score", pct(v.risk_probability.max(), 1), icon="target")
    st.write("")
    t = v.head(int(topn))
    l, r = st.columns([3, 2])
    show(l, tier_bar_fig(t, "drug", "risk_probability", "risk_level", "The 15 highest-risk medicines", 500,
                         f"Chance of a new FDA shortage in the next {ew['horizon']} months"))
    bycat = v.groupby("category").agg(mean=("risk_probability", "mean"), n=("drug", "size")).query("n>=3").sort_values("mean").tail(10).reset_index()
    f = hbar_fig(bycat.category, bycat["mean"] * 100, "Average risk by category", 500, "Categories with at least 3 medicines, in % chance", fmt="{:.1f}", unit="%")
    f.update_traces(customdata=bycat.n, hovertemplate="%{y}<br>average %{x:.1f}% across %{customdata} medicines<extra></extra>")
    show(r, f)
    st.markdown(band_html("banner_podium.svg", "Start at the <em>top</em> of the list", "Medicines are ranked by their chance of a new shortage. Filter to one therapeutic area, then download the list for your team."), unsafe_allow_html=True)
    tbl = t[["drug", "category", "form", "risk_probability", "risk_level", "vs_baseline", "c12", "months_since_last", "n_records"]]
    st.dataframe(tbl, hide_index=True, **WIDE, column_config={
        "drug": "Medicine", "category": "Category", "form": "Form",
        "risk_probability": st.column_config.ProgressColumn("Risk", format="%.2f", min_value=0, max_value=1),
        "risk_level": "Level", "vs_baseline": st.column_config.NumberColumn("× avg", format="%.1f"),
        "c12": "Postings 12 mo", "months_since_last": "Months since last", "n_records": "All-time records"})
    st.download_button("Download watchlist (CSV)", v.to_csv(index=False).encode(), "shortage_watchlist.csv", "text/csv")


def horizon_table(A):
    """Per-medicine chance of a new FDA shortage at each horizon, plus how reliable each horizon was on unseen months.
    Falls back to the single early-warning horizon when an older training run has no multi-horizon file."""
    rep, sc, art = A["report"], A["scores"], A["art"]
    ew = rep["early_warning"]
    H = ew["horizon"]
    hf = rep.get("horizon_forecast") or {}
    info = {x["horizon"]: x for x in hf.get("horizons", [])}
    ctx = ["drug", "category", "form", "c12", "months_since_last", "n_records"]
    if A.get("hz") is None or not info:
        d = sc[ctx + ["risk_probability"]].rename(columns={"risk_probability": f"p_{H}"})
        b = next(r for r in ew["models"] if r["model"] == ew["best"])
        rule = next((r for r in ew["models"] if r["model"].startswith("Rule")), None)
        info = {H: dict(horizon=H, threshold=art["threshold"], base_rate=art["base_rate"],
                        test={k: b["test"][k] for k in ("recall", "precision", "f2", "pr_auc", "roc_auc", "positive_rate")},
                        rule={k: rule["test"][k] for k in ("recall", "precision", "f2")} if rule else {})}
        return d, info, False
    return A["hz"].merge(sc[ctx], on="drug", how="left"), info, True


def reliability_table(info):
    rows = []
    for h in sorted(info):
        t, r = info[h]["test"], info[h].get("rule", {})
        rows.append({"Look-ahead": f"{h} months", "Shortages caught (recall)": t["recall"], "Alerts that were right (precision)": t["precision"],
                     "PR-AUC": t["pr_auc"], "PR-AUC by pure guessing": t["positive_rate"],
                     "Simple rule: caught": r.get("recall"), "Simple rule: right": r.get("precision")})
    return pd.DataFrame(rows)


def show_reliability(info):
    t = reliability_table(info)
    st.dataframe(t.style.format({c: "{:.0%}" if c not in ("PR-AUC", "PR-AUC by pure guessing") else "{:.2f}" for c in t.columns if c != "Look-ahead"}, na_rep="n/a"),
                 hide_index=True, **WIDE)
    st.caption("Measured on months the model never saw. 'Shortages caught' is the share of medicines that really got a new FDA shortage "
               "that the forecast had flagged. 'Alerts that were right' is the share of flagged medicines that really did. The simple rule is "
               "'a medicine with a posting in the last 12 months will have another'. A forecast is only useful if it beats pure guessing.")


def page_forecast(A):
    rep = A["report"]
    demo_banner(rep)
    d, info, multi = horizon_table(A)
    hs = sorted(info)
    c1, c2, c3 = st.columns([3, 3, 1.2], vertical_alignment="bottom")
    h = c1.radio("Look ahead", hs, index=hs.index(6) if 6 in hs else 0, horizontal=True, format_func=lambda x: f"Next {x} months")
    cat = c2.multiselect("Therapeutic category", sorted(d.category.dropna().unique()))
    topn = int(c3.number_input("Show top", 5, 300, 25, step=5))
    if not multi:
        st.info(f"This training run produced one look-ahead window ({hs[0]} months). Open **Setup** and press **Train models** again "
                "to add the 3, 6, 9 and 12 month forecasts.")
    col = f"p_{h}"
    I = info[h]
    thr, base = I["threshold"], I["base_rate"]
    lo, hi = tier_bounds(thr, base)
    v = d.copy()
    if cat:
        v = v[v.category.isin(cat)]
    if v.empty:
        st.warning("No medicines match this category. Clear the filter to see every medicine.")
        return
    v["level"] = np.where(v[col] >= hi, "High", np.where(v[col] >= lo, "Elevated", "Low"))
    v["x_avg"] = v[col] / max(base, 1e-9)
    v = v.sort_values(col, ascending=False).reset_index(drop=True)
    st.markdown(f"<div class='callout'>Each percentage is the model's estimate that the FDA will post a <b>new shortage for that medicine "
                f"within the next {h} months</b>, based on its own shortage history, how busy its category is and the time of year. "
                f"The average medicine scores about {base*100:.1f}%. Longer windows give higher percentages because there is more time for a shortage to appear.</div>",
                unsafe_allow_html=True)
    k = st.columns(4)
    top = v.iloc[0]
    kpi(k[0], f"Flagged high risk, next {h} months", f"{int((v.level == 'High').sum())}", f"of {len(v):,} medicines shown", icon="alert")
    kpi(k[1], "Most likely to be next", f"{top[col]*100:.0f}%", esc(str(top.drug).title()), icon="target")
    kpi(k[2], "Average medicine", f"{base*100:.1f}%", f"chance in {h} months", icon="avg")
    t = I["test"]
    kpi(k[3], "Reliability on unseen months", f"{t['recall']*100:.0f}%", f"of real shortages were flagged; {t['precision']*100:.0f}% of alerts were right", icon="shield")
    st.write("")
    t15 = v.head(15)
    l, r = st.columns([1, 1] if multi else [3, 2])
    show(l, tier_bar_fig(t15, "drug", col, "level", f"The 15 medicines most likely to be next", 520,
                         f"Chance of a new FDA shortage in the next {h} months"))
    if multi:
        pc = [f"p_{x}" for x in hs]
        z = t15[pc].values * 100
        g = go.Figure(go.Heatmap(z=z, x=[f"{x} months" for x in hs], y=t15.drug.str.title(), xgap=2, ygap=2,
                                 colorscale=[[i / 4, c] for i, c in enumerate(SEQ[:5])], showscale=False,
                                 text=[[f"{a:.0f}%" for a in row] for row in z], texttemplate="%{text}", textfont=dict(size=12, color=INK),
                                 hovertemplate="%{y}<br>%{x}: %{z:.1f}%<extra></extra>"))
        g.update_yaxes(autorange="reversed")
        g.update_xaxes(side="top", showline=False, ticks="")
        g.update_yaxes(showgrid=False)
        show(r, fig_style(g.update_layout(title="When the risk builds"), 520, False, "The same medicines at every look-ahead window, in % chance"))

    if multi:
        st.markdown(band_html("banner_timeline.svg", "The longer the window, the <em>higher</em> the chance", "More time means more opportunity for a shortage to appear. A steep line marks a medicine that is likely to be hit sooner."), unsafe_allow_html=True)
        section("How the risk <em>builds</em> over time", "Pick medicines to compare. A line that climbs steeply means the shortage is likely to arrive sooner.")
        opts = v.drug.head(40).tolist()
        pick = st.multiselect("Medicines to compare", opts, default=opts[:5], format_func=lambda x: str(x).title())
        if pick:
            pal = [BLUE, ORANGE, AQUA, "#6250d6", "#eda100", "#e87ba4", "#008300", "#e34948"]
            f = go.Figure()
            for i, dname in enumerate(pick):
                row = v[v.drug == dname].iloc[0]
                f.add_scatter(x=hs, y=[row[f"p_{x}"] * 100 for x in hs], name=str(dname).title(), mode="lines+markers",
                              line=dict(width=2.5, color=pal[i % len(pal)]), marker=dict(size=9, line=dict(width=2, color="#fff")),
                              hovertemplate="%{y:.1f}%<extra>" + str(dname).title() + "</extra>")
            f.add_scatter(x=hs, y=[info[x]["base_rate"] * 100 for x in hs], name="Average medicine", mode="lines",
                          line=dict(width=2, color=GREY, dash="dot"), hovertemplate="%{y:.1f}%<extra>Average medicine</extra>")
            f.update_xaxes(title="Months ahead", tickvals=hs, ticktext=[f"{x}" for x in hs])
            f.update_yaxes(title="Chance of a new shortage (%)", rangemode="tozero")
            show(st, fig_style(f.update_layout(title="Chance of a new FDA shortage by look-ahead", hovermode="x unified"), 420, True,
                               "Each line is one medicine; the dotted grey line is the average medicine"))

    section("Full <em>forecast</em> table", f"Ranked by the chance in the next {h} months.")
    tbl = v.head(topn)
    cfg = {"drug": "Medicine", "category": "Category", "form": "Form", "level": "Level",
           "x_avg": st.column_config.NumberColumn("× avg", format="%.1f"), "c12": "Postings, last 12 mo", "months_since_last": "Months since last",
           "n_records": "All-time records"}
    for x in hs:
        cfg[f"p_{x}"] = st.column_config.ProgressColumn(f"Next {x} mo", format="%.2f", min_value=0, max_value=1)
    st.dataframe(tbl[["drug", "category", "form"] + [f"p_{x}" for x in hs] + ["level", "x_avg", "c12", "months_since_last"]],
                 hide_index=True, column_config=cfg, **WIDE)
    st.download_button("Download forecast (CSV)", v.to_csv(index=False).encode(), "medicine_shortage_forecast.csv", "text/csv")
    with st.expander("How reliable is this forecast?"):
        show_reliability(info)
        st.caption("Only medicines that have appeared in the FDA shortage database can be scored. Scores come from FDA posting patterns, "
                   "not live stock levels, and are not medical advice.")



# ---- model comparison -------------------------------------------------------------------------------
def metric_table(models, split="test", cols=("accuracy", "precision", "recall", "f1", "f2", "roc_auc", "pr_auc")):
    rows = []
    for r in models:
        x = r[split]
        row = {"Model": r["model"]}
        row.update({c.upper().replace("_", "-"): x.get(c) for c in cols})
        if "cv_pr_auc_mean" in r:
            row["CV PR-AUC"] = r["cv_pr_auc_mean"]
        rows.append(row)
    return pd.DataFrame(rows)


def style_table(df, best):
    num_cols = [c for c in df.columns if c != "Model"]
    sty = df.style.format({c: "{:.3f}" for c in num_cols}, na_rep="n/a")
    sty = sty.apply(lambda s: ["font-weight:700;background:#EEF5FD" if s["Model"] == best else "" for _ in s], axis=1)
    return sty.highlight_max(subset=num_cols, color="#d7f0d7")


def evidence_text(res, best, key="pr_auc"):
    """Plain-language justification generated only from the stored metrics."""
    b = next(r for r in res if r["model"] == best)
    rule = next((r for r in res if r["model"].startswith("Rule")), None)
    base = next((r for r in res if r["model"].startswith("Baseline")), None)
    rest = sorted([r for r in res if r["model"] not in (best,) and not r["model"].startswith(("Baseline", "Rule"))],
                  key=lambda r: -(r["val"][key] or 0))
    t = b["test"]
    out = [f"**{best}** was selected because it had the highest validation PR-AUC ({num(b['val'][key])}) - model choice never "
           f"looked at the test set."]
    out.append(f"On the untouched test set it reaches recall **{pct(t['recall'])}** at precision **{pct(t['precision'])}** "
               f"(F2 {num(t['f2'])}, ROC-AUC {num(t['roc_auc'])}, PR-AUC {num(t['pr_auc'])}).")
    if base:
        out.append(f"The do-nothing baseline has PR-AUC {num(base['test']['pr_auc'])}, so the model adds {num((t['pr_auc'] or 0) - (base['test']['pr_auc'] or 0))} PR-AUC.")
    if rule:
        out.append(f"The simple rule “posted in the last 12 months” gets recall {pct(rule['test']['recall'])} / precision {pct(rule['test']['precision'])} / "
                   f"F2 {num(rule['test']['f2'])}; the model is {'better' if t['f2'] > rule['test']['f2'] else 'not better'} on F2 "
                   f"({num(t['f2'])} vs {num(rule['test']['f2'])}).")
    if rest:
        r2 = rest[0]
        gap = abs((b["cv_pr_auc_mean"] or 0) - (r2["cv_pr_auc_mean"] or 0))
        spread = max(b["cv_pr_auc_std"] or 0, r2["cv_pr_auc_std"] or 0)
        verdict = ("The gap to the runner-up is **smaller than the fold-to-fold variation**, so the two are statistically very close; "
                   "the simpler/faster one is a reasonable choice." if gap <= spread else
                   "The gap to the runner-up is larger than the fold-to-fold variation, which supports the choice.")
        out.append(f"Runner-up: **{r2['model']}** (validation PR-AUC {num(r2['val'][key])}). Cross-validated PR-AUC: "
                   f"{num(b['cv_pr_auc_mean'])} ± {num(b['cv_pr_auc_std'])} vs {num(r2['cv_pr_auc_mean'])} ± {num(r2['cv_pr_auc_std'])}. {verdict}")
    return out


def confusion_fig(x):
    z = [[x["tn"], x["fp"]], [x["fn"], x["tp"]]]
    f = go.Figure(go.Heatmap(z=z, x=["Predicted: no", "Predicted: yes"], y=["Actual: no", "Actual: yes"], showscale=False,
                             colorscale=[[0, "#EAF6EF"], [1, "#0F5B38"]], text=z, texttemplate="%{text}", textfont=dict(size=18),
                             hovertemplate="%{y} / %{x}: %{z}<extra></extra>"))
    f.update_yaxes(autorange="reversed")
    f.update_xaxes(side="top", showline=False, ticks="")
    out = fig_style(f.update_layout(title="Confusion matrix"), 340, False, "Rows: what really happened. Columns: what the model said")
    return out.update_layout(margin=dict(t=132))


def curves_fig(res, kind, best=None):
    f = go.Figure()
    pal = [BLUE, ORANGE, AQUA, "#6250d6", "#eda100", "#e87ba4", "#008300", "#e34948"]
    for i, r in enumerate(x for x in res if not x["model"].startswith("Baseline")):
        c = r["curves"]
        xs, ys = (c["fpr"], c["tpr"]) if kind == "roc" else (c["recall"], c["precision"])
        f.add_scatter(x=xs, y=ys, name=r["model"], line=dict(width=3.6 if r["model"] == best else 1.8, color=pal[i % len(pal)]), hovertemplate="%{x:.2f}, %{y:.2f}<extra>" + r["model"] + "</extra>")
    if kind == "roc":
        f.add_scatter(x=[0, 1], y=[0, 1], line=dict(color=GREY, dash="dot", width=1.6), name="Pure guessing", hoverinfo="skip")
    f.update_xaxes(title="False-positive rate" if kind == "roc" else "Recall")
    f.update_yaxes(title="True-positive rate" if kind == "roc" else "Precision")
    f.update_xaxes(range=[0, 1], showgrid=False)
    f.update_yaxes(range=[0, 1.02])
    return fig_style(f.update_layout(title="ROC curves" if kind == "roc" else "Precision-Recall curves"), 400, True,
                     "Closer to the top-left is better; the selected model is the thickest line" if kind == "roc" else
                     "Staying high as recall grows is better; the selected model is the thickest line", legend_pos="right")


def clf_section(T, caption):
    if T.get("skipped"):
        st.warning(T["skipped"])
        return
    st.markdown(f"<div class='callout'>{caption}</div>", unsafe_allow_html=True)
    pr = T["positive_rate"]
    st.caption(f"Train {T['n_train']:,} · validation {T['n_val']:,} · test {T['n_test']:,} observations · share of positives "
               f"train {pct(pr['train'],1)} / val {pct(pr['val'],1)} / test {pct(pr['test'],1)}")
    st.markdown(winner_html(T), unsafe_allow_html=True)
    tab = metric_table(T["models"])
    st.dataframe(style_table(tab, T["best"]), hide_index=True, **WIDE)
    st.caption("Metrics at the decision threshold chosen on validation data to maximise F2 (recall counts double - a missed shortage costs more than a false alarm). Best value per column is shaded. Note how accuracy flatters the do-nothing baseline on rare events - that is why recall, precision, F2 and PR-AUC drive the decision.")
    mets = st.multiselect("Compare on", ["PR-AUC", "ROC-AUC", "RECALL", "PRECISION", "F1", "F2"], default=["PR-AUC", "RECALL", "F2"], key=caption[:20])
    f = go.Figure()
    for i, mname in enumerate(mets):
        f.add_bar(x=tab.Model, y=tab[mname].fillna(0), name=mname, marker_color=[BLUE, ORANGE, AQUA, "#6250d6", "#eda100", "#e87ba4"][i % 6])
    f.update_xaxes(tickangle=-18)
    f.update_yaxes(range=[0, 1.02])
    show(st, fig_style(f.update_layout(barmode="group", title="Test-set comparison", bargap=0.32), 430, True, "Higher is better; scored on months the models never saw", legend_pos="top"))
    a, b = st.columns(2)
    show(a, curves_fig(T["models"], "roc", T["best"]))
    show(b, curves_fig(T["models"], "pr", T["best"]))
    best = next(r for r in T["models"] if r["model"] == T["best"])
    a, b = st.columns(2)
    show(a, confusion_fig(best["test"]))
    imp = pd.DataFrame(T["importance"])[::-1]
    show(b, hbar_fig(imp.feature.map(flabel), imp.importance, f"What {T['best']} relies on", 340, "Drop in score when that fact is scrambled", fmt="{:.3f}",
                     hover="%{y}<br>score drops by %{x:.3f}<extra></extra>"))
    st.markdown("#### Why this model")
    for line in evidence_text(T["models"], T["best"]):
        st.markdown(f"- {line}")


def page_models(A):
    rep = A["report"]
    page_header("Model comparison", "Every model we trained, scored on months it had never seen, with the evidence for the one we selected.")
    demo_banner(rep)
    ew, sc_, du, fc, cl = rep["early_warning"], rep["status_classification"], rep["duration"], rep["forecast"], rep["clustering"]
    t = st.tabs(["★ Early-warning (final model)", "Status classification", "Duration regression", "Monthly volume (context)", "Segmentation", "Method & limits"])
    with t[0]:
        sp = ew["split_months"]
        clf_section(ew, f"<b>Task:</b> for each medicine and month, predict whether the FDA will post a <b>new</b> shortage for it within the next "
                        f"<b>{ew['horizon']} months</b>, using only information available up to that month. <b>Split by time:</b> train through {sp['train_end']}, "
                        f"validation {sp['val'][0]}–{sp['val'][1]}, test {sp['test'][0]}–{sp['test'][1]} (with {ew['horizon']}-month gaps so labels never overlap).")
        hf = rep.get("horizon_forecast") or {}
        if hf.get("horizons"):
            st.markdown("#### The same model family at every look-ahead window")
            st.caption("The Forecast page uses one model per window (3, 6, 9 and 12 months), each trained and tested on its own time-based split.")
            show_reliability({x["horizon"]: x for x in hf["horizons"]})
    with t[1]:
        clf_section(sc_, "<b>Task (Phase 1 objective 1):</b> classify whether a shortage record is still active (<code>is_shortage</code>) from facts known when it was "
                         "posted. Status is a snapshot label, so the primary split is stratified-random. Leaky fields (update_type, dates of update/discontinuation, related_info) are excluded.")
        if sc_.get("temporal"):
            tp = sc_["temporal"]
            with st.expander("Secondary check: chronological hold-out"):
                st.write(tp["note"])
                st.write(f"Positive share - train {pct(tp['positive_rate']['train'])}, val {pct(tp['positive_rate']['val'])}, test {pct(tp['positive_rate']['test'])}.")
                st.dataframe(style_table(metric_table(tp["models"]).drop(columns=[]), tp["best"]), hide_index=True, **WIDE)
    with t[2]:
        if du.get("skipped"):
            st.warning(du["skipped"])
        else:
            st.markdown(f"<div class='callout'><b>Task (Phase 1 objective 3):</b> predict how many days a shortage lasts until resolution. {du['note']}</div>", unsafe_allow_html=True)
            rows = [{"Model": r["model"], "MAE (days)": r["test"]["mae_days"], "RMSE (days)": r["test"]["rmse_days"],
                     "Median AE (days)": r["test"]["median_ae_days"], "R² (log)": r["test"]["r2_log"], "R² (days)": r["test"]["r2_days"]} for r in du["models"]]
            d = pd.DataFrame(rows)
            st.dataframe(d.style.format({c: "{:.2f}" for c in d.columns if c != "Model"}).apply(
                lambda s: ["font-weight:700;background:#EEF5FD" if s["Model"] == du["best"] else "" for _ in s], axis=1), hide_index=True, **WIDE)
            f = go.Figure(go.Bar(x=d.Model, y=d["MAE (days)"], marker_color=[ORANGE if m == du["best"] else PRIMARY for m in d.Model],
                                 text=d["MAE (days)"].round(0).astype(int), textposition="outside", cliponaxis=False, textfont=dict(size=12, color=INK2)))
            show(st, fig_style(f.update_layout(title="Test MAE in days", bargap=0.45), 340, False, "Lower is better; the orange bar is the selected model"))
            b0 = next(r for r in du["models"] if r["model"].startswith("Baseline"))
            bb = next(r for r in du["models"] if r["model"] == du["best"])
            st.markdown(f"- **{du['best']}** was chosen on validation MAE. Test MAE {bb['test']['mae_days']:.0f} d vs {b0['test']['mae_days']:.0f} d for always predicting the median "
                        f"({(1-bb['test']['mae_days']/b0['test']['mae_days'])*100:.0f}% {'lower' if bb['test']['mae_days']<b0['test']['mae_days'] else 'higher'}). Median duration in the data: {du['median_days']:.0f} days.")
    with t[3]:
        if fc.get("skipped"):
            st.warning(fc["skipped"])
        else:
            st.markdown(f"<div class='callout'><b>Background analysis (Phase 1 objective 2):</b> the overall number of new shortage postings per month. The <b>Forecast</b> page is where individual medicines are forecast. Trained through {fc['train_end']}, "
                        f"tested on the following {fc['test_months']} months ({fc['test_start']} onward). Naive baselines are included on purpose: a model must beat them to be worth using.</div>", unsafe_allow_html=True)
            rows = [{"Model": r["model"], "MAE": r["holdout_12m"]["mae"], "RMSE": r["holdout_12m"]["rmse"], "sMAPE %": r["holdout_12m"]["smape"],
                     "MASE": r["holdout_12m"]["mase"], "1-step MAE (rolling)": r["rolling_1step"]["mae"], "1-step RMSE": r["rolling_1step"]["rmse"]} for r in fc["models"]]
            d = pd.DataFrame(rows)
            st.dataframe(d.style.format({c: "{:.2f}" for c in d.columns if c != "Model"}).apply(
                lambda s: ["font-weight:700;background:#EEF5FD" if s["Model"] == fc["best"] else "" for _ in s], axis=1).highlight_min(
                subset=[c for c in d.columns if c != "Model"], color="#d7f0d7"), hide_index=True, **WIDE)
            f = go.Figure()
            f.add_bar(x=d.Model, y=d.MAE, name="12-month hold-out MAE", marker_color=PRIMARY)
            f.add_bar(x=d.Model, y=d["1-step MAE (rolling)"], name="Rolling 1-step MAE", marker_color=ORANGE)
            show(st, fig_style(f.update_layout(barmode="group", title="Mean absolute error", bargap=0.32), 400, True, "Postings per month; lower is better", legend_pos="top"))
            sn = next((r for r in fc["models"] if r["model"].startswith("Seasonal naive")), None)
            bb = next(r for r in fc["models"] if r["model"] == fc["best"])
            txt = f"- **{fc['best']}** has the lowest hold-out MAE among the non-trivial models: {bb['holdout_12m']['mae']:.2f} postings/month (mean monthly volume {fc['mean_monthly']:.1f}). "
            if sn:
                txt += f"The seasonal-naive baseline scores {sn['holdout_12m']['mae']:.2f}. "
            txt += "MASE < 1 means it beats the in-sample seasonal-naive error. With only 12 test points, small differences between models are within noise."
            st.markdown(txt)
    with t[4]:
        ch = cl["chosen"]
        st.markdown(f"<div class='callout'><b>Unsupervised:</b> medicines grouped by shortage behaviour (records, manufacturers, formulations, share active, typical duration, years active, recent activity). "
                    f"Chosen: <b>{ch['algorithm']}, k={ch['k']}</b> - silhouette <b>{ch['silhouette']:.3f}</b>. {cl['note']}</div>", unsafe_allow_html=True)
        g = pd.DataFrame(cl["grid"])
        f = go.Figure()
        for i, (a, d) in enumerate(g.groupby("algorithm")):
            f.add_scatter(x=d.k, y=d.silhouette, name=a, mode="lines+markers", line=dict(width=2.5, color=[BLUE, ORANGE, AQUA][i]), marker=dict(size=8, line=dict(width=2, color="#fff")))
        show(st, fig_style(f.update_layout(title="Silhouette score by number of clusters"), 360, True, "Higher means clearer groups").update_xaxes(dtick=1, title="k"))
        st.dataframe(pd.DataFrame(cl["segment_summary"]).round(2), hide_index=True, **WIDE)
        st.dataframe(g.round(3), hide_index=True, **WIDE)
    with t[5]:
        st.markdown("""
**What makes these numbers trustworthy**
- **Time-based evaluation.** Train → validation → test always move forward in time; for the early-warning model a gap equal to the prediction horizon separates the sets so labels cannot overlap.
- **Model choice uses validation data only**; the test set is looked at once, for reporting. Decision thresholds are also set on validation data.
- **No label leakage.** Fields written as a shortage evolves (`update_type`, `update_date`, `discontinued_date`, `related_info`) are never used as inputs. Company frequency is learned from training rows only.
- **Imbalance** is handled by threshold tuning (F2) and reported with PR-AUC, which is much more honest than accuracy for rare events.
- **Baselines** (base rate, and a one-line rule) are shown so every model has to earn its place.

**Limits to keep in mind**
- The model covers medicines that have appeared in the FDA database at least once; it cannot score a medicine with no history.
- Scores come from FDA *posting* patterns, not live hospital inventory or demand. Phase 2 found essentially no relationship between (synthetic) sales and shortages, so no sales data is used.
- Duration is only observable for resolved shortages (right-censoring).
- Forecast intervals are approximate and the 12-month test window is short.
""")


def page_explorer(A):
    df = A["data"]
    page_header("Data explorer", "Browse and filter the FDA shortage records behind the models.")
    demo_banner(A["report"])
    c = st.columns(4)
    cat = c[0].multiselect("Category", sorted(df.primary_category.unique()))
    stt = c[1].multiselect("Status", sorted(df.status.unique()))
    frm = c[2].multiselect("Dosage form", sorted(df.form_group.unique()))
    y0, y1 = int(df.initial_posting_date.dt.year.min()), int(df.initial_posting_date.dt.year.max())
    yr = c[3].slider("Posted between", y0, y1, (y0, y1)) if y0 < y1 else (y0, y1)
    txt = st.text_input("Search medicine / company / brand").strip().lower()
    v = df[df.initial_posting_date.dt.year.between(*yr)]
    for col, sel in [("primary_category", cat), ("status", stt), ("form_group", frm)]:
        if sel:
            v = v[v[col].isin(sel)]
    if txt:
        v = v[v.generic_name.str.lower().str.contains(txt, na=False) | v.company_name.str.lower().str.contains(txt, na=False)
              | v.brand_name.fillna("").str.lower().str.contains(txt)]
    if v.empty:
        st.warning("No records match. Loosen a filter or clear the search box.")
        return
    st.caption(f"{len(v):,} of {len(df):,} records")
    l, r = st.columns(2)
    s = v.groupby("status").size().reset_index(name="n")
    f = go.Figure(go.Bar(x=s.status, y=s.n, marker_color=PRIMARY, text=s.n, textposition="outside", cliponaxis=False, textfont=dict(size=12, color=INK2),
                         hovertemplate="%{x}: %{y} records<extra></extra>"))
    f.update_layout(bargap=0.5)
    show(l, fig_style(f.update_layout(title="Records by status"), 340, False, "Of the records that match your filters"))
    heat = v.groupby(["primary_category", "form_group"]).size().unstack(fill_value=0)
    heat = heat.loc[heat.sum(1).sort_values().tail(10).index]
    f = go.Figure(go.Heatmap(z=heat.values, x=heat.columns, y=heat.index, colorscale=[[i / 6, c] for i, c in enumerate(SEQ)],
                             hovertemplate="%{y} · %{x}: %{z}<extra></extra>"))
    f.update_yaxes(autorange="reversed")
    show(r, fig_style(f.update_layout(title="Category by dosage form"), 340, False, "Darker squares hold more records"))
    st.dataframe(v[["generic_name", "brand_name", "company_name", "primary_category", "dosage_form", "status", "initial_posting_date", "update_date"]],
                 hide_index=True, **WIDE)
    st.download_button("Download filtered records (CSV)", v.to_csv(index=False).encode(), "filtered_shortages.csv", "text/csv")


# ----------------------------------------------------------------------------------------------- router
PAGES = {"Overview": page_overview, "Risk checker": page_checker, "Watchlist": page_watchlist,
         "Forecast": page_forecast, "Models": page_models, "Data": page_explorer}
SETUP = "Setup"
BRAND = ('<div class="brand"><svg width="36" height="36" viewBox="0 0 32 32" aria-hidden="true"><rect width="32" height="32" rx="9" fill="#2E9E5B"/>'
         '<g transform="rotate(-40 16 16)"><rect x="5.5" y="11" width="21" height="10" rx="5" fill="none" stroke="#fff" stroke-width="2"/>'
         '<path d="M16 11v10" stroke="#fff" stroke-width="2"/><rect x="5.5" y="11" width="10.5" height="10" rx="5" fill="#fff"/></g></svg>'
         '<div><b>MedShortage</b><span>Shortage early warning</span></div></div>')

A = artifacts()
names = list(PAGES) + [SETUP] if A else [SETUP]
_qp = st.query_params.get("page")
_idx = names.index(_qp) if _qp in names else 0
chip = ""
if A:
    _m = A["report"]["meta"]
    _demo = bool(_m.get("is_demo"))
    chip = (f'<div class="chip"><span class="d" style="background:{"#fab219" if _demo else "#34D17F"}"></span><div>'
            f'<b>{"Demo data, not real" if _demo else "openFDA data"}</b>'
            f'<span>{_m["n_clean"]:,} records through {esc(_m["last_posting"])}</span></div></div>')
with st.container(key="topnav"):
    _c1, _c2, _c3 = st.columns([1.3, 5.4, 1.7], vertical_alignment="center")
    _c1.markdown(BRAND, unsafe_allow_html=True)
    choice = _c2.radio("Navigate", names, index=_idx, horizontal=True, label_visibility="collapsed", key="nav")
    _c3.markdown(chip, unsafe_allow_html=True)
st.query_params["page"] = choice

_H = A["report"]["early_warning"]["horizon"] if A else 6
HEROES = {
    "Overview": ("See shortages <em>before</em> they reach the shelf",
                 f"Machine learning on the FDA's shortage history flags the medicines most likely to run short in the next {_H} months.", True),
    "Risk checker": ("Check a medicine's <em>risk</em>",
                     f"Search any medicine for its estimated chance of a new FDA shortage in the next {_H} months.", False),
    "Watchlist": ("The medicines to <em>watch</em> next", f"Every medicine with FDA shortage history, ranked by its chance of a new shortage in the next {_H} months.", False),
    "Forecast": ("Which medicines are <em>next</em>", "Each medicine's chance of a new FDA shortage in the next 3, 6, 9 or 12 months, learned from its own shortage history.", False),
    "Models": ("How the models <em>compare</em>", "Every model we trained, scored on months it had never seen, with the evidence for the one we selected.", False),
    "Data": ("The <em>data</em> behind the scores", "Browse and filter the FDA shortage records the models learn from.", False),
    "Setup": ("Bring in the <em>FDA data</em>", "Choose where the shortage data comes from, then train the models. Everything runs on this computer.", False),
}
_t, _s, _big = HEROES[choice]
_ctas = [("Check a medicine", "Risk checker", "cta"), ("See the watchlist", "Watchlist", "ghost")] if (choice == "Overview" and A) else None
st.markdown(hero_html(_t, _s, _big, _ctas, ART_FOR.get(choice)), unsafe_allow_html=True)

with st.container(key="sheet"):
    try:
        if choice == SETUP or not A:
            page_setup()
        else:
            with st.spinner("Loading ..."):
                PAGES[choice](A)
    except pl.DataError as e:
        st.error(str(e))
    except Exception as e:  # noqa: BLE001
        st.error("This page hit an unexpected problem. Try retraining on the **Setup** page; if it persists, share the details below.")
        with st.expander("Technical details"):
            st.exception(e)
st.markdown(footer_html(A), unsafe_allow_html=True)
