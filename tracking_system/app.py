import streamlit as st
import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.colors import LinearSegmentedColormap
import json
import os
import time
import cv2
import requests
from websockets.sync.client import connect  # Synchronous websocket connection for streamlit
import threading

st.set_page_config(
    page_title="SYNAPSE · Tactical Engine",
    layout="wide",
    initial_sidebar_state="expanded",
    page_icon="⚽",
)

# ─────────────────────────────────────────────────────────────────────────────
#  GLOBAL CSS — Premium Dark Sports Analytics
# ─────────────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Bebas+Neue&family=Space+Grotesk:wght@300;400;500;600;700&family=JetBrains+Mono:wght@400;600;700&family=DM+Sans:wght@300;400;500;600;700&display=swap');

/* ── RESET & BASE ── */
*, *::before, *::after { box-sizing: border-box; margin: 0; padding: 0; }

:root {
    --bg-deep:    #020810;
    --bg-mid:     #040e1c;
    --bg-surface: #071220;
    --bg-raised:  #0a1828;
    --bg-hover:   #0d1f32;
    --accent-green: #00e87a;
    --accent-blue:  #1e8fff;
    --accent-amber: #f59e0b;
    --accent-violet:#a78bfa;
    --accent-red:   #f43f5e;
    --border-subtle: rgba(255,255,255,0.055);
    --border-glow-g: rgba(0,232,122,0.18);
    --border-glow-b: rgba(30,143,255,0.18);
    --text-primary: rgba(255,255,255,0.92);
    --text-secondary: rgba(255,255,255,0.45);
    --text-dim: rgba(255,255,255,0.2);
    --font-display: 'Bebas Neue', sans-serif;
    --font-body: 'DM Sans', sans-serif;
    --font-mono: 'JetBrains Mono', monospace;
}

html, body, [class*="css"], .stApp {
    font-family: var(--font-body) !important;
    background: var(--bg-deep) !important;
}

/* ── DEEP SPACE BACKGROUND ── */
.stApp {
    background:
        radial-gradient(ellipse 60% 40% at 15% 5%, rgba(0,232,122,0.05) 0%, transparent 55%),
        radial-gradient(ellipse 50% 35% at 85% 90%, rgba(30,143,255,0.06) 0%, transparent 55%),
        radial-gradient(ellipse 120% 100% at 50% 50%, var(--bg-deep) 0%, #010710 100%) !important;
    min-height: 100vh;
}

/* Subtle noise texture overlay */
.stApp::before {
    content: '';
    position: fixed;
    inset: 0;
    background-image: url("data:image/svg+xml,%3Csvg viewBox='0 0 256 256' xmlns='http://www.w3.org/2000/svg'%3E%3Cfilter id='n'%3E%3CfeTurbulence type='fractalNoise' baseFrequency='0.9' numOctaves='4' stitchTiles='stitch'/%3E%3C/filter%3E%3Crect width='100%25' height='100%25' filter='url(%23n)' opacity='0.02'/%3E%3C/svg%3E");
    pointer-events: none;
    z-index: 0;
    opacity: 0.4;
}

/* ── HIDE STREAMLIT CHROME ── */
#MainMenu, header, footer { visibility: hidden; }
.block-container {
    padding-top: 0 !important;
    padding-bottom: 2rem !important;
    padding-left: 1.5rem !important;
    padding-right: 1.5rem !important;
    max-width: 100% !important;
}

/* ══════════════════════════════════════════════
   SIDEBAR — FIXED SCROLLING + UPGRADED DESIGN
   ══════════════════════════════════════════════ */
section[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #030c1a 0%, #020810 100%) !important;
    border-right: 1px solid rgba(0,232,122,0.07) !important;
    box-shadow: 4px 0 60px rgba(0,0,0,0.7) !important;
    /* FIX: prevent sidebar from creating scroll context issues */
    overflow: hidden !important;
}
section[data-testid="stSidebar"] > div {
    /* FIX: the inner div handles scroll, not the outer container */
    overflow-y: auto !important;
    overflow-x: hidden !important;
    height: 100vh !important;
    /* FIX: use padding-bottom so bottom content isn't clipped */
    padding-bottom: 24px !important;
    scrollbar-width: thin;
    scrollbar-color: rgba(0,232,122,0.15) transparent;
}
section[data-testid="stSidebar"] > div::-webkit-scrollbar {
    width: 3px;
}
section[data-testid="stSidebar"] > div::-webkit-scrollbar-track {
    background: transparent;
}
section[data-testid="stSidebar"] > div::-webkit-scrollbar-thumb {
    background: rgba(0,232,122,0.15);
    border-radius: 3px;
}
section[data-testid="stSidebar"] * { font-family: var(--font-body) !important; }
section[data-testid="stSidebar"] .block-container {
    padding-top: 0 !important;
    padding-left: 1rem !important;
    padding-right: 1rem !important;
    /* FIX: ensure full height content is accessible */
    min-height: unset !important;
}

/* ── GLOBAL SCROLLBAR ── */
::-webkit-scrollbar { width: 4px; height: 4px; }
::-webkit-scrollbar-track { background: var(--bg-deep); }
::-webkit-scrollbar-thumb { background: rgba(0,232,122,0.15); border-radius: 4px; }
::-webkit-scrollbar-thumb:hover { background: rgba(0,232,122,0.3); }

/* ── SLIDER ── */
.stSlider [data-baseweb="slider"] { padding: 0 4px; }
.stSlider [data-testid="stThumbValue"] {
    font-family: var(--font-mono) !important;
    font-weight: 700 !important;
    color: var(--accent-green) !important;
    background: var(--bg-raised) !important;
    border: 1px solid rgba(0,232,122,0.25) !important;
    border-radius: 4px !important;
    padding: 2px 6px !important;
    font-size: 11px !important;
}
/* Slider track color */
div[data-baseweb="slider"] div[data-testid="stSlider"] {
    accent-color: var(--accent-green);
}

/* ══════════════════════════════
   TOP NAV BAR
   ══════════════════════════════ */
.top-nav {
    background: rgba(3, 12, 26, 0.95);
    border-bottom: 1px solid var(--border-subtle);
    padding: 0;
    display: flex;
    align-items: stretch;
    height: 60px;
    margin: -1.5rem -1.5rem 0;
    position: sticky;
    top: 0;
    z-index: 999;
    backdrop-filter: blur(24px);
    -webkit-backdrop-filter: blur(24px);
}
.nav-brand {
    display: flex;
    align-items: center;
    gap: 12px;
    padding: 0 28px;
    border-right: 1px solid var(--border-subtle);
    flex-shrink: 0;
}
.nav-logo {
    width: 34px; height: 34px;
    background: linear-gradient(135deg, var(--accent-green) 0%, var(--accent-blue) 100%);
    border-radius: 9px;
    display: flex; align-items: center; justify-content: center;
    font-size: 17px;
    box-shadow: 0 0 24px rgba(0,232,122,0.35);
    flex-shrink: 0;
}
.nav-brand-text {
    font-family: var(--font-display);
    font-size: 22px;
    letter-spacing: 3px;
    color: white;
    line-height: 1;
}
.nav-brand-text span {
    background: linear-gradient(135deg, var(--accent-green), #00c4ff);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
}
.nav-sub {
    font-size: 9px;
    letter-spacing: 2.5px;
    color: var(--text-dim);
    font-weight: 500;
    margin-top: -2px;
    text-transform: uppercase;
}
.nav-tabs {
    display: flex;
    align-items: center;
    padding: 0 20px;
    gap: 1px;
    flex: 1;
}
.nav-tab {
    padding: 8px 16px;
    font-size: 10.5px;
    font-weight: 600;
    letter-spacing: 1.5px;
    text-transform: uppercase;
    color: var(--text-dim);
    border-radius: 7px;
    cursor: pointer;
    transition: all 0.2s;
    border: 1px solid transparent;
    font-family: var(--font-body);
    white-space: nowrap;
}
.nav-tab.active {
    color: var(--accent-green);
    background: rgba(0,232,122,0.07);
    border-color: rgba(0,232,122,0.13);
}
.nav-tab:hover:not(.active) {
    color: rgba(255,255,255,0.5);
    background: rgba(255,255,255,0.03);
}

/* ══════════════════════════════
   TICKER BAR
   ══════════════════════════════ */
.ticker {
    background: linear-gradient(90deg, rgba(0,232,122,0.04), rgba(30,143,255,0.03), rgba(0,232,122,0.04));
    border-bottom: 1px solid rgba(0,232,122,0.07);
    padding: 8px 24px;
    display: flex;
    align-items: center;
    gap: 0;
    margin-left: -1.5rem;
    margin-right: -1.5rem;
    margin-bottom: 24px;
    overflow: hidden;
    flex-wrap: nowrap;
}
.ticker-item {
    display: flex;
    align-items: center;
    gap: 7px;
    padding: 0 20px;
    white-space: nowrap;
    position: relative;
}
.ticker-item:not(:last-child)::after {
    content: '';
    position: absolute;
    right: 0;
    top: 50%;
    transform: translateY(-50%);
    height: 16px;
    width: 1px;
    background: var(--border-subtle);
}
.ticker-label {
    font-size: 9px;
    font-weight: 600;
    letter-spacing: 2px;
    text-transform: uppercase;
    color: var(--text-dim);
    font-family: var(--font-body);
}
.ticker-val {
    font-family: var(--font-mono);
    font-size: 11px;
    font-weight: 700;
}

/* ══════════════════════════════
   METRIC STRIP
   ══════════════════════════════ */
.metric-strip {
    display: grid;
    grid-template-columns: repeat(4, 1fr);
    gap: 10px;
    margin-bottom: 20px;
}
.metric-cell {
    background: var(--bg-surface);
    border: 1px solid var(--border-subtle);
    border-radius: 14px;
    padding: 20px 22px;
    position: relative;
    overflow: hidden;
    transition: transform 0.2s, border-color 0.2s;
}
.metric-cell:hover {
    transform: translateY(-2px);
}
.metric-cell::after {
    content: '';
    position: absolute;
    bottom: 0; left: 0; right: 0;
    height: 1px;
}
.mc1 { border-top: 2px solid rgba(0,232,122,0.5); }
.mc2 { border-top: 2px solid rgba(30,143,255,0.5); }
.mc3 { border-top: 2px solid rgba(245,158,11,0.5); }
.mc4 { border-top: 2px solid rgba(167,139,250,0.5); }

.mc1:hover { border-color: rgba(0,232,122,0.25); }
.mc2:hover { border-color: rgba(30,143,255,0.25); }
.mc3:hover { border-color: rgba(245,158,11,0.25); }
.mc4:hover { border-color: rgba(167,139,250,0.25); }

.metric-label {
    font-size: 9px;
    font-weight: 600;
    letter-spacing: 2.5px;
    text-transform: uppercase;
    color: var(--text-dim);
    margin-bottom: 10px;
    font-family: var(--font-body);
}
.metric-value {
    font-family: var(--font-display);
    font-size: 38px;
    line-height: 1;
    letter-spacing: 1px;
}
.metric-unit {
    font-family: var(--font-mono);
    font-size: 13px;
    color: var(--text-secondary);
    margin-left: 2px;
}
.metric-sub {
    font-size: 10px;
    color: var(--text-dim);
    margin-top: 6px;
    font-family: var(--font-body);
}
.mv-green  { color: var(--accent-green); }
.mv-blue   { color: var(--accent-blue); }
.mv-amber  { color: var(--accent-amber); }
.mv-violet { color: var(--accent-violet); }

/* ══════════════════════════════
   CARDS
   ══════════════════════════════ */
.card {
    background: var(--bg-surface);
    border: 1px solid var(--border-subtle);
    border-radius: 14px;
    overflow: hidden;
}
.card-header {
    display: flex;
    align-items: center;
    justify-content: space-between;
    padding: 16px 20px;
    border-bottom: 1px solid var(--border-subtle);
    background: linear-gradient(90deg, rgba(0,232,122,0.03), transparent);
}
.card-title {
    font-family: var(--font-body);
    font-size: 12px;
    font-weight: 600;
    letter-spacing: 2px;
    text-transform: uppercase;
    color: var(--text-secondary);
    display: flex; align-items: center; gap: 9px;
}

/* ══════════════════════════════
   PASS CARDS
   ══════════════════════════════ */
.pass-card {
    background: var(--bg-raised);
    border: 1px solid var(--border-subtle);
    border-radius: 12px;
    padding: 18px;
    margin-bottom: 10px;
    position: relative;
    overflow: hidden;
    transition: transform 0.2s ease, box-shadow 0.2s ease;
}
.pass-card:hover { transform: translateX(4px); }
.pass-rank-watermark {
    position: absolute; right: -6px; top: -14px; font-family: var(--font-display); font-size: 96px; line-height: 1; opacity: 0.035; color: white; pointer-events: none; letter-spacing: -4px;
}
.pass-card-head { display: flex; align-items: flex-start; justify-content: space-between; margin-bottom: 14px; }
.pass-target-label { font-size: 8px; font-weight: 700; letter-spacing: 2.5px; text-transform: uppercase; color: var(--text-dim); margin-bottom: 4px; font-family: var(--font-body); }
.pass-target-name { font-family: var(--font-display); font-size: 20px; letter-spacing: 3px; line-height: 1; }
.pass-coord { font-family: var(--font-mono); font-size: 9px; color: var(--text-dim); margin-top: 5px; }

.ev-badge { text-align: right; background: var(--bg-surface); border-radius: 8px; padding: 8px 12px; border: 1px solid var(--border-subtle); }
.ev-badge-label { font-size: 8px; font-weight: 700; letter-spacing: 2px; text-transform: uppercase; color: var(--text-dim); margin-bottom: 4px; font-family: var(--font-body); }
.ev-badge-value { font-family: var(--font-mono); font-size: 20px; font-weight: 700; line-height: 1; }

.bar-item { margin-bottom: 10px; }
.bar-item:last-child { margin-bottom: 0; }
.bar-item-head { display: flex; justify-content: space-between; align-items: center; margin-bottom: 5px; }
.bar-item-label { font-size: 9px; font-weight: 600; letter-spacing: 1.5px; text-transform: uppercase; color: var(--text-dim); font-family: var(--font-body); }
.bar-item-val { font-family: var(--font-mono); font-size: 11px; font-weight: 700; color: rgba(255,255,255,0.75); }
.bar-track { height: 4px; background: rgba(255,255,255,0.05); border-radius: 4px; overflow: hidden; }
.bar-fill { height: 100%; border-radius: 4px; transition: width 0.6s ease; }

/* ══════════════════════════════
   STATS GRID
   ══════════════════════════════ */
.stats-grid { display: grid; grid-template-columns: 1fr 1fr; gap: 8px; margin-top: 12px; }
.stat-cell { background: var(--bg-surface); border: 1px solid var(--border-subtle); border-radius: 10px; padding: 14px 16px; transition: border-color 0.2s; }
.stat-cell:hover { border-color: rgba(255,255,255,0.1); }
.stat-cell-label { font-size: 9px; font-weight: 600; letter-spacing: 2px; text-transform: uppercase; color: var(--text-dim); margin-bottom: 6px; font-family: var(--font-body); }
.stat-cell-val { font-family: var(--font-mono); font-size: 22px; font-weight: 700; line-height: 1; }

/* ══════════════════════════════
   SIDEBAR COMPONENTS
   ══════════════════════════════ */
.sb-header { padding: 22px 0 18px; border-bottom: 1px solid rgba(255,255,255,0.05); margin-bottom: 22px; }
.sb-logo-row { display: flex; align-items: center; gap: 12px; margin-bottom: 8px; }
.sb-logo { width: 40px; height: 40px; background: linear-gradient(135deg, var(--accent-green) 0%, var(--accent-blue) 100%); border-radius: 11px; display: flex; align-items: center; justify-content: center; font-size: 20px; box-shadow: 0 0 28px rgba(0,232,122,0.28); flex-shrink: 0; }
.sb-brand { font-family: var(--font-display); font-size: 22px; letter-spacing: 3px; color: white; line-height: 1; }
.sb-brand span { background: linear-gradient(135deg, var(--accent-green), #00ccff); -webkit-background-clip: text; -webkit-text-fill-color: transparent; background-clip: text; }
.sb-subtitle { font-size: 9px; font-weight: 500; letter-spacing: 3px; text-transform: uppercase; color: var(--text-dim); margin-top: 2px; font-family: var(--font-body); }
.sb-section-title { font-family: var(--font-body); font-size: 9px; font-weight: 700; letter-spacing: 3px; text-transform: uppercase; color: var(--text-dim); margin-bottom: 10px; display: flex; align-items: center; gap: 8px; }

/* Status rows */
.sb-status-list { display: flex; flex-direction: column; gap: 0; }
.sb-status-row { display: flex; justify-content: space-between; align-items: center; padding: 10px 0; border-bottom: 1px solid rgba(255,255,255,0.035); }
.sb-status-row:last-child { border-bottom: none; }
.sb-status-name { font-size: 11px; font-weight: 500; color: var(--text-secondary); display: flex; align-items: center; gap: 8px; font-family: var(--font-body); }
.sb-status-val { font-family: var(--font-mono); font-size: 9px; font-weight: 700; letter-spacing: 1px; padding: 2px 7px; border-radius: 4px; }
.val-online { color: var(--accent-green); background: rgba(0,232,122,0.08); border: 1px solid rgba(0,232,122,0.15); }

/* Legend items */
.sb-legend-item { display: flex; align-items: center; gap: 10px; padding: 8px 0; border-bottom: 1px solid rgba(255,255,255,0.035); }
.legend-dot { width: 10px; height: 10px; border-radius: 50%; flex-shrink: 0; }
.legend-label { font-size: 11px; font-weight: 400; color: var(--text-secondary); font-family: var(--font-body); }

/* Powered by */
.sb-powered { text-align: center; padding: 14px 0 4px; margin-top: 6px; border-top: 1px solid rgba(255,255,255,0.04); }
.sb-powered-label { font-size: 8px; font-weight: 600; letter-spacing: 3px; text-transform: uppercase; color: var(--text-dim); margin-bottom: 8px; font-family: var(--font-body); }
.sb-powered-logos { display: flex; justify-content: center; gap: 6px; flex-wrap: wrap; }
.sb-powered-logo { font-size: 9px; font-weight: 600; color: rgba(255,255,255,0.18); letter-spacing: 0.5px; padding: 3px 8px; border: 1px solid rgba(255,255,255,0.06); border-radius: 4px; font-family: var(--font-body); }
</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────
#  DATA LOADING
# ─────────────────────────────────────────

PITCH_LENGTH = 105.0
PITCH_WIDTH  = 68.0

@st.cache_data
def load_tracking_data():
    results_dir = "output"
    
    json_path = os.path.join(results_dir, "tracking_data.json")
    intel_csv_path = os.path.join(results_dir, "tracking_data_intel.csv")
    
    if not os.path.exists(json_path) or not os.path.exists(intel_csv_path):
        st.warning("Tracking data not found in `output/` directory. Please run `main.py` first.")
        return [], pd.DataFrame()
        
    with open(json_path, 'r') as f:
        tracking_json = json.load(f)
        
    intel_df = pd.read_csv(intel_csv_path)
    return tracking_json, intel_df

tracking_data, intel_data = load_tracking_data()

if not tracking_data:
    st.stop()

frames = [f['frame'] for f in tracking_data]
min_frame = min(frames)
max_frame = max(frames)

@st.cache_resource
def load_video_capture():
    video_path = "output/output_tracked_video.mp4"
    if os.path.exists(video_path):
        return cv2.VideoCapture(video_path)
    return None

cap = load_video_capture()

def get_video_frame(cap_obj, frame_num):
    if cap_obj is None:
        return None
    
    cap_obj.set(cv2.CAP_PROP_POS_FRAMES, frame_num)
    ret, frame_bgr = cap_obj.read()
    if ret:
        # Convert BGR to RGB for Streamlit/Matplotlib
        return cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    return None

# ─────────────────────────────────────────
#  PITCH DRAWING (matplotlib)
# ─────────────────────────────────────────
def draw_pitch_lines(ax):
    lc, la, lw = "white", 0.65, 1.4

    for i in range(0, int(PITCH_LENGTH), 10):
        alpha = 0.022 if (i // 10) % 2 == 0 else 0.0
        ax.add_patch(patches.Rectangle((i, 0), 10, PITCH_WIDTH, facecolor="white", alpha=alpha, zorder=1))

    # Boundary
    ax.plot([0, PITCH_LENGTH, PITCH_LENGTH, 0, 0], [0, 0, PITCH_WIDTH, PITCH_WIDTH, 0], color=lc, lw=lw, alpha=la, zorder=2)
    # Halfway
    ax.plot([PITCH_LENGTH/2]*2, [0, PITCH_WIDTH], color=lc, lw=lw*0.8, alpha=la*0.6, zorder=2)
    # Penalty boxes
    for side_x, mirror in [(0, 1), (PITCH_LENGTH, -1)]:
        ax.plot([side_x, side_x+mirror*16.5, side_x+mirror*16.5, side_x], [13.85, 13.85, 54.15, 54.15], color=lc, lw=lw*0.8, alpha=la*0.6, zorder=2)
    # 6-yard boxes
    for side_x, mirror in [(0, 1), (PITCH_LENGTH, -1)]:
        ax.plot([side_x, side_x+mirror*5.5, side_x+mirror*5.5, side_x], [24.84, 24.84, 43.16, 43.16], color=lc, lw=lw*0.6, alpha=la*0.4, zorder=2)
    # Centre circle
    ax.add_patch(patches.Circle((PITCH_LENGTH/2, PITCH_WIDTH/2), 9.15, edgecolor=lc, facecolor="none", lw=lw*0.8, alpha=la*0.6, zorder=2))
    ax.scatter(PITCH_LENGTH/2, PITCH_WIDTH/2, c=lc, s=30, zorder=2, alpha=la*0.7)

def build_pitch_figure(frame_idx, tracking_frame, intel_row):
    fig, ax = plt.subplots(figsize=(12, 7.8))
    fig.patch.set_facecolor("#030e1a")
    ax.set_facecolor("#030e1a")

    ax.imshow(
        np.tile(np.linspace(0, 1, 100).reshape(1, -1), (100, 1)),
        extent=[-2, PITCH_LENGTH+2, -2, PITCH_WIDTH+2],
        cmap=LinearSegmentedColormap.from_list("g", ["#031c0a", "#041f0c"]),
        aspect="auto", zorder=0
    )

    ax.set_aspect("equal")
    ax.set_xlim(-2, PITCH_LENGTH + 2)
    ax.set_ylim(-2, PITCH_WIDTH  + 2)
    draw_pitch_lines(ax)

    players = tracking_frame.get('players', [])
    ball_carrier_id = intel_row['ball_carrier'] if not pd.isna(intel_row.get('ball_carrier')) else None
    best_pass_id = intel_row['best_pass_option'] if not pd.isna(intel_row.get('best_pass_option')) else None
    
    bc_pos = None
    pass_pos = None

    for p in players:
        px, py = p['x'], p['y']
        vx, vy = p.get('vx', 0.0), p.get('vy', 0.0)
        
        is_my_team = p.get('is_my_team', False)
        
        if p['id'] == ball_carrier_id:
            fill_color = "#00e87a" # neon green
            edge_color = "white"
            bc_pos = (px, py)
        elif is_my_team:
            fill_color = "#c62828" # Red
            edge_color = "#ef9a9a"
        else:
            fill_color = "#1565c0" # Blue
            edge_color = "#64b5f6"
            
        if p['id'] == best_pass_id:
            pass_pos = (px, py)

        size = 175
        ax.scatter(px+0.6, py+0.6, c="#000000", s=size*0.8, alpha=0.25, zorder=3)
        ax.scatter(px, py, c=fill_color, s=size, edgecolors=edge_color, linewidths=1.8, zorder=4)
        
        speed = np.hypot(vx, vy)
        if speed > 0.5:
            scale = min(1.5, 0.6 + speed * 0.06)
            ax.annotate("", xy=(px + vx*scale*0.12, py + vy*scale*0.12), xytext=(px, py),
                        arrowprops=dict(arrowstyle="->", color=fill_color, lw=1.2, mutation_scale=9), zorder=5)

    # Draw Pass Arrow
    if bc_pos and pass_pos:
        col = "#f59e0b"
        ax.annotate("", xy=pass_pos, xytext=bc_pos, arrowprops=dict(facecolor=col, edgecolor=col, width=2.8, headwidth=12, headlength=9, shrink=0.06, linewidth=0), zorder=6)
        ax.add_patch(patches.Circle(pass_pos, 3.2, edgecolor=col, facecolor="none", lw=1.8, alpha=0.7, zorder=6))

    ax.axis("off")
    plt.tight_layout(pad=0)
    return fig

# ─────────────────────────────────────────
#  SIDEBAR
# ─────────────────────────────────────────
with st.sidebar:
    st.markdown("""
    <div class="sb-header">
        <div class="sb-logo-row">
            <div class="sb-logo">⚽</div>
            <div>
                <div class="sb-brand"><span>SYNAPSE</span></div>
                <div class="sb-subtitle">Post-Match Dashboard</div>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown('<div class="sb-section-title">Match Timeline</div>', unsafe_allow_html=True)
    frame = st.slider("", min_value=min_frame, max_value=max_frame, value=min_frame, step=1, label_visibility="collapsed")
    st.markdown("<div style='height:18px'></div>", unsafe_allow_html=True)

    st.markdown('<div class="sb-section-title">System Status</div>', unsafe_allow_html=True)
    st.markdown("""
    <div class="sb-status-list">
        <div class="sb-status-row">
            <span class="sb-status-name">Spatial Engine</span>
            <span class="sb-status-val val-online">ONLINE</span>
        </div>
        <div class="sb-status-row">
            <span class="sb-status-name">Tactical Intelligence</span>
            <span class="sb-status-val val-online">ONLINE</span>
        </div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("<div style='height:22px'></div>", unsafe_allow_html=True)
    st.markdown('<div class="sb-section-title">Pitch Legend</div>', unsafe_allow_html=True)
    st.markdown("""
    <div>
        <div class="sb-legend-item">
            <div class="legend-dot" style="background:#00e87a;box-shadow:0 0 8px rgba(0,232,122,0.6);"></div>
            <span class="legend-label">Ball Carrier</span>
        </div>
        <div class="sb-legend-item">
            <div class="legend-dot" style="background:#ef9a9a;"></div>
            <span class="legend-label">Attackers (Home)</span>
        </div>
        <div class="sb-legend-item">
            <div class="legend-dot" style="background:#64b5f6;"></div>
            <span class="legend-label">Defenders (Away)</span>
        </div>
        <div class="sb-legend-item">
            <div style="width:20px;height:3px;border-radius:2px;background:linear-gradient(90deg,#f59e0b,#f59e0b);flex-shrink:0;"></div>
            <span class="legend-label">Best Pass Option</span>
        </div>
    </div>
    """, unsafe_allow_html=True)

# ─────────────────────────────────────────
#  PROCESS CURRENT FRAME
# ─────────────────────────────────────────
tracking_frame = next((item for item in tracking_data if item["frame"] == frame), None)
intel_row = intel_data[intel_data['frame_id'] == frame].iloc[0] if not intel_data[intel_data['frame_id'] == frame].empty else None

if not tracking_frame or intel_row is None:
    st.error("Data missing for this frame.")
    st.stop()

# Helper to safely extract values from Intel row
def safe_get(row, key, default):
    if key not in row or pd.isna(row[key]):
        return default
    return row[key]

# ─────────────────────────────────────────
#  TOP NAV BAR
# ─────────────────────────────────────────
st.markdown("""
<div class="top-nav">
    <div class="nav-brand">
        <div class="nav-logo">⚽</div>
        <div>
            <div class="nav-brand-text"><span>SYNAPSE</span> AI</div>
            <div class="nav-sub">Post-Match Analytics</div>
        </div>
    </div>
    <div class="nav-tabs">
        <div class="nav-tab active">Overview</div>
        <div class="nav-tab">Heatmaps</div>
        <div class="nav-tab">Pass Network</div>
        <div class="nav-tab">Player Intel</div>
        <div class="nav-tab">Timeline</div>
    </div>
    <div class="nav-right">
        <span class="status-live" style="color: #a78bfa; border-color: rgba(167,139,250,0.18); background: rgba(167,139,250,0.07);">
            <span class="live-dot" style="background: #a78bfa;"></span>Replay
        </span>
    </div>
</div>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────
#  METRIC STRIP
# ─────────────────────────────────────────
possession = safe_get(intel_row, 'possession_probability', 0.0)
top_ev = safe_get(intel_row, 'best_pass_EV', 0.0)
c_xg = safe_get(intel_row, 'contextual_xG', 0.0)
momentum = safe_get(intel_row, 'momentum_state', 'Neutral')
best_pass_id = safe_get(intel_row, 'best_pass_option', None)

st.markdown(f"""
<div class="metric-strip">
    <div class="metric-cell mc1">
        <div class="metric-label">Active Frame</div>
        <div class="metric-value mv-green">{frame:,}</div>
        <div class="metric-sub">Match Replay Sequence</div>
    </div>
    <div class="metric-cell mc2">
        <div class="metric-label">Optimal Pass EV</div>
        <div class="metric-value mv-blue">{top_ev:.3f}</div>
        <div class="metric-sub">Target Player ID: {best_pass_id}</div>
    </div>
    <div class="metric-cell mc3">
        <div class="metric-label">Contextual xG</div>
        <div class="metric-value mv-amber">{c_xg:.3f}</div>
        <div class="metric-sub">Pressure adjusted probability</div>
    </div>
    <div class="metric-cell mc4">
        <div class="metric-label">Possession Control</div>
        <div class="metric-value mv-violet">{possession * 100:.1f}<span class="metric-unit">%</span></div>
        <div class="metric-sub">Momentum: {momentum}</div>
    </div>
</div>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────
#  MAIN LAYOUT
# ─────────────────────────────────────────
col_video, col_pitch, col_intel = st.columns([1.5, 1.5, 0.8])

# ──────────── VIDEO PANEL ────────────
with col_video:
    st.markdown(f"""
    <div class="card" style="margin-bottom:0;border-bottom:none;border-radius:14px 14px 0 0;">
        <div class="card-header">
            <div class="card-title">
                Match Footage · Frame {frame:,}
            </div>
            <div style="display:flex;gap:6px;align-items:center;">
                <span class="card-badge badge-blue">Camera 1</span>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    video_frame_img = get_video_frame(cap, frame)
    if video_frame_img is not None:
        # We wrap in a div with border acting as the card body
        st.markdown('<div style="border: 1px solid var(--border-subtle); border-top: none; border-radius: 0 0 14px 14px; overflow: hidden;">', unsafe_allow_html=True)
        st.image(video_frame_img, use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)
    else:
        st.warning("Video frame unavailable.")

# ──────────── PITCH PANEL ────────────
with col_pitch:
    st.markdown(f"""
    <div class="card" style="margin-bottom:0;border-bottom:none;border-radius:14px 14px 0 0;">
        <div class="card-header">
            <div class="card-title">
                Pitch Intelligence · Frame {frame:,}
            </div>
            <div style="display:flex;gap:6px;align-items:center;">
                <span class="card-badge badge-blue">Replay</span>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    fig = build_pitch_figure(frame, tracking_frame, intel_row)
    st.pyplot(fig, use_container_width=True)
    plt.close(fig)

# ──────────── INTEL PANEL ────────────
with col_intel:
    st.markdown("""
    <div style="margin-bottom:14px;"><div class="section-label">Pass Intelligence</div></div>
    """, unsafe_allow_html=True)

    if not pd.isna(best_pass_id) and best_pass_id != 'None':
        rank_label = "1ST"
        col = "#f59e0b"
        opt_ev = top_ev
        pass_risk = intel_row.get('risk_index', 0.0)
        
        st.markdown(f"""
        <div class="pass-card" style="border-left: 3px solid {col};">
            <div class="pass-rank-watermark">1</div>
            <div class="pass-card-head">
                <div>
                    <div class="pass-target-label">Rank {rank_label} · Option</div>
                    <div class="pass-target-name" style="color:{col};">Player ID {int(float(best_pass_id))}</div>
                </div>
                <div class="ev-badge">
                    <div class="ev-badge-label">Exp. Value</div>
                    <div class="ev-badge-value" style="color:{col};">{opt_ev:.3f}</div>
                </div>
            </div>
            <div>
                <div class="bar-item">
                    <div class="bar-item-head">
                        <span class="bar-item-label">Pass Risk Index</span>
                        <span class="bar-item-val">{pass_risk:.2f}</span>
                    </div>
                    <div class="bar-track">
                        <div class="bar-fill" style="width:{min(pass_risk*100,100):.1f}%;background:linear-gradient(90deg,#f43f5e,#9f1239);"></div>
                    </div>
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True)
        
        # Adding some engine telemetry stats below the pass options
        attackers = sum(1 for p in tracking_frame['players'] if p.get('is_my_team', False))
        defenders = sum(1 for p in tracking_frame['players'] if not p.get('is_my_team', False))
        st.markdown("""
        <div style="margin-top:18px;">
            <div class="section-label">Engine Telemetry</div>
        </div>
        """, unsafe_allow_html=True)

        st.markdown(f"""
        <div class="stats-grid">
            <div class="stat-cell">
                <div class="stat-cell-label">Attackers</div>
                <div class="stat-cell-val" style="color:#ef9a9a;">{attackers}</div>
            </div>
            <div class="stat-cell">
                <div class="stat-cell-label">Defenders</div>
                <div class="stat-cell-val" style="color:#64b5f6;">{defenders}</div>
            </div>
            <div class="stat-cell">
                <div class="stat-cell-label">Contextual xG</div>
                <div class="stat-cell-val" style="color:#f59e0b;">{c_xg:.3f}</div>
            </div>
            <div class="stat-cell">
                <div class="stat-cell-label">Best EV</div>
                <div class="stat-cell-val" style="color:#00e87a;">{top_ev:.4f}</div>
            </div>
        </div>
        """, unsafe_allow_html=True)
    else:
        st.info("No viable pass options detected for current frame.")

# ─────────────────────────────────────────
#  FOOTER
# ─────────────────────────────────────────
st.markdown(f"""
<div class="footer-bar">
    <div class="footer-left">
        <div style="width:26px;height:26px;border-radius:7px;
                    background:linear-gradient(135deg,#00e87a,#1e8fff);
                    display:flex;align-items:center;justify-content:center;
                    font-size:13px;box-shadow:0 0 14px rgba(0,232,122,0.3);">⚽</div>
        <span class="footer-copy">© 2026 SYNAPSE AI · Tactical Engine v5.0 · All rights reserved</span>
    </div>
</div>
<div style="height:16px;"></div>
""", unsafe_allow_html=True)
