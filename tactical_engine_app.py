import streamlit as st
import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.colors import LinearSegmentedColormap
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import MinMaxScaler
from statsbombpy import sb
import requests
import io
import warnings
from statsbombpy.api_client import NoAuthWarning
import time

warnings.simplefilter("ignore", NoAuthWarning)

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
.nav-right {
    display: flex;
    align-items: center;
    gap: 12px;
    padding: 0 24px;
    border-left: 1px solid var(--border-subtle);
    flex-shrink: 0;
}
.nav-score-pill {
    display: flex;
    align-items: center;
    gap: 8px;
    background: var(--bg-raised);
    border: 1px solid var(--border-subtle);
    border-radius: 10px;
    padding: 8px 16px;
}
.score-team {
    font-size: 10px;
    font-weight: 600;
    letter-spacing: 1.5px;
    color: var(--text-secondary);
    text-transform: uppercase;
    font-family: var(--font-body);
}
.score-num {
    font-family: var(--font-display);
    font-size: 26px;
    color: white;
    line-height: 1;
    letter-spacing: 0;
}
.score-sep { color: rgba(255,255,255,0.12); font-size: 18px; font-weight: 200; }

/* ── LIVE BADGE ── */
@keyframes pulse-ring {
    0%   { box-shadow: 0 0 0 0 rgba(0,232,122,0.7); }
    70%  { box-shadow: 0 0 0 7px rgba(0,232,122,0); }
    100% { box-shadow: 0 0 0 0 rgba(0,232,122,0); }
}
.live-dot {
    display: inline-block;
    width: 7px; height: 7px;
    background: var(--accent-green);
    border-radius: 50%;
    animation: pulse-ring 2s infinite;
    flex-shrink: 0;
    vertical-align: middle;
}
.status-live {
    display: inline-flex; align-items: center; gap: 7px;
    background: rgba(0,232,122,0.07);
    border: 1px solid rgba(0,232,122,0.18);
    border-radius: 20px; padding: 5px 13px;
    font-size: 10px; font-weight: 700;
    letter-spacing: 1.5px; text-transform: uppercase;
    color: var(--accent-green);
    font-family: var(--font-body);
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

/* Glow pulse on metric cells */
.mc1::before { content: ''; position: absolute; inset: 0; background: radial-gradient(ellipse at 0% 0%, rgba(0,232,122,0.06), transparent 70%); pointer-events: none; }
.mc2::before { content: ''; position: absolute; inset: 0; background: radial-gradient(ellipse at 0% 0%, rgba(30,143,255,0.06), transparent 70%); pointer-events: none; }
.mc3::before { content: ''; position: absolute; inset: 0; background: radial-gradient(ellipse at 0% 0%, rgba(245,158,11,0.06), transparent 70%); pointer-events: none; }
.mc4::before { content: ''; position: absolute; inset: 0; background: radial-gradient(ellipse at 0% 0%, rgba(167,139,250,0.06), transparent 70%); pointer-events: none; }

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
.card-badge {
    font-size: 9px;
    font-weight: 700;
    letter-spacing: 1.5px;
    text-transform: uppercase;
    padding: 4px 9px;
    border-radius: 5px;
}
.badge-green {
    background: rgba(0,232,122,0.09);
    border: 1px solid rgba(0,232,122,0.18);
    color: var(--accent-green);
}
.badge-blue {
    background: rgba(30,143,255,0.09);
    border: 1px solid rgba(30,143,255,0.18);
    color: var(--accent-blue);
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
.pass-card:hover {
    transform: translateX(4px);
}
.pass-rank-watermark {
    position: absolute;
    right: -6px; top: -14px;
    font-family: var(--font-display);
    font-size: 96px;
    line-height: 1;
    opacity: 0.035;
    color: white;
    pointer-events: none;
    letter-spacing: -4px;
}
.pass-card-head {
    display: flex;
    align-items: flex-start;
    justify-content: space-between;
    margin-bottom: 14px;
}
.pass-target-label {
    font-size: 8px;
    font-weight: 700;
    letter-spacing: 2.5px;
    text-transform: uppercase;
    color: var(--text-dim);
    margin-bottom: 4px;
    font-family: var(--font-body);
}
.pass-target-name {
    font-family: var(--font-display);
    font-size: 20px;
    letter-spacing: 3px;
    line-height: 1;
}
.pass-coord {
    font-family: var(--font-mono);
    font-size: 9px;
    color: var(--text-dim);
    margin-top: 5px;
}
.ev-badge {
    text-align: right;
    background: var(--bg-surface);
    border-radius: 8px;
    padding: 8px 12px;
    border: 1px solid var(--border-subtle);
}
.ev-badge-label {
    font-size: 8px;
    font-weight: 700;
    letter-spacing: 2px;
    text-transform: uppercase;
    color: var(--text-dim);
    margin-bottom: 4px;
    font-family: var(--font-body);
}
.ev-badge-value {
    font-family: var(--font-mono);
    font-size: 20px;
    font-weight: 700;
    line-height: 1;
}
.bar-item {
    margin-bottom: 10px;
}
.bar-item:last-child { margin-bottom: 0; }
.bar-item-head {
    display: flex;
    justify-content: space-between;
    align-items: center;
    margin-bottom: 5px;
}
.bar-item-label {
    font-size: 9px;
    font-weight: 600;
    letter-spacing: 1.5px;
    text-transform: uppercase;
    color: var(--text-dim);
    font-family: var(--font-body);
}
.bar-item-val {
    font-family: var(--font-mono);
    font-size: 11px;
    font-weight: 700;
    color: rgba(255,255,255,0.75);
}
.bar-track {
    height: 4px;
    background: rgba(255,255,255,0.05);
    border-radius: 4px;
    overflow: hidden;
}
.bar-fill {
    height: 100%;
    border-radius: 4px;
    transition: width 0.6s ease;
}

/* ══════════════════════════════
   STATS GRID
   ══════════════════════════════ */
.stats-grid {
    display: grid;
    grid-template-columns: 1fr 1fr;
    gap: 8px;
    margin-top: 12px;
}
.stat-cell {
    background: var(--bg-surface);
    border: 1px solid var(--border-subtle);
    border-radius: 10px;
    padding: 14px 16px;
    transition: border-color 0.2s;
}
.stat-cell:hover { border-color: rgba(255,255,255,0.1); }
.stat-cell-label {
    font-size: 9px;
    font-weight: 600;
    letter-spacing: 2px;
    text-transform: uppercase;
    color: var(--text-dim);
    margin-bottom: 6px;
    font-family: var(--font-body);
}
.stat-cell-val {
    font-family: var(--font-mono);
    font-size: 22px;
    font-weight: 700;
    line-height: 1;
}

/* ══════════════════════════════
   CONTROL BAR
   ══════════════════════════════ */
.control-bar {
    background: rgba(0,0,0,0.35);
    border-top: 1px solid var(--border-subtle);
    padding: 11px 20px;
    display: flex;
    align-items: center;
    gap: 14px;
    font-size: 10px;
}
.ctrl-label {
    color: var(--text-dim);
    font-weight: 600;
    letter-spacing: 2px;
    text-transform: uppercase;
    white-space: nowrap;
    font-family: var(--font-body);
    font-size: 9px;
}
.ctrl-gradient {
    flex: 1;
    height: 4px;
    border-radius: 4px;
    background: linear-gradient(90deg, #c0392b 0%, #1a1a2e 50%, #1e5fa8 100%);
    box-shadow: 0 0 8px rgba(0,0,0,0.5) inset;
    position: relative;
}
.ctrl-gradient::after {
    content: '';
    position: absolute;
    top: 0; left: 0; right: 0; bottom: 0;
    border-radius: 4px;
    background: linear-gradient(180deg, rgba(255,255,255,0.1), transparent);
}

/* ══════════════════════════════
   SIDEBAR COMPONENTS
   ══════════════════════════════ */
.sb-header {
    padding: 22px 0 18px;
    border-bottom: 1px solid rgba(255,255,255,0.05);
    margin-bottom: 22px;
}
.sb-logo-row {
    display: flex;
    align-items: center;
    gap: 12px;
    margin-bottom: 8px;
}
.sb-logo {
    width: 40px; height: 40px;
    background: linear-gradient(135deg, var(--accent-green) 0%, var(--accent-blue) 100%);
    border-radius: 11px;
    display: flex; align-items: center; justify-content: center;
    font-size: 20px;
    box-shadow: 0 0 28px rgba(0,232,122,0.28);
    flex-shrink: 0;
}
.sb-brand {
    font-family: var(--font-display);
    font-size: 22px;
    letter-spacing: 3px;
    color: white;
    line-height: 1;
}
.sb-brand span {
    background: linear-gradient(135deg, var(--accent-green), #00ccff);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
}
.sb-subtitle {
    font-size: 9px;
    font-weight: 500;
    letter-spacing: 3px;
    text-transform: uppercase;
    color: var(--text-dim);
    margin-top: 2px;
    font-family: var(--font-body);
}
.sb-section-title {
    font-family: var(--font-body);
    font-size: 9px;
    font-weight: 700;
    letter-spacing: 3px;
    text-transform: uppercase;
    color: var(--text-dim);
    margin-bottom: 10px;
    display: flex;
    align-items: center;
    gap: 8px;
}
.sb-section-title::after {
    content: '';
    flex: 1;
    height: 1px;
    background: rgba(255,255,255,0.05);
}

/* Status rows */
.sb-status-list { display: flex; flex-direction: column; gap: 0; }
.sb-status-row {
    display: flex;
    justify-content: space-between;
    align-items: center;
    padding: 10px 0;
    border-bottom: 1px solid rgba(255,255,255,0.035);
}
.sb-status-row:last-child { border-bottom: none; }
.sb-status-name {
    font-size: 11px;
    font-weight: 500;
    color: var(--text-secondary);
    display: flex; align-items: center; gap: 8px;
    font-family: var(--font-body);
}
.sb-status-val {
    font-family: var(--font-mono);
    font-size: 9px;
    font-weight: 700;
    letter-spacing: 1px;
    padding: 2px 7px;
    border-radius: 4px;
}
.val-online {
    color: var(--accent-green);
    background: rgba(0,232,122,0.08);
    border: 1px solid rgba(0,232,122,0.15);
}
.val-running {
    color: var(--accent-amber);
    background: rgba(245,158,11,0.08);
    border: 1px solid rgba(245,158,11,0.15);
}

/* Legend items */
.sb-legend-item {
    display: flex; align-items: center; gap: 10px;
    padding: 8px 0;
    border-bottom: 1px solid rgba(255,255,255,0.035);
}
.sb-legend-item:last-child { border-bottom: none; }
.legend-dot {
    width: 10px; height: 10px;
    border-radius: 50%;
    flex-shrink: 0;
}
.legend-label {
    font-size: 11px;
    font-weight: 400;
    color: var(--text-secondary);
    font-family: var(--font-body);
}

/* Powered by */
.sb-powered {
    text-align: center;
    padding: 14px 0 4px;
    margin-top: 6px;
    border-top: 1px solid rgba(255,255,255,0.04);
}
.sb-powered-label {
    font-size: 8px;
    font-weight: 600;
    letter-spacing: 3px;
    text-transform: uppercase;
    color: var(--text-dim);
    margin-bottom: 8px;
    font-family: var(--font-body);
}
.sb-powered-logos {
    display: flex;
    justify-content: center;
    gap: 6px;
    flex-wrap: wrap;
}
.sb-powered-logo {
    font-size: 9px;
    font-weight: 600;
    color: rgba(255,255,255,0.18);
    letter-spacing: 0.5px;
    padding: 3px 8px;
    border: 1px solid rgba(255,255,255,0.06);
    border-radius: 4px;
    font-family: var(--font-body);
}

/* ══════════════════════════════
   SECTION LABELS
   ══════════════════════════════ */
.section-label {
    font-family: var(--font-body);
    font-size: 9px;
    font-weight: 700;
    letter-spacing: 3px;
    text-transform: uppercase;
    color: var(--text-dim);
    margin-bottom: 12px;
    display: flex;
    align-items: center;
    gap: 8px;
}
.section-label::after {
    content: '';
    flex: 1;
    height: 1px;
    background: var(--border-subtle);
}

/* ══════════════════════════════
   EXPANDER
   ══════════════════════════════ */
.streamlit-expanderHeader {
    font-family: var(--font-body) !important;
    font-weight: 600 !important;
    font-size: 10px !important;
    letter-spacing: 2.5px !important;
    color: var(--text-secondary) !important;
    text-transform: uppercase !important;
    background: var(--bg-surface) !important;
    border: 1px solid var(--border-subtle) !important;
    border-radius: 8px !important;
    padding: 12px 16px !important;
}
.streamlit-expanderContent {
    background: var(--bg-mid) !important;
    border: 1px solid var(--border-subtle) !important;
    border-top: none !important;
    border-radius: 0 0 8px 8px !important;
}

/* ══════════════════════════════
   DATAFRAME
   ══════════════════════════════ */
.stDataFrame { border-radius: 10px; overflow: hidden; }

/* ══════════════════════════════
   SPINNER
   ══════════════════════════════ */
.stSpinner > div {
    border-color: var(--accent-green) transparent transparent !important;
}

/* ══════════════════════════════
   FOOTER
   ══════════════════════════════ */
.footer-bar {
    border-top: 1px solid var(--border-subtle);
    padding: 18px 0 0;
    display: flex;
    justify-content: space-between;
    align-items: center;
    flex-wrap: wrap;
    gap: 12px;
    margin-top: 8px;
}
.footer-left { display: flex; align-items: center; gap: 12px; }
.footer-copy {
    font-size: 10px;
    color: var(--text-dim);
    font-weight: 400;
    letter-spacing: 0.5px;
    font-family: var(--font-body);
}
.footer-right { display: flex; align-items: center; gap: 20px; }
.footer-link {
    font-size: 9px;
    font-weight: 600;
    letter-spacing: 1.5px;
    text-transform: uppercase;
    color: var(--text-dim);
    display: flex; align-items: center; gap: 6px;
    font-family: var(--font-body);
}

/* ══════════════════════════════
   SCAN LINE ANIMATION
   ══════════════════════════════ */
@keyframes scanDown {
    0%   { top: -2px; opacity: 0; }
    5%   { opacity: 1; }
    95%  { opacity: 1; }
    100% { top: 100vh; opacity: 0; }
}
.scan-line {
    position: fixed;
    left: 0; right: 0;
    height: 1px;
    background: linear-gradient(90deg, transparent 0%, rgba(0,232,122,0.15) 30%, rgba(0,232,122,0.08) 70%, transparent 100%);
    pointer-events: none;
    z-index: 9999;
    animation: scanDown 12s linear infinite;
}

/* ══════════════════════════════
   PITCH SECTION DIVIDER
   ══════════════════════════════ */
.pitch-section-gap { height: 0; margin: 0; }
</style>

<!-- Atmospheric scan line -->
<div class="scan-line"></div>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────
#  CONSTANTS
# ─────────────────────────────────────────
PITCH_LENGTH = 104.0
PITCH_WIDTH  = 68.0
GOAL_CENTER  = np.array([104.0, 34.0])

PASS_COLORS  = ["#00e87a", "#f59e0b", "#1e8fff"]
TARGET_NAMES = ["ALPHA", "BETA", "GAMMA"]

# ─────────────────────────────────────────
#  DATA & ML FUNCTIONS
# ─────────────────────────────────────────
@st.cache_resource(show_spinner=False)
def load_and_train_xg_model():
    model  = LogisticRegression(max_iter=500)
    scaler = MinMaxScaler()
    events = sb.events(match_id=3869685)
    shots  = events[events["type"] == "Shot"].copy()
    shots["x"]       = shots["location"].apply(lambda l: l[0] * (104/120) if isinstance(l, list) else np.nan)
    shots["y"]       = shots["location"].apply(lambda l: l[1] * (68/80)  if isinstance(l, list) else np.nan)
    shots["is_goal"] = (shots["shot_outcome"] == "Goal").astype(int)
    shots["dx"]      = GOAL_CENTER[0] - shots["x"]
    shots["dy"]      = GOAL_CENTER[1] - shots["y"]
    shots["distance"]= np.sqrt(shots["dx"]**2 + shots["dy"]**2)
    shots["angle"]   = np.abs(np.arctan2(shots["dy"], shots["dx"]))
    shots = shots.dropna(subset=["distance", "angle", "is_goal"])
    X = scaler.fit_transform(shots[["distance", "angle"]])
    model.fit(X, shots["is_goal"])
    return model, scaler

def calculate_xg(x, y, model, scaler):
    dx   = GOAL_CENTER[0] - x
    dy   = GOAL_CENTER[1] - y
    dist = np.sqrt(dx**2 + dy**2)
    ang  = np.abs(np.arctan2(dy, dx))
    feat = scaler.transform(pd.DataFrame({"distance": [dist], "angle": [ang]}))
    return model.predict_proba(feat)[0][1]

@st.cache_data(show_spinner=False)
def fetch_metrica_frame(frame_idx):
    BASE = "https://raw.githubusercontent.com/metrica-sports/sample-data/master/data/Sample_Game_1/"
    home_url = BASE + "Sample_Game_1_RawTrackingData_Home_Team.csv"
    away_url = BASE + "Sample_Game_1_RawTrackingData_Away_Team.csv"
    skip = max(2, frame_idx - 10)
    home_df = pd.read_csv(io.StringIO(requests.get(home_url).text), skiprows=skip, nrows=15, header=None)
    away_df = pd.read_csv(io.StringIO(requests.get(away_url).text), skiprows=skip, nrows=15, header=None)
    dt = 0.2
    cur_h, prev_h = home_df.iloc[-1], home_df.iloc[-6]
    cur_a, prev_a = away_df.iloc[-1], away_df.iloc[-6]
    attackers, defenders, ball_carrier = [], [], None
    for i in range(3, 31, 2):
        if i + 1 >= len(cur_h): break
        if any(pd.isna(v) for v in [cur_h[i], cur_h[i+1], prev_h[i], prev_h[i+1]]): continue
        x  = cur_h[i]   * PITCH_LENGTH
        y  = cur_h[i+1] * PITCH_WIDTH
        vx = ((cur_h[i]   - prev_h[i])   * PITCH_LENGTH) / dt
        vy = ((cur_h[i+1] - prev_h[i+1]) * PITCH_WIDTH)  / dt
        p  = {"pos": (x, y), "vel": (vx, vy), "id": i}
        attackers.append(p)
        if ball_carrier is None or abs(x - 52) < abs(ball_carrier["pos"][0] - 52):
            ball_carrier = p
    for i in range(3, 31, 2):
        if i + 1 >= len(cur_a): break
        if any(pd.isna(v) for v in [cur_a[i], cur_a[i+1], prev_a[i], prev_a[i+1]]): continue
        x  = cur_a[i]   * PITCH_LENGTH
        y  = cur_a[i+1] * PITCH_WIDTH
        vx = ((cur_a[i]   - prev_a[i])   * PITCH_LENGTH) / dt
        vy = ((cur_a[i+1] - prev_a[i+1]) * PITCH_WIDTH)  / dt
        defenders.append({"pos": (x, y), "vel": (vx, vy)})
    return ball_carrier, attackers, defenders

def generate_pitch_control(attackers, defenders):
    xs = np.linspace(0, PITCH_LENGTH - 1, int(PITCH_LENGTH))
    ys = np.linspace(0, PITCH_WIDTH  - 1, int(PITCH_WIDTH))
    xx, yy = np.meshgrid(xs, ys)
    att_inf, def_inf = np.zeros_like(xx), np.zeros_like(xx)
    for team, mat in [(attackers, att_inf), (defenders, def_inf)]:
        for p in team:
            px, py = p["pos"]
            vx, vy = p["vel"]
            fx, fy = px + vx * 0.5, py + vy * 0.5
            r = max(4.0, 10.0 - np.hypot(vx, vy) * 0.5)
            mat += np.exp(-((xx - fx)**2 + (yy - fy)**2) / (2 * r**2))
    return 1 / (1 + np.exp(-(att_inf - def_inf)))

def evaluate_passes(bc, attackers, pc, model, scaler):
    opts = []
    for t in attackers:
        if t["pos"] == bc["pos"]: continue
        xg       = calculate_xg(t["pos"][0], t["pos"][1], model, scaler)
        sx, sy   = bc["pos"]
        tx, ty   = t["pos"]
        dist     = np.hypot(tx - sx, ty - sy)
        pts      = max(2, int(dist))
        xv = np.clip(np.linspace(sx, tx, pts).astype(int), 0, int(PITCH_LENGTH) - 1)
        yv = np.clip(np.linspace(sy, ty, pts).astype(int), 0, int(PITCH_WIDTH)  - 1)
        pp = np.mean(pc[yv, xv]) * max(0.01, 1.0 - dist * 0.012)
        opts.append({"pos": t["pos"], "xg": xg, "pass_prob": pp, "ev": pp * xg})
    opts.sort(key=lambda x: x["ev"], reverse=True)
    return opts

# ─────────────────────────────────────────
#  PITCH DRAWING
# ─────────────────────────────────────────
def draw_pitch_lines(ax):
    lc, la, lw = "white", 0.65, 1.4

    # Mow stripes
    for i in range(0, int(PITCH_LENGTH), 10):
        alpha = 0.022 if (i // 10) % 2 == 0 else 0.0
        ax.add_patch(patches.Rectangle((i, 0), 10, PITCH_WIDTH,
                                       facecolor="white", alpha=alpha, zorder=1))

    # Boundary
    ax.plot([0, PITCH_LENGTH, PITCH_LENGTH, 0, 0],
            [0, 0, PITCH_WIDTH, PITCH_WIDTH, 0],
            color=lc, lw=lw, alpha=la, zorder=2)
    # Halfway
    ax.plot([PITCH_LENGTH/2]*2, [0, PITCH_WIDTH], color=lc, lw=lw*0.8, alpha=la*0.6, zorder=2)
    # Penalty boxes
    for side_x, mirror in [(0, 1), (PITCH_LENGTH, -1)]:
        ax.plot([side_x, side_x+mirror*16.5, side_x+mirror*16.5, side_x],
                [13.85, 13.85, 54.15, 54.15], color=lc, lw=lw*0.8, alpha=la*0.6, zorder=2)
    # 6-yard boxes
    for side_x, mirror in [(0, 1), (PITCH_LENGTH, -1)]:
        ax.plot([side_x, side_x+mirror*5.5, side_x+mirror*5.5, side_x],
                [24.84, 24.84, 43.16, 43.16], color=lc, lw=lw*0.6, alpha=la*0.4, zorder=2)
    # Centre circle
    ax.add_patch(patches.Circle((PITCH_LENGTH/2, PITCH_WIDTH/2), 9.15,
                                 edgecolor=lc, facecolor="none", lw=lw*0.8, alpha=la*0.6, zorder=2))
    ax.scatter(PITCH_LENGTH/2, PITCH_WIDTH/2, c=lc, s=30, zorder=2, alpha=la*0.7)
    # Penalty spots
    for gx in [10.97, PITCH_LENGTH - 10.97]:
        ax.scatter(gx, PITCH_WIDTH/2, c=lc, s=20, zorder=2, alpha=la*0.5)
    # Penalty arcs
    for cx in [PITCH_LENGTH*0.16, PITCH_LENGTH*0.84]:
        arc = patches.Arc((cx, PITCH_WIDTH/2), 18.3, 18.3, angle=0,
                           theta1=270 if cx < PITCH_LENGTH/2 else 90,
                           theta2=270+180 if cx < PITCH_LENGTH/2 else 90+180,
                           color=lc, lw=lw*0.6, alpha=la*0.4, zorder=2)
        ax.add_patch(arc)
    # Goals
    for gx, gdir in [(0, -2.6), (PITCH_LENGTH, 2.6)]:
        ax.plot([gx, gx+gdir, gx+gdir, gx],
                [30.34, 30.34, 37.66, 37.66], color=lc, lw=lw*1.2, alpha=la*0.9, zorder=2)

def build_pitch_figure(bc, atts, defs, pc_matrix, ranked_passes):
    fig, ax = plt.subplots(figsize=(12, 7.8))
    fig.patch.set_facecolor("#030e1a")
    ax.set_facecolor("#030e1a")

    # Pitch base with subtle gradient
    ax.imshow(
        np.tile(np.linspace(0, 1, 100).reshape(1, -1), (100, 1)),
        extent=[-2, PITCH_LENGTH+2, -2, PITCH_WIDTH+2],
        cmap=LinearSegmentedColormap.from_list("g", ["#031c0a", "#041f0c"]),
        aspect="auto", zorder=0
    )

    # Pitch control heatmap — refined palette
    cmap_pc = LinearSegmentedColormap.from_list(
        "pc", ["#8b1a1a", "#2d0000", "#0a0a12", "#001428", "#003070"], N=512
    )
    ax.imshow(pc_matrix,
              extent=[0, PITCH_LENGTH, PITCH_WIDTH, 0],
              cmap=cmap_pc, alpha=0.5, vmin=0, vmax=1, zorder=1, aspect="auto")

    ax.set_aspect("equal")
    ax.set_xlim(-2, PITCH_LENGTH + 2)
    ax.set_ylim(-2, PITCH_WIDTH  + 2)

    draw_pitch_lines(ax)

    def draw_player(px, py, vx, vy, fill_color, edge_color, size=190):
        ax.scatter(px+0.6, py+0.6, c="#000000", s=size*0.8, alpha=0.25, zorder=3)
        ax.scatter(px, py, c=fill_color, s=size*2.0, alpha=0.08, zorder=3)
        ax.scatter(px, py, c=fill_color, s=size, edgecolors=edge_color,
                   linewidths=1.8, zorder=4)
        speed = np.hypot(vx, vy)
        if speed > 0.5:
            scale = min(1.5, 0.6 + speed * 0.06)
            ax.annotate("",
                xy=(px + vx*scale*0.12, py + vy*scale*0.12),
                xytext=(px, py),
                arrowprops=dict(arrowstyle="->", color=fill_color, lw=1.2, mutation_scale=9),
                zorder=5)

    # Defenders
    for p in defs:
        draw_player(*p["pos"], *p["vel"], "#1565c0", "#64b5f6", 175)

    # Attackers
    for p in atts:
        draw_player(*p["pos"], *p["vel"], "#c62828", "#ef9a9a", 175)

    # Ball carrier — neon green
    bx, by = bc["pos"]
    for rs, ra in [(1100, 0.05), (650, 0.09), (380, 0.15)]:
        ax.scatter(bx, by, c="#00e87a", s=rs, alpha=ra, zorder=4)
    ax.scatter(bx+0.6, by+0.6, c="black", s=420, alpha=0.25, zorder=4)
    ax.scatter(bx, by, c="#00e87a", s=380, edgecolors="white", linewidths=2.8, zorder=6)
    ax.scatter(bx, by, c="white", s=60, zorder=7, alpha=0.95)

    # Pass arrows
    for i, opt in enumerate(ranked_passes[:3]):
        if opt["ev"] < 0.003:
            continue
        tx, ty = opt["pos"]
        col    = PASS_COLORS[i]

        # Glow layers
        for gw, ga in [(18, 0.04), (11, 0.07), (5, 0.12)]:
            ax.annotate("", xy=(tx, ty), xytext=(bx, by),
                        arrowprops=dict(facecolor=col, edgecolor=col,
                                        alpha=ga, width=gw, headwidth=gw*2.4,
                                        shrink=0.06), zorder=4)
        # Shadow
        ax.annotate("", xy=(tx+0.6, ty+0.6), xytext=(bx+0.6, by+0.6),
                    arrowprops=dict(facecolor="black", edgecolor="black",
                                    alpha=0.12, width=3, headwidth=10,
                                    shrink=0.06), zorder=4)
        # Main arrow
        ax.annotate("", xy=(tx, ty), xytext=(bx, by),
                    arrowprops=dict(facecolor=col, edgecolor=col,
                                    width=2.8, headwidth=12, headlength=9,
                                    shrink=0.06, linewidth=0), zorder=6)

        # Target rings
        for r_size, r_alpha in [(5.0, 0.25), (3.2, 0.7)]:
            ax.add_patch(patches.Circle((tx, ty), r_size,
                                         edgecolor=col, facecolor="none",
                                         lw=1.8, alpha=r_alpha, zorder=6))
        # Halo
        ax.scatter(tx, ty, c=col, s=280, alpha=0.1, zorder=5)

        # Greek label
        lbl = ["α", "β", "γ"][i]
        ax.text(tx + 4.0, ty - 1.5, lbl,
                color=col, fontsize=11, fontweight="black",
                fontfamily="DejaVu Serif", zorder=7, alpha=0.92)

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
                <div class="sb-subtitle">Tactical Engine v5</div>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("""
    <div style="margin-bottom:22px;">
        <span class="status-live"><span class="live-dot"></span>Live Analysis Active</span>
    </div>
    """, unsafe_allow_html=True)

    st.markdown('<div class="sb-section-title">Match Timeline</div>', unsafe_allow_html=True)
    frame = st.slider("", min_value=50000, max_value=55000, value=50000, step=25,
                      label_visibility="collapsed")
    st.markdown("<div style='height:18px'></div>", unsafe_allow_html=True)

    st.markdown('<div class="sb-section-title">System Status</div>', unsafe_allow_html=True)
    st.markdown("""
    <div class="sb-status-list">
        <div class="sb-status-row">
            <span class="sb-status-name"><span class="live-dot"></span>StatsBomb xG Core</span>
            <span class="sb-status-val val-online">ONLINE</span>
        </div>
        <div class="sb-status-row">
            <span class="sb-status-name"><span class="live-dot"></span>Kinematic Arrays</span>
            <span class="sb-status-val val-online">ONLINE</span>
        </div>
        <div class="sb-status-row">
            <span class="sb-status-name"><span class="live-dot"></span>EV Optimizer</span>
            <span class="sb-status-val val-running">RUNNING</span>
        </div>
        <div class="sb-status-row">
            <span class="sb-status-name"><span class="live-dot"></span>Pitch Control ML</span>
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
            <div style="width:20px;height:3px;border-radius:2px;background:linear-gradient(90deg,#00e87a,#1e8fff);flex-shrink:0;"></div>
            <span class="legend-label">Pass Options α β γ</span>
        </div>
        <div class="sb-legend-item">
            <div style="width:20px;height:5px;border-radius:2px;background:linear-gradient(90deg,#8b1a1a,#0a0a12,#003070);flex-shrink:0;"></div>
            <span class="legend-label">Pitch Control Field</span>
        </div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("<div style='height:22px'></div>", unsafe_allow_html=True)
    st.markdown("""
    <div class="sb-powered">
        <div class="sb-powered-label">Powered by</div>
        <div class="sb-powered-logos">
            <span class="sb-powered-logo">StatsBomb</span>
            <span class="sb-powered-logo">Metrica</span>
            <span class="sb-powered-logo">DTU</span>
        </div>
    </div>
    """, unsafe_allow_html=True)

# ─────────────────────────────────────────
#  DATA PROCESSING
# ─────────────────────────────────────────
t0 = time.time()
with st.spinner("Fusing sensor arrays…"):
    xg_model, xg_scaler = load_and_train_xg_model()
    bc, atts, defs      = fetch_metrica_frame(frame)
    pc_matrix           = generate_pitch_control(atts, defs)
    ranked_passes       = evaluate_passes(bc, atts, pc_matrix, xg_model, xg_scaler)
latency_ms = (time.time() - t0) * 1000

if not ranked_passes:
    st.error("No pass options computed — try a different frame.")
    st.stop()

# ─────────────────────────────────────────
#  TOP NAV BAR
# ─────────────────────────────────────────
st.markdown(f"""
<div class="top-nav">
    <div class="nav-brand">
        <div class="nav-logo">⚽</div>
        <div>
            <div class="nav-brand-text"><span>SYNAPSE</span> AI</div>
            <div class="nav-sub">Tactical Engine v5</div>
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
        <span class="status-live"><span class="live-dot"></span>Live</span>
        <div class="nav-score-pill">
            <span class="score-team">Home</span>
            <span class="score-num">2</span>
            <span class="score-sep">–</span>
            <span class="score-num">1</span>
            <span class="score-team">Away</span>
        </div>
    </div>
</div>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────
#  TICKER BAR
# ─────────────────────────────────────────
st.markdown(f"""
<div class="ticker">
    <div class="ticker-item">
        <span class="ticker-label">Frame</span>
        <span class="ticker-val" style="color:#00e87a;">{frame:,}</span>
    </div>
    <div class="ticker-item">
        <span class="ticker-label">Best EV</span>
        <span class="ticker-val" style="color:#1e8fff;">{ranked_passes[0]['ev']:.4f}</span>
    </div>
    <div class="ticker-item">
        <span class="ticker-label">Top Exec.</span>
        <span class="ticker-val" style="color:#f59e0b;">{ranked_passes[0]['pass_prob']*100:.1f}%</span>
    </div>
    <div class="ticker-item">
        <span class="ticker-label">Attackers</span>
        <span class="ticker-val" style="color:#ef9a9a;">{len(atts)}</span>
    </div>
    <div class="ticker-item">
        <span class="ticker-label">Defenders</span>
        <span class="ticker-val" style="color:#64b5f6;">{len(defs)}</span>
    </div>
    <div class="ticker-item">
        <span class="ticker-label">Latency</span>
        <span class="ticker-val" style="color:#a78bfa;">{latency_ms:.0f}ms</span>
    </div>
    <div class="ticker-item">
        <span class="ticker-label">Engine</span>
        <span class="ticker-val" style="color:#00e87a;">NOMINAL</span>
    </div>
</div>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────
#  METRIC STRIP
# ─────────────────────────────────────────
st.markdown(f"""
<div class="metric-strip">
    <div class="metric-cell mc1">
        <div class="metric-label">Active Frame</div>
        <div class="metric-value mv-green">{frame:,}</div>
        <div class="metric-sub">Sample Game 1 · H1</div>
    </div>
    <div class="metric-cell mc2">
        <div class="metric-label">Optimal EV Score</div>
        <div class="metric-value mv-blue">{ranked_passes[0]['ev']:.3f}</div>
        <div class="metric-sub">Target Alpha · Ranked #1</div>
    </div>
    <div class="metric-cell mc3">
        <div class="metric-label">Execution Prob.</div>
        <div class="metric-value mv-amber">{ranked_passes[0]['pass_prob']*100:.1f}<span class="metric-unit">%</span></div>
        <div class="metric-sub">Pitch-adjusted probability</div>
    </div>
    <div class="metric-cell mc4">
        <div class="metric-label">Compute Latency</div>
        <div class="metric-value mv-violet">{latency_ms:.0f}<span class="metric-unit">ms</span></div>
        <div class="metric-sub">Full pipeline round-trip</div>
    </div>
</div>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────
#  MAIN LAYOUT
# ─────────────────────────────────────────
col_pitch, col_intel = st.columns([2.6, 1.0])

# ──────────── PITCH PANEL ────────────
with col_pitch:
    st.markdown(f"""
    <div class="card" style="margin-bottom:0;border-bottom:none;border-radius:14px 14px 0 0;">
        <div class="card-header">
            <div class="card-title">
                <span class="live-dot"></span>
                Pitch Intelligence · Frame {frame:,}
            </div>
            <div style="display:flex;gap:6px;align-items:center;">
                <span class="card-badge badge-green">Live</span>
                <span class="card-badge badge-blue">Pitch Control</span>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    fig = build_pitch_figure(bc, atts, defs, pc_matrix, ranked_passes)
    st.pyplot(fig, use_container_width=True)
    plt.close(fig)

    # Control legend
    st.markdown("""
    <div class="control-bar" style="border-radius:0 0 14px 14px;">
        <span class="ctrl-label">← Defensive</span>
        <div class="ctrl-gradient"></div>
        <span class="ctrl-label">Offensive →</span>
    </div>
    """, unsafe_allow_html=True)

# ──────────── INTEL PANEL ────────────
with col_intel:
    st.markdown("""
    <div style="margin-bottom:14px;">
        <div class="section-label">Pass Intelligence</div>
    </div>
    """, unsafe_allow_html=True)

    for i in range(min(3, len(ranked_passes))):
        opt  = ranked_passes[i]
        col  = PASS_COLORS[i]
        name = TARGET_NAMES[i]
        pp   = opt["pass_prob"] * 100
        xg   = opt["xg"]
        coord = f"({opt['pos'][0]:.1f}, {opt['pos'][1]:.1f})"

        rank_label = ["1ST", "2ND", "3RD"][i]

        st.markdown(f"""
        <div class="pass-card" style="
            border-left: 3px solid {col};
            box-shadow: 0 4px 24px rgba(0,0,0,0.3), inset 0 0 40px {col}08;">
            <div class="pass-rank-watermark">{i+1}</div>
            <div class="pass-card-head">
                <div>
                    <div class="pass-target-label">Rank {rank_label} · Option</div>
                    <div class="pass-target-name" style="color:{col};">Target {name}</div>
                    <div class="pass-coord">⊕ {coord}</div>
                </div>
                <div class="ev-badge">
                    <div class="ev-badge-label">Exp. Value</div>
                    <div class="ev-badge-value" style="color:{col};">{opt['ev']:.3f}</div>
                </div>
            </div>
            <div>
                <div class="bar-item">
                    <div class="bar-item-head">
                        <span class="bar-item-label">Execution Prob.</span>
                        <span class="bar-item-val">{pp:.1f}%</span>
                    </div>
                    <div class="bar-track">
                        <div class="bar-fill" style="width:{min(pp,100):.1f}%;background:linear-gradient(90deg,{col},{col}aa);box-shadow:0 0 8px {col}66;"></div>
                    </div>
                </div>
                <div class="bar-item">
                    <div class="bar-item-head">
                        <span class="bar-item-label">Goal Threat xG</span>
                        <span class="bar-item-val">{xg:.3f}</span>
                    </div>
                    <div class="bar-track">
                        <div class="bar-fill" style="width:{min(xg*100,100):.1f}%;background:linear-gradient(90deg,#1e8fff,#1e8fffaa);box-shadow:0 0 8px rgba(30,143,255,0.4);"></div>
                    </div>
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True)

    # Engine telemetry
    st.markdown("""
    <div style="margin-top:18px;">
        <div class="section-label">Engine Telemetry</div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown(f"""
    <div class="stats-grid">
        <div class="stat-cell">
            <div class="stat-cell-label">Attackers</div>
            <div class="stat-cell-val" style="color:#ef9a9a;">{len(atts)}</div>
        </div>
        <div class="stat-cell">
            <div class="stat-cell-label">Defenders</div>
            <div class="stat-cell-val" style="color:#64b5f6;">{len(defs)}</div>
        </div>
        <div class="stat-cell">
            <div class="stat-cell-label">Pass Options</div>
            <div class="stat-cell-val" style="color:#f59e0b;">{len(ranked_passes)}</div>
        </div>
        <div class="stat-cell">
            <div class="stat-cell-label">Best EV</div>
            <div class="stat-cell-val" style="color:#00e87a;">{ranked_passes[0]['ev']:.4f}</div>
        </div>
    </div>
    """, unsafe_allow_html=True)

# ─────────────────────────────────────────
#  RAW TELEMETRY EXPANDER
# ─────────────────────────────────────────
st.markdown("<div style='height:18px'></div>", unsafe_allow_html=True)
with st.expander("RAW ENGINE TELEMETRY"):
    c_snap, c_data = st.columns([0.22, 0.78])
    with c_snap:
        st.markdown(f"""
        <div style="padding:10px 0;display:flex;flex-direction:column;gap:18px;">
            <div>
                <div style="font-size:9px;font-weight:700;letter-spacing:2.5px;color:rgba(255,255,255,0.2);
                            text-transform:uppercase;margin-bottom:6px;font-family:'DM Sans',sans-serif;">Frame</div>
                <div style="font-family:'JetBrains Mono',monospace;font-size:26px;font-weight:700;
                            color:#00e87a;line-height:1;">{frame:,}</div>
            </div>
            <div>
                <div style="font-size:9px;font-weight:700;letter-spacing:2.5px;color:rgba(255,255,255,0.2);
                            text-transform:uppercase;margin-bottom:6px;font-family:'DM Sans',sans-serif;">Options</div>
                <div style="font-family:'JetBrains Mono',monospace;font-size:26px;font-weight:700;
                            color:#1e8fff;line-height:1;">{len(ranked_passes)}</div>
            </div>
            <div>
                <div style="font-size:9px;font-weight:700;letter-spacing:2.5px;color:rgba(255,255,255,0.2);
                            text-transform:uppercase;margin-bottom:6px;font-family:'DM Sans',sans-serif;">Top EV</div>
                <div style="font-family:'JetBrains Mono',monospace;font-size:24px;font-weight:700;
                            color:#a78bfa;line-height:1;">{ranked_passes[0]['ev']:.4f}</div>
            </div>
        </div>
        """, unsafe_allow_html=True)
    with c_data:
        raw_df = pd.DataFrame(ranked_passes).copy()
        raw_df["pos"] = raw_df["pos"].apply(lambda p: f"({p[0]:.1f}, {p[1]:.1f})")
        raw_df.columns = ["Target (X, Y)", "Receiver xG", "Pass Probability", "EV"]
        st.dataframe(
            raw_df.style
                  .background_gradient(subset=["EV"], cmap="YlGn")
                  .background_gradient(subset=["Pass Probability"], cmap="Blues")
                  .format({
                      "Receiver xG": "{:.4f}",
                      "Pass Probability": "{:.4f}",
                      "EV": "{:.4f}",
                  }),
            use_container_width=True,
        )

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
        <span class="footer-copy">© 2025 SYNAPSE AI · Tactical Engine v5.0 · All rights reserved</span>
    </div>
    <div class="footer-right">
        <span class="footer-link"><span class="live-dot"></span>StatsBomb xG</span>
        <span class="footer-link"><span class="live-dot"></span>Metrica Tracking</span>
        <span class="footer-link"><span class="live-dot"></span>Pitch Control ML</span>
    </div>
</div>
<div style="height:16px;"></div>
""", unsafe_allow_html=True)