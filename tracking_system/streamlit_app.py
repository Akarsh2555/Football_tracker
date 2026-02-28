import os
import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.colors import LinearSegmentedColormap
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import MinMaxScaler

# Import the "engine" modules from the tracking system package
from video_processor import VideoProcessor
from tracker import Tracker
from pitch_mapper import PitchMapper
from visualizer import Visualizer
from exporter import DataExporter
from intelligence.spatial import SpatialEngine
from intelligence.core import TacticalIntelligenceEngine

# ---------------------------------------------------------------------------
# constants & helpers (a small subset of the big CSS app shown by the user)
# ---------------------------------------------------------------------------
PITCH_LENGTH = 104.0
PITCH_WIDTH  = 68.0
GOAL_CENTER  = np.array([104.0, 34.0])

# xG training (statsbomb) is unchanged from the sample app;
# you may comment it out if you do not need it in post‑match mode.

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

# ---------------------------------------------------------------------------
# post‑match analysis helpers
# ---------------------------------------------------------------------------

def compute_pitch_control(attackers, defenders):
    """Simple gaussian-based pitch control field."""
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

# ... (other helper functions such as evaluate_passes, draw_pitch_lines, etc.)
# you can copy/paste them from the big example above if needed for the
# post‑match interface; for brevity they have been omitted here.

# ---------------------------------------------------------------------------
# application logic
# ---------------------------------------------------------------------------

st.set_page_config(page_title="SYNAPSE · Post‑Match Analyzer", layout="wide")

st.title("SYNAPSE Football Post‑Match Dashboard")

# 1. acquire data: either run the tracking pipeline or load precomputed CSV

uploaded_video = st.file_uploader("Upload match video (MP4) for processing", type=["mp4", "avi"])
load_csv      = st.file_uploader("—or, upload existing tracking CSV", type=["csv"])

@st.cache_data

def run_tracking(video_path: str, model_path: str = "yolov8n.pt") -> str:
    """Process a video and return path to the exported CSV file."""
    temp_out = os.path.join("output", "streamlit_run.csv")
    proc = VideoProcessor(video_path)
    tracker = Tracker(model_path=model_path)
    mapper = PitchMapper()
    spatial = SpatialEngine(fps=proc.fps)
    intel = TacticalIntelligenceEngine()
    exporter = DataExporter("output")

    # trivial homography; you should provide real field points
    w, h = proc.width, proc.height
    src = np.array([[w*0.1, h*0.9],[w*0.9, h*0.9],[w*0.3, h*0.4],[w*0.7, h*0.4]], np.float32)
    dst = np.array([[0,68],[105,68],[0,0],[105,0]], np.float32)
    mapper.compute_homography(src, dst)

    for idx, frame in proc.get_frame():
        players, balls = tracker.process_frame(frame)
        # low‑effort: export raw tracked positions only
        positions = []
        for i in range(len(players)):
            bbox = players.xyxy[i]
            cid  = players.tracker_id[i]
            bottom = mapper.extract_bottom_center(bbox)
            coords = mapper.transform_point(bottom)
            if coords:
                positions.append({"player_id": cid, "x": coords[0], "y": coords[1]})
        exporter.add_frame_data(idx, positions)
    exporter.export(file_prefix="streamlit")
    proc.release()
    return os.path.join("output", "streamlit.csv")

track_csv_path = None
if uploaded_video and not load_csv:
    # save the uploaded video to disk for processing
    tmp = os.path.join("output", "temp_video.mp4")
    with open(tmp, "wb") as f:
        f.write(uploaded_video.getbuffer())
    with st.spinner("Running tracker… this can take a few minutes"):
        track_csv_path = run_tracking(tmp)
elif load_csv:
    track_csv_path = load_csv

# once we have a path, read the dataframe
if track_csv_path is not None:
    df = pd.read_csv(track_csv_path)
    st.success(f"Tracking data loaded ({len(df):,} rows)")
else:
    df = None

# 2. post‑match analysis
if df is not None:
    st.header("Overview")
    st.write(df.head())
    # further analysis: heatmaps, pitch control, pass options, etc.
    # you can adapt the drawing helpers from the long sample code above
    # and display them with `st.pyplot(fig)` as in the provided example.

    # example: show x/y scatter of all positions
    fig, ax = plt.subplots(figsize=(6,4))
    ax.scatter(df['pitch_x_meter'], df['pitch_y_meter'], s=2, c='red')
    ax.set_title('All player locations')
    ax.set_xlim(0,105); ax.set_ylim(0,68)
    st.pyplot(fig)

    # more charts or a per‑frame slider can be added here


# ---------------------------------------------------------------------------
# instructions for deployment
# ---------------------------------------------------------------------------

st.markdown(
    """
    **Deployment**

    * Run locally with `streamlit run streamlit_app.py`.
    * For a public instance, push the repo to GitHub and use [Streamlit Community Cloud](https://streamlit.io/cloud) or containerize with Docker and deploy to any cloud service (AWS, Azure, GCP).
    * The caching decorators ensure the video processing is done once per file and results are reused across sessions.

    You could also port the same logic to a lightweight FastAPI/Flask backend if you prefer a traditional web server; Streamlit is convenient for rapid prototyping and interactive post‑match dashboards.
    """,
    unsafe_allow_html=True,
)
