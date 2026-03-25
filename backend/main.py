from fastapi import FastAPI, UploadFile, File, BackgroundTasks, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel
import shutil
import sys
import os
import json
import warnings

# Suppress harmless Pydantic V1 Python 3.14 warnings from cluttering the terminal
warnings.filterwarnings("ignore", message=".*Pydantic V1.*Python 3.14.*", category=UserWarning)

# Add root synapse folder to Python path so 'backend.agents...' imports correctly
root_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if root_path not in sys.path:
    sys.path.insert(0, root_path)

import uuid
import subprocess
import asyncio

# Import the new Multi-Agent Orchestrator
try:
    from backend.agents.tactical_orchestrator import TacticalOrchestrator
    orchestrator = TacticalOrchestrator()
except ImportError as e:
    print(f"Failed to import Orchestrator: {e}")
    orchestrator = None


# Ensure we can import the tracking system modules
# The backend is in synapse/backend, the models are in synapse/tracking_system
tracking_system_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'tracking_system'))
sys.path.append(tracking_system_path)

from intelligence.xg_contextual import ContextualxGModel
import numpy as np

app = FastAPI(title="Tactical AI Engine API")

# Setup CORS for the frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:8001", "http://127.0.0.1:8001", "*"], 
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load the ML model once into memory on startup
print("Initializing xG Model...")
try:
    xg_model_path = os.path.join(tracking_system_path, "intelligence", "xg_model.pkl")
    xg_engine = ContextualxGModel(model_path=xg_model_path)
    print("xG Model loaded into memory successfully.")
except Exception as e:
    print(f"Warning: Could not load xG Model: {e}")
    xg_engine = None

# --- In-Memory Job Store ---
# In production, use Redis or a Database
jobs = {}

class XGRequest(BaseModel):
    player_pos: list[float]  # [x,y]
    defenders_pos: list[list[float]] = [] # [[x,y], [x,y]]

@app.get("/")
def health_check():
    return {"status": "online", "model": "Tactical AI Engine API is running"}

import matplotlib
matplotlib.use('Agg') # Ensure backend operates without a display
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.colors import LinearSegmentedColormap
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import MinMaxScaler
from statsbombpy import sb
import requests
import io
import warnings
from statsbombpy.api_client import NoAuthWarning
import time
import base64

warnings.simplefilter("ignore", NoAuthWarning)

# --- TACTICAL ENGINE CONSTANTS ---
PITCH_LENGTH = 104.0
PITCH_WIDTH  = 68.0
GOAL_CENTER  = np.array([104.0, 34.0])

PASS_COLORS  = ["#00e87a", "#f59e0b", "#1e8fff"]
TARGET_NAMES = ["ALPHA", "BETA", "GAMMA"]

# --- TACTICAL ENGINE ML SETUP ---
print("Initializing Tactical Engine xG Model (StatsBomb)...")
try:
    te_xg_model  = LogisticRegression(max_iter=500)
    te_xg_scaler = MinMaxScaler()
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
    X = te_xg_scaler.fit_transform(shots[["distance", "angle"]])
    te_xg_model.fit(X, shots["is_goal"])
    print("Tactical Engine xG Model loaded successfully.")
except Exception as e:
    print(f"Failed to load TE xG model: {e}")
    te_xg_model = None
    te_xg_scaler = None

# --- TACTICAL ENGINE FUNCTIONS ---
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

def calculate_xg_te(x, y, model, scaler):
    dx   = GOAL_CENTER[0] - x
    dy   = GOAL_CENTER[1] - y
    dist = np.sqrt(dx**2 + dy**2)
    ang  = np.abs(np.arctan2(dy, dx))
    feat = scaler.transform(pd.DataFrame({"distance": [dist], "angle": [ang]}))
    return model.predict_proba(feat)[0][1]

def evaluate_passes(bc, attackers, pc, model, scaler):
    opts = []
    for t in attackers:
        if t["pos"] == bc["pos"]: continue
        xg       = calculate_xg_te(t["pos"][0], t["pos"][1], model, scaler)
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

def draw_pitch_lines(ax):
    lc, la, lw = "white", 0.65, 1.4
    for i in range(0, int(PITCH_LENGTH), 10):
        alpha = 0.022 if (i // 10) % 2 == 0 else 0.0
        ax.add_patch(patches.Rectangle((i, 0), 10, PITCH_WIDTH, facecolor="white", alpha=alpha, zorder=1))
    ax.plot([0, PITCH_LENGTH, PITCH_LENGTH, 0, 0], [0, 0, PITCH_WIDTH, PITCH_WIDTH, 0], color=lc, lw=lw, alpha=la, zorder=2)
    ax.plot([PITCH_LENGTH/2]*2, [0, PITCH_WIDTH], color=lc, lw=lw*0.8, alpha=la*0.6, zorder=2)
    for side_x, mirror in [(0, 1), (PITCH_LENGTH, -1)]:
        ax.plot([side_x, side_x+mirror*16.5, side_x+mirror*16.5, side_x], [13.85, 13.85, 54.15, 54.15], color=lc, lw=lw*0.8, alpha=la*0.6, zorder=2)
    for side_x, mirror in [(0, 1), (PITCH_LENGTH, -1)]:
        ax.plot([side_x, side_x+mirror*5.5, side_x+mirror*5.5, side_x], [24.84, 24.84, 43.16, 43.16], color=lc, lw=lw*0.6, alpha=la*0.4, zorder=2)
    ax.add_patch(patches.Circle((PITCH_LENGTH/2, PITCH_WIDTH/2), 9.15, edgecolor=lc, facecolor="none", lw=lw*0.8, alpha=la*0.6, zorder=2))
    ax.scatter(PITCH_LENGTH/2, PITCH_WIDTH/2, c=lc, s=30, zorder=2, alpha=la*0.7)
    for gx in [10.97, PITCH_LENGTH - 10.97]:
        ax.scatter(gx, PITCH_WIDTH/2, c=lc, s=20, zorder=2, alpha=la*0.5)
    for cx in [PITCH_LENGTH*0.16, PITCH_LENGTH*0.84]:
        arc = patches.Arc((cx, PITCH_WIDTH/2), 18.3, 18.3, angle=0, theta1=270 if cx < PITCH_LENGTH/2 else 90, theta2=270+180 if cx < PITCH_LENGTH/2 else 90+180, color=lc, lw=lw*0.6, alpha=la*0.4, zorder=2)
        ax.add_patch(arc)
    for gx, gdir in [(0, -2.6), (PITCH_LENGTH, 2.6)]:
        ax.plot([gx, gx+gdir, gx+gdir, gx], [30.34, 30.34, 37.66, 37.66], color=lc, lw=lw*1.2, alpha=la*0.9, zorder=2)

def build_pitch_figure(bc, atts, defs, pc_matrix, ranked_passes):
    fig, ax = plt.subplots(figsize=(12, 7.8))
    fig.patch.set_facecolor("#030e1a")
    ax.set_facecolor("#030e1a")

    ax.imshow(np.tile(np.linspace(0, 1, 100).reshape(1, -1), (100, 1)), extent=[-2, PITCH_LENGTH+2, -2, PITCH_WIDTH+2], cmap=LinearSegmentedColormap.from_list("g", ["#031c0a", "#041f0c"]), aspect="auto", zorder=0)
    cmap_pc = LinearSegmentedColormap.from_list("pc", ["#8b1a1a", "#2d0000", "#0a0a12", "#001428", "#003070"], N=512)
    ax.imshow(pc_matrix, extent=[0, PITCH_LENGTH, PITCH_WIDTH, 0], cmap=cmap_pc, alpha=0.5, vmin=0, vmax=1, zorder=1, aspect="auto")
    ax.set_aspect("equal")
    ax.set_xlim(-2, PITCH_LENGTH + 2)
    ax.set_ylim(-2, PITCH_WIDTH  + 2)
    draw_pitch_lines(ax)

    def draw_player(px, py, vx, vy, fill_color, edge_color, size=190):
        ax.scatter(px+0.6, py+0.6, c="#000000", s=size*0.8, alpha=0.25, zorder=3)
        ax.scatter(px, py, c=fill_color, s=size*2.0, alpha=0.08, zorder=3)
        ax.scatter(px, py, c=fill_color, s=size, edgecolors=edge_color, linewidths=1.8, zorder=4)
        speed = np.hypot(vx, vy)
        if speed > 0.5:
            scale = min(1.5, 0.6 + speed * 0.06)
            ax.annotate("", xy=(px + vx*scale*0.12, py + vy*scale*0.12), xytext=(px, py), arrowprops=dict(arrowstyle="->", color=fill_color, lw=1.2, mutation_scale=9), zorder=5)

    for p in defs: draw_player(*p["pos"], *p["vel"], "#1565c0", "#64b5f6", 175)
    for p in atts: draw_player(*p["pos"], *p["vel"], "#c62828", "#ef9a9a", 175)

    bx, by = bc["pos"]
    for rs, ra in [(1100, 0.05), (650, 0.09), (380, 0.15)]:
        ax.scatter(bx, by, c="#00e87a", s=rs, alpha=ra, zorder=4)
    ax.scatter(bx+0.6, by+0.6, c="black", s=420, alpha=0.25, zorder=4)
    ax.scatter(bx, by, c="#00e87a", s=380, edgecolors="white", linewidths=2.8, zorder=6)
    ax.scatter(bx, by, c="white", s=60, zorder=7, alpha=0.95)

    for i, opt in enumerate(ranked_passes[:3]):
        if opt["ev"] < 0.003: continue
        tx, ty = opt["pos"]
        col    = PASS_COLORS[i]
        for gw, ga in [(18, 0.04), (11, 0.07), (5, 0.12)]:
            ax.annotate("", xy=(tx, ty), xytext=(bx, by), arrowprops=dict(facecolor=col, edgecolor=col, alpha=ga, width=gw, headwidth=gw*2.4, shrink=0.06), zorder=4)
        ax.annotate("", xy=(tx+0.6, ty+0.6), xytext=(bx+0.6, by+0.6), arrowprops=dict(facecolor="black", edgecolor="black", alpha=0.12, width=3, headwidth=10, shrink=0.06), zorder=4)
        ax.annotate("", xy=(tx, ty), xytext=(bx, by), arrowprops=dict(facecolor=col, edgecolor=col, width=2.8, headwidth=12, headlength=9, shrink=0.06, linewidth=0), zorder=6)
        for r_size, r_alpha in [(5.0, 0.25), (3.2, 0.7)]:
            ax.add_patch(patches.Circle((tx, ty), r_size, edgecolor=col, facecolor="none", lw=1.8, alpha=r_alpha, zorder=6))
        ax.scatter(tx, ty, c=col, s=280, alpha=0.1, zorder=5)
        lbl = ["α", "β", "γ"][i]
        ax.text(tx + 4.0, ty - 1.5, lbl, color=col, fontsize=11, fontweight="black", fontfamily="DejaVu Serif", zorder=7, alpha=0.92)

    ax.axis("off")
    plt.tight_layout(pad=0)
    
    # Save to base64
    buf = io.BytesIO()
    plt.savefig(buf, format="png", bbox_inches='tight', pad_inches=0, facecolor=fig.get_facecolor(), edgecolor='none')
    buf.seek(0)
    img_b64 = base64.b64encode(buf.read()).decode('utf-8')
    plt.close(fig)
    return img_b64

@app.get("/api/tactical-engine/{frame}")
def get_tactical_engine_data(frame: int):
    t0 = time.time()
    
    # Read generated AI tracking data
    tracking_dir = os.path.join(os.path.dirname(__file__), "..", "tracking_system", "output")
    pos_file = os.path.join(tracking_dir, "tracking_data.csv")
    intel_file = os.path.join(tracking_dir, "tracking_data_intel.csv")
    
    attackers = []
    defenders = []
    ball_carrier = None
    intel_data = {}
    ranked_passes = []
    pc_matrix = np.zeros((int(PITCH_WIDTH), int(PITCH_LENGTH)))
    img_b64 = ""

    try:
        if os.path.exists(pos_file) and os.path.exists(intel_file):
            df_pos = pd.read_csv(pos_file)
            df_intel = pd.read_csv(intel_file)
            
            # Filter for frame
            frame_pos = df_pos[df_pos['frame'] == frame]
            frame_intel = df_intel[df_intel['frame_id'] == frame]
            
            if not frame_intel.empty:
                row = frame_intel.iloc[0]
                intel_data = {
                    "contextual_xg": float(row['contextual_xG']) if pd.notnull(row['contextual_xG']) else 0.0,
                    "momentum": str(row['momentum_state']) if pd.notnull(row['momentum_state']) else "Neutral",
                    "risk_index": float(row['risk_index']) if pd.notnull(row['risk_index']) else 0.0,
                    "possession_probability": float(row['possession_probability']) if pd.notnull(row['possession_probability']) else 0.0
                }
                
            for _, p in frame_pos.iterrows():
                px = float(p['pitch_x_meter']) if pd.notnull(p['pitch_x_meter']) else 0.0
                py = float(p['pitch_y_meter']) if pd.notnull(p['pitch_y_meter']) else 0.0
                vx = float(p['vx']) if pd.notnull(p['vx']) else 0.0
                vy = float(p['vy']) if pd.notnull(p['vy']) else 0.0
                pid = int(p['player_id']) if pd.notnull(p['player_id']) else 0
                
                pos_dict = {"pos": (px, py), "vel": (vx, vy), "id": pid, "speed": float(p['speed']) if pd.notnull(p['speed']) else 0.0}
                
                if p['is_my_team'] == True or p['is_my_team'] == 'True':
                    attackers.append(pos_dict)
                else:
                    defenders.append(pos_dict)
                    
                if p['is_ball_carrier'] == True or p['is_ball_carrier'] == 'True':
                    ball_carrier = pos_dict

            if len(attackers) > 0 and len(defenders) > 0:
                pc_matrix = generate_pitch_control(attackers, defenders)
                
            if te_xg_model and ball_carrier and len(attackers) > 0:
                ranked_passes = evaluate_passes(ball_carrier, attackers, pc_matrix, te_xg_model, te_xg_scaler)

        # Generate Image
        if ball_carrier and len(attackers) > 0:
            img_b64 = build_pitch_figure(ball_carrier, attackers, defenders, pc_matrix, ranked_passes)
            
    except Exception as e:
        print(f"Warning: tactical API logic error on frame {frame}: {e}")

    # Convert positions to lists so they are JSON serializable
    serializable_ranked_passes = []
    for opt in ranked_passes:
        opt_copy = opt.copy()
        if isinstance(opt_copy["pos"], tuple):
            opt_copy["pos"] = list(opt_copy["pos"])
        serializable_ranked_passes.append(opt_copy)
        
    latency_ms = (time.time() - t0) * 1000

    return {
        "frame": frame,
        "latency_ms": latency_ms,
        "attackers_count": len(attackers),
        "defenders_count": len(defenders),
        "ranked_passes": serializable_ranked_passes,
        "pitch_image_b64": img_b64,
        "intel_data": intel_data
    }

@app.post("/api/predict-xg")
def predict_xg(request: XGRequest):
    """Instant inference using the pre-loaded ML pickel file."""
    if not xg_engine:
        return JSONResponse(status_code=500, content={"error": "xG Model is not initialized on the server."})
    
    player_arr = np.array(request.player_pos)
    defenders_arr = [np.array(d) for d in request.defenders_pos]
    
    # Run direct inference
    xg_val = xg_engine.predict(player_arr, defenders_arr)
    
    return {"contextual_xg": round(float(xg_val), 3)}


def process_video_task(job_id: str, input_path: str):
    """Background task to run the heavy main.py pipeline."""
    try:
        jobs[job_id]["status"] = "processing"
        
        # We run the existing main.py script as a subprocess so we don't have to rewrite the entire pipeline
        # We target the output to a specific job folder so the frontend can retrieve it
        output_dir = os.path.join(os.path.dirname(__file__), "static", "jobs", job_id)
        os.makedirs(output_dir, exist_ok=True)
        
        main_script = os.path.join(tracking_system_path, "main.py")
        
        # Determine paths
        annotated_out = os.path.join(output_dir, "annotated_video.mp4")
        map_out = os.path.join(output_dir, "tactical_map_video.mp4")
        
        # Construct the command (assuming main.py can take an output directory argument, 
        # or we just let it output to its default and move it)
        # Note: We need to modify main.py slightly to accept a custom output directory via argparse
        # For now, we'll run it, and then move the files from the default `output/` directory
        
        print(f"Starting tracking job {job_id} on {input_path}...")
        
        # Run the heavy processing
        result = subprocess.run(["python", main_script, "--input", input_path], 
                                cwd=tracking_system_path, 
                                capture_output=True, text=True)
                                
        if result.returncode == 0:
            # Move files from default output to the job static folder
            default_output = os.path.join(tracking_system_path, "output")
            
            src_annotated = os.path.join(default_output, "output_tracked_video.mp4")
            src_map = os.path.join(default_output, "tactical_map_video.mp4")
            src_json = os.path.join(default_output, "tracking_data.json")
            
            if os.path.exists(src_annotated):
                shutil.copy(src_annotated, annotated_out)
            if os.path.exists(src_map):
                shutil.copy(src_map, map_out)
                
            # Auto-ingest into RAG database so the AI Coach can answer questions
            if os.path.exists(src_json):
                ingest_script = os.path.join(os.path.dirname(__file__), "ingest_to_db.py")
                print(f"Auto-ingesting {src_json} into RAG database...")
                subprocess.run(["python", ingest_script], capture_output=True)
                
            jobs[job_id]["status"] = "completed"
            jobs[job_id]["annotated_video_url"] = f"/static/jobs/{job_id}/annotated_video.mp4"
            jobs[job_id]["tactical_map_url"] = f"/static/jobs/{job_id}/tactical_map_video.mp4"
            
            print(f"Job {job_id} completed successfully.")
        else:
            print(f"Job {job_id} failed: {result.stderr}")
            jobs[job_id]["status"] = "failed"
            jobs[job_id]["error"] = result.stderr
            
    except Exception as e:
        print(f"Job {job_id} crashed: {e}")
        jobs[job_id]["status"] = "failed"
        jobs[job_id]["error"] = str(e)
    finally:
        # Cleanup uploaded raw file
        if os.path.exists(input_path):
            os.remove(input_path)

class CalibrationData(BaseModel):
    src_pts: list[list[float]]
    dst_pts: list[list[float]]

from fastapi import Request

@app.post("/api/save-calibration")
async def save_calibration(request: Request):
    """Saves the 4 source/destination points from the Web UI to tracking_system/calibration.json"""
    calib_file = os.path.join(tracking_system_path, "calibration.json")
    body = await request.json()
    print(f"DEBUG: RAW BODY: {body}")
    
    with open(calib_file, "w") as f:
        json.dump(body, f, indent=4)
        
    return {"status": "success", "message": "Calibration points saved."}

@app.post("/api/upload-video")
async def upload_video(background_tasks: BackgroundTasks, file: UploadFile = File(...)):
    """Uploads a video and queues it for AI processing."""
    job_id = str(uuid.uuid4())
    
    # Save the file temporarily
    temp_dir = os.path.join(os.path.dirname(__file__), "temp_uploads")
    os.makedirs(temp_dir, exist_ok=True)
    temp_path = os.path.join(temp_dir, f"{job_id}_{file.filename}")
    
    with open(temp_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)
        
    # Register job
    jobs[job_id] = {
        "id": job_id,
        "filename": file.filename,
        "status": "queued"
    }
    
    # Dispatch heavy processing
    background_tasks.add_task(process_video_task, job_id, temp_path)
    
    return {"job_id": job_id, "status": "queued", "message": "Video uploaded and processing started."}

@app.get("/api/jobs/{job_id}")
def get_job_status(job_id: str):
    """Retrieve the status of a video processing job."""
    if job_id not in jobs:
        return JSONResponse(status_code=404, content={"error": "Job not found"})
    return jobs[job_id]

# --- MULTI-AGENT WEBSOCKET ENDPOINT ---
@app.websocket("/ws/post_match_chat")
async def websocket_post_match_chat(websocket: WebSocket):
    """
    WebSocket endpoint that streams the post-match multi-agent reasoning 
    from the TacticalOrchestrator back to the frontend.
    """
    await websocket.accept()
    if orchestrator is None:
        await websocket.send_json({"agent": "System", "text": "Error: Tactical Orchestrator not loaded."})
        await websocket.close()
        return

    try:
        while True:
            # Wait for user message
            data = await websocket.receive_text()
            print(f"Received WS prompt: {data}")
            
            # Delegate to Orchestrator MAS
            await orchestrator.process_chat_message(data, websocket)
            
    except WebSocketDisconnect:
        print("WebSocket client disconnected from /ws/agent_chat")
    except Exception as e:
        print(f"WebSocket error: {e}")
        try:
            await websocket.close()
        except:
            pass
            
# --- POST-MATCH REPORTING ENDPOINTS (DIRECTIVE 3) ---

@app.post("/api/generate_match_report")
async def generate_match_report():
    """
    Endpoint that triggers the MAS to digest the entire match RAG database 
    and return a comprehensive post-match summary.
    """
    if orchestrator is None:
        return JSONResponse(status_code=500, content={"error": "Orchestrator MAS offline"})
        
    try:
        report = await orchestrator.generate_full_report()
        return {"status": "success", "report": report}
    except Exception as e:
        return JSONResponse(status_code=500, content={"error": str(e)})

@app.websocket("/ws/post_match_chat")
async def websocket_post_match_chat(websocket: WebSocket):
    """
    Dedicated WebSocket endpoint for the Post-Match Analyst Chat UI.
    Streams back-and-forth Q&A about specific match incidents using RAG.
    """
    await websocket.accept()
    if orchestrator is None:
        await websocket.send_json({"agent": "System", "text": "Tactical Assistant Offline."})
        await websocket.close()
        return

    try:
        while True:
            data = await websocket.receive_text()
            # Delegate to the same Orchestrator pipeline, which now handles RAG validation
            await orchestrator.process_chat_message(data, websocket)
            
    except WebSocketDisconnect:
        print("WebSocket client disconnected from /ws/post_match_chat")
    except Exception as e:
        print(f"WebSocket error: {e}")
        try:
            await websocket.close()
        except:
            pass

# Mount static files to serve the output videos
from fastapi.staticfiles import StaticFiles

static_dir = os.path.join(os.path.dirname(__file__), "static")
os.makedirs(static_dir, exist_ok=True)
app.mount("/static", StaticFiles(directory=static_dir), name="static")

output_dir = os.path.join(tracking_system_path, "output")
if os.path.exists(output_dir):
    app.mount("/output", StaticFiles(directory=output_dir), name="output")

if __name__ == "__main__":
    import uvicorn
    # Run the server on port 8000
    uvicorn.run(app, host="0.0.0.0", port=8000)
