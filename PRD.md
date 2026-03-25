# Synapse AI: Product Requirements Document (PRD)

## 1. Executive Summary
**Synapse AI** is an advanced, full-stack football (soccer) tactical analytics engine. It leverages state-of-the-art computer vision (YOLO), spatial intelligence, and generative AI to automatically transcribe broadcast match footage into 2D tactical maps, derive complex analytics (Expected Goals, Pitch Control, Pass Expected Value), and provide an interactive AI Coaching Assistant for post-match breakdown.

## 2. Product Vision & Target Audience
- **Vision:** Democratize elite-level football analytics. Enable amateur clubs, content creators, and scouts to extract multi-million-dollar data points from standard broadcast or drone footage without wearing physical GPS trackers.
- **Target Audience:** Football Coaches, Tactical Analysts, Scouts, and Sports Broadcasters.

## 3. Core System Architecture
The application is decoupled into three primary pillars:

### 3.1. Tracking System Engine (Computer Vision & Kinematics)
The core Python-based CLI processing pipeline running heavy ML workloads.
- **Object Detection & Tracking:** YOLOv8 detects players, referees, and the ball. ByteTrack assigns persistent IDs to players across frames.
- **Homography & Pitch Mapping:** Maps 2D camera pixel coordinates to a mathematically accurate top-down 105m x 68m tactical pitch interface.
- **Team Distinction:** Uses K-Means clustering (sorted deterministically by HSV Value/Brightness on 20+ frame samples) to accurately group players into Home and Away teams.
- **Spatial Engine:** Extracts velocities and accelerations using 2D Kalman Filters.
- **Intelligence Engine:**
  - *Pitch Control:* Calculates territorial dominance dynamically.
  - *Pass Evaluator:* Computes interception risks and ranks passing options based on receiver proximity to defenders.
  - *Expected Goals (xG):* Employs Logistic Regression (trained on StatsBomb open data) to score the threat of the ball carrier's position.
- **Video Rendering:** Uses `imageio[ffmpeg]` to dynamically overlay AR graphics and render smooth, web-compatible `H.264 .mp4` outputs.

### 3.2. Backend Data & AI Orchestrator (FastAPI)
The API layer bridging the hardcore ML python scripts with the web interface.
- **Asynchronous Job Queue:** Manages heavy video processing tasks without hanging the UI.
- **RESTful Endpoints:** Serves video assets (`/static/jobs`), saves calibration data (`/api/save-calibration`), and serves frame-by-frame tactical data.
- **Multi-Agent RAG System:** An advanced AI Orchestrator utilizing Google Gemini.
  - Automatically ingests output tracking JSONs into a local ChromaDB Vector Store.
  - Supports WebSocket streaming (`/ws/post_match_chat`) for a live "Tactical Assistant" that cross-references match events to answer complex tactical questions.

### 3.3. Frontend Web Dashboard (Brutalist "Matchday" UI)
A native HTML/JS/CSS client focusing on striking typography and high contrast.
- **Design System:** Brutalist aesthetic (Neon Lime `#ccff00`, Stark Black, Aggressive Red `#ff2a4d`, and `Oswald` typography) inspired by premium sports broadcasters.
- **Calibration Modal:** An interactive HTML5 Canvas tool where users click the 4 corners of the pitch to generate precise homography matrices.
- **Video Sync:** Plays the broadcast video and the simulated 2D tactical pitch side-by-side.
- **Live xG Inference:** Interactive range sliders allowing users to manually test goal probability from anywhere on the pitch.
- **Post-Match Assistant:** A sliding chat interface communicating natively with the multi-agent WebSocket to ask the AI questions like *"Who had the highest peak speed?"* or *"Why did our passing break down in frame 300?"*

## 4. Key Workflows

1. **Upload & Calibrate:** The user drags a `.mp4` into the dropzone. A modal opens with the first frame; the user clicks the 4 pitch corners.
2. **Processing:** The frontend dispatches the video and points to FastAPI. The backend triggers the Tracking System CLI, which outputs annotated `mp4`s and JSON tracking logs.
3. **Analytics Display:** The user scrubs the timeline to view synced frames while the metrics (xG, Pitch Control %, Momentum) dynamically update on the HUD.
4. **Interrogation:** The user speaks to the Synapse AI Chatbot to draw conclusions from the generated database.

## 5. Technical Stack
- **AI/ML:** PyTorch, Ultralytics (YOLOv8), Scikit-Learn (K-Means, Logistic Regression), OpenCV, FilterPy.
- **Backend:** Python 3, FastAPI, Uvicorn, LangChain, ChromaDB, Google GenAI (Gemini 2.5 Flash / Pro).
- **Frontend:** HTML5, Vanilla JavaScript, CSS3 (Tailwind primitives for structure), Chart.js (Radar Charts).
- **Video:** Ffmpeg (via `imageio`), OpenCV.

## 6. Future Enhancements & Scope
- **Camera Panning Support:** Implement robust optical flow strategies to recalculate homography points on dynamically panning/zooming cameras automatically.
- **Mobile Companion:** Build a React Native app to subscribe to live WebSocket analytical feeds for real-time dugout alerts.
- **Multi-Camera Stitching:** Fuse tracking inputs from two separate camera angles into a single cohesive global coordinate map.
