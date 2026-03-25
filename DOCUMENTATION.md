# Synapse (Football_tracker) Documentation

This document provides an in-depth, technical overview of the Synapse project, its architecture, and the core modules that drive the Football Tracking & Tactical AI Engine.

## Table of Contents
1. [Introduction](#introduction)
2. [Directory Structure](#directory-structure)
3. [Computer Vision & Tracking Pipeline](#computer-vision--tracking-pipeline)
4. [Tactical Intelligence Engine](#tactical-intelligence-engine)
5. [Backend API & Fast Processing](#backend-api--fast-processing)
6. [Frontend UI Architecture](#frontend-ui-architecture)

---

## Introduction
Synapse is built to bring professional-grade tactical analysis to football (soccer) video footage. It takes raw broadcast or tactical camera feeds and turns them into 2D pitch coordinates, computes advanced metrics like Expected Goals (xG), determines ball possession through pixel-level proximity, and calculates pass availability and risk. 

---

## Directory Structure

```text
synapse/
├── tracking_system/       # Core Computer Vision, Tracking & ML engines
│   ├── intelligence/      # Advanced tactical models (xG, ball control, risk, momentum)
│   ├── app.py             # Streamlit legacy tactical app 
│   ├── main.py            # Main entry point for local video processing pipeline
│   ├── tracker.py         # YOLO/ByteTrack wrapper for player/ball tracking
│   ├── pitch_mapper.py    # Homography matrix transformations (2D Video -> 2D Pitch)
│   └── train_xg.py        # ML training script for Expected Goals model
├── backend/               # FastAPI Server configuration
│   └── main.py            # API routes, websocket layer, and video job queues
├── frontend/              # Native CSS/JS/HTML UI Web interface
│   ├── index.html         # Main dashboard layout
│   ├── script.js          # Client-side API interactions and websocket handlers
│   ├── tactical_engine.html # Dedicated tactical visualization UI
│   ├── tactical_engine.css  # High-contrast premium CSS styling
│   └── tactical_engine.js   # Canvas/HUD rendering logic for pitch and videos
├── xGModel/               # Pickled datasets/models for StatsBomb ML predictions
└── output/                # Processed videos and TSV/CSV tracking logs
```

---

## Computer Vision & Tracking Pipeline
Located under `tracking_system/`, this pipeline takes a video file and sequentially extracts tracking data.

1. **Object Detection & Tracking (`tracker.py`)**
   - Utilizes custom-trained **YOLOv8/YOLOv11** models to detect Football Players, Referees, and the Football.
   - Assigns unique IDs to players and smooths trajectories over time using visual tracking algorithms.
   - Clustering algorithms (e.g., KMeans on player jersey colors) categorize entities into "Home" and "Away" teams.

2. **Homography & Pitch Mapping (`pitch_mapper.py`, `calibrate.py`)**
   - Maps 2D pixel coordinates from the camera view onto a standardized 105x68 meter 2D virtual pitch.
   - Requires calibration parameters to translate perspective distortion (broadcaster angle) into a top-down tactical configuration.

3. **Execution Entry (`tracking_system/main.py`)**
   - A CLI-based pipeline that combines `VideoProcessor`, `Tracker`, `PitchMapper`, and the `TacticalIntelligenceEngine`.
   - Exports tracked coordinates and events to `output/tracking_data.csv`.

---

## Tactical Intelligence Engine
Located in `tracking_system/intelligence/`, this component transforms raw spatial data into actionable football insights. 

The `TacticalIntelligenceEngine` orchestrates several sub-models per frame:
1. **Ball Control & Possession (`ball_control.py`)**
   - Uses precise bounding-box intersection calculations (raw supervision Detections) rather than warped 2D pitch coordinates to accurately attribute ball possession to the closest player.
2. **Contextual Expected Goals (`xg_contextual.py`)**
   - Calculates the theoretical probability of scoring (xG) from the ball carrier's current location, dynamically adjusting based on the defensive pressure (distance to opponents).
3. **Pass Engine & Risk Model (`pass_engine.py`, `risk_model.py`)**
   - Evaluates potential passing lanes to teammates. Computes Expected Value (EV) of a pass based on the receiver's xG potential and calculates a `risk_index` based on opponent proximity to the passing lane.
4. **Temporal Momentum (`momentum.py`) & Metrics (`metrics.py`)**
   - Continually monitors which team is dictating the flow of the game, aggregating possession probabilities and team compactness metrics over sliding time windows.

---

## Backend API & Fast Processing
The application is served via a **FastAPI** backend (`backend/main.py`).
- **REST Endpoints**: Endpoints to upload custom video files (`/upload`) and check processing job statuses (`/status/{job_id}`).
- **Background Tasks**: Machine learning inference and video processing are pushed to asynchronous background queues to ensure the API remains responsive.
- **Telematic Real-Time Data**: Uses FastAPI to serve inference results that power the frontend HUD, streaming xG, momentum states, and passing options instantly.
- **In-Memory Models**: The StatsBomb-trained Logistic Regression xG model is loaded into memory on startup for ultra-fast, on-the-fly predictions (`predict_xg()`).

---

## Frontend UI Architecture
The presentation layer (`frontend/`) has moved away from Python-based rendering (Streamlit) towards a modern, native web stack for maximum performance and premium aesthetics.
- **Theme & Styling (`tactical_engine.css`, `style.css`)**: Implements a high-contrast vintage dark mode layout with deep blues, neon greens (`#00e87a`), and ambers to create a sleek "Coaching Engine" vibe.
- **Canvas Rendering (`tactical_engine.js`)**: Leverages the HTML5 `<canvas>` API to redraw the 2D pitch map, player nodes, passing lines, and heatmaps 60 times a second.
- **HUD (Heads-Up Display)**: Displays live metrics—such as Ball Carrier ID, Real-time xG, Best Pass Option, and Team Momentum—synchronously alongside the video playback.

---

*This document was generated automatically to outline the structure and intent of the Synapse Football Engine.*
