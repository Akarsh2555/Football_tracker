# Synapse - Football Tracking & Tactical AI Engine

## Overview
**Synapse** (or Football_tracker) is a comprehensive full-stack AI system designed for advanced football (soccer) video analysis. It processes game footage to track players, map their positions to a 2D tactical pitch, predict game events like Expected Goals (xG), and provide real-time coaching recommendations.

## System Architecture

The project is structured into three primary domains:

### 1. Computer Vision & Tracking (`tracking_system/`)
The core vision layer powered by deep learning and geometric transformations.
* **Detection & Tracking**: Uses advanced YOLO models (YOLOv8, YOLOv11) to accurately detect players, referees, and the ball across video frames.
* **Homography & Pitch Mapping**: Features modules (`pitch_mapper.py`, `calibrate.py`) to warp 2D video pixel coordinates into top-down, fixed-pitch tactical perspectives.
* **Analytical Intelligence**: Contains advanced logic (`intelligence/`, `train_xg.py`) for assigning ball possession, tracking momentum, and evaluating expected goals (xG).

### 2. Backend & Data Processing (`backend/`)
The middle layer designed for robust data serving and real-time processing.
* **API Engine**: Built on FastAPI (`backend/main.py`), it processes video uploads, coordinates model inferences, and handles data pipelines.
* **Real-time Comms**: Employs WebSockets to transmit player coordinates and tactical alerts continuously to the frontend.
* *Legacy Support*: Includes earlier iterations built on Streamlit (`tactical_engine_app.py`, `app.py`) for rapid prototyping of the tactical engine.

### 3. Frontend Tactical UI (`frontend/`)
A custom-built, high-contrast visualizer for coaching and analysis.
* **Architecture**: A native HTML, CSS (`tactical_engine.css`), and JavaScript component architecture running independently of Python UI frameworks.
* **Features**: Displays synchronized video playback side-by-side with 2D pitch tracking plots alongside real-time metrics, dynamically fetched from the FastAPI backend.

## Key Capabilities
* **Full-Match Tracking**: Extracts continuous player trajectories from broadcast footage.
* **Ball Carrier Resolution**: Pixel-level proximity logic to correctly attribute ball possession.
* **AI Coaching System**: Translates spatial coordinates into actionable tactical insights.
* **Custom Models**: Supports robust MLOps patterns for iteratively training and refining custom YOLO object detection models on domain-specific datasets.
