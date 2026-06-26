# Person Detection, Tracking, and Re-identification (Re-ID) Pipeline

A real-time Python pipeline that detects, tracks, and re-identifies individuals across video frames. The pipeline uses **YOLOv8** for person detection, **DeepSORT** for multi-object tracking, **Torchreid (OSNet)** for feature embedding extraction, and an **HSV-based heuristic classifier** to identify clothing colors (upper and lower body). It maintains a persistent gallery database (`gallery.pkl`) to re-identify individuals across different tracking sessions or camera views.

---

## Features

- **Person Detection:** Uses a fast and accurate YOLOv8 detector to locate people.
- **Real-Time Tracking:** Employs DeepSORT to track detected individuals across consecutive frames.
- **Appearance Re-ID:** Extracts 512-dimensional normalized embeddings using Torchreid's OSNet, allowing the system to match people even after they leave and re-enter the camera frame.
- **Clothing Color Classification:** Automatically crops upper-body (torso) and lower-body (legs) regions and determines their dominant color (e.g., Black, White, Red, Blue, Green, Yellow, Grey, Brown, Orange, Purple, Pink) using a vectorized HSV color classifier.
- **Persistent Database:** Saves and loads a Re-ID gallery database to preserve recognized identities across sessions.
- **Interactive UI:** Provides a CV2-based display with semi-transparent label cards, real-time FPS, processing latency, and tracking stats.

---

## Project Structure

- **[detector.py](file:///C:/Users/Vishnu%20Pradhap/Documents/antigravity/dazzling-noether/detector.py)**: Wraps the Ultralytics YOLOv8 model to detect persons and format bounding boxes for the tracker.
- **[embedder.py](file:///C:/Users/Vishnu%20Pradhap/Documents/antigravity/dazzling-noether/embedder.py)**: Employs Torchreid's `FeatureExtractor` (OSNet) to extract normalized 512-dimensional feature vectors from cropped person bounding boxes.
- **[color_classifier.py](file:///C:/Users/Vishnu%20Pradhap/Documents/antigravity/dazzling-noether/color_classifier.py)**: Segments the person crop into upper and lower regions, converts them to HSV, and calculates the dominant clothing colors using pixel counting.
- **[gallery.py](file:///C:/Users/Vishnu%20Pradhap/Documents/antigravity/dazzling-noether/gallery.py)**: Manages the Re-ID database, handles matching using cosine similarity (dot product of L2-normalized embeddings), manages history/thumbnails, and handles saving/loading.
- **[pipeline.py](file:///C:/Users/Vishnu%20Pradhap/Documents/antigravity/dazzling-noether/pipeline.py)**: Orchestrates the workflow: detects people, extracts embeddings and colors, updates the DeepSORT tracker, matches tracks against the global gallery, and annotates the video frames.
- **[run.py](file:///C:/Users/Vishnu%20Pradhap/Documents/antigravity/dazzling-noether/run.py)**: Entry point script to parse CLI arguments, open the video stream, execute the pipeline frame-by-frame, and render the output window.
- **[requirements.txt](file:///C:/Users/Vishnu%20Pradhap/Documents/antigravity/dazzling-noether/requirements.txt)**: Lists all PyPI dependencies required by the project.

---

## Setup Instructions (Windows)

Follow these steps to set up the project on your Windows machine:

### Prerequisite: Python 3.8 - 3.10
It is recommended to use **Python 3.8, 3.9, or 3.10** for optimal compatibility with PyTorch, Torchreid, and DeepSORT dependencies.

### 1. Create a Virtual Environment
Open PowerShell or Command Prompt in the project folder and run:
```powershell
python -m venv venv
```

### 2. Activate the Virtual Environment
- **In PowerShell:**
  ```powershell
  .\venv\Scripts\Activate.ps1
  ```
  *(Note: If you get an execution policy error, run `Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope Process` first)*
  
- **In Command Prompt (cmd):**
  ```cmd
  .\venv\Scripts\activate.bat
  ```

### 3. Install PyTorch (CUDA Optional but Recommended)
If you have an NVIDIA GPU, install PyTorch with CUDA support for faster inference:
```powershell
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
```
If running on CPU only:
```powershell
pip install torch torchvision
```

### 4. Install Project Dependencies
Install the remaining packages listed in `requirements.txt`:
```powershell
pip install -r requirements.txt
```

*Note: The script will automatically download the YOLOv8 weights (`yolov8n.pt`) and the OSNet embedding weights (`osnet_x1_0`) on their first execution.*

---

## Running the Pipeline

Ensure your virtual environment is active.

### Run with Webcam (Device 0)
```powershell
python run.py --input 0
```

### Run with a Video File
```powershell
python run.py --input "path/to/your/video.mp4"
```

### Command Line Arguments
You can customize the pipeline execution with the following optional arguments:
- `--input`: Webcam device index (default: `0`) or path to a video file.
- `--threshold`: Cosine similarity threshold for Re-ID matching (default: `0.75`). Higher values make matching stricter.
- `--conf`: Confidence threshold for YOLOv8 person detection (default: `0.5`).
- `--gallery`: Path to the pickle file where the Re-ID database is saved/loaded (default: `gallery.pkl`).
- `--device`: Target device for inference (`cuda` or `cpu`). Automatically detects CUDA if available.

### Keyboard Controls (While video is running)
- Press **`s`** to manually save the current Re-ID database.
- Press **`q`** to safely quit the window, clean up resources, and save the final database.
