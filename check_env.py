"""
Phase 0: Environment & Hardware Diagnostic Script
Checks Python version, installed libraries, Git status, and webcam availability.
"""
import sys
import platform
import subprocess

def log(msg=""):
    print(msg, flush=True)

def check_python():
    log("=" * 50)
    log("1. PYTHON ENVIRONMENT")
    log("=" * 50)
    log(f"Python Version: {platform.python_version()}")
    log(f"Python Executable: {sys.executable}")
    log(f"Architecture: {platform.architecture()[0]}")
    log(f"Platform: {platform.system()} {platform.release()}")
    
    in_venv = sys.prefix != sys.base_prefix
    log(f"Virtual Environment Active: {'Yes' if in_venv else 'No (Running in Global Python)'}")
    
    major, minor = sys.version_info.major, sys.version_info.minor
    if major == 3 and minor in (10, 11):
        log("-> Status: Ideal Python version (3.10/3.11) for MediaPipe stability.")
    elif major == 3 and minor >= 12:
        log(f"-> Note: Python {major}.{minor} detected. We will check if MediaPipe/OpenCV packages load smoothly.")
    else:
        log(f"-> Warning: Python {major}.{minor} detected. Recommended: Python 3.10 or 3.11.")

def check_git():
    log("\n" + "=" * 50)
    log("2. GIT STATUS")
    log("=" * 50)
    try:
        git_ver = subprocess.check_output(["git", "--version"], text=True, stderr=subprocess.STDOUT).strip()
        log(f"Git Installed: {git_ver}")
        
        status = subprocess.run(["git", "status"], capture_output=True, text=True)
        if status.returncode == 0:
            log("Git Repository: Initialized (.git exists)")
        else:
            log("Git Repository: Not initialized yet (fatal: not a git repository)")
    except Exception as e:
        log(f"Git Check Error: {e}")

def check_libraries():
    log("\n" + "=" * 50)
    log("3. REQUIRED LIBRARIES CHECK")
    log("=" * 50)
    libraries = [
        ("numpy", "NumPy (Angle math & array operations)"),
        ("pandas", "Pandas (Workout data & CSV export)"),
        ("sklearn", "Scikit-learn (ML classification & metrics)"),
        ("streamlit", "Streamlit (Web dashboard & UI)"),
        ("cv2", "OpenCV / opencv-python (Video processing & camera capture)"),
        ("mediapipe", "MediaPipe (Pose estimation & landmark detection)"),
        ("matplotlib", "Matplotlib (Progress graphs & angle plots)"),
        ("gtts", "gTTS (Text-to-speech audio coaching)"),
    ]
    
    installed_count = 0
    missing = []
    
    for module_name, desc in libraries:
        try:
            mod = __import__(module_name)
            ver = getattr(mod, "__version__", "installed")
            log(f"[FOUND]   {desc}: v{ver}")
            installed_count += 1
        except ImportError:
            log(f"[MISSING] {desc}")
            missing.append(module_name)
            
    log(f"\nSummary: {installed_count}/{len(libraries)} required libraries installed.")
    return missing

def check_webcam():
    log("\n" + "=" * 50)
    log("4. WEBCAM HARDWARE CHECK")
    log("=" * 50)
    try:
        import cv2
        log("Testing camera index 0...")
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            log("[WARNING] Could not open camera at index 0. Either webcam is in use by another app or permissions are denied.")
        else:
            ret, frame = cap.read()
            if ret and frame is not None:
                h, w, c = frame.shape
                log(f"[SUCCESS] Camera index 0 is operational! Default resolution: {w}x{h}, Channels: {c}")
            else:
                log("[WARNING] Camera opened, but failed to grab a test frame.")
            cap.release()
    except ImportError:
        log("[SKIPPED] OpenCV (cv2) is not installed yet. Webcam check will run once OpenCV is installed.")
    except Exception as e:
        log(f"[ERROR] Camera check encountered an exception: {e}")

if __name__ == "__main__":
    check_python()
    check_git()
    check_libraries()
    check_webcam()
    log("\n" + "=" * 50)
    log("Phase 0 check complete.")
    log("=" * 50)
