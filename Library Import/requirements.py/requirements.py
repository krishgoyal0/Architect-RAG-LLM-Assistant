import subprocess
import sys

# List of required packages with versions
requirements = [
    "transformers==4.44.2",
    "torch>=2.0.0",
    "pymupdf==1.24.10",
    "pytesseract==0.3.10",
    "Pillow==10.4.0"
    "sentence_transformers==5.1.0"
]

def install(package):
    """Install a package via pip."""
    subprocess.check_call([sys.executable, "-m", "pip", "install", package])

if __name__ == "__main__":
    print("[INFO] Installing required packages...")
    for package in requirements:
        print(f"[INSTALL] {package}")
        install(package)
    print("[DONE] All packages installed successfully!")
