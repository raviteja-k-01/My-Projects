from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from typing import List
import os
import numpy as np
import imageio.v3 as iio
import traceback

# EXR
import OpenEXR
import Imath

app = FastAPI()

# -------------------------------
# CORS
# -------------------------------
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# -------------------------------
# DIRECTORIES
# -------------------------------
UPLOAD_DIR = "uploads"
OUTPUT_DIR = "frames"

os.makedirs(UPLOAD_DIR, exist_ok=True)
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Serve frames
app.mount("/frames", StaticFiles(directory=OUTPUT_DIR), name="frames")


# -------------------------------
# ROOT
# -------------------------------
@app.get("/")
def read_root():
    return {"status": "backend running"}


# -------------------------------
# EXR READER
# -------------------------------
def read_exr(file_path):
    exr = OpenEXR.InputFile(file_path)
    header = exr.header()

    dw = header["dataWindow"]
    width = dw.max.x - dw.min.x + 1
    height = dw.max.y - dw.min.y + 1

    FLOAT = Imath.PixelType(Imath.PixelType.FLOAT)

    try:
        r = np.frombuffer(exr.channel("R", FLOAT), dtype=np.float32)
        g = np.frombuffer(exr.channel("G", FLOAT), dtype=np.float32)
        b = np.frombuffer(exr.channel("B", FLOAT), dtype=np.float32)
    except:
        raise Exception("EXR missing RGB channels")

    r.shape = (height, width)
    g.shape = (height, width)
    b.shape = (height, width)

    return np.stack([r, g, b], axis=-1)


# -------------------------------
# TONEMAP
# -------------------------------
def tonemap(img):
    img = np.nan_to_num(img, nan=0.0, posinf=1.0, neginf=0.0)
    img = np.clip(img, 0, None)
    img = img / (1.0 + img)
    img = np.power(img, 1 / 2.2)
    img = np.clip(img, 0, 1)
    return (img * 255).astype(np.uint8)


# -------------------------------
# STANDARD IMAGE
# -------------------------------
def process_standard_image(file_path):
    img = iio.imread(file_path)

    if img.dtype != np.uint8:
        img = np.clip(img, 0, 1)
        img = (img * 255).astype(np.uint8)

    return img


# -------------------------------
# UPLOAD
# -------------------------------
@app.post("/upload/")
async def upload_files(files: List[UploadFile] = File(...)):
    results = []

    # ✅ 1. CLEAR OLD FRAMES
    for f in os.listdir(OUTPUT_DIR):
        try:
            os.remove(os.path.join(OUTPUT_DIR, f))
        except:
            pass

    for i, file in enumerate(files):
        try:
            file_path = os.path.join(UPLOAD_DIR, file.filename)

            # Save file
            with open(file_path, "wb") as f:
                f.write(await file.read())

            # Process
            if file.filename.lower().endswith(".exr"):
                img = read_exr(file_path)
                img = tonemap(img)
            else:
                img = process_standard_image(file_path)

            # ✅ 2. CONSISTENT NAMING (CRITICAL)
            output_name = f"frame_{i:04d}.png"
            output_path = os.path.join(OUTPUT_DIR, output_name)

            iio.imwrite(output_path, img)

            results.append(output_name)

        except Exception as e:
            traceback.print_exc()
            results.append(f"ERROR: {file.filename}")

    # ✅ 3. SORT (extra safety)
    results = sorted(results)

    return {
        "status": "done",
        "frames": results
    }