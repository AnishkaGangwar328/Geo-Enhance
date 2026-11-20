from pathlib import Path
from uuid import uuid4

import numpy as np
import rasterio
from PIL import Image

from flask import (
    Flask,
    jsonify,
    render_template,
    request,
    send_from_directory
)

from werkzeug.exceptions import RequestEntityTooLarge
from werkzeug.utils import secure_filename

from inference import generate_sr


# ==========================================
# PROJECT DIRECTORIES
# ==========================================

ROOT = Path(__file__).resolve().parent

UPLOADS = ROOT / "uploads"
OUTPUTS = ROOT / "outputs"

UPLOADS.mkdir(parents=True, exist_ok=True)
OUTPUTS.mkdir(parents=True, exist_ok=True)


# ==========================================
# FLASK APPLICATION
# ==========================================

app = Flask(__name__)

app.config["MAX_CONTENT_LENGTH"] = 25 * 1024 * 1024


# ==========================================
# IMAGE PREVIEW GENERATION
# ==========================================

def make_previews(lr_path, sr_path, job_id):

    with rasterio.open(lr_path) as src:
        lr = src.read([1, 2, 3]).astype(np.float32)

    with rasterio.open(sr_path) as src:
        sr = src.read([1, 2, 3]).astype(np.float32)

    # Convert CHW -> HWC
    lr = np.moveaxis(lr, 0, -1)
    sr = np.moveaxis(sr, 0, -1)

    # Same brightness/contrast settings for
    # original and super-resolved images
    combined = np.concatenate(
        [
            lr.reshape(-1, 3),
            sr.reshape(-1, 3)
        ],
        axis=0
    )

    low = np.percentile(combined, 2, axis=0)
    high = np.percentile(combined, 98, axis=0)

    def normalize(image):

        normalized = np.clip(
            (image - low) /
            np.maximum(high - low, 1e-6),
            0.0,
            1.0
        )

        rgb = np.round(
            normalized * 255
        ).astype(np.uint8)

        return Image.fromarray(rgb)

    lr_name = f"{job_id}_LR.png"
    sr_name = f"{job_id}_SR.png"

    normalize(lr).save(OUTPUTS / lr_name)
    normalize(sr).save(OUTPUTS / sr_name)

    return lr_name, sr_name


# ==========================================
# HOME PAGE
# ==========================================

@app.route("/")
def home():

    return render_template("index.html")


# ==========================================
# IMAGE ENHANCEMENT API
# ==========================================

@app.route("/api/enhance", methods=["POST"])
def enhance():

    # Matches script.js:
    # formData.append("file", currentFile)

    if "file" not in request.files:

        return jsonify(
            error="Please upload a GeoTIFF image."
        ), 400

    uploaded = request.files["file"]

    if not uploaded.filename:

        return jsonify(
            error="No file selected."
        ), 400

    filename = secure_filename(uploaded.filename)

    if not filename:

        return jsonify(
            error="Invalid filename."
        ), 400

    if Path(filename).suffix.lower() not in (
        ".tif",
        ".tiff"
    ):

        return jsonify(
            error=(
                "Only 4-band Sentinel-2 "
                "GeoTIFF files are supported."
            )
        ), 400

    # Unique job ID prevents file collisions
    job_id = uuid4().hex

    lr_path = UPLOADS / f"{job_id}_LR.tif"

    sr_path = OUTPUTS / (
        f"{job_id}_SwinIR_SR.tif"
    )

    try:

        # Save uploaded image
        uploaded.save(lr_path)

        # ==================================
        # VALIDATE INPUT IMAGE
        # ==================================

        with rasterio.open(lr_path) as src:

            if (
                src.count != 4
                or src.width != 130
                or src.height != 130
            ):

                return jsonify(
                    error=(
                        "Expected a 4-band, "
                        "130×130 GeoTIFF image."
                    )
                ), 400

            if src.dtypes[0] != "uint16":

                return jsonify(
                    error=(
                        "Expected a uint16 "
                        "Sentinel-2 GeoTIFF."
                    )
                ), 400

        # ==================================
        # ACTUAL TRAINED SWINIR INFERENCE
        # ==================================

        print(
            f"[GeoEnhance] Processing: {filename}",
            flush=True
        )

        generate_sr(
            input_path=lr_path,
            output_path=sr_path
        )

        print(
            "[GeoEnhance] 4x SR generated!",
            flush=True
        )

        # ==================================
        # GENERATE RGB PREVIEWS
        # ==================================

        lr_preview, sr_preview = make_previews(
            lr_path,
            sr_path,
            job_id
        )

        # ==================================
        # RETURN FRONTEND RESPONSE
        # ==================================

        return jsonify(

            success=True,

            filename=filename,

            lr_preview=(
                f"/outputs/{lr_preview}"
            ),

            sr_preview=(
                f"/outputs/{sr_preview}"
            ),

            # Matches script.js download_url
            download_url=(
                f"/outputs/{sr_path.name}"
            ),

            input_resolution="130 × 130",

            output_resolution="520 × 520",

            scale="4×"

        ), 200

    except rasterio.errors.RasterioIOError:

        return jsonify(
            error=(
                "Invalid or unreadable GeoTIFF file."
            )
        ), 400

    except Exception:

        app.logger.exception(
            "GeoEnhance super-resolution failed"
        )

        return jsonify(
            error=(
                "Super-resolution processing failed. "
                "Please check the Flask terminal."
            )
        ), 500


# ==========================================
# DOWNLOAD / DISPLAY OUTPUTS
# ==========================================

@app.route("/outputs/<path:filename>")
def serve_output(filename):

    return send_from_directory(
        OUTPUTS,
        filename
    )


# ==========================================
# UPLOAD SIZE ERROR
# ==========================================

@app.errorhandler(RequestEntityTooLarge)
def handle_large_file(error):

    return jsonify(
        error="File exceeds the 25 MB upload limit."
    ), 413


# ==========================================
# START SERVER
# ==========================================

if __name__ == "__main__":

    print("\n" + "=" * 45)
    print("GEOENHANCE — EARTH OBSERVATION SYSTEM")
    print("=" * 45)

    print("AI Model: Lightweight SwinIR")
    print("Upscaling: 4x")
    print("Input: 4 bands, 130 x 130")
    print("Output: 4 bands, 520 x 520")

    print("\nWebsite:")
    print("http://127.0.0.1:5000")

    print("=" * 45 + "\n")

    app.run(
        debug=False,
        host="127.0.0.1",
        port=5000
    )