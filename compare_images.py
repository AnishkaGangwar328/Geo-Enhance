from pathlib import Path
import tkinter as tk
from tkinter import filedialog

import numpy as np
import torch
import rasterio
import matplotlib.pyplot as plt

from restore_swinir import LightweightSwinIR

# -----------------------------------
# 1. Project paths
# -----------------------------------

ROOT = Path(__file__).resolve().parent

SAMPLES = ROOT / "samples"
OUTPUTS = ROOT / "outputs"
MODEL_PATH = ROOT / "models" / "GeoEnhance_SwinIR_4x_Final.pth"

OUTPUTS.mkdir(exist_ok=True)

# -----------------------------------
# 2. Choose satellite image
# -----------------------------------

root = tk.Tk()
root.withdraw()

selected_file = filedialog.askopenfilename(
    title="Select Sentinel-2 Satellite Image",
    initialdir=str(SAMPLES),
    filetypes=[
        ("GeoTIFF Images", "*.tif *.tiff")
    ]
)

root.destroy()

if not selected_file:
    print("No image selected.")
    raise SystemExit

image_path = Path(selected_file)

print("\nSelected image:", image_path.name)

# -----------------------------------
# 3. Read satellite image
# -----------------------------------

with rasterio.open(image_path) as src:
    lr = src.read()
    profile = src.profile.copy()
    transform = src.transform

if lr.shape[0] != 4:
    raise ValueError("Please select a 4-band Sentinel-2 GeoTIFF.")

if lr.shape[1:] != (130, 130):
    raise ValueError(
        f"Expected 130x130 input, got {lr.shape[1:]}"
    )

print("Original LR shape:", lr.shape)

# -----------------------------------
# 4. Load trained SwinIR model
# -----------------------------------

device = torch.device("cpu")

model = LightweightSwinIR(
    in_channels=4,
    dim=48,
    scale=4
).to(device)

checkpoint = torch.load(
    MODEL_PATH,
    map_location=device,
    weights_only=True
)

model.load_state_dict(checkpoint, strict=True)
model.eval()

print("Trained SwinIR model loaded!")

# -----------------------------------
# 5. Actual model inference
# -----------------------------------

lr_normalized = np.clip(
    lr.astype(np.float32) / 10000.0,
    0,
    1
)

input_tensor = torch.from_numpy(
    lr_normalized
).unsqueeze(0).to(device)

with torch.inference_mode():
    sr_tensor = model(input_tensor)

sr = sr_tensor.squeeze(0).cpu().numpy()
sr = np.clip(sr, 0, 1)

print("Generated SR shape:", sr.shape)

# -----------------------------------
# 6. Save generated SR GeoTIFF
# -----------------------------------

sr_uint16 = np.round(
    sr * 10000
).astype(np.uint16)

sr_path = OUTPUTS / f"{image_path.stem}_SwinIR_SR.tif"

profile.update(
    height=sr.shape[1],
    width=sr.shape[2],
    count=4,
    dtype="uint16",
    transform=transform * transform.scale(0.25, 0.25)
)

with rasterio.open(sr_path, "w", **profile) as dst:
    dst.write(sr_uint16)

print("SR image saved:", sr_path)

# -----------------------------------
# 7. Convert to RGB for display
# -----------------------------------

def to_rgb(image):
    return np.moveaxis(
        image[:3].astype(np.float32),
        0,
        -1
    )

lr_rgb = to_rgb(lr)
sr_rgb = to_rgb(sr_uint16)

# Same color stretch for fair comparison
combined = np.concatenate([
    lr_rgb.reshape(-1, 3),
    sr_rgb.reshape(-1, 3)
])

low = np.percentile(combined, 2, axis=0)
high = np.percentile(combined, 98, axis=0)

def display_rgb(image):
    return np.clip(
        (image - low) / np.maximum(high - low, 1e-6),
        0,
        1
    )

# -----------------------------------
# 8. Compact side-by-side comparison
# -----------------------------------

fig, axes = plt.subplots(
    1, 2,
    figsize=(10, 6.5)
)

axes[0].imshow(
    display_rgb(lr_rgb),
    interpolation="bicubic"
)

axes[0].set_title(
    "Original LR\n130 x 130",
    fontsize=8
)
axes[0].axis("off")

axes[1].imshow(
    display_rgb(sr_rgb),
    interpolation="nearest"
)

axes[1].set_title(
    "GeoEnhance SwinIR SR\n520 x 520",
    fontsize=8
)
axes[1].axis("off")

plt.tight_layout()

comparison_path = OUTPUTS / (
    f"{image_path.stem}_comparison.png"
)

plt.savefig(
    comparison_path,
    dpi=100,
    bbox_inches="tight"
)

plt.show()

print("\nComparison saved:", comparison_path)
print("SUCCESS: Real 4x SwinIR inference completed!")