from pathlib import Path
import torch
import rasterio
import numpy as np

from restore_swinir import LightweightSwinIR

ROOT = Path(__file__).resolve().parent

MODEL_PATH = ROOT / "models" / "GeoEnhance_SwinIR_4x_Final.pth"
IMAGE_PATH = ROOT / "samples" / "test_sample_5666_LR.tif"

device = torch.device("cpu")

# 1. Initialize architecture
model = LightweightSwinIR(
    in_channels=4,
    dim=48,
    scale=4
).to(device)

# 2. Load trained weights
state_dict = torch.load(
    MODEL_PATH,
    map_location=device,
    weights_only=True
)

model.load_state_dict(state_dict, strict=True)
model.eval()

print("Model loaded successfully!")

# 3. Read original LR satellite image
with rasterio.open(IMAGE_PATH) as src:
    lr = src.read().astype(np.float32)

print("Original LR shape:", lr.shape)

# 4. Normalize reflectance
lr = np.clip(lr / 10000.0, 0.0, 1.0)

# 5. Convert to PyTorch tensor
input_tensor = torch.from_numpy(lr).unsqueeze(0).to(device)

print("Input tensor:", tuple(input_tensor.shape))

# 6. Run actual SwinIR inference
with torch.inference_mode():
    sr_tensor = model(input_tensor)

print("Generated SR tensor:", tuple(sr_tensor.shape))

assert sr_tensor.shape == (1, 4, 520, 520), (
    f"Unexpected output shape: {sr_tensor.shape}"
)

# 7. Save generated SR GeoTIFF
sr = sr_tensor.squeeze(0).cpu().numpy()
sr = np.clip(sr, 0, 1)
sr_uint16 = np.round(sr * 10000).astype(np.uint16)

output_dir = ROOT / "outputs"
output_dir.mkdir(exist_ok=True)

output_path = output_dir / "test_5666_SwinIR_SR.tif"

with rasterio.open(IMAGE_PATH) as src:
    profile = src.profile.copy()
    profile.update(
        height=520,
        width=520,
        transform=src.transform * src.transform.scale(0.25, 0.25),
        dtype="uint16"
    )

with rasterio.open(output_path, "w", **profile) as dst:
    dst.write(sr_uint16)

print("Saved:", output_path)
print("SUCCESS: 4x super-resolution generated!")