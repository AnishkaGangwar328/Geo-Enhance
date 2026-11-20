from pathlib import Path

import numpy as np
import torch
import rasterio

from restore_swinir import LightweightSwinIR


ROOT = Path(__file__).resolve().parent
MODEL_PATH = ROOT / "models" / "GeoEnhance_SwinIR_4x_Final.pth"

DEVICE = torch.device("cpu")

_model = None


def get_model():
    global _model

    if _model is None:
        model = LightweightSwinIR(
            in_channels=4,
            dim=48,
            scale=4
        ).to(DEVICE)

        checkpoint = torch.load(
            MODEL_PATH,
            map_location=DEVICE,
            weights_only=True
        )

        model.load_state_dict(
            checkpoint,
            strict=True
        )

        model.eval()
        _model = model

        print("GeoEnhance trained SwinIR loaded!")

    return _model


def generate_sr(input_path, output_path):
    input_path = Path(input_path)
    output_path = Path(output_path)

    # Read 4-band satellite image
    with rasterio.open(input_path) as src:
        lr = src.read()
        profile = src.profile.copy()
        transform = src.transform

    if lr.shape != (4, 130, 130):
        raise ValueError(
            f"Expected 4x130x130 GeoTIFF, got {lr.shape}"
        )

    # Normalize satellite reflectance
    lr_normalized = np.clip(
        lr.astype(np.float32) / 10000.0,
        0.0,
        1.0
    )

    tensor = torch.from_numpy(
        lr_normalized
    ).unsqueeze(0).to(DEVICE)

    # Actual SwinIR inference
    model = get_model()

    with torch.inference_mode():
        sr_tensor = model(tensor)

    sr = sr_tensor.squeeze(0).cpu().numpy()
    sr = np.clip(sr, 0.0, 1.0)

    # Convert back to reflectance scale
    sr_uint16 = np.round(
        sr * 10000
    ).astype(np.uint16)

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    # Save 4-band GeoTIFF
    profile.update(
        driver="GTiff",
        height=sr.shape[1],
        width=sr.shape[2],
        count=4,
        dtype="uint16",
        transform=transform * transform.scale(0.25, 0.25)
    )

    with rasterio.open(output_path, "w", **profile) as dst:
        dst.write(sr_uint16)

    return output_path