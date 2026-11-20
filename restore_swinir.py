import torch
import torch.nn as nn
import torch.nn.functional as F

# ---------- WINDOW OPERATIONS ----------

def window_partition(x, window_size):
    B, H, W, C = x.shape
    x = x.view(
        B, H // window_size, window_size,
        W // window_size, window_size, C
    )
    return x.permute(0, 1, 3, 2, 4, 5).contiguous().view(
        -1, window_size * window_size, C
    )

def window_reverse(windows, window_size, H, W, B):
    C = windows.shape[-1]
    x = windows.view(
        B, H // window_size, W // window_size,
        window_size, window_size, C
    )
    return x.permute(0, 1, 3, 2, 4, 5).contiguous().view(
        B, H, W, C
    )

# ---------- SWIN TRANSFORMER BLOCK ----------

class SwinBlock(nn.Module):
    def __init__(self, dim=48, heads=4, window_size=8, shift_size=0):
        super().__init__()
        self.window_size = window_size
        self.shift_size = shift_size

        self.norm1 = nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(
            embed_dim=dim,
            num_heads=heads,
            batch_first=True
        )
        self.norm2 = nn.LayerNorm(dim)
        self.mlp = nn.Sequential(
            nn.Linear(dim, dim * 2),
            nn.GELU(),
            nn.Linear(dim * 2, dim)
        )

    def forward(self, x):
        B, C, H, W = x.shape
        ws = self.window_size

        shortcut = x
        x = x.permute(0, 2, 3, 1)
        x = self.norm1(x)

        pad_h = (ws - H % ws) % ws
        pad_w = (ws - W % ws) % ws
        x = F.pad(x, (0, 0, 0, pad_w, 0, pad_h))

        Hp, Wp = x.shape[1:3]

        if self.shift_size:
            x = torch.roll(
                x,
                shifts=(-self.shift_size, -self.shift_size),
                dims=(1, 2)
            )

        windows = window_partition(x, ws)

        attended, _ = self.attn(
            windows, windows, windows,
            need_weights=False
        )

        x = window_reverse(
            attended, ws, Hp, Wp, B
        )

        if self.shift_size:
            x = torch.roll(
                x,
                shifts=(self.shift_size, self.shift_size),
                dims=(1, 2)
            )

        x = x[:, :H, :W, :]
        x = x.permute(0, 3, 1, 2).contiguous()

        x = shortcut + x

        y = self.norm2(x.permute(0, 2, 3, 1))
        y = self.mlp(y).permute(0, 3, 1, 2).contiguous()

        return x + y

# ---------- LIGHTWEIGHT SWINIR ----------

class LightweightSwinIR(nn.Module):
    def __init__(self, in_channels=4, dim=48, scale=4):
        super().__init__()

        self.scale = scale

        self.conv_first = nn.Conv2d(
            in_channels, dim, 3, padding=1
        )

        self.blocks = nn.Sequential(
            SwinBlock(dim, shift_size=0),
            SwinBlock(dim, shift_size=4),
            SwinBlock(dim, shift_size=0),
            SwinBlock(dim, shift_size=4)
        )

        self.conv_after_body = nn.Conv2d(
            dim, dim, 3, padding=1
        )

        self.upsample = nn.Sequential(
            nn.Conv2d(dim, dim * scale * scale, 3, padding=1),
            nn.PixelShuffle(scale),
            nn.Conv2d(dim, in_channels, 3, padding=1)
        )

    def forward(self, x):
        features = self.conv_first(x)
        body = self.blocks(features)
        features = features + self.conv_after_body(body)

        # Residual prediction over bicubic interpolation
        base = F.interpolate(
            x,
            scale_factor=self.scale,
            mode="bicubic",
            align_corners=False
        )

        return base + self.upsample(features)

# ---------- INITIALIZE AND TEST ----------

device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

model = LightweightSwinIR().to(device)

parameters = sum(
    p.numel() for p in model.parameters()
)

print("========== SWINIR MODEL ==========")
print("Device:", device)
print("Parameters:", f"{parameters:,}")
print("Model size:", f"{parameters / 1e6:.2f} M")

# Use a small input for initial architecture verification
dummy = torch.randn(1, 4, 32, 32, device=device)

with torch.no_grad():
    output = model(dummy)

print("\nInput shape:", tuple(dummy.shape))
print("Output shape:", tuple(output.shape))

assert output.shape == (1, 4, 128, 128)

print("\nLIGHTWEIGHT SWINIR INITIALIZED SUCCESSFULLY!")

import torch
import torch.nn.functional as F

# Correct attention mask construction
def corrected_make_mask(self, H, W, device):
    ws = self.window_size
    shift = self.shift_size

    if shift == 0:
        return None

    img_mask = torch.zeros((1, H, W, 1), device=device)

    h_slices = (
        slice(0, -ws),
        slice(-ws, -shift),
        slice(-shift, None)
    )
    w_slices = (
        slice(0, -ws),
        slice(-ws, -shift),
        slice(-shift, None)
    )

    count = 0
    for h in h_slices:
        for w in w_slices:
            img_mask[:, h, w, :] = count
            count += 1

    mask_windows = window_partition(img_mask, ws).squeeze(-1)

    mask = (
        mask_windows.unsqueeze(1)
        - mask_windows.unsqueeze(2)
    )

    return mask.masked_fill(
        mask != 0, float("-inf")
    ).masked_fill(mask == 0, 0.0)


def corrected_forward(self, x):
    B, C, H, W = x.shape
    ws = self.window_size

    shortcut = x
    x = self.norm1(x.permute(0, 2, 3, 1))

    pad_h = (ws - H % ws) % ws
    pad_w = (ws - W % ws) % ws
    x = F.pad(x, (0, 0, 0, pad_w, 0, pad_h))

    Hp, Wp = x.shape[1:3]

    if self.shift_size:
        x = torch.roll(
            x,
            shifts=(-self.shift_size, -self.shift_size),
            dims=(1, 2)
        )

    windows = window_partition(x, ws)
    mask = self.make_mask(Hp, Wp, x.device)

    if mask is not None:
        heads = self.attn.num_heads

        # Each window needs one mask per attention head
        mask = mask.repeat_interleave(
            B * heads, dim=0
        )

    attended, _ = self.attn(
        windows, windows, windows,
        attn_mask=mask,
        need_weights=False
    )

    x = window_reverse(attended, ws, Hp, Wp, B)

    if self.shift_size:
        x = torch.roll(
            x,
            shifts=(self.shift_size, self.shift_size),
            dims=(1, 2)
        )

    x = x[:, :H, :W, :]
    x = x.permute(0, 3, 1, 2).contiguous()

    x = shortcut + x

    y = self.norm2(x.permute(0, 2, 3, 1))
    y = self.mlp(y).permute(0, 3, 1, 2).contiguous()

    return x + y


SwinBlock.make_mask = corrected_make_mask
SwinBlock.forward = corrected_forward

print("========== CORRECTED SWINIR TEST ==========")

model.eval()

with torch.no_grad():
    for size in [32, 130]:
        dummy = torch.randn(
            1, 4, size, size,
            device=device
        )

        output = model(dummy)

        print(
            f"Input: {tuple(dummy.shape)} "
            f"-> Output: {tuple(output.shape)}"
        )

        assert output.shape == (1, 4, size * 4, size * 4)
        assert torch.isfinite(output).all()

print("\nSWINIR ATTENTION MASK DIMENSIONS FIXED!")

import torch
import torch.nn.functional as F

def final_swin_forward(self, x):
    B, C, H, W = x.shape
    ws = self.window_size
    shift = self.shift_size

    shortcut = x
    x = self.norm1(x.permute(0, 2, 3, 1))

    pad_h = (ws - H % ws) % ws
    pad_w = (ws - W % ws) % ws

    x = F.pad(x, (0, 0, 0, pad_w, 0, pad_h))
    Hp, Wp = x.shape[1:3]

    # Track valid pixels separately from padded pixels
    valid = torch.zeros(
        (1, Hp, Wp, 1),
        dtype=torch.bool,
        device=x.device
    )
    valid[:, :H, :W, :] = True

    if shift:
        x = torch.roll(
            x, shifts=(-shift, -shift), dims=(1, 2)
        )
        valid = torch.roll(
            valid, shifts=(-shift, -shift), dims=(1, 2)
        )

    windows = window_partition(x, ws)
    valid_windows = window_partition(valid, ws).squeeze(-1)

    nW = valid_windows.shape[0]
    tokens = ws * ws

    # Block attention to padded keys
    padding_mask = (~valid_windows).unsqueeze(1).expand(
        nW, tokens, tokens
    )

    # Block artificial connections caused by cyclic shifts
    shift_mask = self.make_mask(Hp, Wp, x.device)

    if shift_mask is not None:
        blocked = padding_mask | torch.isneginf(shift_mask)
    else:
        blocked = padding_mask

    # Avoid fully masked attention rows for padded queries.
    # Their outputs are discarded after cropping.
    invalid_queries = (~valid_windows).unsqueeze(-1)
    blocked = blocked & (~invalid_queries)

    additive_mask = torch.zeros(
        (nW, tokens, tokens),
        dtype=x.dtype,
        device=x.device
    ).masked_fill(blocked, float("-inf"))

    # Correct order: batch -> window -> attention head
    heads = self.attn.num_heads
    additive_mask = (
        additive_mask.unsqueeze(0)
        .unsqueeze(2)
        .expand(B, nW, heads, tokens, tokens)
        .reshape(B * nW * heads, tokens, tokens)
    )

    attended, _ = self.attn(
        windows, windows, windows,
        attn_mask=additive_mask,
        need_weights=False
    )

    x = window_reverse(attended, ws, Hp, Wp, B)

    if shift:
        x = torch.roll(
            x, shifts=(shift, shift), dims=(1, 2)
        )

    x = x[:, :H, :W, :]
    x = x.permute(0, 3, 1, 2).contiguous()

    x = shortcut + x

    y = self.norm2(x.permute(0, 2, 3, 1))
    y = self.mlp(y).permute(0, 3, 1, 2).contiguous()

    return x + y


SwinBlock.forward = final_swin_forward

print("========== FINAL ATTENTION VALIDATION ==========")

model.eval()

with torch.no_grad():
    for batch_size, size in [(1, 32), (2, 64), (1, 130)]:
        dummy = torch.randn(
            batch_size, 4, size, size,
            device=device
        )

        output = model(dummy)

        expected = (batch_size, 4, size * 4, size * 4)

        assert tuple(output.shape) == expected
        assert torch.isfinite(output).all()

        print(
            f"Input {tuple(dummy.shape)} "
            f"-> Output {tuple(output.shape)} | PASS"
        )

print("\nFINAL SWINIR ATTENTION CHECK PASSED!")

