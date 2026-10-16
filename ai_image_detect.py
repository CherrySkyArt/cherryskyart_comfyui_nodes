import json
import os
import zlib

import numpy as np
import torch
from PIL import Image

from comfy_api.latest import io

# PNG text chunks these tools write into their outputs.
GENERATOR_PNG_KEYS = {
    "parameters": "AUTOMATIC1111 / Forge",
    "prompt": "ComfyUI",
    "workflow": "ComfyUI",
    "sd-metadata": "InvokeAI",
    "invokeai_metadata": "InvokeAI",
    "invokeai_graph": "InvokeAI",
    "Dream": "InvokeAI (legacy)",
    "Software": "generic",
    "aigc": "generic AIGC tag",
}

GENERATOR_TEXT_MARKERS = [
    ("stable diffusion", "Stable Diffusion"),
    ("stable-diffusion", "Stable Diffusion"),
    ("comfyui", "ComfyUI"),
    ("automatic1111", "AUTOMATIC1111"),
    ("novelai", "NovelAI"),
    ("midjourney", "Midjourney"),
    ("dall-e", "DALL-E"),
    ("dalle", "DALL-E"),
    ("firefly", "Adobe Firefly"),
    ("adobe firefly", "Adobe Firefly"),
    ("flux", "FLUX"),
    ("imagen", "Google Imagen"),
    ("grok", "Grok / xAI"),
    ("nano banana", "Google Nano Banana"),
    ("seedream", "Seedream"),
    ("qwen-image", "Qwen-Image"),
    ("trainedmodel", "generic model tag"),
    ("c2pa", "C2PA manifest"),
    ("contentcredentials", "C2PA Content Credentials"),
]

# EXIF tags a real camera capture almost always carries.
CAMERA_EXIF_TAGS = [
    "Make",
    "Model",
    "DateTimeOriginal",
    "ExposureTime",
    "FNumber",
    "ISOSpeedRatings",
    "FocalLength",
]


def _read_file_bytes(path, max_bytes=80 * 1024 * 1024):
    size = os.path.getsize(path)
    if size > max_bytes:
        raise ValueError(f"file is {size} bytes, above the {max_bytes} byte limit")
    with open(path, "rb") as fh:
        return fh.read()


def _scan_png_chunks(data):
    """Walk PNG chunks, returning (text_pairs, chunk_types). Tolerant of truncation."""
    text_pairs = []
    chunk_types = []
    pos = 8
    while pos + 8 <= len(data):
        length = int.from_bytes(data[pos:pos + 4], "big")
        ctype = data[pos + 4:pos + 8].decode("latin-1", "replace")
        body = data[pos + 8:pos + 8 + length]
        chunk_types.append(ctype)
        if ctype in ("tEXt", "iTXt", "zTXt"):
            try:
                if ctype == "tEXt":
                    key, _, val = body.partition(b"\x00")
                    text_pairs.append((key.decode("latin-1", "replace"), val.decode("latin-1", "replace")))
                elif ctype == "zTXt":
                    key, _, rest = body.partition(b"\x00")
                    val = zlib.decompress(rest[1:]) if len(rest) > 1 else b""
                    text_pairs.append((key.decode("latin-1", "replace"), val.decode("utf-8", "replace")))
                else:
                    key, _, rest = body.partition(b"\x00")
                    # iTXt: compression flag, method, lang, translated key, then text.
                    comp_flag = rest[0:1]
                    rest = rest[2:]
                    _, _, rest = rest.partition(b"\x00")
                    _, _, val = rest.partition(b"\x00")
                    if comp_flag == b"\x01":
                        val = zlib.decompress(val)
                    text_pairs.append((key.decode("latin-1", "replace"), val.decode("utf-8", "replace")))
            except Exception:
                pass
        if ctype == "IEND":
            break
        pos += 12 + length
    return text_pairs, chunk_types


def _check_generator_metadata(data, notes):
    """Explicit generator signatures. The strongest single signal when present."""
    hits = []

    if data[:8] == b"\x89PNG\r\n\x1a\n":
        text_pairs, chunk_types = _scan_png_chunks(data)
        for key, val in text_pairs:
            if key in GENERATOR_PNG_KEYS:
                hits.append(f"PNG text chunk '{key}' ({GENERATOR_PNG_KEYS[key]})")
            blob = f"{key} {val}".lower()
            for marker, label in GENERATOR_TEXT_MARKERS:
                if marker in blob:
                    hits.append(f"PNG text mentions {label}")
        if "caBX" in chunk_types:
            hits.append("PNG carries a C2PA manifest chunk (caBX)")

    # JPEG/WebP and anything else: scan raw bytes outside the pixel data for
    # marker strings. Cheap and catches XMP, APP segments and embedded JSON.
    head = data[:262144].lower()
    tail = data[-262144:].lower() if len(data) > 262144 else b""
    for marker, label in GENERATOR_TEXT_MARKERS:
        needle = marker.encode("latin-1")
        if needle in head or needle in tail:
            hits.append(f"raw bytes mention {label}")

    hits = sorted(set(hits))
    if hits:
        for h in hits[:8]:
            notes.append(f"[generator] {h}")
        c2pa = any("C2PA" in h or "Content Credentials" in h for h in hits)
        # A C2PA manifest alone is ambiguous: cameras sign captures with it too.
        return (0.45, 0.35) if c2pa and len(hits) == 1 else (0.95, 0.9)

    notes.append("[generator] no generator signature found")
    return (0.0, 0.0)


def _check_camera_exif(data, notes):
    """Absence of camera EXIF is weak evidence; a full plausible set is weak counter-evidence."""
    try:
        import piexif
    except ImportError:
        notes.append("[exif] piexif not installed, EXIF check skipped")
        return (0.0, 0.0)

    try:
        exif = piexif.load(data)
    except Exception:
        notes.append("[exif] no readable EXIF block")
        return (0.5, 0.25)

    present = []
    for tag in CAMERA_EXIF_TAGS:
        for ifd in ("0th", "Exif"):
            tag_id = getattr(piexif.ImageIFD, tag, None) if ifd == "0th" else getattr(piexif.ExifIFD, tag, None)
            if tag_id is not None and tag_id in exif.get(ifd, {}):
                present.append(tag)
                break

    make = exif.get("0th", {}).get(piexif.ImageIFD.Make, b"")
    model = exif.get("0th", {}).get(piexif.ImageIFD.Model, b"")
    make = make.decode("latin-1", "replace").strip("\x00 ") if isinstance(make, bytes) else str(make)
    model = model.decode("latin-1", "replace").strip("\x00 ") if isinstance(model, bytes) else str(model)

    ratio = len(present) / len(CAMERA_EXIF_TAGS)
    missing = [t for t in CAMERA_EXIF_TAGS if t not in present]

    if ratio >= 0.85:
        notes.append(f"[exif] complete camera EXIF set (Make={make!r}, Model={model!r}) - but EXIF is trivially forged")
        return (0.15, 0.2)
    if ratio >= 0.4:
        notes.append(f"[exif] partial camera EXIF, missing {', '.join(missing)}")
        return (0.5, 0.15)

    notes.append(f"[exif] little or no camera EXIF (missing {', '.join(missing)})")
    return (0.65, 0.25)


def _check_jpeg_tables(data, notes):
    """Most camera JPEGs use device-specific quantization tables; many tools use the
    standard IJG tables. A standard table on a claimed camera photo is suspicious."""
    if data[:2] != b"\xff\xd8":
        notes.append("[jpeg] not a JPEG, quantization check skipped")
        return (0.0, 0.0)

    try:
        import io as _io
        img = Image.open(_io.BytesIO(data))
        tables = getattr(img, "quantization", None)
    except Exception:
        notes.append("[jpeg] could not read quantization tables")
        return (0.0, 0.0)

    if not tables:
        notes.append("[jpeg] no quantization tables found")
        return (0.0, 0.0)

    luma = list(tables.get(0, []))
    if not luma:
        notes.append("[jpeg] no luma quantization table")
        return (0.0, 0.0)

    # Pillow/libjpeg at common quality settings produce a highly regular table.
    # Camera ISPs typically ship irregular, hand-tuned ones.
    arr = np.array(luma, dtype=np.float64)
    dc = arr[0] if arr.size else 0
    spread = float(np.std(arr))

    if dc <= 3 and spread < 12:
        notes.append(f"[jpeg] very low quantization (DC={dc:.0f}) - high-quality re-encode, not a camera default")
        return (0.6, 0.3)
    if spread < 6:
        notes.append(f"[jpeg] unusually flat quantization table (std={spread:.1f}) - suggests a software encoder")
        return (0.6, 0.3)

    notes.append(f"[jpeg] quantization table looks irregular (DC={dc:.0f}, std={spread:.1f}), consistent with a camera encoder")
    return (0.3, 0.15)


def _tensor_to_gray(image):
    arr = image[0].cpu().numpy() if image.ndim == 4 else image.cpu().numpy()
    if arr.ndim == 3 and arr.shape[-1] >= 3:
        arr = arr[..., :3] @ np.array([0.299, 0.587, 0.114])
    elif arr.ndim == 3:
        arr = arr[..., 0]
    return arr.astype(np.float64)


def _check_frequency(gray, notes):
    """Diffusion upsamplers leave a weak periodic signature and an unusually clean
    high-frequency noise floor. This is the softest check here - treat as a hint."""
    if gray.size < 64 * 64:
        notes.append("[frequency] image too small for frequency analysis")
        return (0.0, 0.0)

    h, w = gray.shape
    side = min(h, w, 1024)
    crop = gray[:side, :side]

    spec = np.abs(np.fft.fftshift(np.fft.fft2(crop - crop.mean())))
    spec = np.log1p(spec)

    cy, cx = side // 2, side // 2
    yy, xx = np.ogrid[:side, :side]
    radius = np.sqrt((yy - cy) ** 2 + (xx - cx) ** 2)
    high = spec[radius > side * 0.35]
    mid = spec[(radius > side * 0.12) & (radius <= side * 0.35)]

    if high.size == 0 or mid.size == 0:
        return (0.0, 0.0)

    ratio = float(high.mean() / (mid.mean() + 1e-9))

    # Residual high-frequency energy: real sensors carry read noise, so a very
    # smooth residual suggests generated or heavily denoised content.
    resid = crop[1:, 1:] - 0.5 * crop[:-1, 1:] - 0.5 * crop[1:, :-1]
    resid_std = float(np.std(resid))

    score = 0.0
    if ratio < 0.82:
        notes.append(f"[frequency] high-frequency energy falls off fast (ratio={ratio:.2f})")
        score = max(score, 0.55)
    if resid_std < 0.004:
        notes.append(f"[frequency] very low sensor-noise residual (std={resid_std:.4f})")
        score = max(score, 0.6)

    if score == 0.0:
        notes.append(f"[frequency] spectrum unremarkable (ratio={ratio:.2f}, noise={resid_std:.4f})")
        return (0.35, 0.1)

    return (score, 0.15)


class AIImageDetect(io.ComfyNode):
    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="AIImageDetect_Cherry",
            display_name="AI Image Detect 🍒",
            category="image",
            search_aliases=["cherry", "ai detect", "synthetic", "forensics", "exif", "c2pa", "provenance"],
            description=(
                "Heuristic check for whether an image was AI-generated. Combines generator metadata, "
                "camera EXIF completeness, JPEG quantization fingerprinting and frequency analysis into a "
                "0-1 score. Pass image_path for the full suite; an IMAGE tensor alone only supports the "
                "pixel checks, since decoding strips metadata. This is a triage signal, not proof."
            ),
            inputs=[
                io.String.Input(
                    "image_path",
                    default="",
                    tooltip="Path to the original file on disk. Required for metadata, EXIF and JPEG checks.",
                    optional=True,
                ),
                io.Image.Input(
                    "image",
                    tooltip="Decoded image, used for the frequency checks. Optional if image_path is set.",
                    optional=True,
                ),
                io.Float.Input(
                    "threshold",
                    default=0.5,
                    min=0.0,
                    max=1.0,
                    step=0.01,
                    tooltip="Score at or above which is_ai_generated becomes true.",
                ),
            ],
            outputs=[
                io.Boolean.Output(display_name="is_ai_generated"),
                io.Float.Output(display_name="score"),
                io.String.Output(display_name="report"),
            ],
        )

    @classmethod
    def execute(cls, image_path="", image=None, threshold=0.5) -> io.NodeOutput:
        notes = []
        weighted = []

        path = (image_path or "").strip().strip('"')
        data = None
        if path:
            if not os.path.isfile(path):
                notes.append(f"[input] path not found: {path}")
            else:
                try:
                    data = _read_file_bytes(path)
                except Exception as exc:
                    notes.append(f"[input] could not read file: {exc}")

        if data is not None:
            for check, weight in (
                (_check_generator_metadata, 1.0),
                (_check_camera_exif, 1.0),
                (_check_jpeg_tables, 1.0),
            ):
                score, confidence = check(data, notes)
                if confidence > 0:
                    weighted.append((score, confidence * weight))
        else:
            notes.append("[input] no file bytes available, metadata and EXIF checks skipped")

        gray = None
        if image is not None:
            gray = _tensor_to_gray(image)
        elif data is not None:
            try:
                import io as _io
                pil = Image.open(_io.BytesIO(data)).convert("L")
                gray = np.asarray(pil, dtype=np.float64) / 255.0
            except Exception as exc:
                notes.append(f"[frequency] could not decode pixels: {exc}")

        if gray is not None:
            score, confidence = _check_frequency(gray, notes)
            if confidence > 0:
                weighted.append((score, confidence))
        else:
            notes.append("[frequency] no pixels available, frequency check skipped")

        if weighted:
            total_conf = sum(c for _, c in weighted)
            final = sum(s * c for s, c in weighted) / total_conf
        else:
            final = 0.0
            notes.append("[result] no checks could run, score is meaningless")

        verdict = "LIKELY AI" if final >= threshold else "NO STRONG AI SIGNAL"
        report = "\n".join(
            [f"verdict: {verdict}  score={final:.3f}  threshold={threshold:.2f}", ""]
            + notes
            + [
                "",
                "Heuristic only. Metadata is trivially forged and stripped by most upload "
                "pipelines, so a low score is not evidence an image is a real photograph.",
            ]
        )

        return io.NodeOutput(final >= threshold, float(final), report)
