"""Red-team counterpart to ai_image_detect: writes plausible camera EXIF onto an
image so the detector's metadata layer can be tested against forged input.

Scope note: this writes metadata only. It does not attempt to forge the
pixel-level signals (sensor noise/PRNU, device quantization tables, ISP
artifacts) that actual image forensics relies on. If a detector can be fooled by
this node alone, that is the finding: it is leaning too hard on metadata.
"""

import io as _io
import os
import random
from datetime import datetime, timedelta
from fractions import Fraction

import numpy as np
from PIL import Image

from comfy_api.latest import io

# make, model, lens, software, focal_mm, f_number, iso_choices, sensor_width_mm
DEVICE_PRESETS = {
    "iPhone 15 Pro (main)": ("Apple", "iPhone 15 Pro", "iPhone 15 Pro back triple camera 6.765mm f/1.78", "17.4.1", 6.765, 1.78, [40, 50, 64, 80, 100, 125], 9.8),
    "iPhone 14": ("Apple", "iPhone 14", "iPhone 14 back dual camera 5.7mm f/1.5", "16.6", 5.7, 1.5, [32, 40, 50, 64, 80], 9.0),
    "iPhone 13 Pro": ("Apple", "iPhone 13 Pro", "iPhone 13 Pro back triple camera 5.7mm f/1.5", "15.5", 5.7, 1.5, [32, 50, 64, 100], 9.0),
    "Pixel 8 Pro": ("Google", "Pixel 8 Pro", "Pixel 8 Pro back camera 6.9mm f/1.68", "HDR+ 1.0.paradise.2023", 6.9, 1.68, [47, 55, 68, 92, 120], 9.6),
    "Pixel 7": ("Google", "Pixel 7", "Pixel 7 back camera 6.81mm f/1.85", "HDR+ 1.0.540864448", 6.81, 1.85, [50, 61, 79, 101], 9.6),
    "Samsung Galaxy S23": ("samsung", "SM-S911B", "Samsung Galaxy S23 Rear Camera", "S911BXXU2AWF1", 5.4, 1.8, [50, 64, 80, 125, 160], 8.0),
    "Samsung Galaxy S21": ("samsung", "SM-G991B", "Samsung Galaxy S21 Rear Camera", "G991BXXU3AUme", 5.4, 1.8, [50, 80, 125, 200], 8.0),
    "Canon EOS R6": ("Canon", "Canon EOS R6", "RF24-105mm F4 L IS USM", "Firmware Version 1.8.2", 50.0, 4.0, [100, 200, 400, 800, 1600], 36.0),
    "Sony A7 III": ("SONY", "ILCE-7M3", "FE 24-70mm F2.8 GM", "ILCE-7M3 v4.01", 35.0, 2.8, [100, 200, 320, 640, 1250], 35.6),
}

EXPOSURE_CHOICES = [
    (1, 60), (1, 80), (1, 100), (1, 120), (1, 125),
    (1, 160), (1, 200), (1, 250), (1, 320), (1, 400), (1, 500),
]


def _rational(value, denom=1000):
    """piexif wants (numerator, denominator) integer pairs."""
    frac = Fraction(value).limit_denominator(denom)
    return (frac.numerator, frac.denominator)


def _gps_rational(deg):
    deg = abs(deg)
    d = int(deg)
    m_full = (deg - d) * 60
    m = int(m_full)
    s = (m_full - m) * 60
    return [(d, 1), (m, 1), (int(round(s * 1000)), 1000)]


def _build_exif(preset_name, width, height, when, gps_lat, gps_lon, seed):
    import piexif

    make, model, lens, software, focal, fnum, iso_choices, _sensor = DEVICE_PRESETS[preset_name]
    rng = random.Random(seed)

    iso = rng.choice(iso_choices)
    exp_num, exp_den = rng.choice(EXPOSURE_CHOICES)
    stamp = when.strftime("%Y:%m:%d %H:%M:%S")
    sub_sec = f"{rng.randint(0, 999):03d}"

    zeroth = {
        piexif.ImageIFD.Make: make.encode(),
        piexif.ImageIFD.Model: model.encode(),
        piexif.ImageIFD.Software: software.encode(),
        piexif.ImageIFD.DateTime: stamp.encode(),
        piexif.ImageIFD.Orientation: 1,
        piexif.ImageIFD.XResolution: (72, 1),
        piexif.ImageIFD.YResolution: (72, 1),
        piexif.ImageIFD.ResolutionUnit: 2,
        piexif.ImageIFD.YCbCrPositioning: 1,
    }

    exif = {
        piexif.ExifIFD.DateTimeOriginal: stamp.encode(),
        piexif.ExifIFD.DateTimeDigitized: stamp.encode(),
        piexif.ExifIFD.SubSecTimeOriginal: sub_sec.encode(),
        piexif.ExifIFD.ExposureTime: (exp_num, exp_den),
        piexif.ExifIFD.FNumber: _rational(fnum, 100),
        piexif.ExifIFD.ExposureProgram: 2,
        piexif.ExifIFD.ISOSpeedRatings: iso,
        piexif.ExifIFD.ExifVersion: b"0232",
        piexif.ExifIFD.ShutterSpeedValue: _rational(np.log2(exp_den / exp_num), 100),
        piexif.ExifIFD.ApertureValue: _rational(2 * np.log2(fnum), 100),
        piexif.ExifIFD.BrightnessValue: _rational(rng.uniform(0.5, 8.0), 100),
        piexif.ExifIFD.ExposureBiasValue: (0, 1),
        piexif.ExifIFD.MeteringMode: 5,
        piexif.ExifIFD.Flash: 16,
        piexif.ExifIFD.FocalLength: _rational(focal, 1000),
        piexif.ExifIFD.ColorSpace: 1,
        piexif.ExifIFD.PixelXDimension: width,
        piexif.ExifIFD.PixelYDimension: height,
        piexif.ExifIFD.SensingMethod: 2,
        piexif.ExifIFD.SceneType: b"\x01",
        piexif.ExifIFD.ExposureMode: 0,
        piexif.ExifIFD.WhiteBalance: 0,
        piexif.ExifIFD.DigitalZoomRatio: (1, 1),
        piexif.ExifIFD.SceneCaptureType: 0,
        piexif.ExifIFD.LensMake: make.encode(),
        piexif.ExifIFD.LensModel: lens.encode(),
    }

    gps = {}
    if gps_lat != 0.0 or gps_lon != 0.0:
        gps = {
            piexif.GPSIFD.GPSVersionID: (2, 3, 0, 0),
            piexif.GPSIFD.GPSLatitudeRef: (b"N" if gps_lat >= 0 else b"S"),
            piexif.GPSIFD.GPSLatitude: _gps_rational(gps_lat),
            piexif.GPSIFD.GPSLongitudeRef: (b"E" if gps_lon >= 0 else b"W"),
            piexif.GPSIFD.GPSLongitude: _gps_rational(gps_lon),
            piexif.GPSIFD.GPSAltitudeRef: 0,
            piexif.GPSIFD.GPSAltitude: _rational(rng.uniform(5, 200), 100),
            piexif.GPSIFD.GPSTimeStamp: [(when.hour, 1), (when.minute, 1), (when.second, 1)],
            piexif.GPSIFD.GPSDateStamp: when.strftime("%Y:%m:%d").encode(),
        }

    return piexif.dump({"0th": zeroth, "Exif": exif, "GPS": gps, "1st": {}, "thumbnail": None})


class ExifSpoof(io.ComfyNode):
    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="ExifSpoof_Cherry",
            display_name="EXIF Spoof (red team) 🍒",
            category="image",
            search_aliases=["cherry", "exif", "metadata", "spoof", "red team", "camera metadata"],
            description=(
                "Writes plausible camera EXIF (make, model, lens, exposure, ISO, timestamp, optional GPS) "
                "onto an image and saves it as JPEG, for testing whether a detector leans too hard on "
                "metadata. Metadata only - it does not forge sensor noise, device quantization tables or "
                "ISP artifacts, so real pixel-level forensics still sees through it."
            ),
            inputs=[
                io.Image.Input("image", tooltip="Image to tag."),
                io.Combo.Input(
                    "device",
                    options=list(DEVICE_PRESETS.keys()),
                    default="iPhone 15 Pro (main)",
                    tooltip="Device profile to imitate.",
                ),
                io.String.Input(
                    "output_path",
                    default="",
                    tooltip="Where to write the JPEG. Leave blank to skip writing and only return the report.",
                    optional=True,
                ),
                io.Int.Input(
                    "jpeg_quality",
                    default=95,
                    min=1,
                    max=100,
                    step=1,
                    tooltip="JPEG quality. Note this uses the standard encoder, not the device's own tables.",
                ),
                io.String.Input(
                    "datetime_original",
                    default="",
                    tooltip="Capture time as 'YYYY:MM:DD HH:MM:SS'. Blank picks a random recent time.",
                    optional=True,
                ),
                io.Float.Input(
                    "gps_latitude",
                    default=0.0,
                    min=-90.0,
                    max=90.0,
                    step=0.000001,
                    tooltip="Latitude. Leave both GPS fields at 0 to omit location entirely.",
                ),
                io.Float.Input(
                    "gps_longitude",
                    default=0.0,
                    min=-180.0,
                    max=180.0,
                    step=0.000001,
                    tooltip="Longitude. Leave both GPS fields at 0 to omit location entirely.",
                ),
                io.Int.Input(
                    "seed",
                    default=0,
                    min=0,
                    max=0xFFFFFFFF,
                    tooltip="Seeds the randomised exposure/ISO/subsecond values.",
                ),
            ],
            outputs=[
                io.Image.Output(display_name="image"),
                io.String.Output(display_name="saved_path"),
                io.String.Output(display_name="report"),
            ],
        )

    @classmethod
    def execute(
        cls,
        image,
        device,
        output_path="",
        jpeg_quality=95,
        datetime_original="",
        gps_latitude=0.0,
        gps_longitude=0.0,
        seed=0,
    ) -> io.NodeOutput:
        try:
            import piexif
        except ImportError:
            return io.NodeOutput(
                image,
                "",
                "piexif is not installed in this ComfyUI environment.\nInstall it with: pip install piexif",
            )

        frame = image[0] if image.ndim == 4 else image
        arr = np.clip(frame.cpu().numpy() * 255.0, 0, 255).astype(np.uint8)
        pil = Image.fromarray(arr).convert("RGB")
        height, width = arr.shape[0], arr.shape[1]

        stamp = (datetime_original or "").strip()
        if stamp:
            try:
                when = datetime.strptime(stamp, "%Y:%m:%d %H:%M:%S")
            except ValueError:
                return io.NodeOutput(
                    image, "", f"datetime_original must look like 'YYYY:MM:DD HH:MM:SS', got {stamp!r}"
                )
        else:
            rng = random.Random(seed)
            when = datetime.now() - timedelta(
                days=rng.randint(0, 45), hours=rng.randint(0, 23), minutes=rng.randint(0, 59)
            )

        exif_bytes = _build_exif(device, width, height, when, gps_latitude, gps_longitude, seed)

        saved = ""
        path = (output_path or "").strip().strip('"')
        if path:
            parent = os.path.dirname(os.path.abspath(path))
            if parent and not os.path.isdir(parent):
                return io.NodeOutput(image, "", f"output directory does not exist: {parent}")
            pil.save(path, format="JPEG", quality=jpeg_quality, exif=exif_bytes)
            saved = path
        else:
            buf = _io.BytesIO()
            pil.save(buf, format="JPEG", quality=jpeg_quality, exif=exif_bytes)

        make, model, lens, software, focal, fnum, _iso, _sensor = DEVICE_PRESETS[device]
        lines = [
            f"device:    {make} {model}",
            f"software:  {software}",
            f"lens:      {lens}",
            f"focal:     {focal}mm  f/{fnum}",
            f"captured:  {when.strftime('%Y:%m:%d %H:%M:%S')}",
            f"gps:       {'omitted' if gps_latitude == 0.0 and gps_longitude == 0.0 else f'{gps_latitude:.6f}, {gps_longitude:.6f}'}",
            f"saved:     {saved or '(not written to disk)'}",
            "",
            "Metadata layer only. Sensor noise, device quantization tables and ISP",
            "artifacts are untouched, so pixel-level forensics is unaffected.",
        ]
        return io.NodeOutput(image, saved, "\n".join(lines))
