import os
import sys
from pathlib import Path
from contextlib import nullcontext

import numpy as np
import torch
from PIL import Image, ImageFilter
from torch.hub import download_url_to_file

import folder_paths
import comfy.model_management

from comfy_api.latest import io
from .AILab_ImageMaskTools import pil2tensor, tensor2pil


_models_dir = Path(__file__).parent / "models"
SAM3_LOCAL_DIR = _models_dir / "sam3"
if str(_models_dir) not in sys.path:
    sys.path.insert(0, str(_models_dir))

from sam3.model_builder import build_sam3_image_model
from sam3.model.sam3_image_processor import Sam3Processor

# Copied from comfyui-RMBG, ~April 2026
# https://github.com/1038lab/ComfyUI-RMBG/blob/112fd6141d07663d5e4a35d66d269542b58f4438/py/AILab_SAM3Segment.py

# from sam3.model_builder import build_sam3_image_model # I FIXED THIS
# from sam3.model.sam3_image_processor import Sam3Processor

# Copied from comfyui-RMBG, ~April 2026
# https://github.com/1038lab/ComfyUI-RMBG/blob/112fd6141d07663d5e4a35d66d269542b58f4438/py/AILab_SAM3Segment.py

CURRENT_DIR = Path(__file__).resolve().parent
REPO_ROOT = CURRENT_DIR.parent
# SAM3_LOCAL_DIR = REPO_ROOT / "models" / "sam3"
# if str(SAM3_LOCAL_DIR) not in sys.path:
#     sys.path.insert(0, str(SAM3_LOCAL_DIR))
# MODELS_ROOT = REPO_ROOT / "models"
# if str(MODELS_ROOT) not in sys.path:
#     sys.path.insert(0, str(MODELS_ROOT))



SAM3_BPE_PATH = SAM3_LOCAL_DIR / "assets" / "bpe_simple_vocab_16e6.txt.gz"
if not os.path.isfile(SAM3_BPE_PATH):
    raise RuntimeError("SAM3 assets missing; ensure sam3/assets/bpe_simple_vocab_16e6.txt.gz exists.")

_DEFAULT_PT_ENTRY = {
    "model_url": "https://huggingface.co/1038lab/sam3/resolve/main/sam3.pt",
    "filename": "sam3.pt",
}

SAM3_MODELS = {
    "sam3": _DEFAULT_PT_ENTRY.copy(),
}


def get_sam3_pt_models():
    entry = SAM3_MODELS.get("sam3")
    if entry and entry.get("filename", "").endswith(".pt"):
        return {"sam3": entry}
    for key, value in SAM3_MODELS.items():
        if value.get("filename", "").endswith(".pt"):
            return {"sam3": value}
        if "sam3" in key and value:
            candidate = value.copy()
            candidate["model_url"] = _DEFAULT_PT_ENTRY["model_url"]
            candidate["filename"] = _DEFAULT_PT_ENTRY["filename"]
            return {"sam3": candidate}
    return {"sam3": _DEFAULT_PT_ENTRY.copy()}


def process_mask(mask_image, invert_output=False, mask_blur=0, mask_offset=0):
    if invert_output:
        mask_np = np.array(mask_image)
        mask_image = Image.fromarray(255 - mask_np)
    if mask_blur > 0:
        mask_image = mask_image.filter(ImageFilter.GaussianBlur(radius=mask_blur))
    if mask_offset != 0:
        filt = ImageFilter.MaxFilter if mask_offset > 0 else ImageFilter.MinFilter
        size = abs(mask_offset) * 2 + 1
        for _ in range(abs(mask_offset)):
            mask_image = mask_image.filter(filt(size))
    return mask_image


def apply_background_color(image, mask_image, background="Alpha", background_color="#222222"):
    rgba_image = image.copy().convert("RGBA")
    rgba_image.putalpha(mask_image.convert("L"))
    if background == "Color":
        hex_color = background_color.lstrip("#")
        r, g, b = int(hex_color[0:2], 16), int(hex_color[2:4], 16), int(hex_color[4:6], 16)
        bg_image = Image.new("RGBA", image.size, (r, g, b, 255))
        composite = Image.alpha_composite(bg_image, rgba_image)
        return composite.convert("RGB")
    return rgba_image


def mask_to_bbox(mask_tensor):
    binary = (mask_tensor > 0.5).cpu()
    rows = binary.any(dim=1)
    cols = binary.any(dim=0)
    if not rows.any():
        return {"x": 0.0, "y": 0.0, "width": 0.0, "height": 0.0}
    y_min = int(rows.nonzero(as_tuple=True)[0][0])
    y_max = int(rows.nonzero(as_tuple=True)[0][-1])
    x_min = int(cols.nonzero(as_tuple=True)[0][0])
    x_max = int(cols.nonzero(as_tuple=True)[0][-1])
    return {
        "x": float(x_min),
        "y": float(y_min),
        "width": float(x_max - x_min + 1),
        "height": float(y_max - y_min + 1),
    }


def get_or_download_model_file(filename, url):
    local_path = None
    if hasattr(folder_paths, "get_full_path"):
        local_path = folder_paths.get_full_path("sam3", filename)
    if local_path and os.path.isfile(local_path):
        return local_path
    base_models_dir = getattr(folder_paths, "models_dir", os.path.join(CURRENT_DIR, "models"))
    models_dir = os.path.join(base_models_dir, "sam3")
    os.makedirs(models_dir, exist_ok=True)
    local_path = os.path.join(models_dir, filename)
    if not os.path.exists(local_path):
        print(f"Downloading {filename} from {url} ...")
        download_url_to_file(url, local_path)
    return local_path


def _resolve_device(user_choice):
    auto_device = comfy.model_management.get_torch_device()
    if user_choice == "CPU":
        return torch.device("cpu")
    if user_choice == "GPU":
        if auto_device.type != "cuda":
            raise RuntimeError("GPU unavailable")
        return torch.device("cuda")
    return auto_device


class CherrySAM3Segment(io.ComfyNode):
    _processor_cache = {}

    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="CherrySAM3Segment",
            display_name="SAM3 Segment (Cherry)",
            category="🧪AILab/🧽RMBG",
            inputs=[
                io.Image.Input("image"),
                io.String.Input("prompt", default="", multiline=True, placeholder="Describe the concept"),
                io.Combo.Input("output_mode", options=["Merged", "Separate"], default="Merged"),
                io.Float.Input("confidence_threshold", default=0.5, min=0.05, max=0.95, step=0.01),
                io.Int.Input("max_segments", default=0, min=0, max=128, step=1, optional=True),
                io.Int.Input("segment_pick", default=0, min=0, max=128, step=1, optional=True),
                io.Int.Input("mask_blur", default=0, min=0, max=64, step=1, optional=True),
                io.Int.Input("mask_offset", default=0, min=-64, max=64, step=1, optional=True),
                io.Combo.Input("device", options=["Auto", "CPU", "GPU"], default="Auto", optional=True),
                io.Boolean.Input("invert_output", default=False, optional=True),
                io.Boolean.Input("unload_model", default=False, optional=True),
                io.Combo.Input("background", options=["Alpha", "Color"], default="Alpha", optional=True),
                io.Custom("COLORCODE").Input("background_color", optional=True, extra_dict={"default": "#222222"}),
            ],
            outputs=[
                io.Image.Output("IMAGE"),
                io.Mask.Output("MASK"),
                io.Image.Output("MASK_IMAGE"),
                io.BoundingBox.Output("bboxes"),
            ],
        )

    @classmethod
    def _load_processor(cls, device_choice):
        torch_device = _resolve_device(device_choice)
        device_str = "cuda" if torch_device.type == "cuda" else "cpu"
        cache_key = ("sam3", device_str)
        if cache_key not in cls._processor_cache:
            model_info = SAM3_MODELS["sam3"]
            ckpt_path = get_or_download_model_file(model_info["filename"], model_info["model_url"])
            model = build_sam3_image_model(
                bpe_path=SAM3_BPE_PATH,
                device=device_str,
                eval_mode=True,
                checkpoint_path=ckpt_path,
                load_from_HF=False,
                enable_segmentation=True,
                enable_inst_interactivity=False,
            )
            processor = Sam3Processor(model, device=device_str)
            cls._processor_cache[cache_key] = processor
        return cls._processor_cache[cache_key], torch_device

    @staticmethod
    def _empty_result(img_pil, background, background_color):
        w, h = img_pil.size
        mask_image = Image.new("L", (w, h), 0)
        result_image = apply_background_color(img_pil, mask_image, background, background_color)
        result_image = result_image.convert("RGBA") if background == "Alpha" else result_image.convert("RGB")
        empty_mask = torch.zeros((1, h, w), dtype=torch.float32)
        mask_rgb = empty_mask.reshape((-1, 1, h, w)).movedim(1, -1).expand(-1, -1, -1, 3)
        return result_image, empty_mask, mask_rgb

    @staticmethod
    def _empty_batch(img_pil):
        w, h = img_pil.size
        empty_imgs = torch.zeros((0, h, w, 3), dtype=torch.float32)
        empty_masks = torch.zeros((0, h, w), dtype=torch.float32)
        empty_mask_images = torch.zeros((0, h, w, 3), dtype=torch.float32)
        return empty_imgs, empty_masks, empty_mask_images

    @classmethod
    def _run_single_per_instance(cls, processor, img_tensor, prompt, confidence, max_segments, segment_pick, mask_blur, mask_offset, invert, unload_model, background, background_color):
        img_pil = tensor2pil(img_tensor)
        text = prompt.strip() or "object"
        state = processor.set_image(img_pil)
        processor.reset_all_prompts(state)
        processor.set_confidence_threshold(confidence, state)
        state = processor.set_text_prompt(text, state)
        masks = state.get("masks")
        logits = state.get("masks_logits")
        if masks is None or masks.numel() == 0:
            return cls._empty_batch(img_pil)
        masks = masks.float()
        if masks.ndim == 4:
            masks = masks.squeeze(1)
        scores = None
        if logits is not None:
            logits = logits.float()
            if logits.ndim == 4:
                logits = logits.squeeze(1)
            scores = logits.mean(dim=(-2, -1))
        if scores is None:
            scores = torch.ones((masks.shape[0],), device=masks.device)
        if max_segments > 0 and masks.shape[0] > max_segments:
            topk = torch.topk(scores, k=max_segments)
            masks = masks[topk.indices]
            scores = scores[topk.indices]
        sorted_idx = torch.argsort(scores, descending=True)
        masks = masks[sorted_idx]
        if segment_pick > 0:
            idx = segment_pick - 1
            if idx >= masks.shape[0]:
                return cls._empty_batch(img_pil)
            masks = masks[idx : idx + 1]

        mask_imgs, mask_tensors, mask_rgb_list = [], [], []
        for single_mask in masks:
            mask_np = (single_mask.clamp(0, 1).cpu().numpy() * 255).astype(np.uint8)
            mask_image = Image.fromarray(mask_np, mode="L")
            mask_image = process_mask(mask_image, invert, mask_blur, mask_offset)
            composed = apply_background_color(img_pil, mask_image, background, background_color)
            composed = composed.convert("RGBA") if background == "Alpha" else composed.convert("RGB")
            mask_tensor = torch.from_numpy(np.array(mask_image).astype(np.float32) / 255.0).unsqueeze(0)
            mask_rgb = mask_tensor.reshape((1, mask_image.height, mask_image.width, 1)).expand(-1, -1, -1, 3)
            mask_imgs.append(pil2tensor(composed))
            mask_tensors.append(mask_tensor)
            mask_rgb_list.append(mask_rgb)
        return (
            torch.cat(mask_imgs, dim=0),
            torch.cat(mask_tensors, dim=0),
            torch.cat(mask_rgb_list, dim=0),
        )

    @classmethod
    def _run_single_merged(cls, processor, img_tensor, prompt, confidence, max_segments, segment_pick, mask_blur, mask_offset, invert, unload_model, background, background_color):
        img_pil = tensor2pil(img_tensor)
        imgs, masks, _ = cls._run_single_per_instance(
            processor,
            img_tensor,
            prompt,
            confidence,
            max_segments,
            segment_pick,
            mask_blur,
            mask_offset,
            invert,
            unload_model,
            background,
            background_color,
        )
        if masks.shape[0] == 0:
            return cls._empty_result(img_pil, background, background_color)
        merged = masks.amax(dim=0)
        mask_np = (merged.clamp(0, 1).cpu().numpy() * 255).astype(np.uint8)
        mask_image = Image.fromarray(mask_np, mode="L")
        mask_image = process_mask(mask_image, invert, mask_blur, mask_offset)
        result_image = apply_background_color(img_pil, mask_image, background, background_color)
        result_image = result_image.convert("RGBA") if background == "Alpha" else result_image.convert("RGB")
        mask_tensor = torch.from_numpy(np.array(mask_image).astype(np.float32) / 255.0).unsqueeze(0)
        mask_rgb = mask_tensor.reshape((1, mask_image.height, mask_image.width, 1)).expand(-1, -1, -1, 3)
        return result_image, mask_tensor, mask_rgb

    @classmethod
    def execute(cls, image, prompt, output_mode, confidence_threshold, max_segments=0, segment_pick=0, mask_blur=0, mask_offset=0, device="Auto", invert_output=False, unload_model=False, background="Alpha", background_color="#222222") -> io.NodeOutput:
        if image.ndim == 3:
            image = image.unsqueeze(0)
        processor, torch_device = cls._load_processor(device)
        autocast_device = comfy.model_management.get_autocast_device(torch_device)
        autocast_enabled = torch_device.type == "cuda" and not comfy.model_management.is_device_mps(torch_device)
        ctx = torch.autocast(autocast_device, dtype=torch.bfloat16) if autocast_enabled else nullcontext()
        result_images, result_masks, result_mask_images, result_bboxes = [], [], [], []
        with ctx:
            for tensor_img in image:
                if output_mode == "Separate":
                    imgs_batch, masks_batch, mask_imgs_batch = cls._run_single_per_instance(
                        processor,
                        tensor_img,
                        prompt,
                        confidence_threshold,
                        max_segments,
                        segment_pick,
                        mask_blur,
                        mask_offset,
                        invert_output,
                        unload_model,
                        background,
                        background_color,
                    )
                    result_images.append(imgs_batch)
                    result_masks.append(masks_batch)
                    result_mask_images.append(mask_imgs_batch)
                    result_bboxes.extend([mask_to_bbox(masks_batch[i]) for i in range(masks_batch.shape[0])])
                else:
                    img_pil, mask_tensor, mask_rgb = cls._run_single_merged(
                        processor,
                        tensor_img,
                        prompt,
                        confidence_threshold,
                        max_segments,
                        segment_pick,
                        mask_blur,
                        mask_offset,
                        invert_output,
                        unload_model,
                        background,
                        background_color,
                    )
                    result_images.append(pil2tensor(img_pil))
                    result_masks.append(mask_tensor)
                    result_mask_images.append(mask_rgb)
                    result_bboxes.append(mask_to_bbox(mask_tensor[0]))

        if unload_model:
            device_str = "cuda" if torch_device.type == "cuda" else "cpu"
            cache_key = ("sam3", device_str)
            if cache_key in cls._processor_cache:
                del cls._processor_cache[cache_key]
            if torch_device.type == "cuda":
                torch.cuda.empty_cache()

        # Handle empty results
        final_images = torch.cat(result_images, dim=0)
        final_masks = torch.cat(result_masks, dim=0)
        final_mask_images = torch.cat(result_mask_images, dim=0)

        # If no segments found in Separate mode, return at least one empty result
        if final_images.shape[0] == 0:
            # Use the first input image to get dimensions
            img_pil = tensor2pil(image[0])
            empty_img, empty_mask, empty_mask_img = cls._empty_result(img_pil, background, background_color)
            final_images = pil2tensor(empty_img)
            final_masks = empty_mask
            final_mask_images = empty_mask_img
            result_bboxes = [{"x": 0.0, "y": 0.0, "width": 0.0, "height": 0.0}]

        # Merged → single dict; Separate → flat list of dicts
        final_bboxes = result_bboxes[0] if output_mode == "Merged" and len(result_bboxes) == 1 else result_bboxes
        return io.NodeOutput(final_images, final_masks, final_mask_images, final_bboxes)