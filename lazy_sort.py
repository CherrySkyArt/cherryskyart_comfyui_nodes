
import logging
import torch
import numpy as np
from comfy_api.latest import io

# 🚬😎 vibe coded. I dont understand the math of the numpy and torch stuff. And frankly, i dont care ot understand it 😎 (‿∣‿) 🖐️slap 🤤
class LazySort(io.ComfyNode):
    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="LazySort",
            display_name="Lazy Sort 🍒",
            is_input_list=True,
            search_aliases=["left", "right", "top", "bottom", "order", "position", "sort", "center", "mass"],
            description=(
                "Sorts segmented objects by mask center-of-mass. "
                "Robust to outstretched limbs — a hand tip is a few pixels vs thousands in the torso."
            ),
            category="image/batch",
            inputs=[
                io.Image.Input("images", tooltip="Batch of segmented images"),
                io.Mask.Input(
                    "masks",
                    optional=True,
                    tooltip="Per-object masks. Used to compute center-of-mass. Falls back to bbox center if absent.",
                ),
                io.BoundingBox.Input(
                    "bboxes",
                    optional=True,
                    tooltip="Bounding boxes in original-image coords. Required to offset cropped-mask centroids into global space.",
                ),
                io.Combo.Input(
                    "sort_direction",
                    options=["Left → Right", "Right → Left", "Top → Bottom", "Bottom → Top"],
                    default="Left → Right",
                    optional=True,
                ),
            ],
            outputs=[
                io.Image.Output("images"),
                io.Mask.Output("masks"),
                io.BoundingBox.Output("bboxes"),
            ],
        )

    @classmethod
    def execute(cls, images, masks=None, bboxes=None, sort_direction="Left → Right") -> io.NodeOutput:
        # is_input_list=True: everything arrives as a list; unwrap widget scalar
        if isinstance(sort_direction, list):
            sort_direction = sort_direction[0]

        # Stack tensors into a single batch
        if isinstance(images, list):
            images = torch.cat(images, dim=0)
        if isinstance(masks, list):
            masks = torch.cat(masks, dim=0)

        n = images.shape[0]

        # Normalize bboxes to a flat list of dicts
        flat_bboxes = None
        if bboxes is not None:
            flat = []
            items = bboxes if isinstance(bboxes, list) else [bboxes]
            for b in items:
                if isinstance(b, dict):
                    flat.append(b)
                elif isinstance(b, list):
                    flat.extend(b)
            # Count must line up 1:1 with images, or bbox-based offsetting is meaningless.
            if flat and len(flat) != n:
                logging.warning(
                    "LazySort: got %d bboxes for %d images, ignoring bboxes (mask/zero fallback used)",
                    len(flat), n,
                )
                flat = []
            flat_bboxes = flat or None

        use_y = sort_direction in ("Top → Bottom", "Bottom → Top")
        reverse = sort_direction in ("Right → Left", "Bottom → Top")

        keys = [
            cls._center(
                masks[i] if (masks is not None and i < masks.shape[0]) else None,
                flat_bboxes[i] if (flat_bboxes is not None and i < len(flat_bboxes)) else None,
                use_y=use_y,
            )
            for i in range(n)
        ]

        order = sorted(range(n), key=lambda i: keys[i], reverse=reverse)
        idx = torch.tensor(order, dtype=torch.long)

        out_images = images[idx]
        out_masks = masks[idx] if masks is not None else None
        out_bboxes = [[flat_bboxes[i] for i in order]] if flat_bboxes is not None else None

        return io.NodeOutput(out_images, out_masks, out_bboxes)

    @classmethod
    def _center(cls, mask, bbox, use_y=False) -> float:
        # Mass-weighted centroid so torso bulk dominates over outstretched limbs; offsets cropped masks into global coords.
        if mask is not None:
            m = mask.cpu().float().numpy()
            if m.ndim == 3:
                m = m[0]

            total = float(m.sum())
            if total > 1e-6:
                if use_y:
                    mask_h = m.shape[0]
                    row_sums = m.sum(axis=1)
                    local_c = float((np.arange(mask_h, dtype=np.float32) * row_sums).sum() / total)
                    if bbox is not None and abs(mask_h - bbox["height"]) <= 4:
                        return bbox["y"] + local_c
                    return local_c
                else:
                    mask_w = m.shape[1]
                    col_sums = m.sum(axis=0)
                    local_c = float((np.arange(mask_w, dtype=np.float32) * col_sums).sum() / total)
                    if bbox is not None and abs(mask_w - bbox["width"]) <= 4:
                        return bbox["x"] + local_c
                    return local_c

        # Fallback: bbox midpoint
        if bbox is not None:
            return (bbox["y"] + bbox["height"] * 0.5) if use_y else (bbox["x"] + bbox["width"] * 0.5)

        return 0.0
