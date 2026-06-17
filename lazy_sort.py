
import torch
import numpy as np
from comfy_api.latest import io

# 🚬😎 vibe coded. I dont understand the math of the numpy and torch stuff. And frankly, i dont care ot understand it 😎 (‿∣‿) 🖐️slap 🤤
class LazySort(io.ComfyNode):
    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="LazySort",
            display_name="Lazy Sort (L→R) 🍒",
            search_aliases=["cherry" "left", "right", "order", "position", "sort", "center", "mass"],
            description=(
                "Sorts segmented objects left-to-right by mask center-of-mass. "
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
                io.Boolean.Input("is_left_to_right", default=True, optional=True),
            ],
            outputs=[
                io.Image.Output("images"),
                io.Mask.Output("masks"),
                io.BoundingBox.Output("bboxes"),
            ],
        )

    @classmethod
    def execute(cls, images, masks=None, bboxes=None, is_left_to_right=True) -> io.NodeOutput:
        n = images.shape[0]

        # Normalize bboxes: [[{...}, ...]] -> [{...}, ...]
        flat_bboxes = None
        if bboxes is not None:
            flat_bboxes = (
                bboxes[0]
                if isinstance(bboxes, list) and bboxes and isinstance(bboxes[0], list)
                else bboxes
            )

        keys = [
            cls._center_x(
                masks[i] if (masks is not None and i < masks.shape[0]) else None,
                flat_bboxes[i] if (flat_bboxes is not None and i < len(flat_bboxes)) else None,
            )
            for i in range(n)
        ]

        order = sorted(range(n), key=lambda i: keys[i], reverse=not is_left_to_right)
        idx = torch.tensor(order, dtype=torch.long)

        out_images = images[idx]
        out_masks = masks[idx] if masks is not None else None
        out_bboxes = [[flat_bboxes[i] for i in order]] if flat_bboxes is not None else None

        return io.NodeOutput(out_images, out_masks, out_bboxes)

    @classmethod
    def _center_x(cls, mask, bbox) -> float:
        """
        Weighted x centroid of a single mask in original-image coordinates.

        The centroid is mass-weighted: a person's torso contributes thousands of
        pixels while an outstretched hand contributes only a few, so the result
        is naturally anchored to the body bulk rather than the extremities.

        Detection of cropped vs full-res mask: if mask width is within 4px of
        bbox["width"], it's a crop and we offset by bbox["x"].
        """
        if mask is not None:
            m = mask.cpu().float().numpy()
            if m.ndim == 3:
                m = m[0]  # drop leading dim if present

            total = float(m.sum())
            if total > 1e-6:
                mask_w = m.shape[1]
                col_sums = m.sum(axis=0)                          # [W]
                local_cx = float((np.arange(mask_w, dtype=np.float32) * col_sums).sum() / total)

                if bbox is not None and abs(mask_w - bbox["width"]) <= 4:
                    # Cropped mask — shift into global image coords
                    return bbox["x"] + local_cx
                # Full-res mask — centroid already in global coords
                return local_cx

        # Fallback: bbox horizontal midpoint
        if bbox is not None:
            return bbox["x"] + bbox["width"] * 0.5

        return 0.0
