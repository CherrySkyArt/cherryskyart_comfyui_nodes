
import torch
import torch.nn.functional as F
from comfy_api.latest import io


class DepthMeasure(io.ComfyNode):
    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="DepthMeasure",
            display_name="Depth Measure 🍒",
            is_input_list=True,
            category="image/depth",
            search_aliases=["cherry", "depth", "distance", "depth map", "measure distance", "closest", "furthest", "controlnet"],
            description=(
                "Returns furthest/closest object: Given individually-segmented depth-map images (one object per image), "
                "sorts them by depth. (Depth-AnythingV2, ZoeDepth, ect)"
            ),
            inputs=[
                io.Image.Input(
                    "imgs_depthmap",
                    tooltip="Depth-map images, one segmented object per image.",
                ),
                io.Image.Input(
                    "images",
                    tooltip="Original images. Sorted in same order as the depth maps.",
                    optional=True,
                ),
                io.Mask.Input(
                    "masks",
                    tooltip="Optional per-object masks",
                    optional=True,
                ),
                io.Combo.Input("sort_output", options=["Furthest", "Closest"], default="Closest"),
            ],
            outputs=[
                io.Image.Output("imgs_depth_sorted", tooltip="Depth-map images sorted by distance."),
                io.Image.Output("images_sorted", tooltip="Original images sorted in the same order as the depth maps."),
                io.Mask.Output("masks_sorted", tooltip="Masks sorted in the same order (zeros if none provided)."),
            ],
        )

    @classmethod
    def execute(cls, imgs_depthmap, images=None, masks=None, sort_output="Closest"):
        if isinstance(sort_output, list):
            sort_output = sort_output[0]
        if isinstance(imgs_depthmap, list):
            imgs_depthmap = torch.cat(imgs_depthmap, dim=0)
        if isinstance(images, list):
            images = torch.cat(images, dim=0)
        if isinstance(masks, list):
            masks = torch.cat(masks, dim=0)

        n = imgs_depthmap.shape[0]

        # Brighter = closer (Depth Anything v2 relative, MiDaS disparity convention).
        # Invert so higher value = farther, then sort ascending.
        depth = 1.0 - imgs_depthmap.mean(dim=-1)

        mean_depths: list[float] = []
        for i in range(n):
            if masks is not None:
                m = masks[i]
                if m.dim() == 3:
                    m = m.squeeze(0)  # [1,H,W] → [H,W]
                h, w = depth[i].shape
                if m.shape != (h, w):
                    m = F.interpolate(m.unsqueeze(0).unsqueeze(0), size=(h, w), mode="nearest").squeeze(0).squeeze(0)
                obj_mask = m > 0.5
            else:
                obj_mask = imgs_depthmap[i].mean(dim=-1) > (10.0 / 255.0)

            n_valid = obj_mask.sum().item()
            mean_depths.append(depth[i][obj_mask].mean().item() if n_valid > 0 else float("inf"))

        # Lower value = closer. "Closest": ascending. "Furthest": negate so far sorts first, inf stays last.
        def sort_key(i):
            d = mean_depths[i]
            return float("inf") if d == float("inf") else (d if sort_output == "Closest" else -d)

        sorted_indices = sorted(range(n), key=sort_key)

        print(f"[DepthMeasure] sort={sort_output}")
        print(f"[DepthMeasure] mean depth per image: {[round(d, 4) for d in mean_depths]}")
        print(f"[DepthMeasure] output order: {sorted_indices}")

        idx_t = torch.tensor(sorted_indices, dtype=torch.long)
        sorted_depthmaps = imgs_depthmap[idx_t]
        sorted_images = images[idx_t] if images is not None else imgs_depthmap[idx_t]

        if masks is not None:
            sorted_masks = masks[idx_t]
        else:
            sorted_masks = torch.zeros(n, imgs_depthmap.shape[1], imgs_depthmap.shape[2], dtype=torch.float32)

        return io.NodeOutput(sorted_depthmaps, sorted_images, sorted_masks)
