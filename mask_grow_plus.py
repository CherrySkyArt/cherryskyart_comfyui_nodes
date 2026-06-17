import numpy as np
import scipy.ndimage
import torch

from comfy_api.latest import io
import nodes
import logging


# vibe c0der, and AI proumptur 😎🤙 you can shove your college degree and code architecture and cloud architecture up your ass
class MaskGrowPlus(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="MaskGrowPlus",
            search_aliases=["cherry", "territorial grow mask", "voronoi mask", "expand mask no overlap"],
            display_name="Grow Masks+ (Territory) 🍒",
            description="Grow a batch of masks together while preventing overlap.",
            category="mask",
            inputs=[
                io.Mask.Input("masks"),  # Expects a batch [N, H, W]
                io.Int.Input("expand", default=10, min=0, max=nodes.MAX_RESOLUTION, step=1),
                io.Int.Input(
                    "group_size",
                    default=2,
                    min=0,
                    max=64,
                    step=1,
                    tooltip="How many masks compete per run. eg. group_size=2 → groups [0,1], [2,3], [4,5]. And 0 = all simultaneously",
                ),
                io.Boolean.Input("tapered_corners", default=True, advanced=True)
            ],
            outputs=[
                io.Mask.Output(),                          # output 0: full grown batch (all N masks).  Returns the grown batch [N, H, W]
                io.Mask.Output(display_name="nth_masks"), # output 1: one mask per group
            ]
        )

    # ------------------------------------------------------------------ #
    #  Core territorial dilation - operates on a single group of masks    #
    # ------------------------------------------------------------------ #
    @staticmethod
    def _grow_group(masks_np: np.ndarray, expand: int, kernel: np.ndarray) -> np.ndarray:
        """
        Territorially grow a group of masks (shape [G, H, W], values 0-1).
        Returns grown masks with the same shape.
        """
        logging.info("Computing mask.... might take 10s of seconds")
        G, H, W = masks_np.shape

        # Seed label map (0 = empty, i+1 = owned by mask i, -1 = border)
        label_map = np.zeros((H, W), dtype=np.int32)
        for i in range(G):
            owned = masks_np[i] > 0.5
            conflict = owned & (label_map > 0)
            label_map[conflict] = -1
            label_map[owned & (label_map == 0)] = i + 1

        for _ in range(expand):
            # No unclaimed pixels left - further iterations are no-ops, bail out early.
            if not np.any(label_map == 0):
                break
            claims = np.zeros((H, W), dtype=np.int32)

            for i in range(G):
                current = (label_map == i + 1).astype(np.float32)
                dilated = scipy.ndimage.grey_dilation(current, footprint=kernel) > 0.5
                wants = dilated & (label_map == 0)
                contested = wants & (claims != 0)
                claims[contested] = -1
                claims[wants & (claims == 0)] = i + 1

            label_map[claims > 0] = claims[claims > 0]
            label_map[claims == -1] = -1

            if not np.any(claims != 0):
                break

        return np.stack(
            [(label_map == i + 1).astype(np.float32) for i in range(G)],
            axis=0,
        )

    # ------------------------------------------------------------------ #
    #  Execute                                                             #
    # ------------------------------------------------------------------ #
    @classmethod
    def execute(cls, masks, expand, tapered_corners, group_size) -> io.NodeOutput:
        c = 0 if tapered_corners else 1
        kernel = np.array([[c, 1, c],
                           [1, 1, 1],
                           [c, 1, c]])

        masks = masks.reshape((-1, masks.shape[-2], masks.shape[-1]))
        N = masks.shape[0]
        masks_np = masks.numpy()

        # group_size=0 (or >= N) means "all masks compete together"
        effective_group = N if (group_size <= 0 or group_size >= N) else group_size

        out_np = np.empty_like(masks_np)
        nth_masks = []
        

        # Slice into groups, process each independently, write results back
        for start in range(0, N, effective_group):
            end = min(start + effective_group, N)
            group_result = cls._grow_group(masks_np[start:end], expand, kernel)
            out_np[start:end] = group_result
            nth_masks.append(start)

        all_masks = torch.stack([torch.from_numpy(out_np[i]) for i in range(N)], dim=0)
        group_leaders = torch.stack([torch.from_numpy(out_np[i]) for i in nth_masks], dim=0)


        return io.NodeOutput(all_masks, group_leaders)



