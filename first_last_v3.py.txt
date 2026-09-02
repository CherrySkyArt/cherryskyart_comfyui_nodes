import torch
import comfy.model_management
import comfy.latent_formats
from nodes import MAX_RESOLUTION
from comfy_api.latest import io
import node_helpers


class WanImageToVideoSVIProFLF_Cherry(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="WanImageToVideoSVIProFLF_Cherry",
            category="conditioning/video_models",
            display_name="WanI2V_SVI_Cherry Mask v3 🍒",
            inputs=[
                io.Conditioning.Input("positive"),
                io.Conditioning.Input("negative"),
                io.Int.Input(
                    "length",
                    default=81,
                    min=1,
                    max=MAX_RESOLUTION,
                    step=4,
                    tooltip=(
                        "Target video length in frames. "
                        "Wan video latents use a temporal stride of 4."
                    ),
                ),
                io.Latent.Input(
                    "prev_samples",
                    optional=True,
                    tooltip=(
                        "Previous segment latents (B,C,T,H,W). "
                        "The last motion_latent_count temporal slots will be "
                        "used to continue motion (SVI Pro style)."
                    ),
                ),
                io.Latent.Input(
                    "start_samples",
                    optional=True,
                    tooltip=(
                        "Anchor latent(s) for the current segment (B,C,T,H,W). "
                        "If omitted, no frame is hard-locked at the start - the segment "
                        "runs purely off motion continuation from prev_samples (SVI Pro style), "
                        "and only end_samples (if given) is locked."
                    ),
                ),
                io.Latent.Input(
                    "end_samples",
                    optional=True,
                    tooltip=(
                        "Optional target end latent(s), in Wan video latent format (B,C,T,H,W). "
                        "The last temporal slots of this tensor will define the last slots "
                        "of the clip (FLF-style hard lock)."
                    ),
                ),
                io.Int.Input(
                    "motion_latent_count",
                    default=1,
                    min=0,
                    max=16,
                    step=1,
                    tooltip=(
                        "How many last temporal latent slots to copy from prev_samples. "
                        "0 = no motion continuity; 1-2 is typical for smooth transitions."
                    ),
                ),
                io.Float.Input(
                    "end_mask",
                    default=1.0,
                    min=0.0,
                    max=1.0,
                    step=0.01,
                    tooltip=(
                        "How strongly to lock the end frames. 1 = fully hard-locked to "
                        "end_samples (max change from base), 0 = end_samples ignored and the "
                        "base/generated content is kept (locked in). Wan2.2 latents can't be "
                        "spatially masked, so this is a uniform strength, not a region mask."
                    ),
                ),
            ],
            outputs=[
                io.Conditioning.Output(display_name="positive"),
                io.Conditioning.Output(display_name="negative"),
                io.Latent.Output(display_name="latent"),
            ],
        )

    @classmethod
    def execute(
        cls,
        positive,
        negative,
        length,
        prev_samples=None,
        anchor_samples=None,
        end_samples=None,
        use_end_samples=True,
        motion_latent_count=1,
    ) -> io.NodeOutput:

        anchor_latent = anchor_samples["samples"].clone()
        B, C, T_anchor, H, W = anchor_latent.shape

        total_latents = (length - 1) // 4 + 1
        device = anchor_latent.device
        dtype = anchor_latent.dtype

        if prev_samples is None or motion_latent_count == 0:
            base_latent = anchor_latent
        else:
            prev_latent = prev_samples["samples"].clone()
            T_prev = prev_latent.shape[2]

            motion_count = min(motion_latent_count, T_prev)

            if motion_count > 0:
                motion_latent = prev_latent[:, :, -motion_count:]
                base_latent = torch.cat([anchor_latent, motion_latent], dim=2)
            else:
                base_latent = anchor_latent

        T_base = base_latent.shape[2]
        padding_size = max(total_latents - T_base, 0)

        if padding_size > 0:
            padding = torch.zeros(B, C, padding_size, H, W, dtype=dtype, device=device)
            padding = comfy.latent_formats.Wan21().process_out(padding)
            image_cond_latent = torch.cat([base_latent, padding], dim=2)
        else:
            image_cond_latent = base_latent[:, :, :total_latents]

        if image_cond_latent.shape[2] != total_latents:
            image_cond_latent = image_cond_latent[:, :, :total_latents]

        end_t_fix = 0
        if use_end_samples and end_samples is not None:
            end_latent = end_samples["samples"].clone()

            if end_latent.shape[0] == 1 and B > 1:
                end_latent = end_latent.repeat(B, 1, 1, 1, 1)

            if (
                end_latent.shape[1] == C
                and end_latent.shape[3] == H
                and end_latent.shape[4] == W
            ):
                T_end = end_latent.shape[2]

                end_t_fix = min(T_end, total_latents)

                if end_t_fix > 0:
                    image_cond_latent[:, :, -end_t_fix:] = end_latent[:, :, -end_t_fix:]
            else:
                end_t_fix = 0

        empty_latent = torch.zeros(
            [B, 16, total_latents, H, W],
            device=comfy.model_management.intermediate_device(),
        )

        mask = torch.ones((1, 1, total_latents, H, W), device=device, dtype=dtype)

        mask[:, :, :1] = 0.0

        if end_t_fix > 0:
            mask[:, :, -end_t_fix:] = 0.0

        positive = node_helpers.conditioning_set_values(
            positive,
            {
                "concat_latent_image": image_cond_latent,
                "concat_mask": mask,
            },
        )
        negative = node_helpers.conditioning_set_values(
            negative,
            {
                "concat_latent_image": image_cond_latent,
                "concat_mask": mask,
            },
        )

        out_latent = {"samples": empty_latent}
        return io.NodeOutput(positive, negative, out_latent)

