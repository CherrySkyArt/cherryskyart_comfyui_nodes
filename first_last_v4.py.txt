import torch

# import comfy.model_management
from comfy import model_management
import comfy.latent_formats

from nodes import MAX_RESOLUTION
from comfy_api.latest import io
import node_helpers



class WanImageToVideoSVIProFLF_v4_Cherry(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="WanImageToVideoSVIProFLF_v4_Cherry",
            category="conditioning/video_models",
            display_name="WanI2V_SVI_Cherry Mask v4 🍒",
            search_aliases=["cherry", "wan", "svi", "first", "last", "frame"],
            inputs=[
                io.Conditioning.Input("positive"),
                io.Conditioning.Input("negative"),
                io.Int.Input("length", default=81, min=1, max=MAX_RESOLUTION, step=4),
                io.Latent.Input("anchor_samples"),
                io.Latent.Input("prev_samples", optional=True),
                io.Latent.Input("end_samples", optional=True, tooltip=("End samples, optional."),
                ),
                io.Int.Input("motion_latent_count", default=1, min=0, max=128, step=1),

            ],
            outputs=[
                io.Conditioning.Output(display_name="positive"),
                io.Conditioning.Output(display_name="negative"),
                io.Latent.Output(display_name="latent"),
            ],
        )

    @classmethod
    def execute(cls, positive, negative, length, motion_latent_count, anchor_samples, prev_samples=None, end_samples=None) -> io.NodeOutput:
        anchor_latent = anchor_samples["samples"].clone()

        B, C, T, H, W = anchor_latent.shape
        # empty_latent = torch.zeros([B, 16, ((length - 1) // 4) + 1, H, W], device=model_management.intermediate_device())
        empty_latent = anchor_samples["samples"].clone()

        total_latents = (length - 1) // 4 + 1
        print("total_latents: ", total_latents)
        device = anchor_latent.device
        dtype = anchor_latent.dtype

        if prev_samples is None or motion_latent_count == 0:
            padding_size = total_latents - anchor_latent.shape[2]
            image_cond_latent = anchor_latent
        else:
            motion_latent = prev_samples["samples"][:, :, -motion_latent_count:].clone()
            padding_size = total_latents - anchor_latent.shape[2] - motion_latent.shape[2]
            image_cond_latent = torch.cat([anchor_latent, motion_latent], dim=2)

        print("anchor_latent.shape[2]: " , anchor_latent.shape[2])
        # if motion_latent:
        #     print("motion_latent.shape[2]: " , motion_latent.shape[2])
        print("empty_latent.shape[2]: " , empty_latent.shape[2])
        print("padding_size: " , padding_size)
        padding = torch.zeros(1, C, padding_size, H, W, dtype=dtype, device=device)
        padding = comfy.latent_formats.Wan21().process_out(padding)
        image_cond_latent = torch.cat([image_cond_latent, padding], dim=2)

        mask = torch.ones((1, 1, empty_latent.shape[2], H, W), device=device, dtype=dtype)


        # Lock the last temporal slot (anchor).
        if end_samples:
            end_latent = end_samples["samples"].clone()
            T_end = end_latent.shape[2]
            mask[:, :, -T_end:] = 0.0 # Lock the last

        # prev_latent = prev_samples["samples"].clone()
        # T_prev = prev_latent.shape[2]
        # motion_count = min(motion_latent_count, T_prev)
        # if motion_count > 0:
        #     motion_latent = prev_latent[:, :, -motion_count:]
        #     # base_latent = torch.cat([anchor_latent, motion_latent], dim=2)
        #     anchor_latent = torch.cat([anchor_latent, motion_latent], dim=2)
        mask[:, :, :1] = 0.0

        positive = node_helpers.conditioning_set_values(positive, {"concat_latent_image": image_cond_latent, "concat_mask": mask})
        negative = node_helpers.conditioning_set_values(negative, {"concat_latent_image": image_cond_latent, "concat_mask": mask})

        out_latent = {}
        out_latent["samples"] = empty_latent
        return io.NodeOutput(positive, negative, out_latent)