from comfy_api.latest import io
import torch
import node_helpers
import comfy
import comfy.utils
import comfy.clip_vision
import comfy.latent_formats
import logging

import torch

import comfy.model_management
import comfy.latent_formats

from nodes import MAX_RESOLUTION
from comfy_api.latest import io
import node_helpers


def _to_latent_mask(m, lat_w, lat_h):
    # MASK comes in as (frames,H,W) or (H,W); downsample to latent spatial res
    if m.dim() == 2:
        m = m.unsqueeze(0)
    return comfy.utils.common_upscale(
        m.unsqueeze(1), lat_w, lat_h, "bilinear", "center"
    ).squeeze(1)  # -> (frames, H_lat, W_lat)

def _resize_mask(m, target_w, target_h, n):
    # MASK -> (n, target_h, target_w), white = keep
    if m.dim() == 2:
        m = m.unsqueeze(0)
    if m.shape[0] == 1 and n > 1:
        m = m.repeat(n, 1, 1)
    return comfy.utils.common_upscale(
        m.unsqueeze(1), target_w, target_h, "bilinear", "center"
    ).squeeze(1)

class WanFLF2VMask_Cherry_old(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="WanFLF2VMask_Cherry_old",
            category="conditioning/video_models",
            display_name="WanFLF2V Mask OLD (DEP) 🍒",
            search_aliases=["cherry", "first", "middle", "last", "frame"],
            inputs=[
                io.Conditioning.Input("positive"),
                io.Conditioning.Input("negative"),
                io.Vae.Input("vae"),
                io.Int.Input("width", default=832, min=16, max=MAX_RESOLUTION, step=16),
                io.Int.Input("height", default=480, min=16, max=MAX_RESOLUTION, step=16),
                io.Int.Input("length", default=81, min=1, max=MAX_RESOLUTION, step=4),
                io.Int.Input("batch_size", default=1, min=1, max=4096),
                io.Float.Input("middle_frame_position", default=0.5, min=0.0, max=1.0, step=0.01),
                io.Image.Input("start_image", optional=True),
                io.Image.Input("middle_image", optional=True),
                io.Image.Input("end_image", optional=True),
                io.Mask.Input("start_image_mask", optional=True),
                io.Mask.Input("middle_image_mask", optional=True),
                io.Mask.Input("end_image_mask", optional=True),

                io.Float.Input("start_noise",  default=0.0, min=0.0, max=1.0, step=0.1),
                io.Float.Input("middle_noise", default=0.0, min=0.0, max=1.0, step=0.1),
                io.Float.Input("end_noise",    default=0.0, min=0.0, max=1.0, step=0.1),
                io.Int.Input("middle_hold", default=1, min=1, max=8, tooltip=("DEAD ATM"),),
            ],
            outputs=[
                io.Conditioning.Output(display_name="positive"),
                io.Conditioning.Output(display_name="negative"),
                io.Latent.Output(display_name="latent"),
            ],
        )

    @classmethod
    def _calculate_aligned_position(cls, ratio, total_frames):
        desired_idx = int(total_frames * ratio)
        aligned_idx = (desired_idx // 4) * 4              # snap to latent-group boundary
        return max(0, min(aligned_idx, total_frames - 1))

    @classmethod
    def execute(cls, positive, negative, vae, width, height, length, batch_size,
                middle_frame_position=0.5,
                start_image=None, middle_image=None, end_image=None,
                start_image_mask=None, middle_image_mask=None, end_image_mask=None,
                start_noise=0.0, middle_noise=0.0, end_noise=0.0,
                middle_hold=1) -> io.NodeOutput:

        spacial_scale = vae.spacial_compression_encode()
        latent = torch.zeros(
            [batch_size, vae.latent_channels, ((length - 1) // 4) + 1,
             height // spacial_scale, width // spacial_scale],
            device=comfy.model_management.intermediate_device())

        lat_h, lat_w = latent.shape[-2], latent.shape[-1]

        if start_image is not None:
            start_image = comfy.utils.common_upscale(start_image[:length].movedim(-1, 1), width, height, "bilinear", "center").movedim(1, -1)
        if middle_image is not None:
            middle_image = comfy.utils.common_upscale(middle_image[:length].movedim(-1, 1), width, height, "bilinear", "center").movedim(1, -1)
        if end_image is not None:
            end_image = comfy.utils.common_upscale(end_image[-length:].movedim(-1, 1), width, height, "bilinear", "center").movedim(1, -1)

        image = torch.ones((length, height, width, 3)) * 0.5
        mask = torch.ones((1, 1, latent.shape[2] * 4, lat_h, lat_w))

        # MASK OG #########################################################################
        # helper: write the conditioning mask for a keyframe occupying mask
        # positions [a:b]. spatial_mask (white = keep) -> 0 where kept, 1 elsewhere.
        # def write_mask(a, b, spatial_mask):
        #     if spatial_mask is not None:
        #         sm = _to_latent_mask(spatial_mask, lat_w, lat_h)
        #         mask[:, :, a:b] = (1.0 - sm)[None, None]
        #     else:
        #         mask[:, :, a:b] = 0.0
        # MASK 2 new #######################################################################
        def write_mask(a, b, spatial_mask, noise_strength):
            # noise_strength in [0,1]: 0 = full anchor, 1 = no anchor
            if spatial_mask is not None:
                sm = _to_latent_mask(spatial_mask, lat_w, lat_h)
                mask[:, :, a:b] = (1.0 - sm * (1.0 - noise_strength))[None, None]
            else:
                mask[:, :, a:b] = noise_strength

        # --- start (pixel 0..n; also claims the 3 causal-padding positions) ---
        if start_image is not None:
            n = start_image.shape[0]
            image[:n] = start_image
            write_mask(0, n + 3, start_image_mask, start_noise)
            # write_mask(0, n + 3, start_image_mask)

        # --- middle (single keyframe placed by fractional position) ---
        # 1 ####################################################
        # if middle_image is not None:
        #     n = middle_image.shape[0]
        #     f1 = int(round(middle_frame_position * (length - 1)))
        #     f = max(0, min(f1, length - n))
        #     print("n: ", n)
        #     print("f1: ", f1)
        #     print("f: ", f)
        #     print("f+n: ", f+n)
        #     print("f+3+n: ", f+3+n)
        #     image[f:f + n] = middle_image
        #     write_mask(f + 3, f + 3 + n, middle_image_mask)
        # 1.1 ##################################################
        if middle_image is not None:
            n = middle_image.shape[0]
            f = int(round(middle_frame_position * (length - 1)))
            f = max(0, min(f, length - n))
            image[f:f + n] = middle_image
            write_mask(f + 3, f + 3 + n, middle_image_mask, middle_noise)
        
        # --- end (pixel length-n .. length) ---  applied last so it owns the tail
        if end_image is not None:
            n = end_image.shape[0]
            image[-n:] = end_image
            write_mask(length - n + 3, length + 3, end_image_mask, end_noise)

        concat_latent_image = vae.encode(image[:, :, :, :3])
        mask = mask.view(1, mask.shape[2] // 4, 4, mask.shape[3], mask.shape[4]).transpose(1, 2)
        positive = node_helpers.conditioning_set_values(positive, {"concat_latent_image": concat_latent_image, "concat_mask": mask})
        negative = node_helpers.conditioning_set_values(negative, {"concat_latent_image": concat_latent_image, "concat_mask": mask})

        out_latent = {}
        out_latent["samples"] = latent
        return io.NodeOutput(positive, negative, out_latent)