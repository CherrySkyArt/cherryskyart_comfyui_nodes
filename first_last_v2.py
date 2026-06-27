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
import torch.nn.functional as F

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

def _feather(m, r):           # soften edges; m: (n,H,W)
    if r <= 0:
        return m
    k = int(r) * 2 + 1
    x = m.unsqueeze(1)
    x = F.avg_pool2d(F.pad(x, (r, r, r, r), mode="replicate"), kernel_size=k, stride=1)
    return x.squeeze(1)

def _erode(m, r):             # shrink the keep region; m: (H,W)
    if r <= 0:
        return m
    k = int(r) * 2 + 1
    x = (-m)[None, None]
    x = F.max_pool2d(F.pad(x, (r, r, r, r), mode="replicate"), kernel_size=k, stride=1)
    return (-x)[0, 0]


class WanFLF2VMask_Cherry(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="WanFLF2VMask_Cherry",
            category="conditioning/video_models",
            display_name="WanFLF2V Mask v2 feather 🍒",
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
                io.Int.Input("mask_feather", default=4, min=0, max=64),
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
                start_noise=0.0, middle_noise=0.0, end_noise=0.0,
                start_image=None, middle_image=None, end_image=None,
                start_image_mask=None, middle_image_mask=None, end_image_mask=None,
                middle_hold=None, mask_feather=0) -> io.NodeOutput:

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

        # fade keyframe toward gray where the (white=keep) mask is low
        def gray_blend(frame_img, spatial_mask):
            if spatial_mask is None:
                return frame_img
            keep = _resize_mask(spatial_mask, width, height, frame_img.shape[0]).to(frame_img.dtype)
            keep = keep.unsqueeze(-1)  # (n,H,W,1)
            return frame_img * keep + 0.5 * (1.0 - keep)

        # write the concat mask over temporal slice [a:b]
        # 0 = fully given, 1 = fully generate; never goes negative by construction
        def write_concat_mask(a, b, spatial_mask, noise_strength):
            if spatial_mask is None:
                mask[:, :, a:b] = noise_strength
            else:
                keep = _resize_mask(spatial_mask, lat_w, lat_h, 1)[0]      # (H_lat,W_lat)
                mask[:, :, a:b] = 1.0 - keep * (1.0 - noise_strength)      # broadcasts over time

        def place(img_a, img_b, frame_img, sm, noise, ma, mb, feather):
            n = frame_img.shape[0]
            if sm is None:
                image[img_a:img_b] = frame_img
                mask[:, :, ma:mb] = noise
                return
            keep_px = _resize_mask(sm, width, height, n)                     # (n,H,W) white = keep
            if feather > 0:
                keep_px = _feather(keep_px, feather)
            keep_px = keep_px.to(frame_img.dtype)

            # image: real content out to the FULL mask edge
            kp = keep_px.unsqueeze(-1)
            image[img_a:img_b] = frame_img * kp + 0.5 * (1.0 - kp)

            # concat mask: keep region pulled in 1 latent px so the bled seam is regenerated
            keep_lat = comfy.utils.common_upscale(keep_px.unsqueeze(1), lat_w, lat_h, "bilinear", "center").squeeze(1)[0]
            keep_lat = _erode(keep_lat, 1)
            mask[:, :, ma:mb] = 1.0 - keep_lat * (1.0 - noise)


        # if start_image is not None:
        #     n = start_image.shape[0]
        #     image[:n] = gray_blend(start_image, start_image_mask)
        #     write_concat_mask(0, n + 3, start_image_mask, start_noise)

        # if middle_image is not None:
        #     n = middle_image.shape[0]
        #     f = int(round(middle_frame_position * (length - 1)))
        #     f = max(0, min(f, length - n))
        #     image[f:f + n] = gray_blend(middle_image, middle_image_mask)
        #     write_concat_mask(f + 3, f + 3 + n, middle_image_mask, middle_noise)

        # if end_image is not None:
        #     n = end_image.shape[0]
        #     image[-n:] = gray_blend(end_image, end_image_mask)
        #     write_concat_mask(length - n + 3, length + 3, end_image_mask, end_noise)
        # mask_feather = 5
        if start_image is not None:
            n = start_image.shape[0]
            place(0, n, start_image, start_image_mask, start_noise, 0, n + 3, mask_feather)

        if middle_image is not None:
            n = middle_image.shape[0]
            f = int(round(middle_frame_position * (length - 1)))
            f = max(0, min(f, length - n))
            place(f, f + n, middle_image, middle_image_mask, middle_noise, f + 3, f + 3 + n, mask_feather)

        if end_image is not None:
            n = end_image.shape[0]
            place(length - n, length, end_image, end_image_mask, end_noise, length - n + 3, length + 3, mask_feather)

        concat_latent_image = vae.encode(image[:, :, :, :3])
        mask = mask.view(1, mask.shape[2] // 4, 4, mask.shape[3], mask.shape[4]).transpose(1, 2)
        positive = node_helpers.conditioning_set_values(positive, {"concat_latent_image": concat_latent_image, "concat_mask": mask})
        negative = node_helpers.conditioning_set_values(negative, {"concat_latent_image": concat_latent_image, "concat_mask": mask})

        out_latent = {}
        out_latent["samples"] = latent
        return io.NodeOutput(positive, negative, out_latent)