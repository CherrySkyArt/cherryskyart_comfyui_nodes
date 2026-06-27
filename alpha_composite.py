import torch
from comfy_api.latest import io


class AlphaComposite(io.ComfyNode):
    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="AlphaComposite_Cherry",
            display_name="Alpha Composite 🍒",
            search_aliases=["overlay", "layer", "paste", "alpha", "transparency", "composite"],
            description="Composites source onto destination using the source's own alpha channel as the blend mask.",
            category="image/compositing",
            inputs=[
                io.Image.Input("destination"),
                io.Image.Input("source", tooltip="Must have 4 channels (RGBA). Alpha channel drives the blend."),
                io.Int.Input("x", default=0, min=0, max=16384, step=1),
                io.Int.Input("y", default=0, min=0, max=16384, step=1),
                io.Boolean.Input("resize_source", default=False, tooltip="Resize source to match destination before compositing."),
            ],
            outputs=[io.Image.Output()],
        )

    @classmethod
    def execute(cls, destination, source, x: int, y: int, resize_source: bool) -> io.NodeOutput:
        # destination: [B, H, W, C]  C=3 or 4
        # source:      [B, H, W, C]  C must be 4 (RGBA)

        dest = destination.clone()
        src = source.clone()

        # Ensure source has an alpha channel
        if src.shape[-1] < 4:
            raise ValueError("AlphaComposite: source must have 4 channels (RGBA). Got shape: " + str(src.shape))

        # Resize source to dest dims if requested
        if resize_source:
            dh, dw = dest.shape[1], dest.shape[2]
            # comfy uses [B,C,H,W] for upscale
            src_nchw = src.permute(0, 3, 1, 2)
            import comfy.utils
            src_nchw = comfy.utils.common_upscale(src_nchw, dw, dh, "bicubic", "center")
            src = src_nchw.permute(0, 2, 3, 1)

        # Batch-broadcast: if one side has B=1, expand to match the other
        db, dh, dw, dc = dest.shape
        sb, sh, sw, sc = src.shape
        b = max(db, sb)
        if db == 1 and b > 1:
            dest = dest.expand(b, -1, -1, -1).clone()
        if sb == 1 and b > 1:
            src = src.expand(b, -1, -1, -1).clone()

        # Ensure dest has 4 channels (add opaque alpha if needed)
        if dest.shape[-1] == 3:
            ones = torch.ones(b, dh, dw, 1, dtype=dest.dtype, device=dest.device)
            dest = torch.cat([dest, ones], dim=-1)

        src_rgb   = src[..., :3]        # [B, sh, sw, 3]
        src_alpha = src[..., 3:4]       # [B, sh, sw, 1]  in [0,1]

        # Clamp paste region to destination bounds
        dh, dw = dest.shape[1], dest.shape[2]
        x0 = max(x, 0)
        y0 = max(y, 0)
        x1 = min(x + sw, dw)
        y1 = min(y + sh, dh)

        # Corresponding slice into source
        sx0 = x0 - x
        sy0 = y0 - y
        sx1 = sx0 + (x1 - x0)
        sy1 = sy0 + (y1 - y0)

        if x1 <= x0 or y1 <= y0:
            # Source is fully outside destination — return dest as-is (drop alpha if dest originally had 3ch)
            out = dest[..., :dc]
            return io.NodeOutput(out)

        dest_patch  = dest[:, y0:y1, x0:x1, :3]   # [B, ph, pw, 3]
        dest_a_patch = dest[:, y0:y1, x0:x1, 3:4]  # [B, ph, pw, 1]
        src_patch   = src_rgb[:, sy0:sy1, sx0:sx1, :]
        alpha_patch = src_alpha[:, sy0:sy1, sx0:sx1, :]  # [0,1]

        # Standard alpha-over composite
        blended_rgb = src_patch * alpha_patch + dest_patch * (1.0 - alpha_patch)
        # Output alpha: src_alpha over dest_alpha
        blended_a   = alpha_patch + dest_a_patch * (1.0 - alpha_patch)

        dest[:, y0:y1, x0:x1, :3] = blended_rgb
        dest[:, y0:y1, x0:x1, 3:4] = blended_a

        # Return same number of channels the destination came in with
        out = dest[..., :dc]
        return io.NodeOutput(out)
