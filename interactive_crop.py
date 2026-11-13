from comfy_api.latest import io, ui


MAX_CROP = 16384


def clamp_crop(width: int, height: int, left: int, top: int, right: int, bottom: int) -> tuple[int, int, int, int]:
    """Clamp per-edge trims so at least a 1x1 region always remains.
    Mirrored in web/interactive_crop/interactive_crop.js (clampCrop) so the
    on-node box always shows exactly what execute() will cut."""
    left = max(0, min(left, width - 1))
    right = max(0, min(right, width - 1 - left))
    top = max(0, min(top, height - 1))
    bottom = max(0, min(bottom, height - 1 - top))
    return left, top, right, bottom


class InteractiveCrop(io.ComfyNode):
    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="CherryInteractiveCrop",
            display_name="Interactive Crop 🍒",
            category="image/transform",
            search_aliases=["cherry", "crop", "interactive crop", "crop box", "trim", "cut"],
            description=(
                "Crops the image with a box you drag on the node. Run once to see the image, "
                "drag the edges/corners (or the middle to move), then run again. "
                "Lossless: pixels are sliced, never resampled. The whole batch uses the same box."
            ),
            inputs=[
                io.Image.Input("image"),
                # Amount trimmed from each edge, in pixels. Defaults of 0 = no crop, so the
                # first run passes the full image through and shows it on the node.
                io.Int.Input("crop_left", default=0, min=0, max=MAX_CROP, step=1, tooltip="Pixels removed from the left edge."),
                io.Int.Input("crop_top", default=0, min=0, max=MAX_CROP, step=1, tooltip="Pixels removed from the top edge."),
                io.Int.Input("crop_right", default=0, min=0, max=MAX_CROP, step=1, tooltip="Pixels removed from the right edge."),
                io.Int.Input("crop_bottom", default=0, min=0, max=MAX_CROP, step=1, tooltip="Pixels removed from the bottom edge."),
            ],
            outputs=[
                io.Image.Output(display_name="image"),
                io.Int.Output(display_name="x"),
                io.Int.Output(display_name="y"),
                io.Int.Output(display_name="width"),
                io.Int.Output(display_name="height"),
            ],
            # Output node so it runs (and shows the preview) even with nothing wired downstream.
            is_output_node=True,
        )

    @classmethod
    def execute(cls, image, crop_left, crop_top, crop_right, crop_bottom) -> io.NodeOutput:
        _, height, width, _ = image.shape
        left, top, right, bottom = clamp_crop(width, height, crop_left, crop_top, crop_right, crop_bottom)

        # Plain tensor slicing: exact source pixel values, no resampling or quantizing.
        cropped = image[:, top:height - bottom, left:width - right, :].contiguous()

        # Full, uncropped first frame for the on-node editor. Sent under a custom key
        # (not "images") so ComfyUI's default preview doesn't draw it a second time.
        preview = ui.ImageSaveHelper.save_images(
            image[0:1], filename_prefix="cherry_crop_preview", folder_type=io.FolderType.temp, cls=None, compress_level=1,
        )
        return io.NodeOutput(
            cropped,
            left,
            top,
            cropped.shape[2],
            cropped.shape[1],
            ui={"cherry_crop": [dict(r, width=width, height=height) for r in preview]},
        )
