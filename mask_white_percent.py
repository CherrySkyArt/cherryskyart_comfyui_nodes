from comfy_api.latest import io


class MaskWhitePercent(io.ComfyNode):
    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="MaskWhitePercent",
            display_name="Mask White Percent 🍒",
            category="mask",
            search_aliases=["cherry", "mask coverage", "mask threshold", "white percentage"],
            description="Calculates what percentage of a mask is white, using a threshold to decide when a non-black pixel counts, and compares it against a target percent.",
            inputs=[
                io.Mask.Input("mask"),
                io.Int.Input(
                    "pixel_value_white",
                    default=1,
                    min=0,
                    max=255,
                    step=1,
                    tooltip="Pixel values (0-255) at or above this are counted as white.",
                ),
                io.Float.Input(
                    "percent_white",
                    default=50.0,
                    min=0.0,
                    max=100.0,
                    step=0.1,
                    tooltip="Percent threshold to compare the mask's white percentage against.",
                ),
            ],
            outputs=[
                io.Boolean.Output(display_name="is_above_percent"),
                io.Float.Output(display_name="white_percent"),
            ],
        )

    @classmethod
    def execute(cls, mask, pixel_value_white, percent_white) -> io.NodeOutput:
        mask_255 = mask * 255.0
        white_pixels = (mask_255 >= pixel_value_white).sum().item()
        total_pixels = mask_255.numel()

        white_percent = (white_pixels / total_pixels) * 100.0 if total_pixels > 0 else 0.0
        is_above_percent = white_percent > percent_white

        return io.NodeOutput(is_above_percent, white_percent)
