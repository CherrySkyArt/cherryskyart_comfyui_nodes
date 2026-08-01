from comfy_api.latest import io


class BoundingBoxExpand(io.ComfyNode):
    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="BoundingBoxExpand",
            display_name="Bounding Box Expand 🍒",
            category="utilities/primitive",
            search_aliases=["cherry", "bbox", "grow" "coordinates"],
            description="Expands a bounding box to a target size, clamped to image bounds, growing more on the opposite side when near an edge.",
            inputs=[
                io.BoundingBox.Input("bounding_box"),
                io.Image.Input("image", optional=True, tooltip="If connected, overrides image_width and image_height."),
                io.Int.Input("image_width", default=512, tooltip="width of image that original bbox is on, ignored if using image input"),
                io.Int.Input("image_height", default=512, tooltip="height of image that original bbox is on, ignored if using image input"),
                io.Int.Input("target_width", default=0, tooltip="0 = keep original bbox width."),
                io.Int.Input("target_height", default=0, tooltip="0 = keep original bbox height."),
            ],
            outputs=[
                io.BoundingBox.Output("bounding_box"),
            ],
        )

    @classmethod
    def execute(cls, bounding_box, image_width, image_height, target_width, target_height, image=None) -> io.NodeOutput:
        if image is not None:
            image_height, image_width = image.shape[1], image.shape[2]

        bb = bounding_box[0][0] if isinstance(bounding_box, list) else bounding_box
        x = int(bb["x"])
        y = int(bb["y"])
        w = int(bb["width"])
        h = int(bb["height"])

        if target_width == 0:
            target_width = w
        if target_height == 0:
            target_height = h

        new_x, new_w = _expand_axis(x, w, image_width, target_width)
        new_y, new_h = _expand_axis(y, h, image_height, target_height)

        result = {"x": new_x, "y": new_y, "width": new_w, "height": new_h}
        return io.NodeOutput(result)


def _expand_axis(pos: int, size: int, image_size: int, target_size: int) -> tuple[int, int]:
    """Expand a 1D segment [pos, pos+size] to target_size within [0, image_size].

    Grows equally on both sides; if one side hits the boundary, the remaining
    growth is applied to the other side.
    """
    if target_size <= size:
        return pos, size

    extra = target_size - size
    half = extra // 2
    remainder = extra - half  # remainder >= half (handles odd extra)

    # Try growing: pos - half on the left, pos + size + remainder on the right
    new_start = pos - half
    new_end = pos + size + remainder

    if new_start < 0:
        # Shift the deficit to the right
        new_end += (-new_start)
        new_start = 0
    if new_end > image_size:
        # Shift the deficit to the left
        new_start -= (new_end - image_size)
        new_end = image_size
        if new_start < 0:
            new_start = 0  # target_size > image_size; clamp

    new_size = new_end - new_start
    return new_start, new_size
