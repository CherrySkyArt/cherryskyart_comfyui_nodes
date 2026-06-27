from comfy_api.latest import io


class BoundingBoxUnpack(io.ComfyNode):
    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="BoundingBoxUnpack",
            display_name="Bounding Box Unpack 🍒",
            category="utilities/primitive",
            description="Unpacks a BoundingBox into x, y, width, height integers.",
            inputs=[
                io.BoundingBox.Input("bounding_box"),
            ],
            outputs=[
                io.Int.Output("x"),
                io.Int.Output("y"),
                io.Int.Output("width"),
                io.Int.Output("height"),
            ],
        )

    @classmethod
    def execute(cls, bounding_box) -> io.NodeOutput:
        bb = bounding_box[0][0] if isinstance(bounding_box, list) else bounding_box
        return io.NodeOutput(int(bb["x"]), int(bb["y"]), int(bb["width"]), int(bb["height"]))
