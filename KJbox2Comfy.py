
from comfy_api.latest import io

class KJBboxToComfyBBox(io.ComfyNode):
    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="KJBboxToComfyBBox",
            display_name="KJ BBox --> ComfyUI BBox 🍒",
            category="KJNodes/masking",
            description="Converts KJNodes BBOX array (list of x,y,w,h tuples) to ComfyUI BoundingBox format.",
            search_aliases=["cherry"],
            inputs=[
                io.Custom("BBOX").Input("bboxes"),
                # io.Custom("*").Input("bboxes")
            ],
            outputs=[ 
                io.BoundingBox.Output("bboxes"),
            ],
        )


    @classmethod
    def execute(cls, bboxes):
        # Mirror the batch structure: list of lists, one inner list per image
        bbox_dicts = [
            {
                "x": float(x),
                "y": float(y),
                "width": float(w),
                "height": float(h),
            }
            for (x, y, w, h) in bboxes
        ]

        # Wrap in outer list to match batch format [[{...}, {...}], ...]
        all_bbox_dicts = [bbox_dicts]

        return io.NodeOutput(all_bbox_dicts)