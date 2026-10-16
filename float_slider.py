import sys

from comfy_api.latest import io


STEP_OPTIONS = ["1", "0.1", "0.01", "0.001"]


class FloatSlider(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="CherryFloatSlider",
            display_name="Float 🍒",
            category="utils",
            description="Outputs the chosen float value. The step dropdown changes the arrow/drag step size.",
            search_aliases=["float", "slider", "number", "value", "cherry"],
            inputs=[
                # step/round here are the finest allowed; web/float_slider/float_slider.js
                # swaps them live based on the "step" dropdown
                # Omitting min/max makes the frontend fall back to 0..2048,
                # so use the same "unlimited" bounds as core's PrimitiveFloat
                io.Float.Input(
                    "value",
                    default=0.5,
                    min=-sys.maxsize,
                    max=sys.maxsize,
                    step=0.001,
                    round=0.001,
                ),
                io.Combo.Input(
                    "step",
                    options=STEP_OPTIONS,
                    default="0.1",
                    tooltip="Step size for the arrows and dragging.",
                ),
            ],
            outputs=[
                io.Float.Output("float", display_name="float"),
            ],
        )

    @classmethod
    def execute(cls, value: float, step: str) -> io.NodeOutput:
        return io.NodeOutput(value)
