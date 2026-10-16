from comfy_api.latest import io


class PlaySoundOnFinish(io.ComfyNode):
    """
    Passthrough node that plays a sound in the browser when execution reaches it.
    Drop it after your SaveImage / VHS Combine / etc. node so it fires once
    generation is actually finished.
    """

    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="CherryPlaySoundOnFinish",
            display_name="Play Sound On Finish 🍒",
            category="utils",
            description=(
                "Passthrough node that plays a notification sound in the browser "
                "when the workflow execution reaches it. Connect it after your "
                "final output node (SaveImage, VHS Combine, etc.)."
            ),
            search_aliases=["sound", "notify", "notification", "beep", "alert", "finish", "done", "cherry"],
            inputs=[
                io.Custom("*").Input("anything", optional=True, tooltip="Anything; passed through unchanged. Connect your final output node here."),
            ],
            outputs=[
                io.Custom("*").Output(display_name="ANY"),
            ],
        )

    @classmethod
    def execute(cls, anything=None) -> io.NodeOutput:
        return io.NodeOutput(anything)
