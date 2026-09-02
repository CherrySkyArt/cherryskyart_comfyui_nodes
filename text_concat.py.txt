from comfy_api.latest import io


class TextConcat(io.ComfyNode):
    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="TextConcat_Cherry",
            display_name="Text Concat Infinite 🍒",
            category="utils/text",
            description="Concatenates any number of text inputs into one string.",
            search_aliases=["string", "concat", "join", "text", "merge"],
            inputs=[
                io.String.Input(
                    "text_1",
                    default="",
                    multiline=False,
                    tooltip="Text input 1.",
                ),
                io.String.Input(
                    "separator",
                    default="",
                    multiline=False,
                    tooltip="String inserted between each text input.",
                ),
            ],
            outputs=[
                io.String.Output(display_name="STRING"),
            ],
        )

    @classmethod
    def execute(cls, separator: str = "", **kwargs) -> io.NodeOutput:
        parts = []
        i = 1
        while True:
            key = f"text_{i}"
            if key not in kwargs:
                break
            val = kwargs[key]
            if val is not None:
                parts.append(str(val))
            i += 1
        return io.NodeOutput(separator.join(parts))
