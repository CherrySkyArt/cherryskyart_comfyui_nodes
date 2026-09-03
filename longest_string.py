from comfy_api.latest import io


class LongestString(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="CherryLongestString",
            display_name="Longest String 🍒",
            category="utils",
            description=(
                "Splits a multiline string into lines and outputs the longest one."
            ),
            search_aliases=["longest", "string", "line", "multiline", "cherry", "count", "newline"],
            inputs=[
                io.String.Input(
                    "text",
                    multiline=True,
                    default="",
                    tooltip="Multiline string; each line is compared by length.",
                ),
            ],
            outputs=[
                io.String.Output(display_name="STRING"),
            ],
        )

    @classmethod
    def execute(cls, text: str) -> io.NodeOutput:
        lines = text.split("\n")
        longest = max(lines, key=len) if lines else ""
        return io.NodeOutput(longest)
