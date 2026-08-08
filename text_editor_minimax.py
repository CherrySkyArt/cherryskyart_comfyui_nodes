"""
text_editor_minimax
--------------------
A ComfyUI node that exposes a plain STRING input/output, paired with a
custom frontend widget (see js/text_editor_minimax.js) that renders it as a
syntax-highlighted editor for a small custom markup:

    [keyword]
    <d>...</d>
    **bold**
    00:07.200   (mm:ss.mmm timestamps)

The Python side stays intentionally simple: whatever text the widget holds
is passed straight through as output, so this node can sit at the start of
any workflow that needs hand-authored, tagged text (e.g. subtitle/cue data,
prompt templates with markers, etc).
"""

from comfy_api.latest import io


class TextEditorMinimax(io.ComfyNode):
    """Passes the edited text straight through as a STRING output."""

    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="TextEditorMinimax",
            display_name="Text Editor (Minimax) 🍒",
            category="utils/text",
            description=(
                "Multiline text editor with highlighting for [keywords], "
                "<d>...</d> tags, **bold**, and mm:ss.mmm timestamps."
            ),
            search_aliases=["minimax", "string", "wordwrap", "multiline", "text", "editor", "subtitle", "cue", "highlight"],
            inputs=[
                io.String.Input(
                    "text",
                    multiline=True,
                    default=("Prompting guide: \n https://huggingface.co/MiniMaxAI/MiniMax-H3/tree/main/docs"),
                    tooltip="Text editor for minimax.",
                ),
            ],
            outputs=[
                io.String.Output(display_name="text"),
            ],
        )

    @classmethod
    def execute(cls, text: str = "") -> io.NodeOutput:
        return io.NodeOutput(text)
