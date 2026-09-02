"""
StackableRows — a stack of draggable, reorderable text rows, each its own
STRING input that can be typed or connected. Built on io.Autogrow so rows
grow/shrink as real sockets (no dead schema weight for unused slots),
capped at Autogrow's own sanity limit of 100.

Row display order is user-controlled (drag-and-drop in the frontend) and is
independent of slot name, so it's tracked separately in row_order — a JSON
array of slot names in top-to-bottom order. Slots missing from row_order
(e.g. a freshly connected one the frontend hasn't recorded yet) fall back
to sorted-by-name order, appended after the ordered ones.
"""

import json
from comfy_api.latest import io

ROW_PREFIX = "row_"
MAX_ROWS = 100


class StackableRows(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="CherryStackableRows",
            display_name="☰ Stackable Text Rows 🍒",
            category="cherry",
            search_aliases=["cherry", "concat", "concatenate", "arrange"],
            description="A stack of draggable, reorderable text rows. Each row can be typed text or a connected input. Outputs them concatenated top to bottom into one string.",
            inputs=[
                io.Autogrow.Input(
                    "rows",
                    template=io.Autogrow.TemplatePrefix(
                        input=io.String.Input("row", multiline=False, default=""),
                        prefix=ROW_PREFIX,
                        min=1,
                        max=MAX_ROWS,
                    ),
                ),
                io.String.Input("row_order", default="[]", multiline=False, tooltip="Internal: JSON array of row slot names in display order. Managed by the widget UI."),
                io.String.Input("separator", default="\\n", tooltip="Text inserted between rows. \\n and \\t are supported."),
            ],
            outputs=[
                io.String.Output(display_name="text"),
            ],
        )

    @classmethod
    def execute(cls, rows: io.Autogrow.Type, row_order: str, separator: str) -> io.NodeOutput:
        rows = rows or {}

        try:
            order = json.loads(row_order)
            if not isinstance(order, list):
                order = []
        except (json.JSONDecodeError, TypeError):
            order = []

        ordered_names = [name for name in order if name in rows]
        remaining = sorted(name for name in rows if name not in ordered_names)
        ordered_names.extend(remaining)

        sep = separator.replace("\\n", "\n").replace("\\t", "\t")
        text = sep.join(str(rows[name] or "") for name in ordered_names)
        return io.NodeOutput(text)
