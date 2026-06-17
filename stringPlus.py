from comfy_api.latest import io


class StringPlus(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="StringTemplate",
            display_name="String Plus (Template)",
            category="utils",
            description=(
                "Resolves a template string containing %date:yyyy-MM-dd%, "
                "%NodeName.widgetName%, and other ComfyUI filename prefix tokens "
                "into a plain string output."
            ),
            search_aliases=["string", "template", "format", "string plus"],
            inputs=[
                io.String.Input(
                    "template",
                    multiline=False,
                    default="%date:yyyy-MM-dd%/image",
                    tooltip=(
                        "Template string to resolve. Supports the same tokens as "
                        "filename_prefix: %date:yyyy-MM-dd%, %NodeName.widgetName%, "
                        "%year%, %month%, %day%, %hour%, %minute%, %second%."
                    ),
                ),
            ],
            outputs=[
                io.String.Output(display_name="STRING"),
            ],
        )

    @classmethod
    def execute(cls, template: str) -> io.NodeOutput:
        # The frontend resolves %date:...% and %NodeName.widget% tokens via
        # serializeValue before this node executes, so `template` is already
        # the fully-substituted string by the time it arrives here.
        return io.NodeOutput(template)
