from comfy_api.latest import io


class GetByIndex(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="CherryGetByIndex",
            display_name="Get By Index 🍒",
            category="utils/list",
            description="Outputs the item at the given index of a list. Negative indices count from the end.",
            search_aliases=["index", "list", "get", "item", "pick", "cherry"],
            is_input_list=True,
            inputs=[
                io.Custom("*").Input("items", display_name="list", tooltip="List of anything."),
                io.Int.Input("index", default=0, min=-0xffffffff, max=0xffffffff, tooltip="0 = first item, -1 = last item."),
            ],
            outputs=[
                io.Custom("*").Output(display_name="ANY"),
            ],
        )

    @classmethod
    def execute(cls, items: list, index: list) -> io.NodeOutput:
        # is_input_list wraps every input in a list, so index arrives as [int]
        index = index[0]

        # a single python list passed as one item (e.g. from Append Any To List) -> unwrap it
        if len(items) == 1 and isinstance(items[0], (list, tuple)):
            items = items[0]

        if not -len(items) <= index < len(items):
            raise IndexError(f"Get By Index: index {index} out of range for list of length {len(items)}")

        return io.NodeOutput(items[index])
