import numpy as np
import torch
from PIL import Image

from comfy_api.latest import io


class PngInMemory(io.ComfyNode):
    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="PngInMemory_Cherry",
            display_name="PNG In Memory 🍒",
            category="image",
            search_aliases=["cherry", "save", "save image", "image", "passthrough", "preview image", "no save"],
            description="Processes images exactly like Save Image (the same clip/quantize round-trip), but outputs them instead of writing files to disk.",
            inputs=[
                io.Image.Input("images", tooltip="The images to process."),
            ],
            outputs=[io.Image.Output(display_name="images")],
        )

    @classmethod
    def execute(cls, images) -> io.NodeOutput:
        results = []
        for image in images:
            i = 255. * image.cpu().numpy()
            img = Image.fromarray(np.clip(i, 0, 255).astype(np.uint8))
            out = torch.from_numpy(np.array(img).astype(np.float32) / 255.0)
            results.append(out)

        out_images = torch.stack(results) if results else images
        return io.NodeOutput(out_images)
