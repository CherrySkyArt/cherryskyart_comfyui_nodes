import logging
from typing import List
from comfy_api.latest import io
import torch
import comfy
import sys
import nodes
import re


class AppendAnyToList(io.ComfyNode):
    """
    Input A must be a list.
    Input B can be any datatype.
    Appends B onto A, then returns all outputs.
    """

    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="AppendAnyToList",
            display_name="Append Any To List 🍒",
            search_aliases=["cherry"],
            category="utils/list",
            inputs=[
                io.Custom("*").Input("list_a"),
                io.Custom("*").Input("anything"),
                io.Boolean.Input("dedup", default=False, tooltip="ON = duplicate images are not appended to list -> all images are unique. Helpful b/c rerunning workflow keep all items in list sometimes ¯\_(ツ)_/¯.",), 
            ],
            outputs=[
                io.Image.Output("IMAGE", is_output_list=True),
                io.String.Output("STRING", is_output_list=True),
                io.Int.Output("INT", is_output_list=True),
                io.Float.Output("FLOAT", is_output_list=True),
                io.Conditioning.Output("CONDITIONING", is_output_list=True),
                io.Latent.Output("LATENT", is_output_list=True),
                io.Custom("*").Output("ANY", is_output_list=True),
            ],
        )

    @classmethod
    def execute(cls, list_a: List, anything, dedup: bool = False) -> io.NodeOutput:
        logging.info("APPEND NODE EXECUTE")

        if not isinstance(list_a, list):
            list_a = [list_a]

        logging.info("type: " + str(type(anything)))
        logging.info("class: " + str(anything.__class__.__name__))

        list_a.append(anything)

        # cheeky code to prevent duplicates, b/c re-running the workflow sometimes double stacks the list. (pray no bugs)
        if dedup:
            list_a = cls._deduplicate(list_a)

        # yes, double brackets are needed because of the OUTPUT_IS_LIST... ¯\_(ツ)_/¯
        # shoutout to Crystools https://github.com/crystian/ComfyUI-Crystools/blob/main/nodes/list.py#L86
        merged = [list_a]
        # first_item = merged[0] if merged else None

        return io.NodeOutput(
            merged,   # IMAGE
            merged,   # STRING
            merged,   # INT
            merged,   # FLOAT
            merged,   # CONDITIONING
            merged,   # LATENT
            merged    # ANY
        )
    
    @staticmethod
    def _deduplicate(items: List) -> List:
        unique = []
        for candidate in items:
            is_dup = False
            for seen in unique:
                try:
                    # Tensor comparison
                    if isinstance(candidate, torch.Tensor) and isinstance(seen, torch.Tensor):
                        if torch.equal(candidate, seen):
                            is_dup = True
                            break
                    # Fallback for hashable / comparable types
                    elif candidate == seen:
                        is_dup = True
                        break
                except Exception:
                    pass  # If comparison fails, treat as non-duplicate
            if not is_dup:
                unique.append(candidate)
        return unique
    














    
class Remove1stImageOfBatch(io.ComfyNode):
    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="Remove1stImage",
            display_name="Remove 1st image",
            description="shiiiiiiiiiit",
            category="image/batch",
            inputs=[
                io.Image.Input("IMAGE", tooltip=""),
            ],
            outputs=[
                io.Image.Output("IMAGE"),
            ],
        )

    @classmethod
    def execute(cls, images) -> io.NodeOutput:
        batch_cut = images[1:]
        return io.NodeOutput(batch_cut)
    









    
class ImageListToImageBatch(io.ComfyNode):
    #special unwrap b/c comfyui is w/e
    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="ImageListToImageBatch",
            display_name="'Image List' to Batch",
            description="comfyUI is doing some nutty shit, custom node just to unwrap lists, which aren't actually list b/c comfyUI fucking around with it",
            category="utils/list",
            inputs=[
                io.Image.Input("images", tooltip=""),
            ],
            outputs=[
                io.Image.Output("IMAGE"),  # remove is_output_list=True
            ],
        )

    # claude
    @classmethod
    def execute(cls, images) -> io.NodeOutput:

        # Normalize whatever ComfyUI throws at us into a flat list of tensors
        def flatten(x):
            if isinstance(x, torch.Tensor):
                # Could be [B,H,W,C] — split into list of per-batch tensors
                return [x[i:i+1] for i in range(x.shape[0])] if x.dim() == 4 else [x]
            elif isinstance(x, list):
                result = []
                for item in x:
                    result.extend(flatten(item))
                return result
            else:
                raise TypeError(f"Unexpected type in images: {type(x)}")

        flat = flatten(images)

        if not flat:
            raise ValueError("No tensors found in images input")

        shape = flat[0].shape[1:3]  # [H, W]
        out = []

        for img in flat:
            if img.shape[1:3] != shape:
                img = comfy.utils.common_upscale(
                    img.permute([0, 3, 1, 2]),
                    shape[1], shape[0],
                    upscale_method='bicubic',
                    crop='center'
                ).permute([0, 2, 3, 1])
            out.append(img)

        return io.NodeOutput(torch.cat(out, dim=0))


