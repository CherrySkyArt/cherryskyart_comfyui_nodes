from pathlib import Path
import sys

_current_dir = Path(__file__).parent
_models_dir = _current_dir / "models"
for _p in [_current_dir, _models_dir]:
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

from typing_extensions import override
from comfy_api.latest import ComfyExtension, io
from .first_last import FirstLast
from .KJbox2Comfy import KJBboxToComfyBBox
from .mask_grow_plus import MaskGrowPlus
from .cherry_nodes import AppendAnyToList
from .cherry_nodes import ImageListToImageBatch
from .cherry_nodes import Remove1stImageOfBatch
from .sam3 import CherrySAM3Segment
from .depth_measure import DepthMeasure
from .lazy_sort import LazySort
# from .preview_any import PreviewAny
from .stringPlus import StringPlus
from .first_last_v2 import WanFLF2VMask_Cherry
from .first_last_v3 import WanImageToVideoSVIProFLF_Cherry
from .first_last_v4 import WanImageToVideoSVIProFLF_v4_Cherry
from .first_last_v2_old import WanFLF2VMask_Cherry_old
from .bbox_unpack import BoundingBoxUnpack
from .bbox_expand import BoundingBoxExpand
from .alpha_composite import AlphaComposite
from .text_concat import TextConcat
from .mask_white_percent import MaskWhitePercent

WEB_DIRECTORY = "./web"

NODE_CLASS_MAPPINGS = {
    "AppendAnyToList": AppendAnyToList,
    "FirstLast": FirstLast,
    "KJBboxToComfyBBox": KJBboxToComfyBBox,
    "MaskGrowPlus": MaskGrowPlus,
    "ImageListToImageBatch": ImageListToImageBatch,
    "Remove1stImageOfBatch": Remove1stImageOfBatch,
    "CherrySAM3Segment": CherrySAM3Segment,
    "DepthMeasure": DepthMeasure,
    "LazySort": LazySort,
    "CherryStringTemplate": StringPlus,
    "WanFLF2VMask_Cherry": WanFLF2VMask_Cherry,
    "WanImageToVideoSVIProFLF_Cherry": WanImageToVideoSVIProFLF_Cherry,
    "WanImageToVideoSVIProFLF_v4_Cherry": WanImageToVideoSVIProFLF_v4_Cherry,
    "WanFLF2VMask_Cherry_old": WanFLF2VMask_Cherry_old,
    "BoundingBoxUnpack": BoundingBoxUnpack,
    "AlphaComposite_Cherry": AlphaComposite,
    "TextConcat_Cherry": TextConcat,
    "BoundingBoxExpand": BoundingBoxExpand,
    "MaskWhitePercent": MaskWhitePercent,
    # "MaskBatchSplitter //Inspire": MaskBatchSplitter,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "AppendAnyToList": "Append Any To List",
    "FirstLast": "first last super cool",
    "MaskBatchSplitter //Inspire": "MaskBatchSplitter",
    "KJBboxToComfyBBox": "KJ Bbox To Comfy BBox",
    "MaskGrowPlus": "Grow Mask+",
    "ImageListToImageBatch": "Image List to Image Batch",
    "Remove1stImageOfBatch": "Remove 1st Entry",
    "CherrySAM3Segment": "SAM3 Segment (Cherry)",
    "DepthMeasure": "Depth Measure",
    "LazySort": "Lazy Sort",
    "CherryStringTemplate": "String Plus (Template) 🍒",
    "WanFLF2VMask_Cherry": "WanFLF2VMask_Cherry",
    "WanImageToVideoSVIProFLF_Cherry": "WanImageToVideoSVIProFLF_Cherry",
    "WanImageToVideoSVIProFLF_v4_Cherry": "WanImageToVideoSVIProFLF_v4_Cherry",
    "WanFLF2VMask_Cherry_old": "WanFLF2VMask_Cherry_old",
    "BoundingBoxUnpack": "Bounding Box Unpack 🍒",
    "AlphaComposite_Cherry": "Alpha Composite 🍒",
    "TextConcat_Cherry": "Text Concat 🍒",
    "BoundingBoxExpand": "Bounding Box Expand 🍒",
    "MaskWhitePercent": "Mask White Percent 🍒",
}


class CherryExtension(ComfyExtension):
    @override
    async def get_node_list(self) -> list[type[io.ComfyNode]]:
        return [AppendAnyToList, FirstLast, KJBboxToComfyBBox, MaskGrowPlus, ImageListToImageBatch, Remove1stImageOfBatch, CherrySAM3Segment, DepthMeasure, LazySort, StringPlus, WanFLF2VMask_Cherry, WanImageToVideoSVIProFLF_Cherry, WanFLF2VMask_Cherry_old, BoundingBoxUnpack, AlphaComposite, TextConcat, BoundingBoxExpand, MaskWhitePercent]

async def comfy_entrypoint() -> CherryExtension:
    return CherryExtension()
