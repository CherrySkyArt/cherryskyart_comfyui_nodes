from pathlib import Path
import sys

_current_dir = Path(__file__).parent
_models_dir = _current_dir / "models"
for _p in [_current_dir, _models_dir]:
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

from typing_extensions import override
from comfy_api.latest import ComfyExtension, io
from .bskiconditioning import BskiVideoSmooth
from .first_last import FirstLast
# from .bskiUtils import MaskBatchSplitter
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

WEB_DIRECTORY = "./web"

NODE_CLASS_MAPPINGS = {
    "AppendAnyToList": AppendAnyToList,
    "BskiVideoSmooth": BskiVideoSmooth,
    "FirstLast": FirstLast,
    "KJBboxToComfyBBox": KJBboxToComfyBBox,
    "MaskGrowPlus": MaskGrowPlus,
    "ImageListToImageBatch": ImageListToImageBatch,
    "Remove1stImageOfBatch": Remove1stImageOfBatch,
    "CherrySAM3Segment": CherrySAM3Segment,
    "DepthMeasure": DepthMeasure,
    "LazySort": LazySort,
    "StringTemplate": StringPlus
    # "MaskBatchSplitter //Inspire": MaskBatchSplitter,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "AppendAnyToList": "Append Any To List",
    "BskiVideoSmooth": "Video Smooth :)",
    "FirstLast": "first last super cool",
    "MaskBatchSplitter //Inspire": "MaskBatchSplitter",
    "KJBboxToComfyBBox": "KJ Bbox To Comfy BBox",
    "MaskGrowPlus": "Grow Mask+",
    "ImageListToImageBatch": "Image List to Image Batch",
    "Remove1stImageOfBatch": "Remove 1st Entry",
    "CherrySAM3Segment": "SAM3 Segment (Cherry)",
    "DepthMeasure": "Depth Measure",
    "LazySort": "Lazy Sort",
    "StringTemplate": "String Template",
}


class CherryExtension(ComfyExtension):
    @override
    async def get_node_list(self) -> list[type[io.ComfyNode]]:
        return [AppendAnyToList, BskiVideoSmooth, FirstLast, KJBboxToComfyBBox, MaskGrowPlus, ImageListToImageBatch, Remove1stImageOfBatch, CherrySAM3Segment, DepthMeasure, LazySort, StringPlus]

async def comfy_entrypoint() -> CherryExtension:
    return CherryExtension()
