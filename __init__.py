from pathlib import Path
import sys

_current_dir = Path(__file__).parent
_models_dir = _current_dir / "models"
for _p in [_current_dir, _models_dir]:
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

from typing_extensions import override
from comfy_api.latest import ComfyExtension, io
from .KJbox2Comfy import KJBboxToComfyBBox
from .mask_grow_plus import MaskGrowPlus
from .cherry_nodes import AppendAnyToList
from .cherry_nodes import ImageListToImageBatch
from .cherry_nodes import Remove1stImageOfBatch
from .sam3 import CherrySAM3Segment
from .depth_measure import DepthMeasure
from .lazy_sort import LazySort
from .stringPlus import StringPlus
from .longest_string import LongestString
# from .preview_any import PreviewAny
# from .first_last import FirstLast
# from .first_last_v2 import WanFLF2VMask_Cherry
# from .first_last_v3 import WanImageToVideoSVIProFLF_Cherry
# from .first_last_v4 import WanImageToVideoSVIProFLF_v4_Cherry
# from .first_last_v2_old import WanFLF2VMask_Cherry_old
from .bbox_unpack import BoundingBoxUnpack
from .bbox_expand import BoundingBoxExpand
from .alpha_composite import AlphaComposite
# from .text_concat import TextConcat
from .mask_white_percent import MaskWhitePercent
from .png_in_memory import PngInMemory
from .stackable_rows import StackableRows
# from .text_editor_minimax import TextEditorMinimax

WEB_DIRECTORY = "./web"


class CherryExtension(ComfyExtension):
    @override
    async def get_node_list(self) -> list[type[io.ComfyNode]]:
        return [AppendAnyToList, KJBboxToComfyBBox, MaskGrowPlus, ImageListToImageBatch, Remove1stImageOfBatch, CherrySAM3Segment, DepthMeasure, LazySort, StringPlus, BoundingBoxUnpack, AlphaComposite, BoundingBoxExpand, MaskWhitePercent, PngInMemory, StackableRows, LongestString]

async def comfy_entrypoint() -> CherryExtension:
    return CherryExtension()
