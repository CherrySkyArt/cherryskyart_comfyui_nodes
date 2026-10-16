from typing_extensions import override
from comfy_api.latest import ComfyExtension, io
from .KJbox2Comfy import KJBboxToComfyBBox
from .mask_grow_plus import MaskGrowPlus
from .cherry_nodes import AppendAnyToList
from .cherry_nodes import ImageListToImageBatch
from .cherry_nodes import Remove1stImageOfBatch
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
from .ai_image_detect import AIImageDetect
from .exif_spoof import ExifSpoof
from .play_sound import PlaySoundOnFinish
from .load_txt_prompts import LoadTxtPrompts
from .float_slider import FloatSlider
from .save_nodes import SaveImageCherry, SaveVideoCherry
# from .text_editor_minimax import TextEditorMinimax

WEB_DIRECTORY = "./web"


class CherryExtension(ComfyExtension):
    @override
    async def get_node_list(self) -> list[type[io.ComfyNode]]:
        return [AppendAnyToList, KJBboxToComfyBBox, MaskGrowPlus, ImageListToImageBatch, Remove1stImageOfBatch, DepthMeasure, LazySort, StringPlus, BoundingBoxUnpack, AlphaComposite, BoundingBoxExpand, MaskWhitePercent, PngInMemory, StackableRows, LongestString, AIImageDetect, ExifSpoof, PlaySoundOnFinish, LoadTxtPrompts, FloatSlider, SaveImageCherry, SaveVideoCherry]

async def comfy_entrypoint() -> CherryExtension:
    return CherryExtension()
