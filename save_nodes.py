import json
import os

import numpy as np
from PIL import Image
from PIL.PngImagePlugin import PngInfo

import folder_paths
from comfy.cli_args import args
from comfy_api.latest import io, ui, Input, Types


def reserve_unique_file(folder: str, stem: str, ext: str, start: int = 0) -> tuple[str, int]:
    """
    Atomically create an empty file at the first free name and return (filename, n).

    Tries stem.ext, then stem_01.ext ... stem_99.ext, stem_100.ext, ... The file is
    created with O_EXCL ("xb"), so an existing file is never touched, even if another
    process grabs a name between our check and our write.
    """
    n = start
    while True:
        file = f"{stem}{ext}" if n == 0 else f"{stem}_{n:02}{ext}"
        try:
            with open(os.path.join(folder, file), "xb"):
                pass
            return file, n
        except FileExistsError:
            n += 1


def _remove_quietly(path: str):
    try:
        os.remove(path)
    except OSError:
        pass


class SaveImageCherry(io.ComfyNode):
    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="SaveImage_Cherry",
            display_name="Save Image 🍒",
            category="image",
            search_aliases=["cherry", "save", "save image", "export image", "output image", "write image", "no counter"],
            description="Saves the input images to your ComfyUI output directory. Same as Save Image, but only appends a number (_01, _02, ...) when the filename already exists. Never overwrites.",
            inputs=[
                io.Image.Input("images", tooltip="The images to save."),
                io.String.Input("filename_prefix", default="ComfyUI", tooltip="The filename for the file to save. This may include formatting information such as %date:yyyy-MM-dd% or %Empty Latent Image.width% to include values from nodes. A number is only appended if the file already exists."),
            ],
            outputs=[io.Image.Output(display_name="images")],
            hidden=[io.Hidden.prompt, io.Hidden.extra_pnginfo],
            is_output_node=True,
        )

    @classmethod
    def execute(cls, images, filename_prefix) -> io.NodeOutput:
        full_output_folder, filename, _counter, subfolder, filename_prefix = folder_paths.get_save_image_path(
            filename_prefix, folder_paths.get_output_directory(), images[0].shape[1], images[0].shape[0]
        )
        metadata = None
        if not args.disable_metadata:
            metadata = PngInfo()
            if cls.hidden.prompt is not None:
                metadata.add_text("prompt", json.dumps(cls.hidden.prompt))
            if cls.hidden.extra_pnginfo is not None:
                for x in cls.hidden.extra_pnginfo:
                    metadata.add_text(x, json.dumps(cls.hidden.extra_pnginfo[x]))

        results = []
        next_n = {}  # stem -> first suffix worth trying, so big batches don't rescan taken names
        for (batch_number, image) in enumerate(images):
            i = 255. * image.cpu().numpy()
            img = Image.fromarray(np.clip(i, 0, 255).astype(np.uint8))

            stem = filename.replace("%batch_num%", str(batch_number))
            file, n = reserve_unique_file(full_output_folder, stem, ".png", next_n.get(stem, 0))
            next_n[stem] = n + 1
            path = os.path.join(full_output_folder, file)
            try:
                img.save(path, pnginfo=metadata, compress_level=4)
            except Exception:
                _remove_quietly(path)
                raise
            results.append(ui.SavedResult(file, subfolder, io.FolderType.output))

        return io.NodeOutput(images, ui=ui.SavedImages(results))


class SaveVideoCherry(io.ComfyNode):
    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="SaveVideo_Cherry",
            display_name="Save Video 🍒",
            category="video",
            search_aliases=["cherry", "save", "save video", "export video", "no counter"],
            description="Saves the input video to your ComfyUI output directory. Same as Save Video, but only appends a number (_01, _02, ...) when the filename already exists. Never overwrites.",
            inputs=[
                io.Video.Input("video", tooltip="The video to save."),
                io.String.Input("filename_prefix", default="video/ComfyUI", tooltip="The filename for the file to save. This may include formatting information such as %date:yyyy-MM-dd% or %Empty Latent Image.width% to include values from nodes. A number is only appended if the file already exists."),
                io.Combo.Input("format", options=Types.VideoContainer.as_input(), default="auto", tooltip="The format to save the video as."),
                io.Combo.Input("codec", options=Types.VideoCodec.as_input(), default="auto", tooltip="The codec to use for the video."),
            ],
            outputs=[io.Video.Output(display_name="video")],
            hidden=[io.Hidden.prompt, io.Hidden.extra_pnginfo],
            is_output_node=True,
        )

    @classmethod
    def execute(cls, video: Input.Video, filename_prefix, format: str, codec) -> io.NodeOutput:
        width, height = video.get_dimensions()
        full_output_folder, filename, _counter, subfolder, filename_prefix = folder_paths.get_save_image_path(
            filename_prefix,
            folder_paths.get_output_directory(),
            width,
            height
        )
        saved_metadata = None
        if not args.disable_metadata:
            metadata = {}
            if cls.hidden.extra_pnginfo is not None:
                metadata.update(cls.hidden.extra_pnginfo)
            if cls.hidden.prompt is not None:
                metadata["prompt"] = cls.hidden.prompt
            if len(metadata) > 0:
                saved_metadata = metadata

        file, _n = reserve_unique_file(full_output_folder, filename, f".{Types.VideoContainer.get_extension(format)}")
        path = os.path.join(full_output_folder, file)
        try:
            video.save_to(
                path,
                format=Types.VideoContainer(format),
                codec=codec,
                metadata=saved_metadata
            )
        except Exception:
            _remove_quietly(path)
            raise

        return io.NodeOutput(video, ui=ui.PreviewVideo([ui.SavedResult(file, subfolder, io.FolderType.output)]))
