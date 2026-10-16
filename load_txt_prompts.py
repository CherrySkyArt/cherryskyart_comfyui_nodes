import os
import re

from comfy_api.latest import io


def _natural_key(s: str):
    # "file2" sorts before "file10"
    return [int(t) if t.isdigit() else t.lower() for t in re.split(r"(\d+)", s)]


def _find_txt_files(root: str, depth: int) -> list[str]:
    root = os.path.normpath(root)
    root_level = root.rstrip(os.sep).count(os.sep)
    found = []
    for dirpath, dirnames, filenames in os.walk(root):
        level = dirpath.rstrip(os.sep).count(os.sep) - root_level
        if level >= depth:
            dirnames[:] = []  # don't descend any further
        else:
            dirnames.sort(key=_natural_key)
        for name in sorted(filenames, key=_natural_key):
            if name.lower().endswith(".txt"):
                found.append(os.path.join(dirpath, name))
    return found


def _read_text(path: str) -> str:
    for enc in ("utf-8-sig", "cp1252"):
        try:
            with open(path, "r", encoding=enc) as f:
                return f.read()
        except UnicodeDecodeError:
            continue
    with open(path, "r", encoding="utf-8", errors="replace") as f:
        return f.read()


def _clean_path(directory: str) -> str:
    # Windows "Copy as path" wraps in quotes
    return directory.strip().strip('"').strip("'")


class LoadTxtPrompts(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="CherryLoadTxtPrompts",
            display_name="Load TXT Prompts 🍒",
            category="utils",
            description=(
                "Reads every .txt file in a directory (optionally recursing into subfolders) "
                "and outputs a list of prompts, one prompt per file."
            ),
            search_aliases=["txt", "prompt", "load", "folder", "directory", "batch", "cherry", "text", "files"],
            inputs=[
                io.String.Input(
                    "directory",
                    default="",
                    tooltip="Folder to read .txt files from, e.g. C:/my/path/folder/",
                ),
                io.Int.Input(
                    "depth",
                    default=0,
                    min=0,
                    max=100,
                    tooltip="How many subfolder levels to search. 0 = only the given folder.",
                ),
                io.Boolean.Input(
                    "skip_empty",
                    default=True,
                    tooltip="Skip .txt files that are empty or whitespace only.",
                ),
            ],
            outputs=[
                io.String.Output("prompts", display_name="prompts", is_output_list=True),
                io.String.Output("filenames", display_name="filenames", is_output_list=True),
                io.Int.Output("count", display_name="count"),
            ],
        )

    @classmethod
    def fingerprint_inputs(cls, directory: str, depth: int, skip_empty: bool):
        # Re-run when files are added/removed/edited, not only when inputs change
        root = _clean_path(directory)
        if not os.path.isdir(root):
            return f"missing:{root}"
        sig = []
        for p in _find_txt_files(root, depth):
            try:
                st = os.stat(p)
                sig.append((p, st.st_mtime_ns, st.st_size))
            except OSError:
                pass
        return str((depth, skip_empty, sig))

    @classmethod
    def execute(cls, directory: str, depth: int, skip_empty: bool) -> io.NodeOutput:
        root = _clean_path(directory)
        if not os.path.isdir(root):
            raise ValueError(f"Load TXT Prompts: directory not found: '{root}'")

        prompts, filenames = [], []
        for path in _find_txt_files(root, depth):
            text = _read_text(path).strip()
            if skip_empty and not text:
                continue
            prompts.append(text)
            filenames.append(os.path.relpath(path, root))

        if not prompts:
            raise ValueError(f"Load TXT Prompts: no .txt files found in '{root}' (depth={depth})")

        return io.NodeOutput(prompts, filenames, len(prompts))
