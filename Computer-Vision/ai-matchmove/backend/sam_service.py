"""Headless SAM ViT-B prompts in original image coordinates."""

import os
from functools import lru_cache
from pathlib import Path
from pickle import UnpicklingError
from threading import Lock
from typing import Literal

import numpy as np
from PIL import Image
from pydantic import BaseModel, Field


class SamUnavailable(Exception):
    pass


class Prompt(BaseModel):
    frame_number: int
    box: tuple[float, float, float, float] | None = None
    points: list[tuple[float, float, Literal[0, 1]]] = Field(default_factory=list)
    decision: Literal["auto", "exclude", "keep"] = "auto"


def validate_prompt(prompt, width, height):
    if not prompt.box and not prompt.points:
        raise ValueError("Supply at least one point or a bounding box.")
    if prompt.box:
        x0, y0, x1, y1 = prompt.box
        if not (0 <= x0 < x1 <= width and 0 <= y0 < y1 <= height):
            raise ValueError("Bounding box is outside the source frame or has no area.")
    if any(not (0 <= x < width and 0 <= y < height) for x, y, _ in prompt.points):
        raise ValueError("Prompt point is outside the source frame.")
    if not prompt.box and not any(label for _, _, label in prompt.points):
        raise ValueError("A point-only prompt needs a foreground point.")


class SamService:
    def __init__(self, predictor, torch, device, input_size):
        self.predictor = predictor
        self.torch = torch
        self.device = device
        self.input_size = input_size
        self.lock = Lock()

    def segment(self, image, prompts):
        """Accept a PIL RGB image or uint8 H×W×3 array and return aligned boolean masks."""
        if not prompts:
            return []
        rgb = np.asarray(image.convert("RGB") if isinstance(image, Image.Image) else image)
        if rgb.dtype != np.uint8 or rgb.ndim != 3 or rgb.shape[2] != 3:
            raise ValueError("Expected an 8-bit RGB image.")
        height, width = rgb.shape[:2]
        for prompt in prompts:
            validate_prompt(prompt, width, height)
        results = []
        with self.lock, self.torch.inference_mode():
            try:
                self.predictor.set_image(np.ascontiguousarray(rgb))
                for prompt in prompts:
                    points = np.asarray(prompt.points, dtype=np.float32)
                    masks, scores, _ = self.predictor.predict(
                        point_coords=points[:, :2] if points.size else None,
                        point_labels=points[:, 2].astype(np.int32) if points.size else None,
                        box=np.asarray(prompt.box, dtype=np.float32) if prompt.box else None,
                        multimask_output=prompt.box is None,
                    )
                    best = int(np.argmax(scores))
                    mask = masks[best].astype(bool)
                    if mask.shape != (height, width):
                        raise RuntimeError("SAM returned a mask at the wrong resolution.")
                    results.append((mask, float(scores[best])))
            except self.torch.OutOfMemoryError as exc:
                raise SamUnavailable(
                    "SAM ran out of memory. Retry with MATCHMOVE_SAM_INPUT_SIZE=512 or MATCHMOVE_SAM_DEVICE=cpu."
                ) from exc
            finally:
                self.predictor.reset_image()
        return results


_load_lock = Lock()


@lru_cache(maxsize=1)
def _load(checkpoint, device, input_size):
    try:
        import torch
        from segment_anything import SamPredictor, sam_model_registry
    except (ImportError, OSError, RuntimeError) as exc:
        raise SamUnavailable(f"SAM runtime could not load: {exc}") from exc

    if device == "cuda" and not torch.cuda.is_available():
        raise SamUnavailable("SAM CUDA was requested, but this PyTorch environment cannot use CUDA.")
    try:
        if device == "cpu":
            torch.set_num_threads(min(os.cpu_count() or 1, 4))
        model = sam_model_registry["vit_b"]()
        model.load_state_dict(torch.load(checkpoint, map_location="cpu", weights_only=True))
        if input_size != model.image_encoder.img_size:
            # Keep the ViT-B weights and resize the learned absolute position grid.
            # SAM handles relative-position interpolation and source-coordinate transforms.
            encoder = model.image_encoder
            tokens = input_size // encoder.patch_embed.proj.kernel_size[0]
            grid = torch.nn.functional.interpolate(
                encoder.pos_embed.permute(0, 3, 1, 2), size=(tokens, tokens),
                mode="bicubic", align_corners=False,
            )
            encoder.pos_embed = torch.nn.Parameter(grid.permute(0, 2, 3, 1).detach())
            encoder.img_size = input_size
            model.prompt_encoder.input_image_size = (input_size, input_size)
            model.prompt_encoder.image_embedding_size = (tokens, tokens)
            model.prompt_encoder.mask_input_size = (4 * tokens, 4 * tokens)
        model = model.to(device).eval()
    except torch.OutOfMemoryError as exc:
        raise SamUnavailable("SAM ran out of memory while loading ViT-B. Retry with MATCHMOVE_SAM_DEVICE=cpu.") from exc
    except (OSError, RuntimeError, ValueError, EOFError, UnpicklingError) as exc:
        raise SamUnavailable(f"SAM ViT-B checkpoint could not load: {exc}") from exc
    return SamService(SamPredictor(model), torch, device, input_size)


def get_sam():
    setting = os.environ.get("MATCHMOVE_SAM_CHECKPOINT")
    if not setting or not Path(setting).is_file():
        raise SamUnavailable("Set MATCHMOVE_SAM_CHECKPOINT to the existing SAM ViT-B checkpoint.")
    device = os.environ.get("MATCHMOVE_SAM_DEVICE", "cpu")
    if device not in {"cpu", "cuda"}:
        raise SamUnavailable("MATCHMOVE_SAM_DEVICE must be cpu or cuda.")
    input_size = os.environ.get("MATCHMOVE_SAM_INPUT_SIZE", "512")
    if input_size not in {"512", "1024"}:
        raise SamUnavailable("MATCHMOVE_SAM_INPUT_SIZE must be 512 or 1024.")
    with _load_lock:
        return _load(Path(setting).resolve(), device, int(input_size))


def export_mask(path, mask):
    Image.fromarray(np.asarray(mask, dtype=np.uint8) * 255).save(path)


def save_preview(path, image, candidate, excluded):
    preview = image.copy()
    for region, color in [(candidate, (40, 150, 255)), (excluded, (255, 55, 45))]:
        preview[region] = 0.5 * preview[region] + 0.5 * np.asarray(color)
    Image.fromarray(preview).save(path)
