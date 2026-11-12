import json
import re
from shutil import copyfileobj
from collections import Counter
from fractions import Fraction
from math import isclose
from pathlib import Path

import imageio.v3 as iio
import Imath
import numpy as np
import OpenEXR
from PIL import Image

# A minus after "_" or "." marks a negative number; "shot-001" uses a separator.
FRAME_NUMBER = re.compile(r"(?P<sign>(?<![^_.])-)?(?P<digits>[0-9]+)$")
FRAME_LIMIT = 10**15
MAX_FRAME_PIXELS = 24_000_000


class FrameLimitError(ValueError):
    pass


def check_frame_size(width, height):
    if width < 1 or height < 1 or width * height > MAX_FRAME_PIXELS:
        raise FrameLimitError("Frames must have positive dimensions and at most 24 megapixels.")


def read_exr(source):
    exr = OpenEXR.InputFile(source)
    try:
        header = exr.header()
        data = header["dataWindow"]
        display = header["displayWindow"]
        width = data.max.x - data.min.x + 1
        height = data.max.y - data.min.y + 1
        display_width = display.max.x - display.min.x + 1
        display_height = display.max.y - display.min.y + 1
        check_frame_size(width, height)
        check_frame_size(display_width, display_height)
        pixel_type = Imath.PixelType(Imath.PixelType.FLOAT)
        channels = [
            np.frombuffer(exr.channel(channel, pixel_type), dtype=np.float32).reshape(height, width)
            for channel in "RGB"
        ]
        image = np.stack(channels, axis=-1)
        if data != display:
            canvas = np.zeros((display_height, display_width, 3), dtype=np.float32)
            x0, x1 = max(data.min.x, display.min.x), min(data.max.x, display.max.x) + 1
            y0, y1 = max(data.min.y, display.min.y), min(data.max.y, display.max.y) + 1
            if x0 < x1 and y0 < y1:
                canvas[y0 - display.min.y:y1 - display.min.y, x0 - display.min.x:x1 - display.min.x] = (
                    image[y0 - data.min.y:y1 - data.min.y, x0 - data.min.x:x1 - data.min.x]
                )
            image = canvas
        rate = header.get("framesPerSecond")
        fps = Fraction(rate.n, rate.d) if rate and rate.n > 0 and rate.d > 0 else None
        return image, fps
    finally:
        exr.close()


def tonemap(image):
    image = np.nan_to_num(image, nan=0.0, posinf=1.0, neginf=0.0)
    image = np.clip(image, 0, None)
    image = image / (1.0 + image)
    return (np.power(image, 1 / 2.2) * 255).astype(np.uint8)


def decode_frame(source, suffix):
    if suffix.lower() == ".exr":
        image, rate = read_exr(source)
        return tonemap(image), rate

    with Image.open(source) as header:
        check_frame_size(*header.size)
    if hasattr(source, "seek"):
        source.seek(0)
    image = iio.imread(source, index=0, plugin="pillow")
    if image.ndim not in (2, 3) or (image.ndim == 3 and image.shape[2] not in (2, 3, 4)):
        raise ValueError("Expected a single grayscale or color image.")
    if image.dtype != np.uint8:
        if np.issubdtype(image.dtype, np.integer):
            image = image.astype(np.float64) / np.iinfo(image.dtype).max
        image = (np.clip(image, 0, 1) * 255).astype(np.uint8)
    return image, None


def missing_ranges(numbers, first, last):
    if first is None:
        return []
    previous = first - 1
    gaps = []
    for number in [*numbers, last + 1]:
        if number > previous + 1:
            gaps.append({"start": previous + 1, "end": number - 1})
        previous = number
    return gaps


def build_sequence(uploads, folder, sequence_id, supplied_rate):
    frame_dir = folder / "frames"
    frame_dir.mkdir()
    originals = folder / "originals"
    originals.mkdir()
    candidates = []
    for upload in uploads:
        name = Path((upload.filename or "").replace("\\", "/")).name
        match = FRAME_NUMBER.search(Path(name).stem)
        digits = match["digits"].lstrip("0") if match else ""
        number = None
        if match and len(digits) <= 15:
            number = int((match["sign"] or "") + (digits or "0"))
        candidates.append((upload, name, number, match is not None))

    numbered = any(item[3] for item in candidates)
    if not numbered:
        candidates = [(upload, name, index, False) for index, (upload, name, _, _) in enumerate(candidates)]
    counts = Counter(number for _, _, number, _ in candidates if number is not None)
    first = min(counts, default=None)
    last = max(counts, default=None)

    frames = []
    failures = []
    embedded_rate = None
    for upload, name, number, _ in candidates:
        failure = {"source_name": name, "frame_number": number}
        if number is None:
            failures.append({**failure, "error": "Expected a trailing frame number of at most 15 significant digits."})
            continue
        if counts[number] > 1:
            failures.append({**failure, "error": "Duplicate source frame number."})
            continue

        # COLMAP pairs by filename order. The negative prefix sorts before "_".
        filename = f"frame_{number:015d}.png" if number >= 0 else f"frame-{FRAME_LIMIT + number:015d}.png"
        path = frame_dir / filename
        original_name = f"{Path(filename).stem}{Path(name).suffix.lower()}"
        original_path = originals / original_name
        with original_path.open("wb") as original:
            copyfileobj(upload.file, original, length=1024 * 1024)
        try:
            with original_path.open("rb") as original:
                image, rate = decode_frame(original, Path(name).suffix)
        except (FrameLimitError, Image.DecompressionBombError):
            original_path.unlink()
            failures.append({**failure, "error": "Frame exceeds the 24-megapixel input limit."})
            continue
        except (ValueError, OSError, RuntimeError, TypeError, SyntaxError, OverflowError):
            original_path.unlink()
            failures.append({**failure, "error": "Could not decode this image."})
            continue
        if rate is not None and embedded_rate is not None and rate != embedded_rate:
            raise ValueError("The uploaded EXR frames declare different source frame rates.")
        if rate is not None:
            embedded_rate = rate
        iio.imwrite(path, image, plugin="pillow")
        height, width = image.shape[:2]
        frames.append({
            "frame_number": number,
            "source_name": name,
            "filename": filename,
            "original_media": f"originals/{original_name}",
            "url": f"/frames/{sequence_id}/{filename}",
            "width": width,
            "height": height,
        })

    if embedded_rate and supplied_rate and not isclose(float(embedded_rate), float(supplied_rate), rel_tol=1e-5):
        raise ValueError("The supplied FPS conflicts with the embedded EXR frame rate.")
    rate = embedded_rate or supplied_rate
    frames.sort(key=lambda frame: frame["frame_number"])
    dimensions = {(frame["width"], frame["height"]) for frame in frames}
    resolution = None
    if len(dimensions) == 1:
        width, height = dimensions.pop()
        resolution = {"width": width, "height": height}
    return {
        "sequence_id": sequence_id,
        "status": "partial" if frames and failures else "done" if frames else "failed",
        "fps": float(rate) if rate else None,
        "fps_rational": [rate.numerator, rate.denominator] if rate else None,
        "fps_source": "exr" if embedded_rate else "provided" if supplied_rate else None,
        "resolution": resolution,
        "numbering": "source_filename" if numbered else "upload_order",
        "start_frame": first,
        "end_frame": last,
        "missing_frame_ranges": missing_ranges([frame["frame_number"] for frame in frames], first, last),
        "frames": [f"{sequence_id}/{frame['filename']}" for frame in frames],
        "frame_details": frames,
        "failed_uploads": failures,
    }


def write_json(path, value):
    with path.open("w", encoding="utf-8") as stream:
        json.dump(value, stream, indent=2, allow_nan=False)
        stream.write("\n")


def read_json(path):
    with path.open(encoding="utf-8") as stream:
        return json.load(stream)
