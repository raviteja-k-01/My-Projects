"""Decode video through the installed FFmpeg tools, preserving presentation timestamps."""

import json
import shutil
import subprocess
from fractions import Fraction
from pathlib import Path

from PIL import Image

if __package__:
    from .sequence import check_frame_size
else:
    from sequence import check_frame_size

VIDEO_EXTENSIONS = {".mp4", ".mov", ".mkv", ".webm", ".avi", ".m4v", ".mpeg", ".mpg", ".mts", ".mxf"}
INPUT_FORMATS = "mov,mp4,m4a,3gp,3g2,mj2,matroska,webm,avi,mpeg,mpegts,mxf"


def run_tool(command, timeout):
    try:
        result = subprocess.run(command, capture_output=True, timeout=timeout)
    except FileNotFoundError as exc:
        raise ValueError("Install FFmpeg and ffprobe in the backend environment to upload video.") from exc
    except subprocess.TimeoutExpired as exc:
        raise ValueError("Video processing timed out. Try a shorter clip.") from exc
    if result.returncode:
        raise ValueError("FFmpeg could not decode the video. Check that the file is complete and supported.")
    return result.stdout


def decode_video(upload, folder, sequence_id, stride, supplied_rate=None):
    name = Path(upload.filename.replace(chr(92), "/")).name
    source = folder / ("source" + Path(name).suffix.lower())
    with source.open("wb") as stream:
        shutil.copyfileobj(upload.file, stream, length=1024 * 1024)
    retained = False
    try:
        metadata = json.loads(run_tool([
            "ffprobe", "-v", "error", "-protocol_whitelist", "file", "-format_whitelist", INPUT_FORMATS,
            "-select_streams", "v:0", "-show_frames", "-show_streams",
            "-show_entries", "frame=best_effort_timestamp_time:stream=avg_frame_rate,r_frame_rate,"
            "codec_name,width,height,sample_aspect_ratio,duration,color_space,color_transfer,color_primaries,color_range:"
            "stream_tags=make,model,lens_model,focal_length,creation_time:"
            "format_tags=make,model,lens_model,focal_length,com.apple.quicktime.make,com.apple.quicktime.model",
            "-of", "json", str(source),
        ], 120))
        streams, timing = metadata.get("streams", []), metadata.get("frames", [])
        if not streams or not timing:
            raise ValueError("The upload has no decodable video frames.")
        stream = streams[0]
        width, height = stream.get("width", 0), stream.get("height", 0)
        check_frame_size(width, height)
        numbers = list(range(0, len(timing), stride))
        if len(numbers) > 1200:
            raise ValueError("This video would decode more than 1,200 frames. Increase the video sampling step.")
        if width * height * len(numbers) * 4 > shutil.disk_usage(folder).free:
            raise ValueError("Insufficient free disk space for this decode. Increase the video sampling step.")
        frames_dir = folder / "frames"
        frames_dir.mkdir()
        run_tool([
            "ffmpeg", "-nostdin", "-v", "error", "-xerror", "-protocol_whitelist", "file",
            "-format_whitelist", INPUT_FORMATS, "-threads", "2", "-i", str(source),
            "-map", "0:v:0", "-an", "-sn", "-vf", f"select=not(mod(n\\,{stride}))",
            "-fps_mode", "passthrough", "-threads", "2", "-compression_level", "2", "-start_number", "0",
            str(frames_dir / "decode_%06d.png"),
        ], 900)
        decoded = sorted(frames_dir.glob("decode_*.png"))
        if len(decoded) != len(numbers):
            raise ValueError("Decoded frames do not match the video's presentation metadata.")
        stamps = [float(frame["best_effort_timestamp_time"]) if "best_effort_timestamp_time" in frame else None for frame in timing]
        first_timestamp = stamps[0]
        rate = None
        for key in ("avg_frame_rate", "r_frame_rate"):
            try:
                candidate = Fraction(stream.get(key, "0"))
                if candidate > 0:
                    rate = candidate
                    break
            except (ValueError, ZeroDivisionError):
                pass
        frames = []
        for path, number in zip(decoded, numbers, strict=True):
            filename = f"frame_{number:015d}.png"
            with Image.open(path) as image:
                width, height = image.size
            path.rename(frames_dir / filename)
            timestamp = stamps[number]
            frames.append({
                "frame_number": number, "source_name": f"{name} frame {number}", "filename": filename,
                "url": f"/frames/{sequence_id}/{filename}", "width": width, "height": height,
                "timestamp_seconds": timestamp - first_timestamp if timestamp is not None and first_timestamp is not None else None,
            })
        dimensions = {(frame["width"], frame["height"]) for frame in frames}
        resolution = dict(zip(("width", "height"), next(iter(dimensions)))) if len(dimensions) == 1 else None
        aspect = stream.get("sample_aspect_ratio", "1:1")
        if (resolution and stream.get("width") != stream.get("height")
                and (resolution["width"], resolution["height"]) == (stream.get("height"), stream.get("width"))
                and ":" in aspect and aspect != "0:1"):
            aspect = ":".join(reversed(aspect.split(":")))
        fps_source = "video" if rate else "provided" if supplied_rate else None
        rate = rate or supplied_rate
        retained = True
        return {
            "sequence_id": sequence_id, "status": "done", "input_type": "video", "source_name": name,
            "original_media": source.name,
            "numbering": "video_presentation_order", "source_frame_count": len(timing), "video_stride": stride,
            "fps": float(rate) if rate else None, "fps_rational": [rate.numerator, rate.denominator] if rate else None,
            "fps_source": fps_source, "resolution": resolution,
            "video_metadata": {
                "codec": stream.get("codec_name"), "sample_aspect_ratio": aspect,
                "duration_seconds": float(stream["duration"]) if stream.get("duration") not in (None, "N/A") else None,
                "camera_tags": {**metadata.get("format", {}).get("tags", {}), **stream.get("tags", {})},
                "color": {key: stream[key] for key in ("color_space", "color_transfer", "color_primaries", "color_range")
                          if key in stream},
            },
            "start_frame": 0, "end_frame": len(timing) - 1, "missing_frame_ranges": [],
            "frames": [f"{sequence_id}/{frame['filename']}" for frame in frames], "frame_details": frames,
            "failed_uploads": [],
        }
    finally:
        if not retained:
            source.unlink()
