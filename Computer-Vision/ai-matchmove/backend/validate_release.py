"""Run a real release-validation job on saved footage or a local image set."""

import argparse
from contextlib import ExitStack
from pathlib import Path
from time import sleep

from fastapi.testclient import TestClient

from . import main


def run(args):
    main.DATA_DIR = args.data_root.resolve()
    with TestClient(main.app) as client, ExitStack() as files:
        if args.images or args.video:
            paths = [args.video] if args.video else sorted(
                path for path in args.images.iterdir() if path.suffix.lower() in {".jpg", ".jpeg", ".png", ".exr"}
            )
            upload = client.post("/upload/", files=[
                ("files", (path.name, files.enter_context(path.open("rb")))) for path in paths
            ], data={"video_stride": args.video_stride, **({"fps": args.fps} if args.fps else {})})
            upload.raise_for_status()
            sequence_id = upload.json()["sequence_id"]
        else:
            sequence_id = args.sequence

        payload = {"operation": "solve", "max_frames": args.max_frames}
        if args.foreground:
            payload.update(operation="foreground", foreground={"discover": True})
        if args.compare_mask_id:
            payload.update(operation="compare", mask_set_id=args.compare_mask_id)
        response = client.post(f"/sequences/{sequence_id}/jobs", json=payload)
        response.raise_for_status()
        url = response.json()["status_url"]
        print(url, flush=True)
        previous = None
        while True:
            response = client.get(url)
            response.raise_for_status()
            state = response.json()
            if state["stage"] != previous:
                print(state["stage"], flush=True)
                previous = state["stage"]
            if state["status"] not in {"queued", "running"}:
                break
            sleep(0.6)
        result = state.get("result", {})
        if args.foreground:
            print({key: result.get(key) for key in (
                "analyzed_frames", "frames_with_proposals", "masked_frames", "temporal_proposals", "elapsed_seconds"
            )})
        elif args.compare_mask_id and state["status"] == "done":
            for mode in ("baseline", "masked"):
                print(mode, result[mode]["status"], result[mode].get("error", ""), result[mode]["statistics"])
        else:
            print(result.get("statistics", state.get("measurements", {})))
        print(state["status"], state.get("error", ""), flush=True)
        return state["status"] == "done"


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, default=Path("backend/data"))
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--sequence")
    source.add_argument("--images", type=Path)
    source.add_argument("--video", type=Path)
    operation = parser.add_mutually_exclusive_group()
    operation.add_argument("--foreground", action="store_true")
    operation.add_argument("--compare-mask-id")
    parser.add_argument("--fps")
    parser.add_argument("--video-stride", type=int, default=1)
    parser.add_argument("--max-frames", type=int, default=24)
    arguments = parser.parse_args()
    if arguments.compare_mask_id and not arguments.sequence:
        parser.error("--compare-mask-id requires an existing --sequence")
    raise SystemExit(0 if run(arguments) else 1)
