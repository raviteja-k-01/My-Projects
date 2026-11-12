# Release validation

## Results

| Input | Configuration | Outcome |
| --- | --- | --- |
| Known-motion synthetic scene | Nine images, saved regression | 9/9 registered, 12,907 points, 0.111 px mean error. The saved solution has 12,897 tracks observed in at least three images. |
| 1920×1080 reference clip | 24 sampled frames from 139 source frames | 24/24 registered, 10,874 points, 0.695 px, 102.2 s. Saved camera, tracks, PLY, and Blender ZIP. |
| Mixkit jogging bridge | 24 selected from 57 decoded frames, 284 source frames, 1920×1080 | 89 verified pairs, one connected 24-frame component, zero seed initializations, no model, 63.5 s. |
| Sceaux Castle image set | 11 JPEGs, 2832×2128 originals, 1920-pixel SIFT limit | 11/11 registered, 8,653 points, 0.605 px, 47.8 s. |

The [Sceaux Castle sample](https://github.com/openMVG/ImageDataset_SceauxCastle) is a separate real translating-camera image sequence. It has no video FPS. The source images and generated files are stored locally under the ignored data directory.

## Bridge investigation

The mapper reported no usable initial pair even after its built-in relaxations. The 16 strongest or widest probed pairs had provisional triangulation angles from approximately 0.11° to 1.43°, using a 2,304 px initial focal length. COLMAP classified the 89 stored verified pairs as uncalibrated epipolar geometry. This is evidence of weak triangulation for the tested pairs under that calibration. The exact physical camera motion and lens remain unknown.

The saved SAM comparison had proposals on 10 of 24 analyzed frames and exclusions on five. SAM removed 424 keypoints. Both runs retained 89 verified pairs and failed to initialize. Insufficient proposals and inconclusive motion validation limited coverage. The detector can propose a region in an adjacent frame when enough spatially distributed tracks support the transform. It reruns segmentation and motion checks on the target frame.

## Diagnostics and resource limits

Failed solve jobs retain downloadable JSON with the failure stage, correspondence counts, graph components, geometry types, sampled inlier coverage, provisional angles, mapper callbacks, candidate quality and attempt configuration. The probe is bounded to 16 pairs. Supported models require three registered images, 30 tracks observed in at least three images, finite camera translation, and mean reprojection error within COLMAP's default 4 px filtering scale. These are screening criteria for a sparse model.

A failed sequential solve can add up to 16 previously untested wider pairs once if existing geometry provides sufficient provisional angular evidence. It reuses the feature database and keeps the mapper thresholds. The bridge result did not meet the recovery gate. Uploads are limited to 2 GiB of received file data, 24 megapixels per frame, and 1,200 decoded video frames per sequence. Source dimensions stay available for tracks and exports. SIFT uses at most two CPU workers and a 1,920-pixel image limit.

## Reproduce

Run from the project root in the activated Python environment.

~~~sh
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=2
export MATCHMOVE_SAM_CHECKPOINT=/path/to/sam_vit_b.pth
python -m backend.validate_release --images /path/to/sequence
python -m backend.validate_release --video /path/to/clip.mp4 --foreground
~~~

The reference export ZIP passed integrity and script syntax checks. The exporter reproduced the saved camera and lens data.

## Release regression checks

The WSL run passed 35 tests and four subtests, including original SAM inference and a native pure-rotation rejection. A Chrome test with three flat frames exposed that COLMAP can store empty rows for attempted pairs. Pair metrics now count rows with positive support. The focused Windows suite passed 25 tests after that fix, and Chrome passed the real upload, failed solve, and diagnostics download with zero verified pairs. The production frontend build passed.

WSL interrupted a bridge masking run after 14 of 24 frames. A separate OpenCV probe replayed all 24 selected frames and found the same 10 direct-proposal frames plus one temporal proposal, at source frame 60 from frame 45. Only approved masks change feature extraction.

The targeted two-frame FastAPI run completed with real SAM in 22.6 s. It segmented the temporal proposal at frame 60, then retained the region because zero of its 27 tracks to frame 65 exceeded the motion residual threshold. The run approved zero exclusions. This verifies the temporal proposal and rejection path on the bridge clip. It does not establish improved exclusion coverage.

## Mask review and project recovery

Chrome opened the saved 57-frame bridge project using the project picker. Editing the prompt decision retained the saved mask. On source frame 280, **Keep region** saved a revision using the stored mask. The revised mask excluded zero pixels on that frame. The original excluded 8,016 pixels. Chrome restored the revised selection after refresh. A direct solve request against the unreviewed revision returned HTTP 409 before reconstruction.

In a separate Chrome run, explicitly reviewed masks from the known-motion synthetic fixture produced 9/9 registered frames, 12,164 points, and 0.111 px mean reprojection error. The feature audit found zero retained keypoints inside exclusions. The browser downloaded the real Blender ZIP. No new bridge camera solve was run. Automatic runner coverage remains unverified.

The backend uses PyCOLMAP 4.2.1, FastAPI 0.162.4, OpenCV 4.13.0.92, and PyTorch 2.5.1 CPU in the verified WSL environment. It loads the external original ViT-B checkpoint through `MATCHMOVE_SAM_CHECKPOINT`.

Full WSL regression with the checkpoint configured passed 52 tests and four subtests with no skips. The focused Windows run passed 26 tests. Three frontend policy tests and the Vite build passed. A native preflight on flat input returned zero verified pairs, published its report, and produced no reconstruction. Original PNG bytes were verified separately from working frames. The newly downloaded ZIP passed integrity checks and contains nine mapped poses and nine corrected plates.

A targeted real ViT-B inference on source frame 280 used a manually drawn runner box and point. It returned 6,540 mask pixels at 1920×1080 and zero pixels inside the old railing box. This was a prompted segmentation check. The bridge was not reconstructed again.
## Product workflow validation

Validation used FastAPI, Chrome, SAM ViT-B, PyCOLMAP 4.2.1, and Blender 4.0.2.

| Workflow | Verified behavior | Remaining coverage |
| --- | --- | --- |
| Footage and playback | Video upload, source numbering, resolution, presentation timestamps, browser playback and seek. Existing ingestion tests cover missing and rejected sequence frames and isolated uploads. | No new browser run for EXR. The working EXR frames are tone-mapped. |
| Foreground | Dominant-image-warp or epipolar background checks, source-aligned ViT-B masks, candidate navigation, preview, correction and revision recovery. Automatic exclusions still require explicit approval. | Moving-camera proposal recall is insufficient. Fixed-camera results include off-target suggestions. CUDA inference was not exercised. |
| Camera and quality | SIFT, descriptor matching, geometric verification, pose initialization, incremental registration and bundle adjustment on two successful real scenes. Per-frame observed reprojection residuals use the fitted camera model. Real unsuitable footage produces downloadable failure diagnostics. | Shared square-pixel SIMPLE_RADIAL lens. A calibrated focal length can seed the model. Monocular scale is arbitrary. |
| Jobs and persistence | Native cancellation, retry with original settings, saved solves, selected frame, mask revision and export recovery. Chrome reopened a project after restarting the backend. | One local worker. Interrupted jobs restart from their saved inputs. SAM finishes the current inference call before cancellation takes effect. |
| VFX export | Chrome prepared and retrieved packages. Blender reopened three saved files and evaluated pose keys, source frame mapping, pose availability, intrinsics, sparse point coordinates and projection. An image sequence exported at an explicitly supplied rational source FPS. | Exports contain poses and corrected plates for registered frames. Variable-rate footage uses average Blender FPS and keeps source timestamps. |

| Source | Registered / sampled | Sparse points | Mean reprojection error | Notes |
| --- | ---: | ---: | ---: | --- |
| 1920×1080 reference clip | 18 / 18 | 7,870 | 0.767 px | Chrome imported with step 8, played, sought, cancelled SIFT, retried, downloaded, then restored source frame 8 and the export. |
| Sceaux Castle real image sequence | 11 / 11 | 8,652 | 0.606 px | 29 verified pairs, 95.6% inliers. Chrome exported at the explicitly entered 24000/1001 FPS. |
| Mixkit jogging bridge | 0 / 16 | — | — | 49 verified pairs, 90.4% inliers, 0 initializations, strongest sampled provisional angle 1.47°. |
| OpenCV `vtest.avi` fixed camera | 0 / 12 | — | — | 33 verified pairs. Homographies dominated and no valid camera model was recovered. |
| DAVIS `parkour` | 24 / 24 in both modes | 2,214 baseline, 2,080 masked | 0.299 baseline, 0.306 masked | Real moving camera and subject. Eight annotation-guided SAM masks passed review against the published masks. |

Blender measured maximum undistorted projection differences of 0.00042 px across 233 projections for the supplied video, 0.00067 px across 142 for Sceaux Castle, and 0.00052 px across 310 for masked Parkour. These measure agreement with the exported camera and world coordinates. The supplied-video and Parkour files mark 121 and 76 source frames without solved poses. Sceaux Castle has no missing poses.

Real foreground checks used rough hand-labeled subject boxes at eight OpenCV frames and seven bridge frames, with a 0.20 overlap threshold for the proposal probe. The dominant-warp change increased fixed-camera box coverage from 5/8 to 8/8, with two additional boxes that did not overlap a label. Moving-camera coverage stayed at 1/7. A local-flow prototype produced four more off-target boxes on the bridge without finding the runner on additional frames, so it was left out.

The real SAM path produced candidates on 8/8 fixed-camera frames and 3/7 bridge frames in the global-sampling run. Eight fixed-camera exclusion suggestions overlapped the rough labels, and three fell outside them. One bridge SAM mask overlapped the runner, and zero qualified for exclusion. The rough labels cover the marked subjects. Chrome previewed the fixed-camera SAM output, saved a Keep correction using the stored mask, and restored the correction and the camera-solve failure.

The motion pipeline uses spatially balanced corner sampling to retain low-contrast tracks. An A/B run held the background checks, sampled frames, and SAM weights fixed. The probe scored direct proposals before temporal propagation at mask IoU ≥0.20, using rough labels on the bridge and fixed-camera clips and pixel annotations on three [DAVIS 2016](https://davischallenge.org/davis2016/code.html) clips.

| Clip | Global corner sampling | Tiled corner sampling |
| --- | ---: | ---: |
| Bridge | 1/7 | 2/7 |
| OpenCV fixed camera | 7/8 | 8/8 |
| DAVIS BMX trees | 5/8 | 5/8 |
| DAVIS car roundabout | 1/8 | 0/8 |
| DAVIS parkour | 0/8 | 1/8 |
| Total | 14/39 | 16/39 |

The motion checks suggested 13 overlapping exclusions and four off-target exclusions with global sampling, and 16 overlapping and three off-target with tiled sampling. Those counts are per region. Exclusions still require approval. The roundabout regression and low bridge/parkour coverage preclude a claim of reliable automatic foreground detection on moving cameras.

Chrome submitted a fresh tiled-sampling analysis on five real BMX frames. ViT-B returned candidates on two frames in 25.8 s. Chrome displayed the aligned masks, recorded a Keep correction, and restored the revision after reload. The run proposed zero exclusions. A separate eight-frame FastAPI run generated five candidate frames and one suggested exclusion.

On the real `parkour` sequence, the subject masks generated from annotation-guided prompts had 0.487–0.954 IoU with the labeled masks. Both solves used the same 24 frames and settings. The masked extraction removed 1,309 features and retained zero inside the applied masks. Inlier ratio was 91.69% baseline and 91.56% masked. Mean reprojection error rose from 0.299 to 0.306 px. Solver time was 34.97 s and 33.02 s. Chrome opened the saved masked comparison result as a solve, exported it to Blender, and restored the package after reload. Opening the comparison result reuses the saved camera, tracks and points.

The reviewed synthetic comparison used identical selected frames and solver settings. The masks came from actual SAM inference. Baseline retained 103,662 features, including 4,467 inside exclusion regions. The masked solve retained 99,195 features and zero inside those regions. Geometric inliers changed from 92.29% to 99.23%. Both registered 9/9 frames. Mean reprojection error was 0.11106 px and 0.11078 px, respectively, with 12,232 and 12,227 points. Solver time in this run was 31.71 s and 28.33 s. The fixture measures feature exclusion and geometry. Real-footage pose accuracy against ground truth remains unmeasured.

The SAM runtime used an existing ViT-B checkpoint with Segment Anything 1.0, PyTorch 2.5.1+cpu, torchvision 0.20.1+cpu and a 512-pixel encoder in WSL Python 3.12.3. The GTX 1650 was visible to WSL.

The final automated run passed 58 Python tests and four subtests, including real SAM and PyCOLMAP coverage. The three frontend tests and Vite production build passed. One Starlette TestClient deprecation warning remains.

### Reproduce

Start the backend as in the README with port `8027` and a separate `MATCHMOVE_DATA_DIR` for the browser upload. Run the backend tests from the project root in the activated Python environment:

```sh
python -m pip install pytest httpx
MATCHMOVE_SAM_CHECKPOINT=/path/to/sam_vit_b.pth OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=2 python -m pytest --import-mode=importlib backend -q -rs
```

From the project root in the frontend terminal:

```sh
cd frontend-vite
node --test tests/*.test.mjs
npm run build
```

For a cold reopen, set `MATCHMOVE_CHROME_PROFILE` to the profile printed by the browser test, restart the validation backend with the same data directory, and run `node tests/browserWorkflow.mjs --restore`. The script also accepts `--review <project ID>`, `--analyze-review <project ID>`, `--export-only <project ID>` and `--compare-export <project ID>` for existing validation projects. Set `MATCHMOVE_E2E_BACKEND` when using a different port.

Use `python -m backend.validate_motion path/to/clip` with numeric JPEGs in `images/` and matching binary PNG annotations in `masks/`. The DAVIS probe used eight frame numbers per clip, evenly spaced from source frame 3 through three frames before the end, with neighbors at offsets −3 and +3. Set `MATCHMOVE_SAM_CHECKPOINT` to use the real checkpoint. Use `python -m backend.demo_sam_comparison --output backend/data/sam-demo` to generate and run the synthetic comparison.

To verify a downloaded package, run the following in WSL from its extracted directory with Blender 4.0.2 available:

```sh
blender -b -t 2 --factory-startup --python-exit-code 1 --python import_matchmove.py -- . validated.blend
blender -b validated.blend -t 2 --python-exit-code 1 --python /path/to/ai-matchmove/backend/validate_blender.py -- .
```

The fixed-camera video is the [OpenCV vtest sample](https://github.com/opencv/opencv/blob/master/samples/data/vtest.avi).

## Synthetic sequence validation

The nine-frame synthetic sequence was generated locally and uploaded through Chrome at `24000/1001` fps. PyCOLMAP registered frames 101–109 in 16.3 s, with 12,907 points and 0.111 px mean reprojection error. Against the known synthetic poses after similarity alignment, the camera-center RMSE was 0.00005 scene units and the maximum rotation error was 0.005°. Chrome displayed the feature overlay, retrieved the Blender ZIP, and restored frame 102 and the export link after reload.

Blender 4.0.2 imported the package, reopened the saved scene, and verified nine camera poses, 12,907 points, and the registered plates. The largest undistorted projection difference over 116 observations was 0.00017 px between Blender and the exported camera model.

The Chrome smoke test repeated this workflow and registered 9/9 frames with 12,907 points and 0.111 px mean reprojection error. Blender reopened that export and verified its camera poses, points, and plates.

Generate the fixture as described in the README. To run the Chrome check, start the backend on port `8027` and run this from `frontend-vite/`:

```sh
node tests/browserWorkflow.mjs --demo ../backend/data/demo-frames
```

The script uses Chrome for Windows by default. Set `MATCHMOVE_CHROME` if its executable is elsewhere.
