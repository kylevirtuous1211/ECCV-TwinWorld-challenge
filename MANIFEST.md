# What produced each of the fifty-two entries

The Codabench archive is `submission_v0.30_bcc019.zip`, Codabench submission
**909861**, final phase, 2026-08-31T09:56Z, Final Score 1.534.

It holds 52 entries: 39 rendered images (13 scenes x 3 held-out frames) and 13
point clouds. This table is the map from each of them back to the Gaussian model
that produced it. Sizes are what the artefact occupies on disk, not in the
archive.

## The 39 renders

| entries | native representation | where | size |
|---|---|---|---|
| `<dataset>/<scene>/rgb/*.png`, all 13 scenes | 2DGS rendering models, **13 scenes x 3 seeds**, then a per-pixel mean of the three renders | `native/renders/seed{0,1,2}/<dataset>_<scene>/` — **rendered images and receipts only; see the gaps below** | 420 MB |

The three seeds are averaged rather than one being selected, because selecting
needs ground truth and seven of the thirteen scenes have none. `average_renders.py`
does the mean and refuses to average a frame missing from any input.

## The 9 TUM clouds

| entries | native representation | where | size |
|---|---|---|---|
| `tum/scene_000..005,007,008` | 2DGS geometry models, **9 scenes x 5 seeds**; seed 0 is fused and seeds 1-4 vote on it | `native/tum_geometry/seed{0..4}/tum_<scene>_isect/checkpoint.pt` | 39 GB |
| `tum/scene_006` | the same, but trained from the **released** sparse seed at `--min-views 2` rather than from the dense MVS seed | `native/tum_scene_006_minviews2/seed{0..4}/` | 4 GB |

`scene_006` is the one hand-made exception and the reason is in
`code/README.md` step 3b: the dense MVS seed covers 192 m of that scene's x
extent where the released seed covers 248 m, so a quarter of the ground is
absent from the reconstruction.

## The 4 Gold Coast clouds

| entries | native representation | where | size |
|---|---|---|---|
| `gold_coast/scene_009..012` | 2DGS geometry models, **4 scenes**, each with `colour.npy` beside it; then a random-forest label per point; then a frame shift; then, for 011 and 012 only, the labelled occupancy volume | `native/gold_coast_geometry/gc_<scene>_isect/` and `native/semantic/` | ~10 GB |

**`native/semantic/` is post-shift, not pre-shift**, and it is the state after
`shift_clouds.py`. `scene_009` and `scene_010` carry their own measured offsets;
`scene_011` and `scene_012` carry **(0.45, -1.05)**, the route that extrapolates
from `scene_009` alone. If you start from these clouds rather than re-running the
labelling, `build_gc_clouds.py --from-offset` must say `0.45 -1.05`, not the
`0.6625 -1.07` that a plain `--unmeasured extrapolate` run would leave. Both
routes end at the same absolute position; only the stated starting point differs.

## Known gaps in the native representation

- **The rendering models have no `checkpoint.pt` at all.** Those 39 runs were
  launched without `--save-checkpoint`. Every rendered image survives - all 117
  per-seed frames and the 39 averaged ones, and we checked that the 39 are
  byte-identical to the images in the submitted archive - along with each run's
  `receipt.json`, which carries the full configuration and that run's own
  PSNR/SSIM/LPIPS. `NATIVE_TO_SUBMISSION.md` gives the command that retrains
  them; we do not provide retrained weights as substitutes for the original
  checkpoints.
- **`native/tum_scene_006_minviews2/seed0` has no `checkpoint.pt`.** That run was
  launched without `--save-checkpoint`; its `point_cloud.ply` and `colour.npy`
  are present, and seeds 1-4 have checkpoints. The fused cloud that shipped is
  present in full either way.
- The dev-phase scenes in the archive (`tum/scene_000..003`,
  `gold_coast/scene_009,010`) are **thinned at a 0.2 m voxel** by
  `make_submission.py --thin-unscored 0.2`, because the final phase does not read
  them and they otherwise cost the archive's point budget. The native models are
  unthinned; the thinning is a packing step, reproduced by the same flag.
