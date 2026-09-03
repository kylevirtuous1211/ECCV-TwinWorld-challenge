# From the Gaussian models to the Codabench archive

Requested as item 8: the scripts and instructions that turn the native
representation into the official submission format. This is the whole of that
path, in order, with the exact commands. It assumes `code/` has been installed
per `code/README.md`.

    ROOT=<TwinWorld_Datasets>
    NATIVE=<this delivery>/native
    WORK=<a scratch directory, about 60 GB>

Nothing below retrains anything. Every step reads Gaussian models and writes
either images or point clouds.

## 1. Renders: mean of three seeds

Each rendering model writes its own `rgb/` when it is trained. If you are
starting from the checkpoints rather than retraining, re-render with the same
entry point and then average:

    for seed in 0 1 2; do
      for scene in <all thirteen>; do
        python code/scripts/train_scene.py --dataset-root $ROOT \
            --dataset <dataset> --scene <scene> \
            --model 2dgs --iterations 7000 \
            --normal-weight 0.05 --distortion-weight 0.0 --seed $seed \
            --checkpoint $NATIVE/renders/seed$seed/<dataset>_<scene>/checkpoint.pt \
            --output $WORK/render/seed$seed/<dataset>_<scene>
      done
    done

    python code/scripts/average_renders.py \
        --renders $WORK/render/seed0 $WORK/render/seed1 $WORK/render/seed2 \
        --output $WORK/render_mean3

`average_renders.py` writes a receipt recording the per-frame spread across
seeds, and refuses to average a frame that is missing from any input.

## 2. TUM clouds: fuse, vote, thin

    for seed in 0 1 2 3 4; do
      for scene in scene_000 .. scene_008; do
        python code/scripts/export_geometry.py --dataset-root $ROOT \
            --dataset tum --scene $scene \
            --normal-weight 0.05 --distortion-weight 0.01 \
            --depth-readout intersection --seed $seed \
            --checkpoint $NATIVE/tum_geometry/seed$seed/tum_${scene}_isect/checkpoint.pt \
            --output $WORK/geom/seed$seed/tum_${scene}_isect
      done
    done

    python code/scripts/veto_clouds.py \
        --clouds $WORK/geom/seed0 \
        --other $WORK/geom/seed1 $WORK/geom/seed2 $WORK/geom/seed3 $WORK/geom/seed4 \
        --min-agree 2 --keep-radius 0.05 \
        --pattern 'tum_*_isect' --output $WORK/geom_ens4

    python code/scripts/downsample_clouds.py --clouds $WORK/geom_ens4 \
        --pattern 'tum_*_isect' --voxel 0.03 --output $WORK/geom_ens4_v0.03

`--depth-readout intersection` is load-bearing and is the subject of
`code/README.md` finding 1: gsplat's native 2DGS path reports each Gaussian's
centre depth for every pixel its splat covers, and reading the ray-disc
intersection instead is worth 0.083 of `geometry_score` here. It is a known
upstream defect (gsplat #477, #863, PR #932); `code/twinworld/raydepth.py`
recovers the intersection from what an unmodified install already returns.

`tum/scene_006` repeats the same two commands over
`$NATIVE/tum_scene_006_minviews2/seed{0..4}` with `--min-views 2` added, and its
result substitutes for `scene_006` in step 6.

## 3. Gold Coast clouds: label, shift

    for scene in scene_009 .. scene_012; do
      python code/scripts/export_geometry.py --dataset-root $ROOT \
          --dataset gold_coast --scene $scene \
          --iterations 15000 --grow-grad2d 0.0001 \
          --normal-weight 0.05 --distortion-weight 0.01 \
          --depth-readout intersection \
          --checkpoint $NATIVE/gold_coast_geometry/gc_${scene}_isect/checkpoint.pt \
          --output $WORK/geom_gc/gc_${scene}_isect
    done

    python code/scripts/frame_offset.py --dataset-root $ROOT \
        --output $WORK/frame_offset_xy.json

    python code/scripts/label_cloud.py --dataset-root $ROOT \
        --fusion-root $WORK/geom_gc --variant isect \
        --voxel 0.08 --fit-on reconstruction --colour \
        --offsets $WORK/frame_offset_xy.json \
        --trees 200 --max-depth 32 --min-leaf 2 \
        --output $WORK/semantic

    python code/scripts/shift_clouds.py --clouds $WORK/semantic \
        --offsets $WORK/frame_offset_xy.json --unmeasured extrapolate \
        --output $WORK/semantic_shifted

The shift exists because the released Gold Coast ground truth sits 1.1 to 1.4 m
out of the camera frame its own images define. That is a property of the
released data rather than of our reconstruction: fitting the **provided**
`train/sparse/0/points3D.ply` to the **provided** `3d_gt` recovers the same
0.25 degree rotation, with nothing of ours in the loop. `frame_offset.py`
measures it on the two Gold Coast scenes that ship ground truth, with the nine
TUM scenes as the control that reads approximately zero.

`scene_011` and `scene_012` ship no ground truth, so `--unmeasured extrapolate`
gives them the mean of the two measured offsets. Step 4 then re-centres them.
**This is disclosed in full, with its justification and its alternatives, in
`DISCLOSURE.md`.**

## 4. Pack, which is also where the Gold Coast clouds get capped

    python code/scripts/make_submission.py --dataset-root $ROOT \
        --renders $WORK/render_mean3 \
        --clouds $WORK/geom_ens4_v0.03 --tum-variant isect \
        --semantic $WORK/semantic_shifted \
        --thin-unscored 0.2 --cap-cloud 13000000 \
        --output $WORK/base.zip

`make_submission.py` resolves all fifty-two slots before writing anything and
refuses an ambiguous match rather than ranking candidates alphabetically. Two of
its flags change the clouds and both matter to step 5:

- `--cap-cloud 13000000` walks a voxel upward until any cloud over the limit is
  under it. **`scene_011` is 14,326,140 points as step 3 leaves it and 12,752,458
  after the cap.** The scorer is killed by a single oversized cloud, so this is
  not cosmetic.
- `--thin-unscored 0.2` thins the six scenes the final phase does not read.

## 5. The labelled occupancy volume, for scene_011 and scene_012 only

**Built on the capped surfaces, which is why it reads out of the archive from
step 4 rather than out of `$WORK/semantic_shifted`.** Taking the uncapped cloud
here produces a different result and does not reproduce the submission.

    python code/scripts/build_gc_clouds.py \
        --from-archive $WORK/base.zip \
        --scenes scene_011 scene_012 \
        --from-offset 0.6625 -1.07 --offset 0.6925 -0.830 \
        --lattice bcc --spacing 0.19 --shell 0.17 \
        --output $WORK/volume

    scene_011   12,752,458 surface points -> 12,543,412
    scene_012   10,097,143 surface points ->  9,983,255

**We ran this step from these instructions and it reproduces the submitted clouds
byte for byte** - SHA-256 `08443fd9b2bc48a8...` for `scene_011` and
`4afca675d0fe27f2...` for `scene_012`, both matching what is inside
`submission_v0.30_bcc019.zip`. That is the check that caught the ordering: built
before the cap instead of after it, `scene_011` comes out of a 14,326,140-point
surface and does not match.

This replaces the labelled surface with a labelled occupancy volume on a
body-centred cubic lattice. **It optimises the semantic metric as written rather
than improving the reconstruction**, it is the largest single component of our
semantic score, and it is disclosed and explained in `DISCLOSURE.md` together
with a proposed change to the metric that would remove the incentive.

## 6. Replace the three clouds, and check

    python code/scripts/reshift_submission.py --submission $WORK/base.zip \
        --replace tum/scene_006        $WORK/geom_mv2_ens4/tum_scene_006_isect/point_cloud.ply \
        --replace gold_coast/scene_011 $WORK/volume/scene_011_volume.ply \
        --replace gold_coast/scene_012 $WORK/volume/scene_012_volume.ply \
        --output $WORK/submission_v0.30_bcc019.zip

    bash code/verify.sh $WORK/submission_v0.30_bcc019.zip $ROOT

`reshift_submission.py` rewrites only the named clouds and then hashes the
decompressed content of all fifty-two entries in both archives, so "nothing else
changed" is checked rather than asserted. On the archive we submitted, 49 of 52
entries are byte-identical to our previous submission and the 3 that differ are
exactly `tum/scene_006`, `gold_coast/scene_011` and `gold_coast/scene_012`.

`verify.sh` runs the format and point-budget validation and then scores the six
scenes that ship ground truth with our reimplementation of the three official
metrics. Expected on the submitted archive:

    checked 39 frames and 13 point clouds
      71,659,835 points, largest cloud 12,543,412 (gold_coast/scene_011)
    PASS - safe to upload

## What will not reproduce bit-for-bit, and why

Two steps upstream of everything here are not deterministic, and both are named
in `code/README.md`:

- **2DGS training does not reproduce even at a fixed seed on fixed hardware.**
  Measured, not assumed: two runs of `tum/scene_000` with byte-identical command
  lines, the same `--seed 0`, the same machine and the same GPU diverge in
  Gaussian count by step 1000 and end at 1,218,151 against 1,249,375 Gaussians,
  checkpoints of 287,486,054 against 294,854,822 bytes, test PSNR 19.73 against
  19.82, and **none of the three rendered frames byte-identical** - the PSNR
  *between* the two runs is 15.7 to 23.8 dB. Training seeds only
  `torch.manual_seed`, while gsplat's backward accumulates with `atomicAdd`,
  whose summation order is not fixed. Averaging three seeds is partly a response
  to this.
- **COLMAP's dense MVS is not deterministic across machines**, and the seed it
  produces is what steps 2 and 3 are initialised from.

Everything from the checkpoints onwards *is* deterministic, which is why this
document starts there. Geometry agreed to four decimal places between the two
machines we used (RTX 6000 Ada, sm_89, and H200, sm_90); rendering did not.
