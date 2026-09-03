"""Train a Gaussian scene on one scene's train views and render its test poses.

Both representations live here because they differ in three lines: which
rasteriser is called, which gradient key the densification strategy watches, and
whether the third scale axis means anything. Keeping them in one file makes the
comparison honest - the initialisation, the schedule, the loss and the
densification are identical, so a difference in the renders is a difference
between the representations rather than between two codebases.

The poses are given, so nothing here estimates geometry from scratch: the
Gaussians start on the COLMAP points and the optimiser only has to decide what
they look like and how they spread.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from twinworld.colmap import read_poses
from twinworld.dataset import read_cameras
from twinworld.raydepth import intersection_normal_map


@dataclass
class TrainConfig:
    model: str = "3dgs"                 # or "2dgs"
    iterations: int = 7000
    sh_degree: int = 3
    sh_degree_interval: int = 1000
    ssim_weight: float = 0.2
    init_opacity: float = 0.1
    init_scale_factor: float = 1.0
    densify_from: int = 500
    densify_until_fraction: float = 0.5
    refine_every: int = 100
    reset_opacity_every: int = 3000
    # How readily a gaussian splits, and how faint one has to be before it is
    # pruned. gsplat's defaults are tuned for hundreds of views; here twelve
    # photographs end up explained by 1.2M to 2.3M gaussians, which is enough
    # capacity to memorise them - train PSNR runs 10 dB above test. Geometry is
    # what pays for that: a model that can reproduce the images from any depth
    # is not being told where the surface is. Raising the threshold is the
    # cheapest control on that.
    grow_grad2d: float = 0.0002
    prune_opacity: float = 0.005
    # gsplat's DefaultStrategy grows a gaussian when its view-space gradient is
    # large, and a gaussian near a camera has a huge screen footprint and so a
    # huge NDC gradient - which is a mechanism for *manufacturing* the near-camera
    # floaters this project has measured. MCMC never reads a view-space gradient:
    # it treats the set as MCMC samples, replaces clone/split with a relocation of
    # dead gaussians onto live ones, and adds SGLD noise to positions. Its own
    # ablation on OMMO - large scenes with distant objects, the closest published
    # analogue to ours - is +0.018 of our rendering score over eight scenes with
    # none losing. Note it is *not* a floater fix: the noise is weighted by
    # sigmoid(-100*(opacity - 0.005)), which is 1.35e-39 at opacity 0.9, so it is
    # switched off for exactly the opaque gaussians our floaters are.
    strategy: str = "default"        # "default" or "mcmc"
    cap_max: int = 1_700_000         # MCMC grows to this rather than to a gradient threshold
    noise_lr: float = 5e5
    opacity_reg: float = 0.01        # MCMC's L1 on opacity
    scale_reg: float = 0.01          # MCMC's L1 on the covariance's square-rooted eigenvalues
    random_background: bool = False
    # 2DGS only, and both measured rather than taken from the paper - gsplat's
    # distortion is an L1 variant whose magnitude has nothing to do with the L2
    # one the published weights were tuned for. Normal consistency helps;
    # distortion costs rendering quality at every weight tried. See
    # scripts/train_scene.py --distortion-weight to re-measure.
    normal_weight: float = 0.05
    distortion_weight: float = 0.0
    normal_start: int = 2000
    distortion_start: int = 1000
    # What the normal-consistency term is asked to agree with. "centre" is
    # gsplat's own target and reproduces every number measured before
    # 2026-08-18; it is derived from a depth map that is constant within each
    # splat and therefore points at the camera. "intersection" rebuilds it from
    # the ray-disc hit. See twinworld/raydepth.intersection_normal_map.
    normal_target: str = "centre"
    # A monocular depth prior, applied to the *shape* of the rendered depth map
    # and never to its scale. Handing MoGe's metres to the model directly was
    # measured and lost: its median residual against COLMAP is 0.36 m on TUM and
    # 2.03 m on Gold Coast, against a 5 cm threshold, so as a position it is
    # forty times our tolerance. What it is right about is the arrangement
    # within one view, which is the one thing twelve photographs under-determine
    # - and normalising both sides by their own median and spread keeps exactly
    # that and discards the rest.
    depth_prior_weight: float = 0.0
    depth_prior_start: int = 1000
    # Co-adaptation controls, from "Quantifying and Alleviating Co-Adaptation in
    # Sparse-View 3D Gaussian Splatting" (NeurIPS 2025, arXiv:2508.12720) and the
    # DropGaussian line of work. With twelve photographs the gaussians entangle:
    # train PSNR runs 10 to 13 dB above test on every scene here, which is the
    # symptom that paper names. Dropping a random share of them each step, and
    # perturbing opacity multiplicatively, both force each primitive to be
    # individually right rather than right in company.
    #
    # This matters for geometry and not only for rendering, which is not obvious:
    # the cloud is fused from depth rendered at *test* poses, so a model that has
    # co-adapted to the train views is being asked for depth it never learned to
    # produce.
    dropout: float = 0.0
    opacity_noise: float = 0.0
    seed: int = 0
    learning_rates: dict = field(default_factory=lambda: {
        "means": 1.6e-4, "scales": 5e-3, "quats": 1e-3,
        "opacities": 5e-2, "sh0": 2.5e-3, "shN": 2.5e-3 / 20,
    })


@dataclass
class Frame:
    name: str
    stem: str
    width: int
    height: int
    view_matrix: torch.Tensor      # world to camera, 4x4
    intrinsics: torch.Tensor       # 3x3
    image: torch.Tensor | None     # H W 3 in [0, 1], absent for test poses


def _view_matrix(pose) -> torch.Tensor:
    matrix = np.eye(4)
    matrix[:3, :3] = pose.rotation
    matrix[:3, 3] = pose.translation
    return torch.from_numpy(matrix).float()


def load_frames(split_dir: Path, load_images: bool) -> list[Frame]:
    from PIL import Image

    sparse = split_dir / "sparse" / "0"
    cameras = read_cameras(sparse / "cameras.txt")
    frames = []
    for pose in read_poses(sparse / "images.txt"):
        width, height = cameras[pose.camera_id]
        parameters = _camera_parameters(sparse / "cameras.txt", pose.camera_id)
        fx, fy, cx, cy = parameters
        image = None
        if load_images:
            path = split_dir / "images" / pose.name
            array = np.asarray(Image.open(path).convert("RGB"), dtype=np.float32) / 255.0
            if array.shape[:2] != (height, width):
                raise ValueError(
                    f"{path.name} is {array.shape[1]}x{array.shape[0]} but the camera "
                    f"says {width}x{height}")
            image = torch.from_numpy(array)
        frames.append(Frame(
            name=pose.name, stem=Path(pose.name).stem, width=width, height=height,
            view_matrix=_view_matrix(pose),
            intrinsics=torch.tensor([[fx, 0.0, cx], [0.0, fy, cy], [0.0, 0.0, 1.0]]).float(),
            image=image))
    return frames


def _camera_parameters(path: Path, camera_id: int) -> tuple[float, float, float, float]:
    for line in path.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        parts = line.split()
        if int(parts[0]) != camera_id:
            continue
        model, values = parts[1], [float(v) for v in parts[4:]]
        if model == "PINHOLE":
            return values[0], values[1], values[2], values[3]
        if model == "SIMPLE_PINHOLE":
            return values[0], values[0], values[1], values[2]
        raise ValueError(f"camera model {model} is not handled")
    raise ValueError(f"camera {camera_id} not found in {path}")


def initialise(points: np.ndarray, colours: np.ndarray, config: TrainConfig,
               device: str) -> torch.nn.ParameterDict:
    """Gaussians on the COLMAP points, sized by how far apart those points are."""
    from scipy.spatial import cKDTree

    generator = torch.Generator(device="cpu").manual_seed(config.seed)
    means = torch.from_numpy(points).float()

    # A Gaussian should be about as wide as the gap to its neighbours, or the
    # scene starts as either a fog or a set of disconnected specks.
    distance, _ = cKDTree(points).query(points, k=4, workers=-1)
    spacing = np.clip(distance[:, 1:].mean(axis=1), 1e-7, None) * config.init_scale_factor
    scales = torch.from_numpy(np.log(spacing)).float()[:, None].repeat(1, 3)

    quats = torch.zeros(len(points), 4)
    quats[:, 0] = 1.0
    opacities = torch.logit(torch.full((len(points),), config.init_opacity))

    colours = torch.from_numpy(colours).float()
    sh0 = ((colours - 0.5) / 0.28209479177387814)[:, None, :]   # inverse of SH band 0
    bands = (config.sh_degree + 1) ** 2 - 1
    shN = torch.zeros(len(points), bands, 3)
    del generator

    return torch.nn.ParameterDict({
        "means": torch.nn.Parameter(means),
        "scales": torch.nn.Parameter(scales),
        "quats": torch.nn.Parameter(quats),
        "opacities": torch.nn.Parameter(opacities),
        "sh0": torch.nn.Parameter(sh0),
        "shN": torch.nn.Parameter(shN),
    }).to(device)


def gaussian_window(window: int, sigma: float, device) -> torch.Tensor:
    coords = torch.arange(window, dtype=torch.float32, device=device) - window // 2
    kernel = torch.exp(-(coords ** 2) / (2 * sigma ** 2))
    kernel = kernel / kernel.sum()
    return torch.outer(kernel, kernel)[None, None]


def torch_ssim(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """SSIM over an H W 3 pair, differentiable, for the training loss."""
    window = gaussian_window(11, 1.5, a.device).repeat(3, 1, 1, 1)
    x = a.permute(2, 0, 1)[None]
    y = b.permute(2, 0, 1)[None]
    mu_x = F.conv2d(x, window, padding=5, groups=3)
    mu_y = F.conv2d(y, window, padding=5, groups=3)
    xx = F.conv2d(x * x, window, padding=5, groups=3) - mu_x ** 2
    yy = F.conv2d(y * y, window, padding=5, groups=3) - mu_y ** 2
    xy = F.conv2d(x * y, window, padding=5, groups=3) - mu_x * mu_y
    C1, C2 = 0.01 ** 2, 0.03 ** 2
    smap = ((2 * mu_x * mu_y + C1) * (2 * xy + C2)) / \
           ((mu_x ** 2 + mu_y ** 2 + C1) * (xx + yy + C2))
    return smap.mean()


def _normalise_depth(values: torch.Tensor) -> torch.Tensor:
    """Median-centred and mean-absolute-scaled, so only the arrangement survives.

    This is the normalisation MiDaS uses to compare depths that live in
    different units. The median and the mean absolute deviation are used rather
    than the mean and the standard deviation because a monocular model's tail -
    sky, glass, anything it declines to commit on - is heavy, and a squared
    statistic would let that tail set the scale.
    """
    centre = values.median()
    spread = (values - centre).abs().mean().clamp_min(1e-6)
    return (values - centre) / spread


def depth_prior_loss(rendered_depth: torch.Tensor, alphas: torch.Tensor,
                     prior: torch.Tensor, alpha_min: float = 0.5) -> torch.Tensor | None:
    """How differently the render and the prior arrange one view's depths.

    Only pixels the model has actually committed to are compared - a pixel no
    gaussian covers has no depth to be wrong about - and the comparison is on
    normalised depths, so a prior that is uniformly 24% short costs nothing and
    a prior that puts the wall behind the roof costs a lot.
    """
    # The rasteriser hands back a batch dimension and the prior does not, so
    # drop it here rather than at every call site.
    depth, alpha = rendered_depth[..., 0], alphas[..., 0]
    if depth.dim() == 3:
        depth, alpha = depth[0], alpha[0]
    if depth.shape != prior.shape:
        raise ValueError(
            f"the depth prior is {tuple(prior.shape)} and the render is "
            f"{tuple(depth.shape)}; they must be the same view")

    covered = (alpha > alpha_min) & torch.isfinite(prior) & (prior > 0)
    if int(covered.sum()) < 1024:
        return None
    return (_normalise_depth(depth[covered])
            - _normalise_depth(prior[covered])).abs().mean()


def render(params, frame: Frame, config: TrainConfig, sh_degree: int,
           background: torch.Tensor, device: str, step: int | None = None,
           training: bool = False):
    """One view, through whichever rasteriser the configuration names.

    `step` is only consulted to decide whether the intersection normal target is
    worth computing; None means "compute it if configured", which is what the
    inference paths want.

    `training` gates the co-adaptation controls. They must be off everywhere else
    - a cloud fused from dropped-out renders would be missing surface at random,
    and the point of them is what the optimiser learns, not what it draws.
    """
    from gsplat import rasterization, rasterization_2dgs

    colours = torch.cat([params["sh0"], params["shN"]], dim=1)
    opacities = torch.sigmoid(params["opacities"])
    if training and config.opacity_noise > 0:
        # Multiplicative and centred on one, so the expected opacity is unchanged
        # and only the reliance on any particular value is punished.
        opacities = opacities * (1.0 + config.opacity_noise
                                 * (2.0 * torch.rand_like(opacities) - 1.0))
    if training and config.dropout > 0:
        # Inverted dropout: the survivors are scaled up so the rendered opacity
        # stays in the same range as it will be at inference, exactly as dropout
        # does in a network. Scaling the *opacity* rather than the colour keeps
        # the depth composition consistent, which is what the geometry reads.
        keep = (torch.rand_like(opacities) >= config.dropout).float()
        opacities = opacities * keep / (1.0 - config.dropout)
    common = dict(
        means=params["means"],
        quats=params["quats"],
        scales=torch.exp(params["scales"]),
        opacities=opacities.clamp(0.0, 1.0),
        colors=colours,
        viewmats=frame.view_matrix[None].to(device),
        Ks=frame.intrinsics[None].to(device),
        width=frame.width,
        height=frame.height,
        sh_degree=sh_degree,
        packed=False,
        backgrounds=background[None],
    )
    if config.model == "2dgs":
        # Both extras are opt-in and silently absent otherwise: the depth-derived
        # normal is only computed when a depth channel is rendered, and the
        # distortion only when distloss is asked for. Leaving the defaults gives
        # a 2DGS run with none of 2DGS's constraints and no error to say so.
        colours, alphas, normals, surface_normals, distortion, median, info = \
            rasterization_2dgs(**common, render_mode="RGB+ED", distloss=True)
        live = step is None or step >= config.normal_start
        if config.normal_target == "intersection" and live and config.normal_weight > 0:
            # Gated because this costs a second rasteriser forward per step:
            # median_ids is not returned by the autograd wrapper, so the only
            # way to know which splat set each pixel's depth is to ask again.
            # Before normal_start the term is not applied and the geometry is
            # still nonsense, so the answer would be thrown away.
            surface_normals = intersection_normal_map(
                params["means"], params["quats"], torch.exp(params["scales"]),
                frame.view_matrix.to(device), frame.intrinsics.to(device),
                frame.width, frame.height, info, median[0, ..., 0])
        return colours[..., :3], info, {
            "alphas": alphas, "normals": normals, "depth": colours[..., 3:],
            "median_depth": median,
            "surface_normals": surface_normals, "distortion": distortion,
        }
    outputs = rasterization(**common)
    return outputs[0], outputs[2], {}


def surface_regularisers(extra: dict, config: TrainConfig, step: int,
                         prior: torch.Tensor | None = None) -> dict:
    """2DGS's two geometric terms, which are the reason to use it at all.

    Without these 2DGS is only 3DGS with flattened primitives - the ellipsoids
    become discs and nothing constrains where the discs sit. The normal term ties
    each splat's own orientation to the orientation implied by the depth it
    renders, and the distortion term pulls the splats along a ray together so a
    surface is thin instead of a cloud. Both are exactly the constraint missing
    when a point is seen by two views.

    They start late on purpose: applied from the first step they regularise a
    geometry that is still nonsense, and the run converges to a smooth wrong
    answer.
    """
    losses = {}
    if not extra:
        return losses

    if step >= config.normal_start and config.normal_weight > 0:
        # Weighted by opacity so empty pixels, whose normals are meaningless,
        # do not vote.
        alpha = extra["alphas"].detach()
        agreement = (extra["normals"] * extra["surface_normals"]).sum(dim=-1, keepdim=True)
        losses["normal"] = config.normal_weight * ((1.0 - agreement) * alpha).mean()

    if step >= config.distortion_start and config.distortion_weight > 0:
        losses["distortion"] = config.distortion_weight * extra["distortion"].mean()

    if (prior is not None and config.depth_prior_weight > 0
            and step >= config.depth_prior_start):
        term = depth_prior_loss(extra["depth"], extra["alphas"], prior)
        if term is not None:
            losses["depth_prior"] = config.depth_prior_weight * term

    return losses


def train(points: np.ndarray, colours: np.ndarray, frames: list[Frame],
          config: TrainConfig, device: str = "cuda", log=print,
          depth_priors: dict[str, np.ndarray] | None = None):
    from gsplat.strategy import DefaultStrategy, MCMCStrategy

    torch.manual_seed(config.seed)
    params = initialise(points, colours, config, device)

    centres = torch.stack([torch.linalg.inv(f.view_matrix)[:3, 3] for f in frames])
    scene_scale = float(torch.linalg.norm(centres - centres.mean(0), dim=1).max())
    log(f"{len(points)} gaussians, scene scale {scene_scale:.2f} m, "
        f"{len(frames)} train views, model {config.model}")

    optimizers = {
        name: torch.optim.Adam(
            [{"params": params[name], "lr": rate * (scene_scale if name == "means" else 1.0)}],
            eps=1e-15)
        for name, rate in config.learning_rates.items()
    }
    # The position learning rate decays over training, as in the original 3DGS.
    decay = torch.optim.lr_scheduler.ExponentialLR(
        optimizers["means"], gamma=0.01 ** (1.0 / config.iterations))

    if config.strategy == "mcmc":
        # refine_stop_iter defaults to 25000 upstream against our 7000, so the cap
        # would never be reached; it is put on the same fraction the default
        # strategy uses so the two arms stop densifying at the same step.
        strategy = MCMCStrategy(
            cap_max=config.cap_max,
            noise_lr=config.noise_lr,
            refine_start_iter=config.densify_from,
            refine_stop_iter=int(config.iterations * config.densify_until_fraction),
            refine_every=config.refine_every,
            min_opacity=config.prune_opacity,
            verbose=False,
        )
    else:
        strategy = DefaultStrategy(
            refine_start_iter=config.densify_from,
            refine_stop_iter=int(config.iterations * config.densify_until_fraction),
            refine_every=config.refine_every,
            reset_every=config.reset_opacity_every,
            grow_grad2d=config.grow_grad2d,
            prune_opa=config.prune_opacity,
            key_for_gradient="gradient_2dgs" if config.model == "2dgs" else "means2d",
            verbose=False,
        )
    strategy.check_sanity(params, optimizers)
    # MCMCStrategy has no scene_scale in its state: it never converts a view-space
    # gradient into world units, which is the same reason it cannot manufacture a
    # near-camera floater the way the default strategy does.
    state = (strategy.initialize_state() if config.strategy == "mcmc"
             else strategy.initialize_state(scene_scale=scene_scale))

    images = torch.stack([f.image for f in frames]).to(device)
    priors = None
    if depth_priors and config.depth_prior_weight > 0:
        missing = [f.stem for f in frames if f.stem not in depth_priors]
        if missing:
            raise ValueError(
                f"--depth-prior is missing {len(missing)} of {len(frames)} train views, "
                f"first {missing[0]}. Rebuild it with scripts/dense_init.py --depth-maps.")
        priors = [torch.from_numpy(depth_priors[f.stem].astype(np.float32)).to(device)
                  for f in frames]
        log(f"depth prior on {len(priors)} views, weight {config.depth_prior_weight}")
    order = torch.randperm(len(frames))
    for step in range(config.iterations):
        if step % len(frames) == 0:
            order = torch.randperm(len(frames))
        index = int(order[step % len(frames)])
        frame, truth = frames[index], images[index]

        sh_degree = min(step // config.sh_degree_interval, config.sh_degree)
        background = (torch.rand(3, device=device) if config.random_background
                      else torch.zeros(3, device=device))

        rendered, info, extra = render(params, frame, config, sh_degree,
                                       background, device, step, training=True)
        image = rendered[0].clamp(0.0, 1.0)

        strategy.step_pre_backward(params, optimizers, state, step, info)
        l1 = (image - truth).abs().mean()
        photometric = (1.0 - config.ssim_weight) * l1 + \
                      config.ssim_weight * (1.0 - torch_ssim(image, truth))
        regularisers = surface_regularisers(
            extra, config, step, priors[index] if priors is not None else None)
        loss = photometric + sum(regularisers.values())
        if config.strategy == "mcmc":
            # The two L1 terms MCMC's derivation requires. Without them the
            # relocation has nothing pulling dead gaussians towards low opacity
            # and small scale, and the cap fills with redundant samples.
            loss = loss + config.opacity_reg * torch.sigmoid(params["opacities"]).abs().mean()
            loss = loss + config.scale_reg * torch.exp(params["scales"]).abs().mean()
        loss.backward()

        for optimizer in optimizers.values():
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)
        decay.step()
        if config.strategy == "mcmc":
            strategy.step_post_backward(params, optimizers, state, step, info,
                                        lr=optimizers["means"].param_groups[0]["lr"])
        else:
            strategy.step_post_backward(params, optimizers, state, step, info, packed=False)

        if step % 500 == 0 or step == config.iterations - 1:
            psnr = -10.0 * math.log10(max(float(((image - truth) ** 2).mean()), 1e-12))
            extra_text = "".join(f"  {name} {float(value):.4f}"
                                 for name, value in sorted(regularisers.items()))
            log(f"  step {step:6d}  photo {float(photometric):.4f}{extra_text}  "
                f"train psnr {psnr:5.2f}  gaussians {len(params['means']):,}")

    return params


@torch.no_grad()
def render_frames(params, frames: list[Frame], config: TrainConfig,
                  device: str = "cuda") -> dict[str, np.ndarray]:
    background = torch.zeros(3, device=device)
    output = {}
    for frame in frames:
        rendered, _, _ = render(params, frame, config, config.sh_degree, background, device)
        image = rendered[0].clamp(0.0, 1.0).cpu().numpy()
        output[frame.stem] = (image * 255.0).round().astype(np.uint8)
    return output


@dataclass
class DepthFrame:
    """What one view says about where the surface is."""
    stem: str
    depth: np.ndarray            # H W, expected depth, already divided by alpha
    alpha: np.ndarray            # H W, how much of the pixel any Gaussian covered
    median_depth: np.ndarray | None   # H W, 2DGS only
    # H W 3 in [0, 1]. The rendered colour, not the photograph: it exists at test
    # poses and on the seven withheld scenes, where photographs do not, so the
    # same path produces a colour for every cloud a submission needs.
    colour: np.ndarray | None = None


@torch.no_grad()
def render_depth_frames(params, frames: list[Frame], config: TrainConfig,
                        device: str = "cuda",
                        intersection: bool = False) -> dict[str, DepthFrame]:
    """Depth per view, which is the only thing that turns a trained scene into a surface.

    Two depths come back where the rasteriser offers two, because they fail
    differently. The expected depth is the opacity-weighted mean along the ray,
    so a ray that passes through a half-transparent Gaussian in front of a wall
    reports a distance somewhere between the two, at no surface at all. The
    median depth takes the sample where accumulated opacity crosses one half, so
    it lands on whichever surface actually blocks the ray. 2DGS surface
    extraction normally uses the median for that reason - but which one fuses
    better is measurable, so both are returned rather than one being chosen here.

    The alpha channel comes back too and is not optional. `RGB+ED` divides the
    accumulated depth by alpha, so a pixel no Gaussian covered is a division by
    1e-10 and reports a distance of order 1e9. Fusing without an alpha gate
    scatters those across the scene.
    """
    from gsplat import rasterization, rasterization_2dgs

    colours = torch.cat([params["sh0"], params["shN"]], dim=1)
    output = {}
    for frame in frames:
        common = dict(
            means=params["means"],
            quats=params["quats"],
            scales=torch.exp(params["scales"]),
            opacities=torch.sigmoid(params["opacities"]),
            colors=colours,
            viewmats=frame.view_matrix[None].to(device),
            Ks=frame.intrinsics[None].to(device),
            width=frame.width,
            height=frame.height,
            sh_degree=config.sh_degree,
            packed=False,
        )
        if config.model == "2dgs":
            rendered, alphas, _, _, _, median, meta = rasterization_2dgs(
                **common, render_mode="RGB+ED")
            depth_map = median[0, ..., 0]
            if intersection:
                # gsplat reports the splat's centre depth for every pixel it
                # covers rather than the ray-disc hit; see twinworld/raydepth.py.
                # Worth +0.0829 of geometry_score over the four dev scenes.
                from twinworld.raydepth import intersection_depth

                hit, usable = intersection_depth(
                    common["means"], common["quats"], common["scales"],
                    frame.view_matrix.to(device), frame.width, frame.height, meta)
                depth_map = torch.where(usable, hit, depth_map)
            median_depth = depth_map.float().cpu().numpy()
        else:
            rendered, alphas, _ = rasterization(**common, render_mode="RGB+ED")
            median_depth = None
        output[frame.stem] = DepthFrame(
            stem=frame.stem,
            depth=rendered[0, ..., -1].float().cpu().numpy(),
            alpha=alphas[0, ..., 0].float().cpu().numpy(),
            median_depth=median_depth,
            # Already rendered by the call above and previously dropped on the
            # floor; the rasteriser produced RGB alongside the depth either way.
            colour=rendered[0, ..., :3].clamp(0.0, 1.0).float().cpu().numpy(),
        )
    return output


@torch.no_grad()
def count_supporting_views(params, frames: list[Frame], config: TrainConfig,
                           device: str = "cuda", depth_tolerance: float | None = 0.01,
                           alpha_floor: float = 0.5) -> np.ndarray:
    """How many training views put each Gaussian at the surface that view can see.

    The artefact this is for is the near-camera floater: a Gaussian one view
    needed to explain a few of its own pixels, sitting metres in front of
    anything real. It is not low opacity - it is opaque by construction, because
    it exists to paint pixels - and it is not small, so neither the opacity prune
    nor a scale prune touches it. What it is, is unsupported: in every view but
    the one that made it, it is either outside the frustum or floating in front
    of the depth that view renders.

    So the discriminator is agreement, counted per Gaussian. Each centre is
    projected into every training view and counted when it lands in the image, in
    front of the camera, on a pixel some Gaussian covered, and within
    `depth_tolerance` of that pixel's rendered distance - relative, because these
    are aerial scenes where the same absolute slack means different things at
    50 m and at 400 m.

    The median depth is used where the rasteriser offers it. The expected depth
    is an opacity-weighted mean along the ray, so a ray that passes through the
    floater and then the wall reports a distance at neither, which is exactly the
    ray this function has to judge.

    `depth_tolerance=None` drops the depth test and counts frustum containment
    alone. The two disagree about a Gaussian that is real but occluded - the far
    side of a roof, seen by one view and hidden in the rest - which the depth
    test calls unsupported and the frustum test does not. Which of those two
    errors costs more is measurable, so both are reachable from here.
    """
    depths = render_depth_frames(params, frames, config, device=device)
    means = params["means"].detach()
    counts = torch.zeros(len(means), dtype=torch.int32, device=means.device)

    for frame in frames:
        rendered = depths[frame.stem]
        surface = rendered.median_depth if rendered.median_depth is not None else rendered.depth
        surface = torch.from_numpy(np.asarray(surface)).to(means.device)
        alpha = torch.from_numpy(np.asarray(rendered.alpha)).to(means.device)

        view = frame.view_matrix.to(means.device)
        camera = means @ view[:3, :3].T + view[:3, 3]
        depth = camera[:, 2]
        intrinsics = frame.intrinsics.to(means.device)
        u = camera[:, 0] / depth.clamp(min=1e-6) * intrinsics[0, 0] + intrinsics[0, 2]
        v = camera[:, 1] / depth.clamp(min=1e-6) * intrinsics[1, 1] + intrinsics[1, 2]

        column = u.round().long().clamp(0, frame.width - 1)
        row = v.round().long().clamp(0, frame.height - 1)
        inside = ((depth > 0) & (u >= 0) & (u < frame.width)
                  & (v >= 0) & (v < frame.height))
        supported = inside & (alpha[row, column] >= alpha_floor)
        if depth_tolerance is not None:
            supported &= (depth - surface[row, column]).abs() <= depth_tolerance * depth
        counts += supported.to(counts.dtype)

    return counts.cpu().numpy()


def nearest_camera_distance(params, frames: list[Frame]) -> np.ndarray:
    """How far each Gaussian sits from the closest training camera, in metres.

    Support alone prunes too much. A Gaussian only one view contains is not
    necessarily a floater - it can be a wall the other flight lines never looked
    at, and on TUM that is most of what a support threshold removes. What makes a
    floater a floater is that it is also *near*: it was grown a few metres in
    front of one camera to paint that camera's pixels, in a scene whose surfaces
    are hundreds of metres away. Distance is the second half of the test.
    """
    means = params["means"].detach()
    centres = []
    for frame in frames:
        view = frame.view_matrix.to(means.device)
        centres.append(-view[:3, :3].T @ view[:3, 3])
    centres = torch.stack(centres)
    closest = torch.cdist(means, centres).min(dim=1).values
    return closest.cpu().numpy()


def select_gaussians(params, keep) -> dict:
    """The same Gaussians, restricted to `keep`, ready to render or save."""
    index = torch.as_tensor(np.asarray(keep), device=params["means"].device)
    return {name: value[index] for name, value in params.items()}


def save_checkpoint(params, config: TrainConfig, path: Path) -> None:
    """Keep the Gaussians, so geometry work never has to retrain to look at them."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({
        "params": {name: value.detach().cpu() for name, value in params.items()},
        "model": config.model,
        "sh_degree": config.sh_degree,
    }, path)


def load_checkpoint(path: Path, device: str = "cuda") -> tuple[dict, TrainConfig]:
    stored = torch.load(str(path), map_location=device, weights_only=True)
    params = {name: value.to(device) for name, value in stored["params"].items()}
    return params, TrainConfig(model=stored["model"], sh_degree=stored["sh_degree"])
