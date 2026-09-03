"""The 2D rendering metrics, and the rendering_score they combine into.

    rendering_score = mean(PSNR / 50 capped at 1,  SSIM,  1 - LPIPS)

PSNR and SSIM are numpy so they run in the plain venv with no CUDA anywhere near
them. LPIPS needs torch and a VGG backbone, so it is imported lazily and the
caller is told when it is unavailable rather than silently getting a two-term
mean that looks like a real score.

SSIM follows the Wang et al. formulation the challenge names: an 11-tap Gaussian
window at sigma 1.5, population rather than sample covariance, and the border
cropped by half the window so no pixel is scored against padding. That is the
same convention scikit-image documents as its reproduction of the paper. Other
implementations in this field zero-pad instead and score the border, which reads
a few hundredths higher - so treat these as a yardstick for our own attempts
rather than a prediction of the leaderboard.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.ndimage import gaussian_filter

SSIM_SIGMA = 1.5
SSIM_TRUNCATE = 3.5
PSNR_CAP = 50.0


class RenderingMetricError(RuntimeError):
    pass


@dataclass
class RenderingScore:
    psnr: float
    ssim: float
    lpips: float | None

    @property
    def score(self) -> float | None:
        """The challenge's rendering_score, or None when LPIPS is unavailable."""
        if self.lpips is None:
            return None
        return float(np.mean([min(self.psnr / PSNR_CAP, 1.0), self.ssim, 1.0 - self.lpips]))


def _as_float_image(image: np.ndarray) -> np.ndarray:
    image = np.asarray(image)
    if image.dtype == np.uint8:
        return image.astype(np.float64) / 255.0
    return image.astype(np.float64)


def psnr(rendered: np.ndarray, reference: np.ndarray) -> float:
    a, b = _as_float_image(rendered), _as_float_image(reference)
    if a.shape != b.shape:
        raise RenderingMetricError(f"shapes differ: {a.shape} against {b.shape}")
    mse = float(np.mean((a - b) ** 2))
    if mse == 0.0:
        return float("inf")
    return float(10.0 * np.log10(1.0 / mse))


def ssim(rendered: np.ndarray, reference: np.ndarray) -> float:
    a, b = _as_float_image(rendered), _as_float_image(reference)
    if a.shape != b.shape:
        raise RenderingMetricError(f"shapes differ: {a.shape} against {b.shape}")

    window = 2 * int(SSIM_TRUNCATE * SSIM_SIGMA + 0.5) + 1
    pad = (window - 1) // 2
    if min(a.shape[0], a.shape[1]) <= 2 * pad:
        raise RenderingMetricError(
            f"image is {a.shape[0]}x{a.shape[1]}, too small for an {window}-tap window")

    channels = a.shape[2] if a.ndim == 3 else 1
    a = a.reshape(a.shape[0], a.shape[1], channels)
    b = b.reshape(b.shape[0], b.shape[1], channels)

    C1, C2 = (0.01 ** 2), (0.03 ** 2)   # data range is 1.0
    blur = lambda x: gaussian_filter(x, sigma=SSIM_SIGMA, truncate=SSIM_TRUNCATE, mode="reflect")

    per_channel = []
    for channel in range(channels):
        x, y = a[..., channel], b[..., channel]
        ux, uy = blur(x), blur(y)
        uxx, uyy, uxy = blur(x * x), blur(y * y), blur(x * y)
        vx, vy, vxy = uxx - ux * ux, uyy - uy * uy, uxy - ux * uy
        numerator = (2 * ux * uy + C1) * (2 * vxy + C2)
        denominator = (ux ** 2 + uy ** 2 + C1) * (vx + vy + C2)
        smap = numerator / denominator
        per_channel.append(smap[pad:-pad, pad:-pad].mean())
    return float(np.mean(per_channel))


_lpips_model = None


def lpips_vgg(rendered: np.ndarray, reference: np.ndarray) -> float:
    """LPIPS with the VGG trunk, which is the variant the challenge names."""
    global _lpips_model
    try:
        import torch
        import lpips as lpips_package
    except ImportError as error:  # noqa: PERF203
        raise RenderingMetricError(
            f"LPIPS needs torch and the lpips package ({error}); "
            f"run this from the twinworld environment") from error

    if _lpips_model is None:
        _lpips_model = lpips_package.LPIPS(net="vgg", verbose=False).eval()

    def to_tensor(image):
        array = _as_float_image(image)
        tensor = torch.from_numpy(array).permute(2, 0, 1)[None].float()
        return tensor * 2.0 - 1.0     # the package expects [-1, 1]

    with torch.no_grad():
        return float(_lpips_model(to_tensor(rendered), to_tensor(reference)).item())


def score_pair(rendered: np.ndarray, reference: np.ndarray,
               with_lpips: bool = True) -> RenderingScore:
    value = None
    if with_lpips:
        try:
            value = lpips_vgg(rendered, reference)
        except RenderingMetricError:
            value = None
    return RenderingScore(psnr(rendered, reference), ssim(rendered, reference), value)


def combine(scores: list[RenderingScore]) -> RenderingScore:
    """Average across frames, which is what a per-scene or overall figure means."""
    if not scores:
        raise RenderingMetricError("no frames to combine")
    finite = [s.psnr for s in scores if np.isfinite(s.psnr)]
    lpips_values = [s.lpips for s in scores if s.lpips is not None]
    return RenderingScore(
        psnr=float(np.mean(finite)) if finite else float("inf"),
        ssim=float(np.mean([s.ssim for s in scores])),
        lpips=float(np.mean(lpips_values)) if len(lpips_values) == len(scores) else None,
    )
