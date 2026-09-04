"""The lattice sweep: pooled mIoU against covering radius.

Reads the three sweep result files written by scripts/volume_sweep.py. The
selection criterion was declared before the grid ran: highest pooled mIoU at the
residual frame error the offset correction is expected to leave (0.15 m), among
variants inside the per-cloud point budget.

    uv run --with matplotlib --with numpy python \
        delivery/factsheet_official/fig/make_sweep.py --runs <EditReadyGS_runs/twinworld>
"""
import argparse, json, math
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

PRIMARY = "0.15"                      # the declared selection error
CAP_011, CAP_012 = 12_752_458, 10_097_143   # counts that have already scored
INK, MUTED, ACCENT, WARN = "#111318", "#5f6672", "#1f6feb", "#be123c"


def covering_radius(spec):
    """Radius of the ball each lattice point must cover.

    Body-centred cubic (the A3* lattice) covers at a*sqrt(5)/4; the simple cubic
    grid the dilate variants sit on covers at s*sqrt(3)/2.
    """
    if spec.get("lattice") == "bcc":
        return spec["spacing"] * math.sqrt(5) / 4
    if spec.get("kind") == "dilate":
        return spec["spacing"] * math.sqrt(3) / 2
    return None


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--runs", default=Path("/home/intern_2603056/EditReadyGS_runs/twinworld"),
                   type=Path)
    p.add_argument("--out", default=Path("delivery/factsheet_official/fig"), type=Path)
    args = p.parse_args()

    seen, rows = set(), []
    for directory in ("volume_sweep2", "volume_sweep3", "volume_sweep4"):
        path = args.runs / directory / "sweep.json"
        if not path.exists():
            continue
        for entry in json.loads(path.read_text()):
            if entry["name"] in seen or PRIMARY not in entry["miou"]:
                continue
            radius = covering_radius(entry["spec"])
            if radius is None:
                continue
            seen.add(entry["name"])
            rows.append(dict(
                name=entry["name"], radius=radius * 100, miou=entry["miou"][PRIMARY],
                bcc=entry["spec"].get("lattice") == "bcc",
                fits=entry["points_011"] <= CAP_011 and entry["points_012"] <= CAP_012,
                shipped=entry["spec"].get("lattice") == "bcc"
                        and abs(entry["spec"].get("spacing", 0) - 0.19) < 1e-9))
    rows.sort(key=lambda r: r["radius"])

    fig, ax = plt.subplots(figsize=(3.35, 2.5))
    # Two constructions, kept apart. A body-centered cubic lattice fills space;
    # the dilation variants thicken the surface itself, so at equal covering
    # radius they place points where the truth actually is and score higher.
    # Plotting them on one trend line would invite a comparison across families
    # that the sweep does not support.
    families = ((True, "body-centered cubic lattice", "o"),
                (False, "surface dilation", "^"))
    for is_bcc, label, marker in families:
        inside = [r for r in rows if r["bcc"] is is_bcc and r["fits"]]
        outside = [r for r in rows if r["bcc"] is is_bcc and not r["fits"]]
        ax.scatter([r["radius"] for r in inside], [r["miou"] for r in inside],
                   s=26, marker=marker, facecolors=ACCENT, edgecolors=ACCENT,
                   linewidths=1.0, zorder=3, label=label)
        ax.scatter([r["radius"] for r in outside], [r["miou"] for r in outside],
                   s=26, marker=marker, facecolors="none", edgecolors=WARN,
                   linewidths=1.0, zorder=3,
                   label="over point budget" if is_bcc else None)
    shipped = next((r for r in rows if r["shipped"]), None)
    if shipped:
        ax.scatter([shipped["radius"]], [shipped["miou"]], s=110, marker="o",
                   facecolors="none", edgecolors=INK, linewidths=1.2, zorder=4)
        ax.annotate("submitted", (shipped["radius"], shipped["miou"]),
                    textcoords="offset points", xytext=(9, -3), fontsize=7, color=INK)

    ax.set_xlabel("covering radius (cm)", fontsize=8)
    ax.set_ylabel(f"pooled mIoU at {PRIMARY} m frame error", fontsize=8)
    ax.tick_params(labelsize=7)
    ax.grid(True, linewidth=0.4, color="#e3e5e8")
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_edgecolor("#c8ccd2")
    ax.legend(fontsize=6.6, frameon=False, loc="lower left", handletextpad=0.3)
    fig.tight_layout(pad=0.3)
    args.out.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out / "sweep.pdf", bbox_inches="tight", pad_inches=0.01)
    print(f"wrote sweep.pdf   {len(rows)} of 29 constructions have a defined "
          f"covering radius; {sum(r['fits'] for r in rows)} of those are inside "
          f"the point budget")
    for r in rows:
        print(f"   {r['radius']:5.1f} cm  mIoU {r['miou']:.4f}  "
              f"{'fits' if r['fits'] else 'over'}  {r['name']}")


if __name__ == "__main__":
    main()
