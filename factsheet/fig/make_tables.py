"""LaTeX tables for the factsheet, generated from the Codabench leaderboards.

The two JSON files under data/ are verbatim responses from the public endpoint,
kept so every number in the tables is checkable without a login:

    curl -s https://www.codabench.org/api/phases/29458/get_leaderboard/   # final
    curl -s https://www.codabench.org/api/phases/29457/get_leaderboard/   # development

    uv run --with nothing python delivery/factsheet_official/fig/make_tables.py
"""
import argparse, json, statistics
from pathlib import Path

US = "kylechen1211"
COLUMNS = ("final_score", "psnr", "ssim", "lpips", "fscore", "miou_3d")


def rows(path):
    payload = json.loads(Path(path).read_text())
    out = []
    for submission in payload["submissions"]:
        scores = {s["column_key"]: s["score"] for s in submission["scores"]}
        out.append((submission["owner"], submission["id"], scores))
    out.sort(key=lambda r: float(r[2].get("final_score", 0)), reverse=True)
    return out


def escape(name):
    return name.replace("_", r"\_")


def leaderboard_table(path):
    ranked = rows(path)
    lines = []
    for rank, (owner, _, s) in enumerate(ranked, 1):
        cells = " & ".join(s.get(c, "--") for c in COLUMNS)
        line = f"{rank} & \\code{{{escape(owner)}}} & {cells} \\\\"
        if owner == US:
            line = ("\\rowcolor{black!7}\n" +
                    f"{rank} & \\textbf{{\\code{{{escape(owner)}}}}} & {cells} \\\\")
        lines.append(line)
    median = {c: statistics.median(float(s.get(c, 0)) for _, _, s in ranked)
              for c in COLUMNS}
    lines.append("\\midrule")
    lines.append("\\multicolumn{2}{@{}l}{median of " + str(len(ranked)) + "} & " +
                 " & ".join(f"{median[c]:.2f}" for c in COLUMNS) + " \\\\")
    return "\n".join(lines)


def phase_comparison(final_path, dev_path):
    def find(path):
        for rank, (owner, sid, s) in enumerate(rows(path), 1):
            if owner == US:
                return rank, len(rows(path)), sid, s
        raise SystemExit(f"{US} not on {path}")
    out = []
    for label, path in (("development", dev_path), ("final", final_path)):
        rank, total, sid, s = find(path)
        out.append(f"{label} & {sid} & {rank} of {total} & " +
                   " & ".join(s.get(c, "--") for c in COLUMNS) + " \\\\")
    return "\n".join(out)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data", default=Path("delivery/factsheet_official/data"), type=Path)
    args = p.parse_args()
    final = args.data / "leaderboard_final_29458.json"
    dev = args.data / "leaderboard_dev_29457.json"
    print("% ---- final leaderboard, all entries ----")
    print(leaderboard_table(final))
    print()
    print("% ---- our two phases ----")
    print(phase_comparison(final, dev))


if __name__ == "__main__":
    main()
