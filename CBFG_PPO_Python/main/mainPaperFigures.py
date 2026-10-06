"""Plot main-paper Fig. 7/8/9 layouts from actual CSV logs.

Run from the project root: python -m main.mainPaperFigures
Optional --manifest accepts a JSON list of {method, speed, seed, directory}.
directory points to an experiment with logs/trainingLog.csv and, optionally,
logs/simulationLog.csv and logs/evaluationLog.csv. Relative paths are resolved
against the manifest directory. Missing methods are never fabricated.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import FuncFormatter

METHODS = ("CBFG-PPO", "PPO-R", "PPO", "DDPG")
COLORS = dict(zip(METHODS, ("#ef5656", "#72a3ed", "#65b960", "#b45eb7")))
PROJECT = Path(__file__).resolve().parents[1]


def read_csv(path):
    with Path(path).open(encoding="utf-8-sig", newline="") as stream:
        return list(csv.DictReader(stream))


def trailing_mean(values, width=10):
    values = np.asarray(values, dtype=float)
    cumulative = np.r_[0., np.cumsum(values)]
    end = np.arange(1, len(values) + 1)
    start = np.maximum(0, end - width)
    return (cumulative[end] - cumulative[start]) / (end - start)


def training_series(rows):
    # A final episode interrupted by the step budget is not a failure episode.
    rows = [r for r in rows if int(r.get("episode_complete", bool(r.get("termination_reason"))))]
    if not rows:
        raise ValueError("Training log contains no completed episodes")
    steps = np.array([int(r["global_step"]) for r in rows])
    if np.any(np.diff(steps) <= 0):
        raise ValueError("Completed episode endpoints must strictly increase")
    reward = np.array([float(r["episode_reward"]) for r in rows])
    success = np.array([float(r["success"]) for r in rows])
    # IV-A: successes/completed episodes ending in (t-3000, t].
    start = np.searchsorted(steps, steps - 3000, side="right")
    cumulative = np.r_[0., np.cumsum(success)]
    end = np.arange(1, len(steps) + 1)
    raw_rate = (cumulative[end] - cumulative[start]) / (end - start)
    return steps, reward, trailing_mean(raw_rate), raw_rate


def mean_sd(values):
    values = np.asarray(values, dtype=float)
    # Paper does not identify the SD denominator. Population SD (ddof=0)
    # is an explicit plotting convention. One seed cannot estimate variation.
    return float(values.mean()), float(values.std(ddof=0)) if len(values) > 1 else None


def align_series(series, component):
    # Common observed support only; never extrapolate early/late seed results.
    lower = max(s[0][0] for s in series)
    upper = min(s[0][-1] for s in series)
    if lower > upper:
        raise ValueError("Seeds have no overlapping training-step interval")
    grid = np.unique(np.concatenate([s[0] for s in series]))
    grid = grid[(grid >= lower) & (grid <= upper)]
    values = np.array([np.interp(grid, s[0], s[component]) for s in series])
    return grid, values.mean(axis=0), values.std(axis=0, ddof=0)


def save_figure(fig, output, stem):
    fig.savefig(output / (stem + ".png"), dpi=180, bbox_inches="tight")
    fig.savefig(output / (stem + ".pdf"), bbox_inches="tight")
    plt.close(fig)


def plot_training(runs, output):
    fig = plt.figure(figsize=(13.8, 8.4))
    grid = fig.add_gridspec(2, 3, wspace=.32, hspace=.48)
    summaries = []
    for row, speed in enumerate((5, 10)):
        reward_ax = fig.add_subplot(grid[row, 0])
        rate_ax = fig.add_subplot(grid[row, 1])
        bar_grid = grid[row, 2].subgridspec(2, 1, hspace=.35)
        bar_axes = [fig.add_subplot(bar_grid[i]) for i in (0, 1)]
        groups = {m: [r for r in runs if r["method"] == m and r["speed"] == speed and r.get("series") is not None] for m in METHODS}
        for method, group in groups.items():
            if not group:
                continue
            series = [r["series"] for r in group]
            for ax, component in ((reward_ax, 1), (rate_ax, 2)):
                display_series = [(s[0], trailing_mean(s[1]), s[2], s[3]) for s in series] if component == 1 else series
                x, mean, sd = align_series(display_series, component)
                ax.plot(x, mean, color=COLORS[method], lw=1.1, label=f"{method} (n={len(group)})")
                if len(group) > 1:
                    ax.fill_between(x, mean-sd, mean+sd, color=COLORS[method], alpha=.2)
            # Figure caption: whole-training mean within each run, then across seeds.
            metrics = ([float(s[1].mean()) for s in series], [float(s[3].mean()) for s in series])
            summary = {"speed": speed, "method": method, "seeds": [r["seed"] for r in group], "run_mean_rewards": metrics[0], "run_mean_raw_success_rates": metrics[1]}
            for i, values in enumerate(metrics):
                mean, sd = mean_sd(values)
                key = ("reward", "success_rate")[i]
                summary[key + "_mean"] = mean
                summary[key + "_sd"] = sd
                kwargs = {"yerr": sd, "capsize": 3} if sd is not None else {}
                bar_axes[i].bar(METHODS.index(method), mean, color=COLORS[method], width=.7, **kwargs)
            summaries.append(summary)
        for ax, label, letter in ((reward_ax, "Average episodic reward", "ad"[row]), (rate_ax, "Success rate", "be"[row])):
            ax.set(xlabel="Training step", ylabel=label, title=f"({letter}) Minimum EV speed {speed} m/s")
            ax.xaxis.set_major_formatter(FuncFormatter(lambda x, _: f"{x/1000:g}k"))
            ax.grid(alpha=.2)
            if any(groups.values()):
                ax.legend(fontsize=7, loc="best")
        rate_ax.set_ylim(0, 1.05)
        for i, ax in enumerate(bar_axes):
            ax.set_xticks(range(4), METHODS, fontsize=8)
            ax.set_xlim(-.6, 3.6)
            ax.set_ylabel(("Mean training reward", "Mean success rate")[i], fontsize=9)
            ax.grid(axis="y", alpha=.2)
            for j, method in enumerate(METHODS):
                if not groups[method]:
                    ax.text(j, .08, "N/A", transform=ax.get_xaxis_transform(), ha="center", fontsize=8, color="gray")
            if i == 1:
                ax.set_ylim(0, 1.05)
            else:
                ax.set_title(f"({'cf'[row]}) Whole-training averages", fontsize=10)
        counts = [len(g) for g in groups.values() if g]
        if counts and min(counts) == 1:
            bar_axes[1].text(.5, -.36, "n=1: seed SD unavailable", transform=bar_axes[1].transAxes, ha="center", fontsize=8)
    fig.suptitle("Fig. 7 layout — measured implementation results", fontsize=14)
    fig.text(.5, .012, "Missing baselines: N/A. No paper curves copied. Success: last 3000 steps, trailing width 10. Reward display: trailing width 10 (plotting assumption).", ha="center", fontsize=9)
    save_figure(fig, output, "fig7_learning_comparison")
    return summaries


def plot_trace(run, output):
    rows = read_csv(run["directory"] / "logs" / "simulationLog.csv")
    if not rows:
        return None
    time = np.array([float(r["time"]) for r in rows])
    vehicle_ids = sorted({int(k.split("_")[0][2:]) for k in rows[0] if k.startswith("sv") and k.endswith("_x")})
    fig, axes = plt.subplots(3, 1, figsize=(8.8, 7), sharex=True)
    for ax, field, ylabel, letter in zip(axes, ("x", "y", "speed"), ("x (m)", "y (m)", "Speed (m/s)"), "abc"):
        ax.plot(time, [float(r["ego_"+field]) for r in rows], color="#3575c8", lw=1.7, label="EV")
        for i in vehicle_ids:
            ax.plot(time, [float(r[f"sv{i}_"+field]) for r in rows], lw=1, ls=("--", "-.", ":")[(i-1)%3], label=f"SV{i}")
        ax.set_ylabel(ylabel)
        ax.set_title(f"({letter})", fontsize=10)
        ax.grid(alpha=.35)
        if field == "y":
            for boundary in (2, 6, 10):
                ax.axhline(boundary, color="gray", lw=.5, alpha=.6)
            ax.set_ylim(1.5, 10.5)
        ax.legend(fontsize=7, ncol=min(len(vehicle_ids)+1, 7), loc="best")
    axes[-1].set_xlabel("Time (s)")
    number = 8 if run["method"] == "CBFG-PPO" else 9
    stem = f"fig{number}_{run['method'].lower().replace('-', '_')}_speed{run['speed']}_seed{run['seed']}"
    fig.suptitle(f"Fig. {number} layout — {run['method']}, minimum EV speed {run['speed']} m/s", fontsize=12)
    fig.text(.5, .015, "Recorded evaluation episode; full time interval and all SVs. Not the paper's identical condensed traffic case.", ha="center", fontsize=8)
    fig.tight_layout(rect=(0, .035, 1, .96))
    save_figure(fig, output, stem)
    return {"file": stem, "method": run["method"], "speed": run["speed"], "seed": run["seed"], "sv_count": len(vehicle_ids), "start_time": float(time[0]), "end_time": float(time[-1])}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path)
    parser.add_argument("--output-dir", type=Path, default=PROJECT / "results" / "figures_paper")
    args = parser.parse_args()
    if args.manifest:
        spec = json.loads(args.manifest.read_text(encoding="utf-8-sig"))
        base = args.manifest.resolve().parent
    else:
        spec = [{"method": "CBFG-PPO", "speed": s, "seed": 0, "directory": str(PROJECT / "results" / "runs" / f"reference_v4_speed{s}_seed0")} for s in (5, 10)]
        base = PROJECT
    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=True)
    runs, sources, identities = [], [], set()
    for item in spec:
        run = dict(item)
        if run["method"] not in METHODS or run["speed"] not in (5, 10):
            raise ValueError("Supported methods: CBFG-PPO, PPO-R, PPO, DDPG; speeds: 5, 10")
        key = (run["method"], run["speed"], run["seed"])
        if key in identities:
            raise ValueError(f"Duplicate seed: {key}")
        identities.add(key)
        run["directory"] = (base / run["directory"]).resolve()
        train = run["directory"] / "logs" / "trainingLog.csv"
        if train.exists():
            run["series"] = training_series(read_csv(train))
        else:
            run["series"] = None
        for name in ("trainingLog.csv", "simulationLog.csv", "evaluationLog.csv", "evaluationManifest.json"):
            path = run["directory"] / "logs" / name
            if path.exists():
                sources.append({"path": str(path), "sha256": hashlib.sha256(path.read_bytes()).hexdigest()})
        runs.append(run)
    summaries = plot_training(runs, output)
    traces = [plot_trace(r, output) for r in runs if r["method"] in ("CBFG-PPO", "PPO") and (r["directory"] / "logs" / "simulationLog.csv").exists()]
    payload = {"training_statistics": summaries, "traces": traces, "sources": sources,
               "conventions": {"success_window_steps": 3000, "success_smoothing_samples": 10, "reward_smoothing": "trailing width 10, plotting assumption; bars use raw returns", "curve_alignment": "linear interpolation on common observed support", "bar_mean": "equal-weight mean of run means, completed episode endpoints", "bar_success": "raw 3000-step rates, before display smoothing", "standard_deviation_ddof": 0, "single_seed_sd": None}}
    (output / "plot_manifest.json").write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    (output / "input_manifest.json").write_text(json.dumps(spec, indent=2, ensure_ascii=False), encoding="utf-8")
    (output / "README.md").write_text("""# 主論文との比較用グラフ

- fig7_learning_comparison.png / .pdf：Fig.7と同じ2行3列構成。左は報酬、中央は成功率、右は学習全体の平均。現在CBFG-PPOのみ各1 seed。PPO-R・PPO・DDPGはN/A。1 seedのためseed間SD・帯は描かない。
- fig8_cbfg_ppo_speed5_seed0.png / .pdf：Fig.8と同じEV・SVのx座標、y座標、速度の3段構成。10 m/sも別ファイルに保存。
- Fig.9はPPOの実測ログが必要。現在は未学習なので未生成。PPOのログをmanifestに追加すると同じ3段構成で生成する。

Fig.8は既存評価で選ばれた代表エピソードを使う。主論文の凝縮ケースとSV数・配置・軌跡が一致したものではない。SV番号はログの車両IDを保持し、近傍車だけへの恣意的な選び直しや時間軸の移動をしない。時間0の行がないので0.2 sから実測値を表示する。

成功率は直近3000 stepに完了したエピソードの成功割合を各完了時点で計算し、幅10の移動平均を表示する。報酬の線は判読のため幅10の移動平均を使う（本文には指定がない描画上の仮定）。棒の報酬集計には平滑化前の全エピソード報酬を使う。右の棒は各runの全学習平均を先に計算し、seedを等重みで平均する。成功率の棒は表示平滑化前の率を使う。完了時点でのサンプリング頻度、seedの曲線間の線形補間、SDの分母（ddof=0）は描画上の仮定。途中で学習予算に達した未完了エピソードは除外する。

学習データは各150000 stepのv4 run。評価軌跡は既存bestPolicyのもの。論文の平均的な学習returnのモデルを選ぶ手順とは異なる。論文値・未実施の比較手法・10 seedの統計は補っていない。描画変更に伴う再学習は行っていない。

再生成（プロジェクト直下）：
`python -m main.mainPaperFigures`

比較手法やseed追加：input_manifest.jsonを編集したコピーに各runを追加し、
`python -m main.mainPaperFigures --manifest PATH_TO_MANIFEST`

各runのdirectory/logsにtrainingLog.csvを置く。simulationLog.csvがあればFig.8/9も生成する。元CSVのSHA256、集計値、描画規約はplot_manifest.jsonで確認できる。
""", encoding="utf-8")
    print(f"Saved paper-layout figures and provenance to {output}")


if __name__ == "__main__":
    main()
