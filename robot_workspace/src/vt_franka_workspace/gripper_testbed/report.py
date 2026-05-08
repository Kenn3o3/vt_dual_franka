from __future__ import annotations

import json
from pathlib import Path
from statistics import mean
from typing import Any


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    rows: list[dict[str, Any]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line:
            rows.append(json.loads(line))
    return rows


def write_gripper_testbed_report(run_dir: str | Path, output_path: str | Path | None = None) -> Path:
    run_dir = Path(run_dir)
    output_path = Path(output_path) if output_path is not None else run_dir / "gripper_testbed_report.html"
    telemetry = []
    for stream_path in sorted(run_dir.glob("episodes/episode_*/streams/gripper_telemetry.jsonl")):
        telemetry.extend(load_jsonl(stream_path))
    states = []
    for stream_path in sorted(run_dir.glob("episodes/episode_*/streams/gripper_states.jsonl")):
        states.extend(load_jsonl(stream_path))

    plot_path = output_path.with_suffix(".png")
    plot_written = False
    if telemetry or states:
        plot_written = _write_plot(telemetry, states, plot_path)

    target_widths = [float(row["target_width"]) for row in telemetry if row.get("target_width") is not None]
    triggers = [float(row["trigger_depth"]) for row in telemetry if row.get("trigger_depth") is not None]
    force_limits = [float(row["force_limit"]) for row in telemetry if row.get("force_limit") is not None]
    width_errors = [abs(float(row["width_error"])) for row in states if row.get("width_error") is not None]
    summary = {
        "target_commands": len(telemetry),
        "state_samples": len(states),
        "trigger_min": min(triggers) if triggers else None,
        "trigger_max": max(triggers) if triggers else None,
        "target_width_min": min(target_widths) if target_widths else None,
        "target_width_max": max(target_widths) if target_widths else None,
        "mean_abs_width_error": mean(width_errors) if width_errors else None,
        "max_abs_width_error": max(width_errors) if width_errors else None,
        "force_limit_values": sorted(set(force_limits)),
        "force_limit_mean": mean(force_limits) if force_limits else None,
    }
    html = _render_report(run_dir, summary, plot_path if plot_written else None)
    output_path.write_text(html, encoding="utf-8")
    (output_path.with_suffix(".summary.json")).write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return output_path


def _write_plot(target_rows: list[dict[str, Any]], state_rows: list[dict[str, Any]], plot_path: Path) -> bool:
    try:
        import matplotlib
    except ImportError:
        return False

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    all_rows = target_rows or state_rows
    t0 = float(all_rows[0].get("wall_time", all_rows[0].get("recorded_at_wall_time", 0.0)))
    target_times = [float(row.get("wall_time", row.get("recorded_at_wall_time", t0))) - t0 for row in target_rows]
    triggers = [float(row.get("trigger_depth", 0.0)) for row in target_rows]
    targets = [float(row.get("target_width", 0.0)) for row in target_rows]
    forces = [float(row.get("force_limit", 0.0)) for row in target_rows]
    state_times = [float(row.get("wall_time", row.get("recorded_at_wall_time", t0))) - t0 for row in state_rows]
    measured_widths = [float(row.get("measured_width", 0.0)) for row in state_rows]
    width_errors = [float(row.get("width_error", 0.0)) for row in state_rows]

    fig, axes = plt.subplots(3, 1, figsize=(11, 8), sharex=True)
    axes[0].plot(target_times, triggers, label="trigger depth", color="tab:blue")
    axes[0].set_ylabel("trigger")
    axes[0].set_ylim(-0.05, 1.05)
    axes[0].grid(True, alpha=0.25)
    axes[0].legend(loc="upper right")

    axes[1].plot(target_times, targets, label="target width m", color="tab:green")
    axes[1].plot(state_times, measured_widths, label="measured width m", color="tab:brown", alpha=0.85)
    axes[1].set_ylabel("width (m)")
    axes[1].grid(True, alpha=0.25)
    axes[1].legend(loc="upper right")

    axes[2].step(target_times, forces, label="force limit N", color="tab:orange", where="post")
    axes[2].plot(state_times, width_errors, label="width error m", color="tab:red", alpha=0.7)
    axes[2].set_ylabel("force / error")
    axes[2].set_xlabel("time (s)")
    axes[2].grid(True, alpha=0.25)
    axes[2].legend(loc="upper right")

    fig.tight_layout()
    fig.savefig(plot_path, dpi=150)
    plt.close(fig)
    return True


def _render_report(run_dir: Path, summary: dict[str, Any], plot_path: Path | None) -> str:
    image_html = ""
    if plot_path is not None:
        image_html = f'<img src="{plot_path.name}" alt="Gripper testbed plot">'
    rows = "\n".join(
        f"<tr><th>{key}</th><td>{value}</td></tr>"
        for key, value in summary.items()
    )
    return f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Gripper Testbed Report</title>
  <style>
    body {{ font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif; margin: 32px; color: #18212b; background: #f8fafc; }}
    main {{ max-width: 980px; margin: 0 auto; background: white; border: 1px solid #d9e0e8; border-radius: 12px; padding: 24px; }}
    h1 {{ margin-top: 0; }}
    table {{ width: 100%; border-collapse: collapse; margin: 18px 0; }}
    th, td {{ text-align: left; border-bottom: 1px solid #e6ebf0; padding: 9px 10px; }}
    th {{ width: 220px; color: #52606d; }}
    img {{ width: 100%; border: 1px solid #d9e0e8; border-radius: 8px; }}
    code {{ background: #eef3f6; padding: 2px 5px; border-radius: 4px; }}
  </style>
</head>
<body>
  <main>
    <h1>Gripper Testbed Report</h1>
    <p>Run directory: <code>{run_dir}</code></p>
    <table>{rows}</table>
    {image_html}
  </main>
</body>
</html>
"""
