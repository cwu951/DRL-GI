from pathlib import Path
import math

import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
from matplotlib import rcParams
from matplotlib.cm import get_cmap
from matplotlib.gridspec import GridSpec
import numpy as np
from scipy import stats


ROOT = Path(__file__).resolve().parent


def set_plot_style():
    config = {
        "font.family": "serif",
        "font.size": 12,
        "axes.titlesize": 12,
        "axes.labelsize": 12,
        "xtick.labelsize": 10,
        "ytick.labelsize": 10,
        "legend.fontsize": 12,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "mathtext.fontset": "stix",
        "font.serif": ["Times New Roman"],
        "axes.unicode_minus": False,
    }
    rcParams.update(config)


def load_raindata():
    return np.load(ROOT / "rainfall" / "training_raindata.npy", allow_pickle=True).tolist()


def resolve_record_path(path_like):
    path = Path(path_like)
    if path.is_absolute():
        return path
    return ROOT / path


def load_record(path_like):
    record_path = resolve_record_path(path_like)
    if not record_path.exists():
        raise FileNotFoundError(
            f"Result file not found: {record_path}\n"
            "For GLI/GCI plots you must provide both the Base result file and the GI-only result file, "
            "in addition to the model result files."
        )
    return np.load(record_path, allow_pickle=True).tolist()


def validate_required_records(path_map):
    missing = []
    resolved = {}
    for label, path_like in path_map.items():
        resolved_path = resolve_record_path(path_like)
        resolved[label] = resolved_path
        if not resolved_path.exists():
            missing.append((label, resolved_path))

    if missing:
        details = "\n".join([f"- {label}: {path}" for label, path in missing])
        raise FileNotFoundError(
            "Required result files are missing.\n"
            f"{details}\n"
            "For GLI/GCI plotting you need:\n"
            "- one Base result file\n"
            "- one GI-only result file\n"
            "- all model result files to compare\n"
        )
    return resolved


def _find_first_rainfall_key(record):
    for key in sorted(record.keys()):
        if str(key).startswith("rainfall"):
            return key
    raise KeyError("No rainfall* key found in result record.")


def _sorted_rainfall_keys(record):
    rainfall_keys = [key for key in record.keys() if str(key).startswith("rainfall")]
    if not rainfall_keys:
        raise KeyError("No rainfall* key found in result record.")
    return sorted(rainfall_keys, key=lambda key: int(str(key).replace("rainfall", "")))


def _resolve_rainfall_key(record, rainfall_key):
    if rainfall_key in record:
        return rainfall_key

    rainfall_keys = _sorted_rainfall_keys(record)
    desired_index = int(str(rainfall_key).replace("rainfall", ""))

    # Real-rainfall result files usually contain only a few events; if the requested
    # key is missing but the ordinal index matches the stored event order, fall back
    # to the corresponding rainfall key instead of crashing.
    if 0 <= desired_index < len(rainfall_keys):
        return rainfall_keys[desired_index]

    raise KeyError(f"{rainfall_key!r} not found in record. Available keys: {rainfall_keys[:10]}")


def infer_data_path(record):
    rainfall_key = _find_first_rainfall_key(record)
    rainfall_data = record[rainfall_key]
    if "flooding" in rainfall_data and "CSO" in rainfall_data:
        return []
    candidate_keys = [key for key, value in rainfall_data.items() if isinstance(value, dict)]
    if len(candidate_keys) == 1:
        return [candidate_keys[0]]
    raise ValueError("Unable to infer result path automatically. Please provide data_path explicitly.")


def _resolve_rainfall_data(record, rainfall_key, data_path):
    resolved_key = _resolve_rainfall_key(record, rainfall_key)
    current = record[resolved_key]
    for key in data_path:
        current = current[key]
    return current


def get_metric_series(record, rainfall_key, data_path, metric_name, skip=2, scale=1000.0):
    rainfall_data = _resolve_rainfall_data(record, rainfall_key, data_path)
    return np.asarray(rainfall_data[metric_name], dtype=float)[skip:] / scale


def get_flooding_cso_series(record, rainfall_key, data_path, skip=2, scale=1000.0):
    flooding = get_metric_series(record, rainfall_key, data_path, "flooding", skip=skip, scale=scale)
    cso = get_metric_series(record, rainfall_key, data_path, "CSO", skip=skip, scale=scale)
    return flooding + cso


def get_inflow_series(record, rainfall_key, data_path, skip=2, scale=1000.0):
    return get_metric_series(record, rainfall_key, data_path, "inflow", skip=skip, scale=scale)


def build_model_specs(model_specs, results_dir="results", cmap_name="tab10"):
    cmap = get_cmap(cmap_name, max(10, len(model_specs)))
    built_specs = []
    for index, spec in enumerate(model_specs):
        if "path" in spec:
            result_path = spec["path"]
        elif "file" in spec:
            result_path = Path(results_dir) / spec["file"]
        else:
            raise KeyError("Each model spec must provide either 'path' or 'file'.")
        record = load_record(result_path)
        data_path = spec.get("data_path")
        if data_path is None:
            data_path = infer_data_path(record)
        built_specs.append(
            {
                "label": spec["label"],
                "file": spec.get("file", str(result_path)),
                "record": record,
                "data_path": data_path,
                "color": spec.get("color", mcolors.to_hex(cmap(index))),
                "linestyle": spec.get("linestyle", "-"),
                "marker": spec.get("marker", "o"),
                "linewidth": spec.get("linewidth", 1.8),
                "scatter_size": spec.get("scatter_size", 100),
            }
        )
    return built_specs


def pick_events_by_peak_gradient(raindata, start_idx, num_rainfalls, n_pick):
    peak_values = []
    for rainfall_index in range(start_idx, start_idx + num_rainfalls):
        rain_event = raindata[rainfall_index]
        intensities = [float(value[1]) for value in rain_event]
        peak_values.append(max(intensities) if intensities else 0.0)

    peak_values = np.asarray(peak_values, dtype=float)
    sorted_indices = np.argsort(peak_values)
    if n_pick == 1:
        chosen_local_idx = [len(sorted_indices) // 2]
    else:
        chosen_local_idx = np.linspace(0, len(sorted_indices) - 1, n_pick).round().astype(int)
    chosen_indices = sorted_indices[chosen_local_idx]
    chosen_events = [start_idx + int(i) for i in chosen_indices]
    chosen_peaks = [float(peak_values[i]) for i in chosen_indices]
    return chosen_events, chosen_peaks


def compute_improvements_vs_baseline(base_record, base_path, model_specs, rain_ids):
    improvement_map = {spec["label"]: [] for spec in model_specs}
    for rain_id in rain_ids:
        rainfall_key = f"rainfall{rain_id}"
        base_last = get_flooding_cso_series(base_record, rainfall_key, base_path)[-1]
        for spec in model_specs:
            model_last = get_flooding_cso_series(spec["record"], rainfall_key, spec["data_path"])[-1]
            improvement_map[spec["label"]].append(base_last - model_last)
    return {label: np.asarray(values, dtype=float) for label, values in improvement_map.items()}


def plot_control_performance_figure(
    model_specs,
    base_record,
    base_path,
    rain_ids,
    chosen_events,
    chosen_peaks,
    one_step_minutes=5,
    real_model_specs=None,
    real_events=None,
    save_path=None,
):
    improvements = compute_improvements_vs_baseline(base_record, base_path, model_specs, rain_ids)

    design_event_count = len(chosen_events)
    real_events = list(real_events or [])
    mixed_events = list(chosen_events) + real_events
    n_pick = len(mixed_events)
    n_col = 2
    n_row = math.ceil(n_pick / n_col)

    fig_height = 2.0 * n_row + 1.0
    fig = plt.figure(figsize=(7, fig_height), dpi=150, constrained_layout=True)
    outer = GridSpec(nrows=2, ncols=1, figure=fig, height_ratios=[4, 1], hspace=0.1)
    gs_top = outer[0].subgridspec(nrows=n_row, ncols=n_col, wspace=0.05, hspace=0.05)
    axes_top = np.empty((n_row, n_col), dtype=object)
    for row in range(n_row):
        for col in range(n_col):
            axes_top[row, col] = fig.add_subplot(gs_top[row, col])
    ax_bar = fig.add_subplot(outer[1, 0])

    line_handles = []
    line_labels = []

    for index, event_id in enumerate(mixed_events):
        rainfall_key = f"rainfall{event_id}"
        col = index // n_row
        row = index % n_row
        ax = axes_top[row, col]
        is_real = index >= design_event_count
        active_specs = real_model_specs if (is_real and real_model_specs is not None) else model_specs

        for spec in active_specs:
            series = get_flooding_cso_series(spec["record"], rainfall_key, spec["data_path"])
            minutes = np.arange(len(series)) * one_step_minutes
            handle, = ax.plot(
                minutes,
                series,
                label=spec["label"],
                color=spec["color"],
                linestyle=spec["linestyle"],
                linewidth=1.2 if spec["linestyle"] == "--" else 1.0,
            )
            line_handles.append(handle)
            line_labels.append(spec["label"])

        if is_real:
            ax.set_title(f"Real Rainfall {event_id + 1}", fontweight="bold")
        else:
            peak = chosen_peaks[index]
            ax.set_title(
                f"Design Rainfall {event_id - rain_ids[0] + 1} (peak = {peak:.2f} mm/hour)",
                fontweight="bold",
            )
        ax.set_ylabel("FC (×10³ m³)")
        ax.set_xlabel("Time (minutes)")
        ax.grid(True, alpha=0.3)

    for index in range(n_pick, n_row * n_col):
        fig.delaxes(axes_top[index // n_col, index % n_col])

    unique = {}
    for handle, label in zip(line_handles, line_labels):
        if label not in unique:
            unique[label] = handle

    gap_factor = max(1.2, len(model_specs) * 0.33)
    x_base = np.arange(len(rain_ids)) * gap_factor
    bar_width = min(0.14, 0.9 / max(1, len(model_specs)))
    offsets = (np.arange(len(model_specs)) - (len(model_specs) - 1) / 2.0) * bar_width

    for idx, spec in enumerate(model_specs):
        ax_bar.bar(
            x_base + offsets[idx],
            improvements[spec["label"]],
            width=bar_width,
            color=spec["color"],
            edgecolor="black",
            linewidth=0.4,
            alpha=0.9,
            label=spec["label"],
        )

    tick_step = 2 if len(rain_ids) <= 60 else 5
    ax_bar.set_xticks(x_base[::tick_step])
    ax_bar.set_xticklabels([str(i + 1) for i in range(0, len(rain_ids), tick_step)])
    ax_bar.set_xlabel(f"Rainfall event index (1-{len(rain_ids)})")
    ax_bar.set_ylabel("Reduction in FC\nvs. baseline (×10³ m³)")
    ax_bar.grid(axis="y", linestyle="--", alpha=0.3)
    ax_bar.margins(x=0.01)

    fig.legend(
        list(unique.values()),
        list(unique.keys()),
        loc="upper center",
        ncol=min(len(unique), 6),
        frameon=False,
        bbox_to_anchor=(0.5, 1.04),
    )
    fig.text(-0.05, 1.00, "(a)", ha="left", va="top", fontsize=14, fontweight="bold")
    fig.text(-0.05, 0.25, "(b)", ha="left", va="top", fontsize=14, fontweight="bold")

    if save_path is not None:
        save_path = ROOT / save_path
        save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=300, bbox_inches="tight")
    return fig, improvements


def compute_delta1(base_record, base_path, gi_record, gi_path, rain_ids):
    delta_1 = []
    for rain_id in rain_ids:
        rainfall_key = f"rainfall{rain_id}"
        base_inflow = get_inflow_series(base_record, rainfall_key, base_path)
        gi_inflow = get_inflow_series(gi_record, rainfall_key, gi_path)
        delta_1.append(np.sum(base_inflow - gi_inflow))
    return np.asarray(delta_1, dtype=float)


def compute_delta2(model_record, model_path, base_record, base_path, rain_ids):
    delta_2 = []
    for rain_id in rain_ids:
        rainfall_key = f"rainfall{rain_id}"
        base_fc = get_flooding_cso_series(base_record, rainfall_key, base_path)
        model_fc = get_flooding_cso_series(model_record, rainfall_key, model_path)
        delta_2.append(float(np.sum(base_fc - model_fc)))
    return np.asarray(delta_2, dtype=float)


def plot_delta_regression_figure(
    model_specs,
    base_record,
    base_path,
    gi_record,
    gi_path,
    rain_ids,
    compare_name="Model",
    save_path=None,
):
    delta_1 = compute_delta1(base_record, base_path, gi_record, gi_path, rain_ids)
    delta_2_map = {
        spec["label"]: compute_delta2(spec["record"], spec["data_path"], base_record, base_path, rain_ids)
        for spec in model_specs
    }

    fig, ax = plt.subplots(figsize=(7, 6.5), dpi=150)
    pending_annotations = []

    for spec in model_specs:
        yvals = delta_2_map[spec["label"]]
        ax.scatter(
            delta_1,
            yvals,
            c=spec["color"],
            marker=spec["marker"],
            s=spec["scatter_size"],
            alpha=0.6,
            edgecolors="white",
            linewidth=0.8,
            label=spec["label"],
            zorder=3,
        )

        mask = np.isfinite(delta_1) & np.isfinite(yvals)
        if mask.sum() >= 2:
            slope, intercept, r_val, _, _ = stats.linregress(delta_1[mask], yvals[mask])
            x_line = np.array([0, max(np.max(delta_1[mask]) * 1.08, 1e-6)])
            y_line = slope * x_line + intercept
            ax.plot(x_line, y_line, color=spec["color"], linewidth=1.0, alpha=0.7, zorder=2)

            pending_annotations.append(
                {
                    "text": (
                        f"{spec['label']}: $y = {slope:.2f}x + {intercept:.2f}$, $R^2 = {r_val**2:.2f}$"
                        if intercept >= 0
                        else f"{spec['label']}: $y = {slope:.2f}x - {abs(intercept):.2f}$, $R^2 = {r_val**2:.2f}$"
                    ),
                    "color": tuple(max(0.0, val * 0.6) for val in mcolors.to_rgb(spec["color"])),
                }
            )

    ax.set_xlabel("Δ1 = Base Inflow − GI Inflow (×10³ m³)")
    ax.set_ylabel(f"Δ2 = Base FC − {compare_name} FC (×10³ m³)")
    ax.grid(True, alpha=0.3)
    ax.legend(loc="upper left", ncol=2, frameon=False)

    if np.isfinite(delta_1).any():
        x_max = max(1e-9, np.nanmax(delta_1)) * 1.1
        ax.set_xlim(0, x_max if x_max > 0 else 1.0)
    ax.set_ylim(0, 1000)

    y_start = 0.02
    line_gap = 0.055
    for idx, ann in enumerate(reversed(pending_annotations)):
        ax.text(
            0.98,
            y_start + idx * line_gap,
            ann["text"],
            transform=ax.transAxes,
            color=ann["color"],
            fontsize=12,
            fontweight="bold",
            ha="right",
            va="bottom",
            zorder=4,
        )

    if save_path is not None:
        save_path = ROOT / save_path
        save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=300, bbox_inches="tight")
    return fig, delta_1, delta_2_map


def _safe_ratio(num, den):
    num = np.asarray(num, dtype=float)
    den = np.asarray(den, dtype=float)
    return np.divide(num, den, out=np.zeros_like(num), where=(den != 0))


def _gci_curve(base_record, base_path, gi_record, gi_path, model_record, model_path, rainfall_key):
    base_in = get_inflow_series(base_record, rainfall_key, base_path)
    gi_in = get_inflow_series(gi_record, rainfall_key, gi_path)
    delta1_cum = np.cumsum(base_in - gi_in)

    base_fc = get_flooding_cso_series(base_record, rainfall_key, base_path)
    model_fc = get_flooding_cso_series(model_record, rainfall_key, model_path)
    delta2_cum = np.cumsum(base_fc - model_fc)

    ratio = _safe_ratio(delta2_cum, delta1_cum)
    ratio = np.clip(ratio, -700, 700)
    return -np.exp(-ratio)


def plot_gli_process_figure(
    model_specs,
    base_record,
    base_path,
    gi_record,
    gi_path,
    chosen_events,
    chosen_peaks,
    rain_start,
    one_step_minutes=5,
    real_model_specs=None,
    real_base_record=None,
    real_base_path=None,
    real_gi_record=None,
    real_gi_path=None,
    real_events=None,
    save_path=None,
):
    design_event_count = len(chosen_events)
    real_events = list(real_events or [])
    mixed_events = list(chosen_events) + real_events
    n_pick = len(mixed_events)
    n_col = 2
    n_row = math.ceil(n_pick / n_col)

    fig_height = 1.75 * n_row
    fig, axes = plt.subplots(n_row, n_col, figsize=(7, fig_height), dpi=150, squeeze=False)
    handles = None
    labels = None

    for index, event_id in enumerate(mixed_events):
        rainfall_key = f"rainfall{event_id}"
        col = index // n_row
        row = index % n_row
        ax = axes[row, col]
        is_real = index >= design_event_count

        current_model_specs = real_model_specs if (is_real and real_model_specs is not None) else model_specs
        current_base_record = real_base_record if (is_real and real_base_record is not None) else base_record
        current_base_path = real_base_path if (is_real and real_base_path is not None) else base_path
        current_gi_record = real_gi_record if (is_real and real_gi_record is not None) else gi_record
        current_gi_path = real_gi_path if (is_real and real_gi_path is not None) else gi_path

        local_handles = []
        for spec in current_model_specs:
            curve = _gci_curve(
                current_base_record,
                current_base_path,
                current_gi_record,
                current_gi_path,
                spec["record"],
                spec["data_path"],
                rainfall_key,
            )
            minutes = np.arange(len(curve)) * one_step_minutes
            handle, = ax.plot(
                minutes,
                curve,
                label=spec["label"],
                color=spec["color"],
                linestyle=spec["linestyle"],
                linewidth=1.2 if spec["linestyle"] == "--" else 1.0,
            )
            local_handles.append(handle)

        if handles is None:
            handles = local_handles
            labels = [h.get_label() for h in local_handles]

        if is_real:
            ax.set_title(f"Real Rainfall {event_id + 1}", fontweight="bold")
        else:
            peak = chosen_peaks[index]
            ax.set_title(
                f"Design Rainfall {event_id - rain_start + 1} (peak = {peak:.2f} mm/hour)",
                fontweight="bold",
            )
        ax.set_ylim(-1.1, 0.1)
        ax.set_ylabel(r"$-e^{-\mathrm{GCI}}$")
        ax.set_xlabel("Time (minutes)")
        ax.grid(True, alpha=0.3)

    for index in range(n_pick, n_row * n_col):
        fig.delaxes(axes[index // n_col, index % n_col])

    if handles is not None:
        fig.legend(handles, labels, loc="upper center", bbox_to_anchor=(0.5, 1.04), ncol=len(labels), frameon=False)

    plt.tight_layout()
    if save_path is not None:
        save_path = ROOT / save_path
        save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=300, bbox_inches="tight")
    return fig
