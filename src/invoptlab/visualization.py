from __future__ import annotations

from typing import Any, Iterable

import numpy as np

from .core import ForwardProblem, InverseDataset
from .experiments import ExperimentResult
from .geometry import (
    ConsistencyConstraints,
    GeometrySnapshot,
    feasible_polygon_2d,
    sample_feasible_region,
)
from .losses import InverseLoss, evaluate_losses


COLORS = {
    "feasible": "#4C78A8",
    "constraint": "#9AA0A6",
    "estimate": "#F58518",
    "incenter": "#E45756",
    "truth": "#54A24B",
    "loss": "#7A5195",
}


def _plt():
    try:
        import matplotlib.pyplot as plt
    except ImportError as exc:
        raise ImportError("Static plots require `pip install invoptlab[plots]`") from exc
    return plt


def _go():
    try:
        import plotly.graph_objects as go
    except ImportError as exc:
        raise ImportError("Interactive plots require `pip install invoptlab[plots]`") from exc
    return go


def plot_cone_2d(
    problem: ForwardProblem,
    snapshot: GeometrySnapshot,
    *,
    true_theta: np.ndarray | None = None,
    show_constraints: bool = True,
    ax: Any | None = None,
):
    if problem.parameter_space.dimension != 2:
        raise ValueError("plot_cone_2d requires a two-dimensional parameter")
    plt = _plt()
    if ax is None:
        _, ax = plt.subplots(figsize=(7, 7))
    polygon = feasible_polygon_2d(problem.parameter_space, snapshot.constraints)
    radius = problem.parameter_space.radius
    if problem.parameter_space.kind == "box":
        extent = float(max(np.max(np.abs(problem.parameter_space.lower)), np.max(np.abs(problem.parameter_space.upper))))
    else:
        extent = 1.12 * radius
    if polygon.size:
        closed = np.vstack([polygon, polygon[0]])
        ax.fill(closed[:, 0], closed[:, 1], color=COLORS["feasible"], alpha=0.25, label="feasible region")
        ax.plot(closed[:, 0], closed[:, 1], color=COLORS["feasible"], linewidth=2)
    if show_constraints:
        for normal in snapshot.constraints.normalized_matrix:
            direction = np.asarray([-normal[1], normal[0]])
            segment = np.vstack([-extent * direction, extent * direction])
            ax.plot(segment[:, 0], segment[:, 1], color=COLORS["constraint"], alpha=0.3, linewidth=0.8)
    if snapshot.theta is not None:
        ax.scatter(*snapshot.theta, color=COLORS["estimate"], s=75, marker="o", label="estimate", zorder=5)
    if snapshot.incenter is not None:
        ax.scatter(*snapshot.incenter, color=COLORS["incenter"], s=90, marker="*", label="incenter", zorder=6)
        if snapshot.inradius is not None and snapshot.inradius > 0:
            circle = plt.Circle(snapshot.incenter, snapshot.inradius, color=COLORS["incenter"], fill=False, linestyle="--")
            ax.add_patch(circle)
    if true_theta is not None:
        ax.scatter(*true_theta, color=COLORS["truth"], s=90, marker="X", label=r"true $\theta$", zorder=6)
    ax.axhline(0, color="black", linewidth=0.6)
    ax.axvline(0, color="black", linewidth=0.6)
    ax.set(xlim=(-extent, extent), ylim=(-extent, extent), xlabel=r"$\theta_1$", ylabel=r"$\theta_2$")
    ax.set_aspect("equal", adjustable="box")
    ax.set_title(f"Consistency region after {snapshot.step} observations")
    ax.grid(alpha=0.15)
    ax.legend(loc="best")
    return ax.figure


def animate_cone_2d(
    problem: ForwardProblem,
    snapshots: list[GeometrySnapshot],
    *,
    true_theta: np.ndarray | None = None,
):
    if problem.parameter_space.dimension != 2:
        raise ValueError("animate_cone_2d requires a two-dimensional parameter")
    go = _go()
    frames = []
    for snapshot in snapshots:
        polygon = feasible_polygon_2d(problem.parameter_space, snapshot.constraints)
        traces = []
        if polygon.size:
            closed = np.vstack([polygon, polygon[0]])
            traces.append(
                go.Scatter(
                    x=closed[:, 0], y=closed[:, 1], fill="toself", mode="lines",
                    line={"color": COLORS["feasible"]}, fillcolor="rgba(76,120,168,0.25)", name="feasible region"
                )
            )
        if snapshot.theta is not None:
            traces.append(go.Scatter(x=[snapshot.theta[0]], y=[snapshot.theta[1]], mode="markers", marker={"size": 11, "color": COLORS["estimate"]}, name="estimate"))
        if snapshot.incenter is not None:
            traces.append(go.Scatter(x=[snapshot.incenter[0]], y=[snapshot.incenter[1]], mode="markers", marker={"size": 14, "symbol": "star", "color": COLORS["incenter"]}, name="incenter"))
        if true_theta is not None:
            traces.append(go.Scatter(x=[true_theta[0]], y=[true_theta[1]], mode="markers", marker={"size": 12, "symbol": "x", "color": COLORS["truth"]}, name="true theta"))
        frames.append(go.Frame(data=traces, name=str(snapshot.step)))
    initial = frames[0].data if frames else []
    radius = problem.parameter_space.radius * 1.12
    figure = go.Figure(data=initial, frames=frames)
    figure.update_layout(
        title="Evolution of the normalized consistency region",
        xaxis={"range": [-radius, radius], "scaleanchor": "y", "title": "theta_1"},
        yaxis={"range": [-radius, radius], "title": "theta_2"},
        sliders=[{
            "active": 0,
            "steps": [{"label": frame.name, "method": "animate", "args": [[frame.name], {"mode": "immediate", "frame": {"duration": 300, "redraw": True}}]} for frame in frames],
            "currentvalue": {"prefix": "observations: "},
        }],
        updatemenus=[{
            "type": "buttons",
            "buttons": [
                {"label": "Play", "method": "animate", "args": [None, {"frame": {"duration": 500, "redraw": True}, "fromcurrent": True}]},
                {"label": "Pause", "method": "animate", "args": [[None], {"mode": "immediate", "frame": {"duration": 0, "redraw": False}}]},
            ],
        }],
        template="plotly_white",
    )
    return figure


def plot_cone_3d(
    problem: ForwardProblem,
    snapshot: GeometrySnapshot,
    *,
    true_theta: np.ndarray | None = None,
    samples: int = 20_000,
    seed: int = 0,
):
    if problem.parameter_space.dimension != 3:
        raise ValueError("plot_cone_3d requires a three-dimensional parameter")
    go = _go()
    feasible = sample_feasible_region(
        problem.parameter_space, snapshot.constraints, count=samples, seed=seed, boundary=True
    )
    figure = go.Figure()
    if feasible.shape[0] >= 4:
        try:
            from scipy.spatial import ConvexHull

            hull = ConvexHull(feasible)
            figure.add_trace(
                go.Mesh3d(
                    x=feasible[:, 0], y=feasible[:, 1], z=feasible[:, 2],
                    i=hull.simplices[:, 0], j=hull.simplices[:, 1], k=hull.simplices[:, 2],
                    color=COLORS["feasible"], opacity=0.3, name="sampled feasible region"
                )
            )
        except Exception:
            figure.add_trace(go.Scatter3d(x=feasible[:, 0], y=feasible[:, 1], z=feasible[:, 2], mode="markers", marker={"size": 2, "color": COLORS["feasible"]}, name="feasible directions"))
    if snapshot.theta is not None:
        figure.add_trace(go.Scatter3d(x=[snapshot.theta[0]], y=[snapshot.theta[1]], z=[snapshot.theta[2]], mode="markers", marker={"size": 7, "color": COLORS["estimate"]}, name="estimate"))
    if snapshot.incenter is not None:
        figure.add_trace(go.Scatter3d(x=[snapshot.incenter[0]], y=[snapshot.incenter[1]], z=[snapshot.incenter[2]], mode="markers", marker={"size": 8, "color": COLORS["incenter"], "symbol": "diamond"}, name="incenter"))
    if true_theta is not None:
        figure.add_trace(go.Scatter3d(x=[true_theta[0]], y=[true_theta[1]], z=[true_theta[2]], mode="markers", marker={"size": 8, "color": COLORS["truth"], "symbol": "x"}, name="true theta"))
    figure.update_layout(
        title=f"Sampled consistency region after {snapshot.step} observations",
        scene={"xaxis_title": "theta_1", "yaxis_title": "theta_2", "zaxis_title": "theta_3", "aspectmode": "cube"},
        template="plotly_white",
    )
    return figure


def plot_parameter_history(result: ExperimentResult):
    plt = _plt()
    parameters = result.history.parameters
    if not parameters.size:
        raise ValueError("The result has no parameter history")
    steps = [record.step for record in result.history.steps]
    figure, ax = plt.subplots(figsize=(8, 4.5))
    for index in range(parameters.shape[1]):
        ax.plot(steps, parameters[:, index], label=fr"$\theta_{index + 1}$")
    ax.set(xlabel="step", ylabel="parameter value", title="Parameter evolution")
    ax.grid(alpha=0.2)
    ax.legend(ncol=min(4, parameters.shape[1]))
    return figure


def plot_training_loss(result: ExperimentResult):
    plt = _plt()
    records = [record for record in result.history.steps if record.loss is not None]
    if not records:
        raise ValueError("The result has no recorded training losses")
    figure, ax = plt.subplots(figsize=(8, 4.5))
    ax.plot([record.step for record in records], [record.loss for record in records], color=COLORS["loss"])
    ax.set(xlabel="training step", ylabel="loss", title="Inverse-optimization training loss")
    ax.grid(alpha=0.2)
    return figure


def plot_regret(result: ExperimentResult):
    plt = _plt()
    values = [row.get("true_regret") for row in result.per_observation]
    if not any(value is not None for value in values):
        values = [row["surrogate_suboptimality"] for row in result.per_observation]
        label = "surrogate suboptimality"
    else:
        values = [0.0 if value is None else value for value in values]
        label = "true regret"
    values_array = np.asarray(values, dtype=float)
    figure, axes = plt.subplots(1, 2, figsize=(11, 4))
    axes[0].plot(np.arange(1, values_array.size + 1), values_array, label=label)
    axes[0].set(xlabel="observation", ylabel=label, title="Instantaneous performance")
    axes[1].plot(np.arange(1, values_array.size + 1), np.cumsum(values_array), color=COLORS["incenter"])
    axes[1].set(xlabel="observation", ylabel=f"cumulative {label}", title="Cumulative performance")
    for ax in axes:
        ax.grid(alpha=0.2)
    figure.tight_layout()
    return figure


def plot_geometry_history(result: ExperimentResult):
    plt = _plt()
    if not result.geometry_history:
        raise ValueError("The result has no geometry history")
    steps = [snapshot.step for snapshot in result.geometry_history]
    fraction = [snapshot.statistics["feasible_direction_fraction"] for snapshot in result.geometry_history]
    constraints = [snapshot.statistics["constraint_count"] for snapshot in result.geometry_history]
    radii = [snapshot.inradius for snapshot in result.geometry_history]
    figure, axes = plt.subplots(1, 3, figsize=(14, 4))
    axes[0].plot(steps, fraction)
    axes[0].set(title="Remaining parameter directions", ylabel="sampled fraction")
    axes[1].plot(steps, constraints)
    axes[1].set(title="Constraint growth", ylabel="unique constraints")
    if any(radius is not None for radius in radii):
        axes[2].plot(steps, [np.nan if radius is None else radius for radius in radii])
    axes[2].set(title="Inradius", ylabel="radius")
    for ax in axes:
        ax.set_xlabel("observations")
        ax.grid(alpha=0.2)
    figure.tight_layout()
    return figure


def plot_loss_landscape_2d(
    problem: ForwardProblem,
    dataset: InverseDataset,
    loss: InverseLoss,
    *,
    resolution: int = 41,
    result: ExperimentResult | None = None,
    surface: bool = False,
):
    if problem.parameter_space.dimension != 2:
        raise ValueError("A 2D loss landscape requires a two-dimensional parameter")
    go = _go()
    space = problem.parameter_space
    if space.kind == "box":
        x_values = np.linspace(space.lower[0], space.upper[0], resolution)
        y_values = np.linspace(space.lower[1], space.upper[1], resolution)
    else:
        x_values = np.linspace(-space.radius, space.radius, resolution)
        y_values = np.linspace(-space.radius, space.radius, resolution)
    z = np.full((resolution, resolution), np.nan)
    for row, y in enumerate(y_values):
        for column, x in enumerate(x_values):
            theta = np.asarray([x, y])
            if not space.contains(theta, tolerance=1e-7):
                continue
            values, _, _ = evaluate_losses(loss, problem, theta, dataset.observations)
            z[row, column] = float(np.average(values, weights=[obs.weight for obs in dataset]))
    if surface:
        figure = go.Figure(go.Surface(x=x_values, y=y_values, z=z, colorscale="Viridis"))
        figure.update_layout(scene={"xaxis_title": "theta_1", "yaxis_title": "theta_2", "zaxis_title": "loss"})
    else:
        figure = go.Figure(go.Contour(x=x_values, y=y_values, z=z, colorscale="Viridis", contours={"showlabels": True}))
        figure.update_layout(xaxis_title="theta_1", yaxis_title="theta_2")
    if result is not None:
        if surface:
            values, _, _ = evaluate_losses(loss, problem, result.theta, dataset.observations)
            figure.add_trace(go.Scatter3d(x=[result.theta[0]], y=[result.theta[1]], z=[float(np.mean(values))], mode="markers", marker={"size": 7, "color": COLORS["estimate"]}, name="estimate"))
        else:
            figure.add_trace(go.Scatter(x=[result.theta[0]], y=[result.theta[1]], mode="markers", marker={"size": 12, "color": COLORS["estimate"]}, name="estimate"))
    figure.update_layout(title=f"{loss.name.replace('_', ' ').title()} landscape", template="plotly_white")
    return figure


def plot_run_comparison(results: Iterable[ExperimentResult], metric: str):
    plt = _plt()
    results = list(results)
    labels = [result.name for result in results]
    values = [result.metrics.get(metric, np.nan) for result in results]
    figure, ax = plt.subplots(figsize=(max(7, len(results) * 0.7), 4.5))
    ax.bar(labels, values, color=COLORS["feasible"])
    ax.set(ylabel=metric.replace("_", " "), title=f"Run comparison: {metric.replace('_', ' ')}")
    ax.tick_params(axis="x", rotation=35)
    ax.grid(axis="y", alpha=0.2)
    figure.tight_layout()
    return figure


def save_figure(figure: Any, path: str, **kwargs: Any) -> str:
    if hasattr(figure, "write_html") and str(path).lower().endswith(".html"):
        figure.write_html(path, include_plotlyjs="cdn")
    elif hasattr(figure, "write_image") and str(path).lower().endswith((".png", ".pdf", ".svg")):
        figure.write_image(path, **kwargs)
    elif hasattr(figure, "savefig"):
        figure.savefig(path, bbox_inches="tight", dpi=160, **kwargs)
    else:
        raise TypeError("Unsupported figure type")
    return str(path)
