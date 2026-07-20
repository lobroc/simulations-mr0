import MRzeroCore as mr0
import plotly.express as px
import plotly.graph_objects as go
import numpy as np
import torch

from sys import argv
from typing import Union
from types import MethodType
from sys import path
from pathlib import Path

path.append(str(Path.cwd()))

from create_mr0_phantom import MovingVoxelGridPhantom
from src.utils import subsample_compiled_phantom, simdata_to_cuda

def plot_phantom_map(phantom: MovingVoxelGridPhantom, property: str, max_spins: int = 20_000, time_samples: Union[int, None] = None):
    """
    Plot a 3D point cloud from an object, with optional time-stepped deformations.

    Parameters
    ----------
    obj : object
        Must expose the following attributes:
            - xpos : array-like, shape (N,)  — base X coordinates
            - ypos : array-like, shape (N,)  — base Y coordinates
            - zpos : array-like, shape (N,)  — base Z coordinates
            - property : string, shape (N,)  — scalar field used for colouring
    deformations : np.ndarray or None, shape (T, N, 3)
        Displacement vectors per time step. At step t, new coords are:
            x = xpos + deformations[t, :, 0]
            y = ypos + deformations[t, :, 1]
            z = zpos + deformations[t, :, 2]
        If None, a static plot is returned with no slider.

    Returns
    -------
    fig : plotly.graph_objects.Figure
    """
    x0 = phantom.voxel_pos[:, 0].cpu().numpy()
    y0 = phantom.voxel_pos[:, 1].cpu().numpy()
    z0 = phantom.voxel_pos[:, 2].cpu().numpy()
    color = getattr(phantom, property).cpu().numpy()

    # ── colour scale bounds shared across all frames ──────────────────────────
    cmin, cmax = color.min(), color.max()

    marker_base = dict(
        color=color,
        colorscale="Viridis",
        cmin=cmin.item(),
        cmax=cmax.item(),
        colorbar=dict(title=property),
        size=0.5,
        opacity=0.85,
    )

    # ── static plot (no deformations) ─────────────────────────────────────────
    if phantom.voxel_motion is None:
        fig = go.Figure(
            data=[go.Scatter3d(x=x0, y=y0, z=z0, mode="markers", marker=marker_base)]
        )
        fig.update_layout(
            scene=dict(xaxis_title="X", yaxis_title="Y", zaxis_title="Z"),
            margin=dict(l=0, r=0, b=0, t=30),
        )
        return fig

    # ── animated plot with slider ──────────────────────────────────────────────
    frames = []
    T = time_samples if time_samples is not None else 0
    computed_motions = phantom.voxel_motion(torch.linspace(phantom.time_curve.time_scale_min, phantom.time_curve.time_scale_max, T).cuda()).cpu().numpy()
    for t in range(T):
        xt = x0 + computed_motions[t, :, 0]
        yt = y0 + computed_motions[t, :, 1]
        zt = z0 + computed_motions[t, :, 2]
        frames.append(
            go.Frame(
                data=[
                    go.Scatter3d(
                        x=xt, y=yt, z=zt,
                        mode="markers",
                        marker=marker_base,
                    )
                ],
                name=str(t),
            )
        )

    # Initial frame = t=0
    x_init = x0 + computed_motions[0, :, 0]
    y_init = y0 + computed_motions[0, :, 1]
    z_init = z0 + computed_motions[0, :, 2]

    fig = go.Figure(
        data=[
            go.Scatter3d(
                x=x_init, y=y_init, z=z_init,
                mode="markers",
                marker=marker_base,
            )
        ],
        frames=frames,
    )

    # Slider steps
    slider_steps = [
        dict(
            args=[
                [str(t)],
                dict(
                    frame=dict(duration=0, redraw=True),
                    mode="immediate",
                    transition=dict(duration=0),
                ),
            ],
            label=str(t),
            method="animate",
        )
        for t in range(T)
    ]

    fig.update_layout(
        scene=dict(xaxis_title="X", yaxis_title="Y", zaxis_title="Z"),
        margin=dict(l=0, r=0, b=0, t=30),
        updatemenus=[
            dict(
                type="buttons",
                showactive=False,
                y=0,
                x=0.05,
                xanchor="right",
                yanchor="top",
                buttons=[
                    dict(
                        label="▶  Play",
                        method="animate",
                        args=[
                            None,
                            dict(
                                frame=dict(duration=(phantom.time_curve.time_scale_max - phantom.time_curve.time_scale_min).item() / time_samples, redraw=True),
                                fromcurrent=True,
                                transition=dict(duration=0),
                            ),
                        ],
                    ),
                    dict(
                        label="⏸  Pause",
                        method="animate",
                        args=[
                            [None],
                            dict(
                                frame=dict(duration=0, redraw=False),
                                mode="immediate",
                                transition=dict(duration=0),
                            ),
                        ],
                    ),
                ],
            )
        ],
        sliders=[
            dict(
                active=0,
                currentvalue=dict(prefix="Time step: ", visible=True, xanchor="center"),
                pad=dict(t=40),
                steps=slider_steps,
            )
        ],
    )

    return fig

def plot_phantom_map_2d(phantom: MovingVoxelGridPhantom, property: str, max_spins: int = 20_000, time_samples: Union[int, None] = None):
    """
    Plot a 2D point cloud from an object, with optional time-stepped deformations.

    Parameters
    ----------
    obj : object
        Must expose the following attributes:
            - xpos : array-like, shape (N,)  — base X coordinates
            - ypos : array-like, shape (N,)  — base Y coordinates
            - property : string, shape (N,)  — scalar field used for colouring
    deformations : np.ndarray or None, shape (T, N, 2)
        Displacement vectors per time step. At step t, new coords are:
            x = xpos + deformations[t, :, 0]
            y = ypos + deformations[t, :, 1]
        If None, a static plot is returned with no slider.

    Returns
    -------
    fig : plotly.graph_objects.Figure
    """
    x0 = phantom.voxel_pos[:, 0].cpu().numpy()
    y0 = phantom.voxel_pos[:, 1].cpu().numpy()
    color = getattr(phantom, property).cpu().numpy()

    # ── colour scale bounds shared across all frames ──────────────────────────
    cmin, cmax = color.min(), color.max()

    marker_base = dict(
        color=color,
        colorscale="Viridis",
        cmin=cmin.item(),
        cmax=cmax.item(),
        colorbar=dict(title=property),
        size=4,
        opacity=0.85,
    )

    # ── static plot (no deformations) ─────────────────────────────────────────
    if phantom.voxel_motion is None:
        fig = go.Figure(
            data=[go.Scatter(x=x0, y=y0, mode="markers", marker=marker_base)]
        )
        fig.update_layout(
            xaxis_title="X",
            yaxis_title="Y",
            yaxis=dict(scaleanchor="x", scaleratio=1),
            margin=dict(l=0, r=0, b=0, t=30),
        )
        return fig

    # ── animated plot with slider ──────────────────────────────────────────────
    frames = []
    T = time_samples if time_samples is not None else 0
    computed_motions = phantom.voxel_motion(torch.linspace(phantom.time_curve.time_scale_min, phantom.time_curve.time_scale_max, T).cuda()).cpu().numpy()
    for t in range(T):
        xt = x0 + computed_motions[t, :, 0]
        yt = y0 + computed_motions[t, :, 1]
        frames.append(
            go.Frame(
                data=[
                    go.Scatter(
                        x=xt, y=yt,
                        mode="markers",
                        marker=marker_base,
                    )
                ],
                name=str(t),
            )
        )

    # Initial frame = t=0
    x_init = x0 + computed_motions[0, :, 0]
    y_init = y0 + computed_motions[0, :, 1]

    fig = go.Figure(
        data=[
            go.Scatter(
                x=x_init, y=y_init,
                mode="markers",
                marker=marker_base,
            )
        ],
        frames=frames,
    )

    # Slider steps
    slider_steps = [
        dict(
            args=[
                [str(t)],
                dict(
                    frame=dict(duration=0, redraw=True),
                    mode="immediate",
                    transition=dict(duration=0),
                ),
            ],
            label=str(t),
            method="animate",
        )
        for t in range(T)
    ]

    fig.update_layout(
        xaxis_title="X",
        yaxis_title="Y",
        yaxis=dict(scaleanchor="x", scaleratio=1),
        margin=dict(l=0, r=0, b=0, t=30),
        updatemenus=[
            dict(
                type="buttons",
                showactive=False,
                y=0,
                x=0.05,
                xanchor="right",
                yanchor="top",
                buttons=[
                    dict(
                        label="▶  Play",
                        method="animate",
                        args=[
                            None,
                            dict(
                                frame=dict(duration=(phantom.time_curve.time_scale_max - phantom.time_curve.time_scale_min).item() / time_samples, redraw=True),
                                fromcurrent=True,
                                transition=dict(duration=0),
                            ),
                        ],
                    ),
                    dict(
                        label="⏸  Pause",
                        method="animate",
                        args=[
                            [None],
                            dict(
                                frame=dict(duration=0, redraw=False),
                                mode="immediate",
                                transition=dict(duration=0),
                            ),
                        ],
                    ),
                ],
            )
        ],
        sliders=[
            dict(
                active=0,
                currentvalue=dict(prefix="Time step: ", visible=True, xanchor="center"),
                pad=dict(t=40),
                steps=slider_steps,
            )
        ],
    )

    return fig

def plot_motion_interpolation_curve(phantom: MovingVoxelGridPhantom, time_samples: int = 1000):
    time_points = torch.linspace(phantom.time_curve.time_scale_min, 2*phantom.time_curve.time_scale_max, time_samples).cuda()
    chunk_size = 10

    motions_global = []
    for i in range(0, time_points.shape[0], chunk_size):
        t_chunk = time_points[i:i+chunk_size]
        motions_chunk = phantom.voxel_motion(t_chunk)
        motion_global_chunk = motions_chunk.mean(dim=1)  # (chunk_size, 3)
        motions_global.append(motion_global_chunk)

    motions_global = torch.cat(motions_global, dim=0)  # (time_samples, 3)

    fig = px.line(
        x=time_points.cpu().numpy(),
        y=motions_global.norm(dim=1).cpu().numpy(),
        labels={'x': 'Time', 'y': 'Mean Voxel Displacement (L2 norm)'},
        title='Global Motion Interpolation Curve',
    )
    fig.update_layout(margin=dict(l=0, r=0, b=0, t=30))
    return fig

ph_file = argv[1]
num_spins = int(argv[2]) if len(argv) > 2 else 5000

phantom = MovingVoxelGridPhantom.load(ph_file, cubic_deformations=False, closed_loop_deforms=True)
phantom_compiled = phantom.build(voxel_shape='sinc', flipped_positions=True)

ph_comp_subs = subsample_compiled_phantom(phantom_compiled, subsample_factor=len(phantom_compiled.T1) // num_spins)
ph_comp_subs.voxel_motion = MethodType(phantom.voxel_motion.__func__, ph_comp_subs)
ph_comp_subs_cuda = simdata_to_cuda(ph_comp_subs)

plot_phantom_map(
    phantom=ph_comp_subs_cuda,
    property='T1',
    max_spins=num_spins,
    time_samples=32,
).show()

plot_phantom_map(
    phantom=ph_comp_subs_cuda,
    property='T2',
    max_spins=num_spins,
    time_samples=32,
).show()

plot_phantom_map(
    phantom=ph_comp_subs_cuda,
    property='PD',
    max_spins=num_spins,
    time_samples=32,
).show()

plot_motion_interpolation_curve(
    phantom=ph_comp_subs_cuda,
    time_samples=1000
).show()
