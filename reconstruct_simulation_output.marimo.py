import marimo

__generated_with = "0.23.11"
app = marimo.App(width="medium")


@app.cell
def _():
    import h5py
    import numpy as np
    import cupy as cp
    import pypulseq as pp
    import mrinufft
    import matplotlib.pyplot as plt
    import ipyvolume as ipv

    from scipy.fft import fft, fftfreq, fftshift
    from pathlib import Path
    from tqdm.auto import trange, tqdm

    from typing import Tuple, Union
    from numpy.typing import NDArray

    # BACKEND = 'cufinufft'
    BACKEND = 'gpuNUFFT'

    plots_dir = Path('plots')
    plots_dir.mkdir(exist_ok=True)
    return (
        BACKEND,
        NDArray,
        Path,
        Union,
        cp,
        fft,
        fftfreq,
        mrinufft,
        np,
        plots_dir,
        plt,
        pp,
        trange,
    )


@app.cell
def _(Path, np):
    output_dir = Path('...').resolve()
    coil_signals = np.load(output_dir / 'simulation_outputs' / '....npy').T
    coil_signals = np.squeeze(coil_signals)
    coil_signals = np.conj(coil_signals)
    print(coil_signals)
    print(coil_signals.shape)
    return coil_signals, output_dir


@app.cell
def _(output_dir, pp):
    sequence = pp.Sequence()
    sequence.read(output_dir.parent / 'simulator-data' / 'sequences' / '....seq')
    print(sequence)
    return (sequence,)


@app.cell
def _(sequence):
    sequence_kspace_calc = sequence.calculate_kspace()
    k_traj_adc = sequence_kspace_calc[0] # Kspace trajectory during sampling
    kspace_times = sequence_kspace_calc[4] # Times at which samples are taken
    samples_per_adc = sequence.get_block(10_000 + 3).adc.num_samples
    num_spokes = k_traj_adc.shape[1] // samples_per_adc
    print('Samples per ADC:', samples_per_adc)
    print('Number of spokes:', num_spokes)
    return k_traj_adc, kspace_times, num_spokes, samples_per_adc


@app.cell
def _(coil_signals, k_traj_adc, np):
    print(k_traj_adc.shape, coil_signals.shape)

    output_shape = (np.array((120, 180, 180))).astype(int)
    normed_ktraj = k_traj_adc / (2*np.max(np.abs(k_traj_adc))) # normalize to [-0.5, 0.5]
    flat_traj = normed_ktraj.T
    weights = np.sqrt(np.sum(np.abs(normed_ktraj**2), axis=0))

    recon_regularisation = 0.1
    max_recon_iters = 500
    recon_solver = 'lsqr'
    return flat_traj, normed_ktraj, output_shape, weights


@app.cell
def _(BACKEND, flat_traj, mrinufft, output_shape):
    dcf = mrinufft.density.nufft_based.pipe(flat_traj, shape=output_shape, backend=BACKEND, osf=2)
    return (dcf,)


@app.cell
def _(BACKEND, coil_signals, dcf, mrinufft, normed_ktraj, output_shape, plt):
    nufft = mrinufft.get_operator(BACKEND)(normed_ktraj.T, shape=output_shape, density=dcf, squeeze_dims=True, upsampfac=2.0)
    adjoint_manual = nufft.adj_op(coil_signals)
    _fig, _axs = plt.subplots(1, 3, figsize=(15, 5))
    _axs[0].imshow(abs(adjoint_manual)[adjoint_manual.shape[0] // 2, :, :], cmap='viridis')
    _axs[1].imshow(abs(adjoint_manual)[:, adjoint_manual.shape[1] // 2, :], cmap='viridis')
    _axs[2].imshow(abs(adjoint_manual)[:, :, adjoint_manual.shape[2] // 2], cmap='viridis')
    _fig.suptitle('NuFFT dynamic MRI reconstruction (no phase separation)')
    plt.show()
    return


@app.cell
def _(
    BACKEND,
    coil_signals,
    dcf,
    mrinufft,
    normed_ktraj,
    np,
    output_shape,
    plt,
    weights,
):
    subset_pct = 0.65
    subset_selector = weights < np.max(weights) * subset_pct
    nufft_subs = mrinufft.get_operator(BACKEND)(normed_ktraj.T[subset_selector], shape=output_shape, density=dcf[subset_selector], squeeze_dims=True, upsampfac=2.0)
    adjoint_manual_subs = nufft_subs.adj_op(coil_signals[subset_selector])
    _fig, _axs = plt.subplots(1, 3, figsize=(15, 5))
    _axs[0].imshow(abs(adjoint_manual_subs)[adjoint_manual_subs.shape[0] // 2, :, :], cmap='viridis')
    _axs[1].imshow(abs(adjoint_manual_subs)[:, adjoint_manual_subs.shape[1] // 2, :], cmap='viridis')
    _axs[2].imshow(abs(adjoint_manual_subs)[:, :, int(adjoint_manual_subs.shape[2] * 0.75)], cmap='viridis')
    _fig.suptitle(f'NuFFT dynamic MRI reconstruction (no phase separation, cropped kspace = {int(subset_pct * 100)}%)')
    plt.show()
    return


@app.cell
def _(kspace_times):
    breath_time, breath_phases = 5.0, 32
    breath_idx_per_sample = (kspace_times / breath_time).astype(int)
    phase_per_sample = ((kspace_times % breath_time) * (breath_phases / breath_time)).astype(int)
    return breath_phases, breath_time, phase_per_sample


@app.cell
def _(
    BACKEND,
    breath_phases,
    coil_signals,
    dcf,
    flat_traj,
    mrinufft,
    normed_ktraj,
    np,
    output_shape,
    phase_per_sample,
    plt,
    samples_per_adc,
    trange,
):
    phase_resolution = 2
    ncols = 4
    viz_phases = np.array([0, 8, 16, 24])
    _fig, _axs = plt.subplots(nrows=2, ncols=len(viz_phases))
    rendered_frames = np.zeros((breath_phases // phase_resolution, *output_shape), dtype=np.complex64)
    rendered_frames_optim = np.zeros((breath_phases // phase_resolution, *output_shape), dtype=np.complex64)
    nspokes = []
    for plot_idx, _phase in enumerate(trange(0, breath_phases, phase_resolution, position=0, leave=True)):
        this_phase_traj = normed_ktraj[:flat_traj.shape[0], (phase_per_sample >= _phase) & (phase_per_sample < _phase + phase_resolution)]
        this_weights = dcf[(phase_per_sample >= _phase) & (phase_per_sample < _phase + phase_resolution)]
        this_signals = coil_signals[:flat_traj.shape[0]][(phase_per_sample >= _phase) & (phase_per_sample < _phase + phase_resolution)]
        nspokes.append(this_signals.shape[0] / samples_per_adc)
        this_nufft = mrinufft.get_operator(BACKEND)(this_phase_traj.T, shape=output_shape, density=this_weights, squeeze_dims=True, upsampfac=2.0)
        this_recon = this_nufft.adj_op(this_signals)
        rendered_frames[_phase // phase_resolution] = this_recon
        if _phase in viz_phases:
            ax = _axs[:, *np.argwhere(viz_phases == _phase)[0]].T
            ax[0].imshow(np.abs(this_recon)[this_recon.shape[0] // 2, :, ::-1][20:-20, 20:-20], cmap='gray')
            ax[0].axis('off')
            # ax[1].imshow(np.abs(this_recon)[::-1, this_recon.shape[1] // 2, :][18:-18, 20:-20], cmap='gray')
            # ax[1].axis('off')
            ax[1].imshow(np.abs(this_recon)[::-1, :, int(this_recon.shape[2] * 0.75)].T[20:-20, 18:-18], cmap='gray')
            ax[1].axis('off')

            ax[0].axhline(92, c='red', linestyle='--')
            ax[1].axhline(90, c='red', linestyle='--')
    plt.tight_layout()
    plt.show()
    return phase_resolution, rendered_frames


@app.cell
def _(breath_phases, breath_time, np, phase_resolution, rendered_frames):
    from imageio.v3 import imwrite
    from IPython.display import display, Image

    outfile_anim = '/tmp/phase_animation.gif'
    imwrite(
        outfile_anim,
        255 * np.abs(rendered_frames)[:, rendered_frames.shape[1] // 2, :, :] / np.max(np.abs(rendered_frames)),
        duration=1000*(breath_time / breath_phases * phase_resolution),
        extension='.gif',
        loop=0
    )
    display(Image(outfile_anim))
    return Image, display, imwrite


@app.cell
def _(np, plt, rendered_frames):
    _fig, _axs = plt.subplots(1, 2, figsize=(12, 5))
    scaled_rendered_frames = 255 * (np.abs(rendered_frames) / np.max(np.abs(rendered_frames)))
    hist, bins = np.histogram(scaled_rendered_frames.flatten(), bins=256, range=(0, 255))
    _axs[0].bar(bins[:-1], hist, width=1)
    _axs[0].set_title('Histogram of pixel intensities in rendered frames')
    _axs[0].set_xlabel('Pixel intensity')
    _axs[0].set_ylabel('Frequency')
    cumsum = np.cumsum(hist)
    total = cumsum[-1]
    # Find where cumsum is > 99.9% of total, then clip to that value and replot histogram
    clip_value = bins[np.searchsorted(cumsum, total * 0.999)]
    print('Clipping pixel intensities at:', clip_value)
    scaled_rendered_frames_clipped = np.clip(scaled_rendered_frames, 0, clip_value)
    hist_clipped, bins_clipped = np.histogram(scaled_rendered_frames_clipped.flatten(), bins=256, range=(0, clip_value))
    _axs[1].bar(bins_clipped[:-1], hist_clipped, width=clip_value / 256)
    _axs[1].set_title('Histogram of pixel intensities in rendered frames (clipped at 99.9% cumulative frequency)')
    _axs[1].set_xlabel('Pixel intensity')
    _axs[1].set_ylabel('Frequency')
    _fig.tight_layout()
    plt.show()
    return (clip_value,)


@app.cell
def _(
    Image,
    breath_phases,
    breath_time,
    clip_value,
    display,
    imwrite,
    np,
    phase_resolution,
    plt,
    rendered_frames,
):
    outfile_anim_full = '/tmp/phase_animation_full.gif'
    animation_frames = []
    for _phase, time_sample in enumerate(255 * (np.abs(rendered_frames) / np.max(np.abs(rendered_frames)))):
        time_sample = np.clip(time_sample, 0, clip_value)
        time_fig, time_axs = plt.subplots(1, 3, figsize=(15, 5))
        time_axs[0].imshow(abs(time_sample)[time_sample.shape[0] // 2, :, :], cmap='gray', vmax=clip_value, vmin=0)
        time_axs[1].imshow(abs(time_sample)[:, time_sample.shape[1] // 2, :], cmap='gray', vmax=clip_value, vmin=0)
        time_axs[2].imshow(abs(time_sample)[:, :, time_sample.shape[2] // 2], cmap='gray', vmax=clip_value, vmin=0)
        time_fig.suptitle(f'NuFFT dynamic MRI reconstruction (phase {_phase * phase_resolution} of {breath_phases})')
        time_fig.canvas.draw()
        image = np.frombuffer(time_fig.canvas.tostring_argb(), dtype='uint8')
        image = image.reshape(time_fig.canvas.get_width_height()[::-1] + (4,))
        image = image[..., 1:]
        animation_frames.append(image)
        plt.close(time_fig)
    imwrite(outfile_anim_full, animation_frames, duration=1000 * (breath_time / breath_phases * phase_resolution), extension='.gif', loop=0)
    display(Image(outfile_anim_full))
    return


if __name__ == "__main__":
    app.run()
