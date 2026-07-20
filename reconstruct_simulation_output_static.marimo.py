import marimo

__generated_with = "0.23.11"
app = marimo.App()


@app.cell
def _():
    import numpy as np
    import pypulseq as pp
    import plotly.express as px
    import mrinufft
    import matplotlib.pyplot as plt
    import ipyvolume as ipv
    import cupy as cp

    from scipy.fft import fftshift, ifftshift, fftn, ifftn
    from pathlib import Path
    from tqdm import trange

    BACKEND = 'gpuNUFFT'

    plots_dir = Path('plots')
    plots_dir.mkdir(exist_ok=True)
    return BACKEND, Path, cp, ipv, mrinufft, np, plots_dir, plt, pp


@app.cell
def _(Path, np):
    output_dir = Path('...')
    coil_signals = np.load(output_dir / 'simulation_outputs' / '...').T
    coil_signals = np.squeeze(coil_signals)

    print(coil_signals)
    print(coil_signals.shape)
    return coil_signals, output_dir


@app.cell
def _(output_dir, pp):
    sequence = pp.Sequence()
    sequence.read(output_dir / 'sequences' / '...')
    print(sequence)
    return (sequence,)


@app.cell
def _(sequence):
    sequence_kspace_calc = sequence.calculate_kspace()
    k_traj_adc = sequence_kspace_calc[0] # Kspace trajectory during sampling
    kspace_times = sequence_kspace_calc[4] # Times at which samples are taken
    samples_per_adc = sequence.get_block(10_003).adc.num_samples
    num_spokes = k_traj_adc.shape[1] // samples_per_adc
    print('Samples per ADC:', samples_per_adc)
    print('Number of spokes:', num_spokes)
    return (k_traj_adc,)


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
def _(
    BACKEND,
    coil_signals,
    dcf,
    mrinufft,
    normed_ktraj,
    output_shape,
    plots_dir,
    plt,
):
    nufft = mrinufft.get_operator(BACKEND)(normed_ktraj.T, shape=output_shape, density=dcf, squeeze_dims=True)
    adjoint_manual = nufft.adj_op(coil_signals)
    _fig, _axs = plt.subplots(1, 3, figsize=(15, 5))
    _axs[0].imshow(abs(adjoint_manual)[adjoint_manual.shape[0] // 2, ::-1, :][24:-24, 20:-20], cmap='viridis')
    _axs[1].imshow(abs(adjoint_manual)[:, adjoint_manual.shape[1] // 2, ::-1][18:-18, 20:-20], cmap='viridis')
    _axs[2].imshow(abs(adjoint_manual)[:, ::-1, adjoint_manual.shape[2] // 2].T[24:-24, 18:-18], cmap='viridis')

    for _ax in _axs:
        _ax.axis('off')

    _fig.savefig(plots_dir / 'mrzero_static_with_t2s.svg', transparent=True)

    #_fig.suptitle('NuFFT static MRI reconstruction')
    plt.show()
    return adjoint_manual, nufft


@app.cell(disabled=True)
def _(adjoint_manual, ipv):
    ipv.quickvolshow(abs(adjoint_manual), level=0.1)
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
    _axs[0].imshow(abs(adjoint_manual_subs)[adjoint_manual_subs.shape[0] // 2, ::-1, :][24:-24, 20:-20], cmap='viridis')
    _axs[1].imshow(abs(adjoint_manual_subs)[:, adjoint_manual_subs.shape[1] // 2, ::-1][18:-18, 20:-20], cmap='viridis')
    _axs[2].imshow(abs(adjoint_manual_subs)[:, ::-1, adjoint_manual_subs.shape[2] // 2].T[24:-24, 18:-18], cmap='viridis')
    #_fig.suptitle('NuFFT static MRI reconstruction')
    _fig.suptitle(f'NuFFT static MRI reconstruction (cropped kspace = {int(subset_pct * 100)}%)')
    plt.show()
    return


@app.cell(disabled=True)
def _(coil_signals, cp, nufft, plt):
    pinv_recon = nufft.pinv_solver(cp.asarray(coil_signals), max_iter=500, optim='lsqr').get()
    _fig, _axs = plt.subplots(1, 3, figsize=(15, 5))
    _axs[0].imshow(abs(pinv_recon)[pinv_recon.shape[0] // 2, :, :], cmap='viridis')
    _axs[1].imshow(abs(pinv_recon)[:, pinv_recon.shape[1] // 2, :], cmap='viridis')
    _axs[2].imshow(abs(pinv_recon)[:, :, pinv_recon.shape[2] // 2], cmap='viridis')
    _fig.suptitle(f'NuFFT static MRI reconstruction (pinv method)')
    plt.show()
    return


if __name__ == "__main__":
    app.run()
