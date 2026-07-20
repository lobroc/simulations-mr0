import numpy as np
import pypulseq as pp
import json5

from mrinufft.io.pulseq import acq2opts
from mrinufft.trajectories.utils import Acquisition, Hardware
from pathlib import Path
from argparse import ArgumentParser
from tqdm.auto import tqdm
from decimal import Decimal, ROUND_HALF_UP

from aztek.aztek_interface import AZTEK_trajectory_comp

def parse_args():
    parser = ArgumentParser()
    parser.add_argument('--output-dir', '-o', type=Path, required=True, help='Output directory for the generated sequence.')
    parser.add_argument('--spatial-dims', '-d', nargs='+', type=int, default=[256, 256, 256], help='Spatial dimensions of the image in pixels (e.g. --spatial-dims 256 256 256).')
    parser.add_argument('--runtime', '-t', type=float, default=10.0, help='Total runtime of the sequence in seconds.')
    parser.add_argument('--dummy-time', type=float, default=0.0, help='Additional dummy time to reach steady state.')
    parser.add_argument('--whole-spoke', '-w', action='store_true', help='Make the spoke be fully sampled, and not stop before downwards ramp.')
    parser.add_argument('--fully-sample', '-f', action='store_true', help='Ignore runtime arg, and fully sample space.')
    args = parser.parse_args()
    return args

def compute_scanner_params(spatial_dims=[256, 256, 256]):
    with open('scanner_properties.jsonc', 'r') as f:
        properties = json5.load(f)

    # Sequence parameters
    TE = 12e-6  # s
    TR = 2.0e-3  # s
    FA = 5.0  # degrees
    PULSE_DURATION = 16e-6 # s
    SPATIAL_DIMS_PX = spatial_dims

    acq = Acquisition(
        fov=properties['fov'],  # m
        img_size=SPATIAL_DIMS_PX,
        hardware=Hardware(
            n_coils=1, # KomaMRI
            field_strength=properties['B0'],  # T
            gmax=properties['max_grad'], # T/m
            smax=properties['max_slew'], # T/m/s
            grad_raster_time=properties['grad_raster_time'], # s
        ),
        adc_dwell_time=properties['adc_raster_time'], # s
        norm_factor=0.5
    )
    print(acq)
    opts = acq2opts(acq)
    opts.B0 = properties['B0']  # T
    opts.block_duration_raster=properties['block_duration_raster']
    opts.grad_raster_time=properties['grad_raster_time']
    opts.rf_raster_time=properties['rf_raster_time']
    opts.adc_raster_time=properties['adc_raster_time']
    opts.adc_samples_divisor=properties['adc_oversampling']  # oversampling factor for ADC

    return opts, acq, TE, TR, FA, PULSE_DURATION, SPATIAL_DIMS_PX

if __name__ == '__main__':
    args = parse_args()
    assert len(args.spatial_dims) == 3, 'Spatial dimensions must be a list of three integers (e.g. --spatial-dims 256 256 256).'

    if not 'sequences' in args.output_dir.parts:
        args.output_dir = args.output_dir / 'sequences'

    args.output_dir.mkdir(parents=True, exist_ok=True)

    #
    # ----------------------------------
    # Input parameters
    # ----------------------------------
    #

    opts, acq, TE, TR, FA, PULSE_DURATION, SPATIAL_DIMS_PX = compute_scanner_params(args.spatial_dims)
    print(opts)

    quantise_float = lambda x, raster_time: float(Decimal(str(x)).quantize(Decimal(str(raster_time)), rounding=ROUND_HALF_UP))
    snap_to_raster = lambda x, raster_time: int(np.round(x / raster_time)) * raster_time

    #
    # ----------------------------------
    # Find trajectory, with inferred spokes/samples
    # ----------------------------------
    #

    # Compute k-space parameters
    # Use: https://mriquestions.com/field-of-view-fov.html
    FOV = np.array(acq.fov) # in m
    img_size = np.array(acq.img_size) # in pixels
    delta_w = FOV / img_size  # pixel widths in m
    delta_k = 1 / FOV  # in 1/m
    k_fov = 1 / delta_w  # in 1/m
    k_max = k_fov / 2  # in 1/m

    # Infer number of spokes and samples
    max_fov_dir, med_fov_dir, min_fov_dir = sorted(range(3), key=lambda i: acq.fov[i], reverse=True)

    # Computing Nyquist criterion
    n = 4 # commonly used param
    nyquist_spokes = 16*np.pi*(k_max[max_fov_dir]**2) / (n*np.tan(np.pi/n)) # https://cds.ismrm.org/protected/22MProceedings/PDFfiles/2450.html
    nyquist_size_matrix = 4*np.pi*(img_size[max_fov_dir]**2)

    samples = int(np.ceil(k_max[max_fov_dir] / delta_k[max_fov_dir]))
    dummy_spokes = int(np.ceil(args.dummy_time / TR))
    spokes = int(np.ceil((args.runtime - args.dummy_time) / TR)) if not args.fully_sample else int(np.ceil(nyquist_spokes))

    print('Number of spokes for Nyquist criterion:', np.ceil(nyquist_spokes).astype(int))
    print('Number of spokes for Nyquist (alt):', np.ceil(nyquist_size_matrix).astype(int))
    print(f'We are doing {np.round(100 * spokes / nyquist_spokes, 1)} % sampling at worst case (max fov dir).')

    print('Inferred spokes:', spokes, ' and samples:', samples)

    gx, gy, gz = np.zeros(spokes, dtype=np.int16), np.zeros(spokes, dtype=np.int16), np.zeros(spokes, dtype=np.int16)
    result, iamp = AZTEK_trajectory_comp(
        totalSpoke=np.int32(spokes),
        gxS=gx, gyS=gy, gzS=gz,
        scale_factor=np.int32(1),
        AZTEK_Twist=np.float64(2),
        AZTEK_Shuffle=np.float64(2),
        AZTEK_Speed=np.int32(3)
    )
    if result != 0:
        raise RuntimeError(f"AZTEK_trajectory_comp failed with error code {result}")

    trajectory = np.stack((gx, gy, gz), axis=-1, dtype=np.float32)  # shape (Ns, 3)
    trajectory = ((trajectory / iamp) / 2)
    trajectory = trajectory * (img_size / img_size[max_fov_dir]) # Scale to ratio from input dims.

    # np.save(data_dir / '3d_golden_means_radial_trajectory.npy', trajectory)
    print('Trajectory computed.')

    #
    # ----------------------------------
    # Compute sequence
    # ----------------------------------
    #

    sequence = pp.Sequence(system=opts)

    dt = opts.adc_raster_time # Use adc time for grad alignment to make task easier
    grad_speed_amp = 1

    for i in tqdm(range(dummy_spokes + spokes), desc='Generating sequence blocks'):
        rf_pulse = pp.make_block_pulse(
            flip_angle=np.deg2rad(FA),
            duration=(PULSE_DURATION // opts.rf_raster_time) * opts.rf_raster_time,
            system=opts,
            phase_offset=np.deg2rad((117 * i)) % (2 * np.pi)
        )
        t_rf_center = pp.calc_rf_center(rf_pulse)[0]
        grad_time = quantise_float(samples*opts.grad_raster_time/grad_speed_amp, opts.grad_raster_time)

        final_resting_time = quantise_float(TR - TE - (2 * grad_time) - t_rf_center, opts.grad_raster_time)
        if final_resting_time < 0:
            grad_speed_amp += 1
            if grad_speed_amp >= (opts.grad_raster_time / dt):
                raise RuntimeError("Cannot fit sampling within gradient ramp + flat top. Consider reducing number of samples or increasing grad time.")

            grad_time = quantise_float(samples*opts.grad_raster_time/grad_speed_amp, opts.grad_raster_time)
            final_resting_time = quantise_float(TR - TE - (2 * grad_time) - t_rf_center, opts.grad_raster_time)


        sequence.add_block(rf_pulse)

        if i < dummy_spokes:
            sequence.add_block(pp.make_delay(TR - PULSE_DURATION))
            continue

        sequence.add_block(pp.make_delay(quantise_float(TE - t_rf_center, opts.adc_raster_time)))

        k = (2 * trajectory[i-dummy_spokes]) * k_max # shape (Ns, 3)
        grad_ramp_time = snap_to_raster(grad_time / 8, opts.grad_raster_time)
        grad_fall_time = grad_ramp_time
        grad_flat_time = quantise_float(grad_time - grad_ramp_time - grad_fall_time, opts.grad_raster_time)

        gx = pp.make_trapezoid(
            channel='x',
            area=k[0],
            rise_time=grad_ramp_time,
            flat_time=grad_flat_time,
            fall_time=grad_fall_time,
            system=opts
        )

        gy = pp.make_trapezoid(
            channel='y',
            area=k[1],
            rise_time=grad_ramp_time,
            flat_time=grad_flat_time,
            fall_time=grad_fall_time,
            system=opts
        )
        gz = pp.make_trapezoid(
            channel='z',
            area=k[2],
            rise_time=grad_ramp_time,
            flat_time=grad_flat_time,
            fall_time=grad_fall_time,
            system=opts
        )

        adc_duration = grad_ramp_time + grad_flat_time + (grad_fall_time if args.whole_spoke else 0)
        adc_dwell_time = snap_to_raster(adc_duration / samples, opts.adc_raster_time)
        adc = pp.make_adc(
            num_samples=adc_duration / adc_dwell_time,
            duration=adc_duration,
            delay=0,
            system=opts,
            phase_offset=np.deg2rad((117 * i)) % (2 * np.pi)
        )

        sequence.add_block(gx, gy, gz, adc)

        gx2 = pp.make_trapezoid(
            channel='x',
            area=-k[0],
            rise_time=grad_ramp_time,
            flat_time=grad_flat_time,
            fall_time=grad_fall_time,
            system=opts
        )
        gy2 = pp.make_trapezoid(
            channel='y',
            area=-k[1],
            rise_time=grad_ramp_time,
            flat_time=grad_flat_time,
            fall_time=grad_fall_time,
            system=opts
        )
        gz2 = pp.make_trapezoid(
            channel='z',
            area=-k[2],
            rise_time=grad_ramp_time,
            flat_time=grad_flat_time,
            fall_time=grad_fall_time,
            system=opts
        )

        sequence.add_block(gx2, gy2, gz2)

        sequence.add_block(pp.make_delay(final_resting_time))

    print(sequence.test_report())

    output_path = args.output_dir / f'aztek_radial_v4_compat_{"_".join(map(str, SPATIAL_DIMS_PX))}_px_{int(args.runtime)}s_runtime_{int(args.dummy_time)}s_dummy_{"whole" if args.whole_spoke else "part"}_spoke.seq'
    print('Writing sequence to', output_path)
    sequence.write(output_path, check_timing=False)
