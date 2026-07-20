from os import environ
environ["PYTORCH_ALLOC_CONF"] = "expandable_segments:True"

import MRzeroCore as mr0
import argparse
import numpy as np
import torch
import json

from pathlib import Path

from types import MethodType
from create_mr0_phantom import MovingVoxelGridPhantom
from src.utils import subsample_compiled_phantom, simdata_to_cuda
from src.time_curve import TimeCurve

def get_parser() -> None:
    parser = argparse.ArgumentParser(description='Process sequence and phantom files.')

    parser.add_argument('--sequence-file',
                        type=str,
                        help='Path to the sequence file (.seq, .seqk)')

    parser.add_argument('--phantom-file',
                        type=str,
                        help='Path to the phantom file (.phantom)')

    parser.add_argument('--output-file',
                        type=str,
                        help='Path to save the output .npy file')

    parser.add_argument('--phantom-subsample-factor',
                        type=int,
                        default=1,
                        help='Factor by which to subsample the phantom (default = no subsampling)')

    parser.add_argument('--time-curve-config',
                        type=str,
                        default='',
                        help='Path towards a configuration file to define a more complex time curve for breathing.'
    )
    parser.add_argument('--coil-smaps-path',
                        type=str,
                        default='',
                        help='Path to the coil sensitivity maps file. Shape (ncoils, X, Y, Z). Will be resampled to fit phantom resolution.'
    )

    return parser

def load_simulation_data(args):
    sequence_file = args.sequence_file
    phantom_file = args.phantom_file
    phantom_subsample_factor = args.phantom_subsample_factor

    # Load sequence
    print("Preloading sequence...")
    sequence = mr0.Sequence.import_file(sequence_file, backend="pydisseqt", exact_trajectories=True, print_stats=False)

    print("Preloading phantom...")
    phantom = MovingVoxelGridPhantom.load(phantom_file, cubic_deformations=False, closed_loop_deforms=True)
    print('Data loaded!')

    if args.coil_smaps_path != '':
        if Path(args.coil_smaps_path).exists():
            coil_smaps = np.load(args.coil_smaps_path)
            assert coil_smaps.ndim == 4, "Coil sensitivity maps should be a 4D array with shape (ncoils, X, Y, Z)"
            resampled_smaps = torch.nn.functional.interpolate(
                torch.from_numpy(coil_smaps).unsqueeze(0),  # Add batch dimension
                size=phantom.PD.shape,
                mode='trilinear',
                align_corners=False
            ).squeeze(0)
            phantom.coil_sens = resampled_smaps
        else:
            print(f"Coil sensitivity maps file {args.coil_smaps_path} does not exist. Using default coil sensitivity maps.")

    if args.time_curve_config != '':
        if Path(args.time_curve_config).exists():
            with open(args.time_curve_config, 'r') as f:
                tc_cfg = json.load(f)
            custom_tc = TimeCurve(
                t=tc_cfg['t'],
                t_unit=tc_cfg['t_unit'],
                periodic=True,
                periods=tc_cfg['periods']
            )
            phantom.set_deformations(phantom.deformations, custom_tc)
        else:
            print(f"Time curve config file {args.time_curve_config} does not exist. Using default phantom time curve.")

    phantom_comp = phantom.build(voxel_shape='sinc', flipped_positions=True)

    if phantom_subsample_factor > 1:
        print(f"Subsampling phantom by a factor of {phantom_subsample_factor}")
        print("Original phantom size:", phantom.T1.numel())
        phantom_comp = subsample_compiled_phantom(phantom_comp, phantom_subsample_factor)
        phantom_comp.PD *= (phantom_subsample_factor ** 3) # Scale PD to preserve total signal
        print("Subsampled phantom size:", phantom_comp.T1.numel())

    if phantom_comp.voxel_motion is not None:
        phantom_comp.voxel_motion = MethodType(phantom.voxel_motion.__func__, phantom_comp)

    return sequence, phantom_comp

if __name__ == '__main__':
    args = get_parser().parse_args()
    sequence, phantom = load_simulation_data(args)

    print("Preloading done, computing graph!")
    graph = mr0.compute_graph(sequence, phantom, 10_000, 1e-3)
    print('Executing graph, running simulation.')
    with torch.no_grad():
        signal = mr0.execute_graph(graph, sequence.cuda(), simdata_to_cuda(phantom), min_emitted_signal=1e-3, min_latent_signal=1e-2, print_progress=True, clear_state_mag=True)
    print('Got signal:', signal)

    np.save(args.output_file, signal.cpu().numpy())
