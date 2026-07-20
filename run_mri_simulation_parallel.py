from os import environ
environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

import MRzeroCore as mr0
import numpy as np
import torch

from argparse import ArgumentParser
from types import MethodType
from src.utils import simdata_to_cuda, index_compiled_phantom
from run_mri_simulation import get_parser, load_simulation_data
from os import remove

def add_args_for_parallel_execution(parser: ArgumentParser) -> None:
    parser.add_argument(
        "--job-index",
        type=int,
        required=False,
        help="Index of the current job (1-based)",
    )
    parser.add_argument(
        "--num-jobs",
        type=int,
        required=True,
        help="Total number of parallel jobs",
    )
    parser.add_argument(
        "--stack-signals",
        action="store_true",
        default=False,
        help="Whether to stack signals from different jobs into a single output file (replaces normal execution).",
    )

def stack_signals(args):
    output_files = [args.output_file.replace(".npy", f"_part_{job_idx}.npy") for job_idx in range(1, args.num_jobs + 1)]
    signals = [np.load(output_file) for output_file in output_files]
    stacked_signal = sum(signals)

    np.save(args.output_file, stacked_signal)
    for output_file in output_files:
        remove(output_file)

if __name__ == '__main__':
    args = get_parser()
    add_args_for_parallel_execution(args)
    args = args.parse_args()
    assert args.output_file.endswith(".npy"), "Output file must have a .npy extension when using parallel execution mode."

    if args.stack_signals:
        stack_signals(args)
    else:
        sequence, phantom = load_simulation_data(args)
        k_folds = np.array_split(np.arange(phantom.PD.shape[0]), args.num_jobs)

        print("Data preloaded, starting parallel simulation on worker ", args.job_index)
        phantom = index_compiled_phantom(phantom, k_folds[args.job_index - 1])
        phantom.cubic_motion = phantom.cubic_motion[0]
        phantom.cubic_coefs = phantom.cubic_coefs[0] # Because of a weird bug, these get put into tuples. Unpack them here to fix.
        if phantom.voxel_motion is not None:
            phantom.voxel_motion = MethodType(phantom.voxel_motion.__func__, phantom) # Ensure voxel_motion is correctly bound to the subsampled phantom

        graph = mr0.compute_graph(sequence, phantom, 10_000, 1e-3)
        print('Executing graph, running simulation.')
        with torch.no_grad():
            signal = mr0.execute_graph(graph, sequence.cuda(), simdata_to_cuda(phantom), min_emitted_signal=1e-3, min_latent_signal=1e-2, print_progress=True)
        print('Got signal:', signal)

        np.save(args.output_file.replace(".npy", f"_part_{args.job_index}.npy"), signal.cpu().numpy())
        print('Done!')
