import pypulseq as pp
import MRzeroCore as mr0
import torch
from mrseq.utils import write_sequence

from tempfile import NamedTemporaryFile

def convert_150_to_141(seq_file: str) -> str:
    seq_orig = pp.Sequence()
    seq_orig.read(seq_file)

    temp_file = NamedTemporaryFile('w+', suffix='.seq', prefix='temp_seq_', delete_on_close=False, delete=False)
    write_sequence(seq_orig, temp_file.name, create_signature=True, v141_compatibility=True)
    return temp_file.name

def subsample_compiled_phantom(phantom_comp, subsample_factor):
    new = mr0.SimData(
        PD=phantom_comp.PD[::subsample_factor],
        T1=phantom_comp.T1[::subsample_factor],
        T2=phantom_comp.T2[::subsample_factor],
        T2dash=phantom_comp.T2dash[::subsample_factor],
        D=phantom_comp.D[::subsample_factor],
        B0=phantom_comp.B0[::subsample_factor],
        B1=phantom_comp.B1[:, ::subsample_factor],
        coil_sens=phantom_comp.coil_sens[:, ::subsample_factor],
        size=phantom_comp.size,
        voxel_pos=phantom_comp.voxel_pos[::subsample_factor],
        nyquist=phantom_comp.nyquist, # Assume nyquist is unchanged by subsampling, as a safe bet.
        affine=phantom_comp.affine,
        dephasing_func=phantom_comp.dephasing_func,
        recover_func=phantom_comp.recover_func,
        phantom_motion=phantom_comp.phantom_motion,
        voxel_motion=phantom_comp.voxel_motion,
        tissue_masks={k: v.ravel()[::subsample_factor] for k, v in phantom_comp.tissue_masks.items()},
    )
    if 'deformations' in dir(phantom_comp): # Don't assign and create if absent
        new.deformations = phantom_comp.deformations[:, ::subsample_factor, :] if phantom_comp.deformations is not None else None
    new.time_curve = phantom_comp.time_curve
    new.cubic_motion = phantom_comp.cubic_coefs is not None
    new.cubic_coefs = [x[::subsample_factor] if x.ndim > 1 else x for x in phantom_comp.cubic_coefs] if phantom_comp.cubic_coefs is not None else None
    return new

def simdata_to_cuda(simdata):
    simdata_cuda = simdata.cuda()
    if 'deformations' in dir(simdata): # Don't assign and create if absent
        simdata_cuda.deformations = simdata.deformations.cuda() if simdata.deformations is not None else None
    simdata_cuda.time_curve = simdata.time_curve.cuda() if simdata.time_curve is not None else None
    simdata_cuda.cubic_motion = simdata.cubic_motion
    if simdata.cubic_motion:
        simdata_cuda.cubic_coefs = [x.cuda() for x in simdata.cubic_coefs]
    else:
        simdata_cuda.cubic_coefs = None
    return simdata_cuda

def index_compiled_phantom(phantom_comp, indices):
    indexed = mr0.SimData(
        PD=phantom_comp.PD[indices],
        T1=phantom_comp.T1[indices],
        T2=phantom_comp.T2[indices],
        T2dash=phantom_comp.T2dash[indices],
        D=phantom_comp.D[indices],
        B0=phantom_comp.B0[indices],
        B1=phantom_comp.B1[:, indices],
        coil_sens=phantom_comp.coil_sens[:, indices],
        size=phantom_comp.size,
        voxel_pos=phantom_comp.voxel_pos[indices],
        nyquist=phantom_comp.nyquist,
        affine=phantom_comp.affine,
        dephasing_func=phantom_comp.dephasing_func,
        recover_func=phantom_comp.recover_func,
        phantom_motion=phantom_comp.phantom_motion,
        voxel_motion=phantom_comp.voxel_motion,
        tissue_masks={k: v.ravel()[indices] for k, v in phantom_comp.tissue_masks.items()},
    )
    if 'deformations' in dir(phantom_comp): # Don't assign and create if absent
        indexed.deformations = phantom_comp.deformations[:, indices, :] if phantom_comp.deformations is not None else None
    indexed.time_curve = phantom_comp.time_curve
    indexed.cubic_motion=phantom_comp.cubic_motion,
    indexed.cubic_coefs=[x[indices] if x.ndim > 1 else x for x in phantom_comp.cubic_coefs] if phantom_comp.cubic_coefs is not None else None,
    return indexed

def make_generic_affine(size, shape) -> torch.Tensor:
    size = torch.as_tensor(size)
    shape = torch.as_tensor(shape)
    affine = torch.eye(3,4)
    affine[0, 0] = size[0] / shape[0] * 1000
    affine[1, 1] = size[1] / shape[1] * 1000
    affine[2, 2] = size[2] / shape[2] * 1000
    affine[:, 3] = -size / 2 * 1000
    return affine
