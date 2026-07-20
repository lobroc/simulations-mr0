import argparse
import json5

import numpy as np
import MRzeroCore as mr0
import torch

from pathlib import Path
from types import MethodType
from warnings import warn
from torchcubicspline import natural_cubic_spline_coeffs, NaturalCubicSpline

from numpy.typing import NDArray, ArrayLike
from typing import Literal, Optional, Dict, Union

from src.time_curve import TimeCurve
from src.utils import make_generic_affine

def parse_args():
    parser = argparse.ArgumentParser(description='Process simulator data.')

    parser.add_argument('--data-dir', '-d',
                        type=str,
                        required=True,
                        help='Directory containing simulator data')

    parser.add_argument('--patient-index', '-p',
                        type=int,
                        default=0,
                        help='Index of the patient to process')

    parser.add_argument('--deformation-index', '-i',
                        type=int,
                        default=0,
                        help='Index of the deformation to process')

    parser.add_argument('--static', '-s',
                        action='store_true',
                        help='Use static phantom without deformation')

    parser.add_argument('--skip-phantom-check', '-n',
                        action='store_true',
                        help='Whether to skip the phantom check after building. Not recommended, as it may lead to undefined behaviour if the phantom is not correctly made.'
    )

    args = parser.parse_args()
    return args

class MovingVoxelGridPhantom(mr0.VoxelGridPhantom):
    def __init__(self,
        PD: ArrayLike,
        T1: ArrayLike,
        T2: ArrayLike,
        T2dash: ArrayLike,
        D: ArrayLike,
        B0: ArrayLike,
        B1: ArrayLike,
        coil_sens: ArrayLike,
        size: ArrayLike,
        affine: Union[ArrayLike, None] = None,
        deformations: Union[NDArray, None] = None,
        time_curve: Union[TimeCurve, None] = None,
        cubic_motion: bool = False,
        cubic_coefs: Union[torch.Tensor, None] = None,
        closed_loop_deforms: bool = False,
        **kwargs
    ):
        super().__init__(
            PD=PD,
            T1=T1,
            T2=T2,
            T2dash=T2dash,
            D=D,
            B0=B0,
            B1=B1,
            coil_sens=coil_sens,
            size=size,
            affine=affine if affine is not None else make_generic_affine(size, PD.shape),
            tissue_masks=kwargs.get('tissue_masks', None)
        )
        self.cubic_motion = cubic_motion
        self.cubic_coefs = cubic_coefs

        closed_loop_deforms = closed_loop_deforms

        self.voxel_motion = None # Off by default
        self.deformations = deformations
        if self.deformations is not None:
            self.deformations = torch.as_tensor(self.deformations, dtype=torch.float32)
            coords = self.make_grid(self.PD > 0, self.size, self.PD.device)
            if coords.max() > 0.5:
                self.deformations /= (2 * coords.max())
            if coords.min() < -0.5:
                self.deformations /= (2 * -coords.min())

        if closed_loop_deforms and self.deformations is not None:
            self.deformations[-1, ...] = self.deformations[0, ...]

        if T1.max() > 5.0 or T2.max() > 5.0 or T2dash.max() > 0.5:
            warn('Surprising values in T1/T2/T2dash maps. Check phantom properties.')

        if time_curve is not None and self.deformations is not None:
            self.set_deformations(self.deformations, time_curve)
        else:
            self.time_curve = None

        self.__converted = False

    def activate_motion(self) -> None:
        self.voxel_motion = self.get_voxel_motion if self.deformations is not None else None

    def deactivate_motion(self) -> None:
        self.voxel_motion = None

    def __str__(self) -> str:
        retstr = ""
        for k, v in self.__dict__.items():
            if hasattr(v, 'shape'):
                is_complex = (isinstance(v, torch.Tensor) and v.is_complex()) or \
                            (isinstance(v, np.ndarray) and np.iscomplexobj(v))
                if is_complex:
                    mod = v.abs() if isinstance(v, torch.Tensor) else np.abs(v)
                    arg = v.angle() if isinstance(v, torch.Tensor) else np.angle(v)
                    retstr += (f"{k}: shape={tuple(v.shape)}, "
                        f"|v| min={mod.min():.4f} max={mod.max():.4f}, "
                        f"arg min={arg.min():.4f} max={arg.max():.4f}")
                else:
                    retstr += f"{k}: shape={tuple(v.shape)}, min={v.min():.4f}, max={v.max():.4f}\n"
            else:
                retstr += f"{k}: {v}\n"
        return retstr

    @staticmethod
    def make_grid(mask, size, device) -> torch.Tensor:
        '''
        Cloned part of build() from VoxelGridPhantom
        '''
        shape = torch.tensor(mask.shape)
        pos_x, pos_y, pos_z = torch.meshgrid(
            size[0] *
            torch.fft.fftshift(torch.fft.fftfreq(
                int(shape[0]), device=device)),
            size[1] *
            torch.fft.fftshift(torch.fft.fftfreq(
                int(shape[1]), device=device)),
            size[2] *
            torch.fft.fftshift(torch.fft.fftfreq(
                int(shape[2]), device=device)),
            indexing="ij"
        )

        voxel_pos = torch.stack([
            pos_x[mask].flatten(),
            pos_y[mask].flatten(),
            pos_z[mask].flatten()
        ], dim=1)
        return voxel_pos

    @classmethod
    def from_vgp(cls, vgp) -> 'MovingVoxelGridPhantom':
        return cls(
            PD=vgp.PD,
            T1=vgp.T1,
            T2=vgp.T2,
            T2dash=vgp.T2dash,
            D=vgp.D,
            B0=vgp.B0,
            B1=vgp.B1,
            coil_sens=vgp.coil_sens,
            size=vgp.size,
            affine=vgp.affine
        )

    def set_deformations(self, deformations, time_curve: TimeCurve) -> None:
        self.deformations = torch.as_tensor(deformations, dtype=torch.float32)
        self.time_curve = time_curve

        if self.cubic_motion and self.cubic_coefs is None:
            self.cubic_coefs = natural_cubic_spline_coeffs(
                self.time_curve.sample_time_regular(self.deformations.shape[0]),
                self.deformations.reshape(self.deformations.shape[0], -1, 3).permute(1, 0, 2) # Cubic spline coeffs expect (N, T, D), we have (T, N, D)
            )

        self.activate_motion()

    def get_voxel_motion(self, t) -> torch.Tensor:
        if self.deformations is None:
            raise RuntimeError("Deformations not set for MovingVoxelGridPhantom.")
        if not self.deformations.is_cuda:
            self.deformations = self.deformations.cuda()

        compute_device = torch.device('cuda')
        if self.cubic_coefs is not None and (not all([x.is_cuda for x in self.cubic_coefs])):
            self.cubic_coefs = [x.cuda() for x in self.cubic_coefs]

        # Map time to deformation index
        breath_phase = self.time_curve(t)
        if self.cubic_motion:
            spline = NaturalCubicSpline(self.cubic_coefs)
            deform_field = spline.evaluate(breath_phase).permute(1, 0, 2) # Spline output is (N, T, D), we want (T, N, D)
        else:
            breath_index_max = self.deformations.shape[0] - 1
            breath_phase_fine = (breath_phase * breath_index_max).to(device=compute_device, dtype=torch.float32)

            breath_phase_idx_down = torch.floor(breath_phase * breath_index_max).to(device=compute_device, dtype=torch.int)
            breath_phase_idx_up = torch.ceil((breath_phase * breath_index_max) % (breath_index_max)).to(device=compute_device, dtype=torch.int)
            up_weight = 1 - (breath_phase_idx_up - breath_phase_fine)
            down_weight = 1 - up_weight
            deform_field = down_weight[:, None, None] * self.deformations[breath_phase_idx_down] + up_weight[:, None, None] * self.deformations[breath_phase_idx_up]

        if 'voxel_pos' not in dir(self):
            warn('Defining voxel positions in get_voxel_motion, as was not defined previously.')
            self.voxel_pos, mask = self.__make_voxel_pos()
            self.voxel_pos = self.voxel_pos[mask, :]
        self.voxel_pos = self.voxel_pos.to(compute_device)

        deformed_pos = self.voxel_pos + deform_field.reshape(deform_field.shape[0], -1, 3)

        return deformed_pos

    def save(self, file_name) -> None:
        if self.time_curve is not None:
            filename_move = Path(file_name).parent / (Path(file_name).name.split('.')[0] + '_movement.pth')
            torch.save(self.time_curve.state_dict(), filename_move)
        np.savez(file_name,
            PD_map=self.PD.numpy(),
            T1_map=self.T1.numpy(),
            T2_map=self.T2.numpy(),
            T2dash_map=self.T2dash.numpy(),
            D_map=self.D.numpy(),
            B0_map=self.B0.numpy(),
            B1_map=self.B1.numpy(),
            FOV=self.size.numpy(),
            affine=self.affine.numpy(),
            deformations=self.deformations.numpy() if self.deformations is not None else np.nan,
            time_shape=len(self.time_curve.t) if self.time_curve is not None else np.nan,
            periods_shape=len(self.time_curve.periods) if self.time_curve is not None else np.nan,
        )

    @classmethod
    def load(cls, file_name, cubic_deformations: bool = False, closed_loop_deforms: bool = False) -> 'MovingVoxelGridPhantom':
        data = np.load(file_name)

        filename_move = Path(file_name).parent / (Path(file_name).name.split('.')[0] + '_movement.pth')
        if Path(filename_move).exists():
            time_curve = TimeCurve( # Ensure recieving objects have the right shape
                t = torch.zeros(data['time_shape'], dtype=torch.float32),
                t_unit = torch.zeros(data['time_shape'], dtype=torch.float32),
                periods = torch.zeros(data['periods_shape'], dtype=torch.float32)
            )
            time_curve.load_state_dict(torch.load(filename_move, weights_only=True))
        else:
            time_curve = None

        instance = cls(
            PD=data['PD_map'],
            T1=data['T1_map'],
            T2=data['T2_map'],
            T2dash=data['T2dash_map'],
            D=data['D_map'],
            B0=data['B0_map'],
            B1=data['B1_map'],
            coil_sens=np.ones_like(data['PD_map'])[None, ...],
            size=data['FOV'],
            affine=torch.from_numpy(data['affine']) if 'affine' in data else make_generic_affine(data['FOV'], data['PD_map'].shape),
            deformations = data['deformations'] if not np.isnan(data['deformations']).all() else None,
            time_curve = time_curve,
            cubic_motion = cubic_deformations,
            cubic_coefs = None,
            closed_loop_deforms = closed_loop_deforms
        )

        return instance

    def __make_voxel_pos(self, PD_threshold: float, flipped: bool = False) -> torch.Tensor:
        mask = self.PD > PD_threshold

        shape = torch.tensor(mask.shape)
        pos_x, pos_y, pos_z = torch.meshgrid(
            self.size[0] *
            torch.fft.fftshift(torch.fft.fftfreq(
                int(shape[0]), device=self.PD.device)),
            self.size[1] *
            torch.fft.fftshift(torch.fft.fftfreq(
                int(shape[1]), device=self.PD.device)),
            self.size[2] *
            torch.fft.fftshift(torch.fft.fftfreq(
                int(shape[2]), device=self.PD.device)),
            indexing="ij"
        )

        voxel_pos = torch.stack([
            pos_x[mask].flatten() * (-1 if flipped else 1),
            pos_y[mask].flatten() * (-1 if flipped else 1),
            pos_z[mask].flatten() * (-1 if flipped else 1)
        ], dim=1)

        return voxel_pos, mask

    def build(self, PD_threshold: float = 1e-6,
              voxel_shape: Literal["sinc", "box", "point"] = "sinc",
              flipped_positions: bool = False
              ) -> mr0.SimData:
        """Build a :class:`SimData` instance for simulation.

        Arguments
        ---------
        PD_threshold : float
            All voxels with a proton density below this value are ignored.
        flipped_positions : bool
            Whether to flip the sign of the voxel positions. This can be used to correct for flipping versus other simulators.
        """
        if self.__converted:
            warn('This instance has already been built once. Building again may lead to undefined behaviour.')

        if 'voxel_pos' not in dir(self):
            self.voxel_pos, mask = self.__make_voxel_pos(PD_threshold, flipped=flipped_positions)
        else:
            mask = self.PD > PD_threshold
        shape = torch.tensor(mask.shape)
        voxel_pos = self.voxel_pos

        if voxel_shape == "box":
            def dephasing_func(t, n): return torch.prod(torch.sinc(t * n), dim=-1)
        elif voxel_shape == "sinc":
            def dephasing_func(t, n): return torch.prod(torch.sigmoid(
                    (n - t.abs() + 0.5) * 100
                ), dim=-1)
        elif voxel_shape == "point":
            def dephasing_func(t, _): return torch.ones_like(t[:, 0])
        else:
            raise ValueError(f"Unsupported voxel shape '{voxel_shape}'")

        if not self.tissue_masks:
            self.tissue_masks = {"combined": mask}

        sd = mr0.SimData(
            self.PD[mask],
            self.T1[mask],
            self.T2[mask],
            self.T2dash[mask],
            self.D[mask],
            self.B0[mask],
            self.B1[:, mask],
            self.coil_sens[:, mask],
            self.size,
            voxel_pos=voxel_pos,
            nyquist=torch.as_tensor(shape, device=self.PD.device) / 2 / self.size,
            dephasing_func=dephasing_func,
            affine=self.affine,
            recover_func=None,
            phantom_motion=self.phantom_motion,
            voxel_motion=None,
            tissue_masks = {k: v[mask].flatten() for k, v in self.tissue_masks.items()}
        )
        sd.voxel_motion = MethodType(self.voxel_motion.__func__, sd) if self.voxel_motion is not None else None
        sd.time_curve = self.time_curve
        sd.cubic_coefs = self.cubic_coefs
        sd.cubic_motion = self.cubic_motion

        # Scale deformations: voxels to physical units. Divide by shape to normalise, then multiply by size.
        if self.deformations is not None:
            scaled_deformations = (self.deformations / shape) * self.size
            scaled_deformations = scaled_deformations * (-1 if flipped_positions else 1)

            if mask.sum() == scaled_deformations.shape[1]: # We have already masked deformations sometime prior, so just assign.
                sd.deformations = scaled_deformations
            else:
                sd.deformations = scaled_deformations[:, mask, :]

            if self.cubic_motion and self.cubic_coefs is not None:
                del sd.cubic_coefs
                del self.cubic_coefs # Clean up memory before recomputing cubic coefs
                # Recompute cubic coefs for the scaled deformations
                sd.cubic_coefs = natural_cubic_spline_coeffs(
                    self.time_curve.sample_time_regular(self.deformations.shape[0]),
                    sd.deformations.permute(1, 0, 2) # Cubic spline coeffs expect (N, T, D), we have (T, N, D)
                )
                self.cubic_coefs = sd.cubic_coefs

        self.__converted = True

        return sd

    def interpolate(self, x: int, y: int, z: int) -> 'MovingVoxelGridPhantom':
        """Return a resized copy of this :class:`MovingVoxelGridPhantom` instance.

        This uses torch.nn.functional.interpolate in 'area' mode, which is not
        very good: Assumes pixels are squares -> has strong aliasing.

        Parameters
        ----------
        x : int
            The new resolution along the 1st dimension
        y : int
            The new resolution along the 2nd dimension
        z : int
            The new resolution along the 3rd dimension
        mode : str
            Algorithm used for upsampling (via torch.nn.functional.interpolate)

        Returns
        -------
        MovingVoxelGridPhantom
            A new :class:`MovingVoxelGridPhantom` instance containing resized tensors.
        """
        def resample(tensor: torch.Tensor) -> torch.Tensor:
            # Introduce additional dimensions: mini-batch and channels
            return torch.nn.functional.interpolate(
                tensor[None, None, ...], size=(x, y, z), mode='trilinear'
            )[0, 0, ...]

        def resample_multicoil(tensor: torch.Tensor) -> torch.Tensor:
            coils = tensor.shape[0]
            output = torch.zeros(coils, x, y, z, dtype=tensor.dtype)
            for i in range(coils):
                re = resample(torch.real(tensor[i, ...]))
                im = resample(torch.imag(tensor[i, ...]))
                output[i, ...] = re + 1j * im

            return output

        def resample_masks(tensors: Dict) -> Optional[Dict]:
            output = {}
            for key, mask in tensors.items():
                # Interpolate the mask
                interpolated_mask = torch.nn.functional.interpolate(
                    mask[None, None, ...].float(), size=(x, y, z), mode='area'
                )[0, 0, ...]
                # Store the result
                output[key] = interpolated_mask

            return output

        mvgp = MovingVoxelGridPhantom(
            resample(self.PD),
            resample(self.T1),
            resample(self.T2),
            resample(self.T2dash),
            resample(self.D),
            resample(self.B0),
            resample_multicoil(self.B1),
            resample_multicoil(self.coil_sens),
            self.size.clone(),
            affine=self.affine.clone(),
            tissue_masks=resample_masks(self.tissue_masks),
            deformations=resample_multicoil(self.deformations) if self.deformations is not None else None,
            time_curve=self.time_curve,
            cubic_motion=self.cubic_motion,
            cubic_coefs=resample_multicoil(self.cubic_coefs) if self.cubic_coefs is not None else None,
            closed_loop_deforms=self.closed_loop_deforms
        )
        if mvgp.deformations is not None:
            mvgp.activate_motion()

        return mvgp

    def __getitem__(self, key):
        mvgp = MovingVoxelGridPhantom(
            self.PD[key],
            self.T1[key],
            self.T2[key],
            self.T2dash[key],
            self.D[key],
            self.B0[key],
            self.B1[:, key],
            self.coil_sens[:, key],
            self.size,
            affine=self.affine.clone(),
            tissue_masks={k: mask[key] for k, mask in self.tissue_masks.items()},
            deformations=self.deformations[:, key, :] if self.deformations is not None else None,
            time_curve=self.time_curve,
            cubic_motion=self.cubic_motion,
            cubic_coefs=self.cubic_coefs[:, key, :] if self.cubic_coefs is not None else None,
            closed_loop_deforms=self.closed_loop_deforms
        )

        if mvgp.deformations is not None:
            mvgp.activate_motion()

        return mvgp

    def __len__(self):
        return self.PD.numel()

    def cuda(self) -> None:
        super().cuda()
        if self.deformations is not None:
            self.deformations = self.deformations.cuda()
        if self.time_curve is not None:
            self.time_curve = self.time_curve.cuda()
        if self.cubic_coefs is not None:
            self.cubic_coefs = self.cubic_coefs.cuda()

    def copy(self):
        mvgp = MovingVoxelGridPhantom(
            self.PD.clone(),
            self.T1.clone(),
            self.T2.clone(),
            self.T2dash.clone(),
            self.D.clone(),
            self.B0.clone(),
            self.B1.clone(),
            self.coil_sens.clone(),
            self.size.clone(),
            affine=self.affine.clone(),
            tissue_masks={k: mask.clone() for k, mask in self.tissue_masks.items()},
            deformations=self.deformations.clone() if self.deformations is not None else None,
            time_curve=self.time_curve.copy() if self.time_curve is not None else None,
            cubic_motion=self.cubic_motion,
            cubic_coefs=self.cubic_coefs.clone() if self.cubic_coefs is not None else None,
            closed_loop_deforms=self.closed_loop_deforms
        )

        if mvgp.deformations is not None:
            mvgp.activate_motion()

        return mvgp

if __name__ == "__main__":
    params = parse_args()

    BREATH_LENGTH = np.float32(5.0)  # seconds
    spatial_scales = np.array((0.16, 0.25, 0.25)) # front-back, top-bottom, left-right dims, in meters

    data_dir = Path(params.data_dir)
    patients = sorted(data_dir.iterdir())

    with open('../simulator-data/segmentation/mr_tissue_properties.jsonc', 'r') as f:
        tissue_properties = json5.load(f)

    patient_dir = data_dir / f"patient_{params.patient_index}_deformation_{params.deformation_index}"
    patient_files = sorted(patient_dir.iterdir())

    print("Found data, loading from:", patient_dir)
    mesh_t0 = np.load(patient_dir / "mesh_t0.npy").astype(np.float32)
    deformations = np.load(patient_dir / "relative_deformations.npy").astype(np.float32)
    t1_vol = np.load(patient_dir / "t1_volume.npy").astype(np.float32)
    t2_vol = np.load(patient_dir / "t2_volume.npy").astype(np.float32)
    t2star_vol = np.load(patient_dir / "t2star_volume.npy").astype(np.float32)

    if not params.static:
        relative_deform_fields = np.load(patient_dir / "interpolated_deform_fields_fixed_bones.npy").astype(np.float32)

    proton_density_vol = np.load(patient_dir / "proton_density_volume.npy").astype(np.float32)
    print("Data loaded successfully.")

    proton_density_vol -= np.min(proton_density_vol)
    proton_d_95 = np.quantile(proton_density_vol, 0.98) # Exclude outliers to prevent extreme values dominating the scale.
    proton_d_95_indices = np.where(proton_density_vol > proton_d_95)
    proton_density_vol[proton_d_95_indices] = proton_d_95  # Cap
    proton_density_vol /= proton_d_95

    # Find indices where == 0
    values_in_t1_t2 = ((t1_vol != 0) | (t2_vol != 0) | (t2star_vol != 0))
    lin_inds = np.argwhere(values_in_t1_t2)

    num_timepoints = deformations.shape[0]
    num_points = len(lin_inds)
    N = num_points

    bone_selector = (t2_vol == tissue_properties["bone"]["T2"]) & (t1_vol == tissue_properties["bone"]["T1"])
    heart_selector = (t2_vol == tissue_properties["heart_muscle"]["T2"]) & (t1_vol == tissue_properties["heart_muscle"]["T1"])

    lung_selector = (t2star_vol == tissue_properties["lung"]["T2*"]) & (t1_vol == tissue_properties["lung"]["T1"])
    lung_selector |= (t2_vol == tissue_properties["blood"]["T2"]) & (t1_vol == tissue_properties["blood"]["T1"]) # There is also blood in the lungs
    lung_selector |= (t2_vol == tissue_properties["cartilage"]["T2"]) & (t1_vol == tissue_properties["cartilage"]["T1"]) # Cartilage in airways surrogate signal. Also include.

    xpos = lin_inds[:, 0] * (spatial_scales[0] / t1_vol.shape[0])
    ypos = lin_inds[:, 1] * (spatial_scales[1] / t1_vol.shape[1])
    zpos = lin_inds[:, 2] * (spatial_scales[2] / t1_vol.shape[2])

    # Safe inversion
    t2star_vol_inv = np.full_like(t2star_vol, np.inf)
    t2star_vol_inv[t2star_vol != 0] = 1 / t2star_vol[t2star_vol != 0]
    t2_vol_inv = np.full_like(t2_vol, np.inf)
    t2_vol_inv[t2_vol != 0] = 1 / t2_vol[t2_vol != 0]

    # T2' dephasing effects with 1/T2* = 1/T2 + 1/T2'
    t2dash_vol = np.zeros_like(t2star_vol)
    not_both_inf_mask = ~(np.isinf(t2star_vol_inv) & np.isinf(t2_vol_inv))
    t2dash_vol[not_both_inf_mask] = 1 / (t2star_vol_inv[not_both_inf_mask] - t2_vol_inv[not_both_inf_mask])

    print('Defining phantom with', N, 'points.')
    input_phantom = MovingVoxelGridPhantom(
        PD = proton_density_vol,
        T1 = t1_vol / 1000.0, # Scale from ms -> s
        T2 = t2_vol / 1000.0,
        T2dash=t2dash_vol / 1000.0,
        D=np.zeros_like(proton_density_vol),
        B0=np.zeros_like(proton_density_vol),
        B1=np.ones_like(proton_density_vol)[None, ...], # Important to have ones: otherwise, values will be all-zero.
        coil_sens=np.ones_like(proton_density_vol)[None, ...],
        size=spatial_scales,
        deformations=relative_deform_fields if not params.static else None,
        time_curve=TimeCurve(t=[0, BREATH_LENGTH], t_unit=[0, 1], periodic=True)
    )

    if not params.skip_phantom_check:
        print('Checking phantom was correctly defined...')
        input_phantom_compiled = input_phantom.build()
        static_pos = input_phantom_compiled.voxel_pos.clone() # Check this was defined
    else:
        print('Skipping phantom check. Errors may happen at simulation runtime.')

    print("Phantom defined successfully, writing.")
    if params.static:
        phantom_filename = data_dir / 'phantoms' / f'mr0_phantom_static_patient_{params.patient_index}.mr0phantom'
    else:
        phantom_filename = data_dir / 'phantoms' / f'mr0_phantom_dynamic_patient_{params.patient_index}_deform_{params.deformation_index}.mr0phantom'

    phantom_filename.parent.mkdir(exist_ok=True, parents=True)
    input_phantom.save(phantom_filename)
    print("Phantom written to:", phantom_filename)
