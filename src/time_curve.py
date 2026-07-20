import torch
import matplotlib.pyplot as plt

from numpy.typing import NDArray
from typing import Union, List, Mapping, Any

class TimeCurve(torch.nn.Module):
    def __init__(self, t: Union[torch.Tensor, List], t_unit: Union[torch.Tensor, List], periodic: bool = True, periods: Union[torch.Tensor, None] = None):
        '''
        KomaMRI-like time interface to define motion.

        `t`: time vector (float)
        `t_unit`: state of motion (0 to 1). 0 means start, 1 is the end. For cyclic motion, make sure deformation end converges to start, as time will jump 0 -> 1.
        `periodic`: Whether motion repeats. Time wraps back around.
        `periods`: Relative duration of further repetitions. Irrespective of `periodic`, which will wrap time *after* each of the periods defined here. eg. defining `periods = [0.5, 1.0, 1.5]` means that repetition happens in chunks of these 3 periods.
        '''
        super().__init__()

        t_tensor = torch.as_tensor(t, dtype=torch.float32)
        t_unit_tensor = torch.as_tensor(t_unit, dtype=torch.float32)

        self.register_buffer('t', t_tensor)
        self.register_buffer('t_unit', t_unit_tensor)

        assert self.t_unit.max() <= 1.0
        periodic_tensor = torch.as_tensor(periodic, dtype=torch.bool)
        periods_tensor = torch.as_tensor(periods, dtype=torch.float32) if periods is not None else torch.tensor([1.0], dtype=torch.float32)

        self.register_buffer("periodic", periodic_tensor)
        self.register_buffer("periods", periods_tensor)

        self.time_scale_max = torch.sum(self.periods) * (self.t[-1] - self.t[0])
        self.time_scale_min = 0.0

    def __call__(self, time) -> torch.float32:
        period_times = self.periods * (self.t[-1] - self.t[0])

        total_period = torch.sum(period_times)
        if self.periodic:
            # Wrap time around the defined periods
            time = time % total_period
        elif time > total_period:
            # If not periodic and time exceeds total period, return the last state of motion
            return self.t_unit[-1]

        active_period = torch.argwhere(time < torch.cumsum(period_times, dim=0)).min()
        elapsed_time_before_period = torch.sum(period_times[:active_period])
        resampled_time = (time - elapsed_time_before_period) / self.periods[active_period]

        # Find the corresponding index in the time vector
        # idxl = torch.argwhere(self.t <= resampled_time).max()
        # idxr = torch.argwhere(self.t >= resampled_time).min()

        # t: (N,) sorted ascending, resampled_time: (M,)
        idxr = torch.searchsorted(self.t, resampled_time, right=False)
        idxl = torch.searchsorted(self.t, resampled_time, right=True) - 1

        # clamp in case resampled_time falls outside [t.min(), t.max()]
        idxl = idxl.clamp(0, self.t.numel() - 1)
        idxr = idxr.clamp(0, self.t.numel() - 1)

        # Interpolate the state of motion
        t_left = self.t[idxl]
        t_right = self.t[idxr]
        t_unit_left = self.t_unit[idxl]
        t_unit_right = self.t_unit[idxr]

        # Linear interpolation
        interped = torch.where(
            idxl != idxr,
            (t_unit_right * (resampled_time - t_left) + t_unit_left * (t_right - resampled_time)) / (t_right - t_left),
            t_unit_left
        )

        return interped

    def load_state_dict(self, state_dict: Mapping[str, Any], strict: bool = True, assign: bool = False):
        superret = super().load_state_dict(state_dict, strict, assign)

        # Recompute time scale extremas
        self.time_scale_max = torch.sum(self.periods) * (self.t[-1] - self.t[0])
        self.time_scale_min = 0.0

        return superret

    def sample_time_regular(self, num_pts: int) -> torch.Tensor:
        sample_times = torch.linspace(0, self.time_scale_max, num_pts)
        return self(sample_times)

    def plot(self) -> None:
        plot_t = torch.linspace(0, self.time_scale_max * 1.5, 50_000) # Viz extra
        plot_t_unit = self(plot_t)

        plt.figure(figsize=(10, 4))
        plt.plot(plot_t, plot_t_unit, label='Time Curve')
        plt.xlabel('Time')
        plt.ylabel('State of Motion (t_unit)')
        plt.title('Time Curve Plot')
        plt.grid()
        plt.legend()
        plt.show()
