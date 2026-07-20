import numpy as np
from numpy.typing import NDArray
import ctypes

from typing import Tuple
from pathlib import Path

# File location
so_path = Path(__file__).parent / 'libaztek.so'
if not so_path.exists():
    raise FileNotFoundError(f"Shared library not found at {so_path.resolve()}, run the generation script first.")

aztek_lib = ctypes.cdll.LoadLibrary(str(so_path.resolve()))

def AZTEK_trajectory_comp(
        totalSpoke: np.int32,
        gxS: NDArray[np.int16],
        gyS: NDArray[np.int16],
        gzS: NDArray[np.int16],
        scale_factor: np.int32,
        AZTEK_Twist: np.float64,
        AZTEK_Shuffle: np.float64,
        AZTEK_Speed: np.int32
    ) -> Tuple[bool, np.int16]:
    '''
    Generate the 3D Radial Trajectory

    gxS (modified): gradient scale array in x direction
    gyS (modified): gradient scale array in y direction
    gzS (modified): gradient scale array in z direction
    totalSpoke: total number of spokes
    scale_factor: used for low res scan

    returns  0: success,  1: failure, and max_iamp, the maximum gradient instruction amplitude
    '''

    # Pre-call checks
    assert isinstance(gxS, np.ndarray) and gxS.dtype == np.int16, "gxS must be a numpy array of type int16"
    assert isinstance(gyS, np.ndarray) and gyS.dtype == np.int16, "gyS must be a numpy array of type int16"
    assert isinstance(gzS, np.ndarray) and gzS.dtype == np.int16, "gzS must be a numpy array of type int16"
    assert gxS.ndim == 1 or gyS.ndim == 1 or gzS.ndim == 1, "gxS, gyS, and gzS must be 1-dimensional arrays"

    # Make max_iamp a C pointer
    max_iamp_ptr = ctypes.pointer(ctypes.c_int16(0))

    aztek_lib.AZTEK_trajectory_comp.argtypes = [
        ctypes.c_int,      # totalSpoke
        np.ctypeslib.ndpointer(dtype=np.int16, ndim=1, flags='C_CONTIGUOUS'),  # gxS
        np.ctypeslib.ndpointer(dtype=np.int16, ndim=1, flags='C_CONTIGUOUS'),  # gyS
        np.ctypeslib.ndpointer(dtype=np.int16, ndim=1, flags='C_CONTIGUOUS'),  # gzS
        ctypes.c_int,      # scale_factor
        ctypes.POINTER(ctypes.c_int16),  # max_iamp (single short* OUT)
        ctypes.c_double,   # AZTEK_Twist
        ctypes.c_double,   # AZTEK_Shuffle
        ctypes.c_int       # AZTEK_Speed
    ]

    aztek_lib.AZTEK_trajectory_comp.restype = ctypes.c_int

    result = aztek_lib.AZTEK_trajectory_comp(
        totalSpoke,
        gxS,
        gyS,
        gzS,
        scale_factor,
        max_iamp_ptr,
        AZTEK_Twist,
        AZTEK_Shuffle,
        AZTEK_Speed
    )
    max_iamp = max_iamp_ptr.contents.value
    max_iamp = np.int16(max_iamp)

    return result, max_iamp

if __name__ == "__main__":
    print('Testing example trajectory generation...')

    # Return value storage
    gx = np.zeros(1000, dtype=np.int16)
    gy = np.zeros(1000, dtype=np.int16)
    gz = np.zeros(1000, dtype=np.int16)

    # Parameters
    spokes = np.int32(128)
    scale = np.int32(1)
    twist = np.float64(2)
    shuffle = np.float64(2)
    speed = np.int32(3)

    ret, iamp = AZTEK_trajectory_comp(
        spokes,
        gx,
        gy,
        gz,
        scale,
        twist,
        shuffle,
        speed
    )
    gs = np.stack((gx, gy, gz), axis=0)

    print(f"Code result: {'OK' if ret == 0 else 'ERROR'}, Max Iamp: {iamp}")
    print("First 10 gradient values (x, y, z):", gs[:, :10], sep='\n')
