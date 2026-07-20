from os import environ
environ['TORCH_CUDA_ARCH_LIST'] = '8.6;9.0;9.2;10.0;10.2;11.0;11.8+PTX'  # For compatibility with newer GPUs
environ['CUBLAS_WORKSPACE_CONFIG'] = ':16:8'
environ['CUDA_VISIBLE_DEVICES'] = '0'  # Use only the first GPU, modify as needed

import numpy as np
import faiss
import cupy as cp
import torch

from pathlib import Path
from argparse import ArgumentParser
from typing import Union, List
from tqdm.auto import tqdm
from numpy.typing import NDArray
from cupyx.scipy.special import softmax
from pyvista import PolyData

class FaissNearestNeighbour:
    def __init__(self, points: NDArray, values: NDArray):
        self.points = points.astype(np.float32)
        self.values = values
        d = self.points.shape[1]  # Dimension of points
        self.batch_size = 3_000_000
        self.num_neighbours = 1

        nlist = len(self.points) // 2048

        index = faiss.IndexIVFFlat(faiss.IndexFlatL2(d), d, nlist)
        index.train(self.points)
        index.add(self.points)
        index_gpu = faiss.index_cpu_to_all_gpus(index)

        self.index_gpu = index_gpu

    def __lookup_batch(self, batch_index: int, query_points: NDArray) -> NDArray:
        start = batch_index * self.batch_size
        end = min(start + self.batch_size, len(query_points))
        query_batch = query_points[start:end].astype(np.float32)
        distances, indices = self.index_gpu.search(query_batch, 1)
        return distances, self.values[indices.flatten()]

    def __call__(self, query_points: NDArray) -> NDArray:
        num_batches = (len(query_points) + self.batch_size - 1) // self.batch_size

        all_distances, all_values = [], []
        for i in tqdm(range(num_batches), desc="Performing nearest neighbor lookup"):
            distances, batch_values = self.__lookup_batch(i, query_points)
            all_distances.append(distances)
            all_values.append(batch_values)
        all_distances = np.concatenate(all_distances, axis=0)
        all_values = np.concatenate(all_values, axis=0)
        return all_distances, all_values

class FourierEmbedding(torch.nn.Module):
    def __init__(self, in_features=4, num_frequencies=64, scale=0.3):
        super().__init__()
        # in_features is now 4 (X, Y, Z, T)
        # Lower scale recommended for spatiotemporal fields to prevent temporal jitter
        self.register_buffer('B', torch.randn(in_features, num_frequencies) * scale)

    def forward(self, x):
        # x shape: (..., 4) -> projection shape: (..., num_frequencies)
        projection = torch.matmul(x, self.B)
        # Returns shape: (..., num_frequencies * 2) bounded between [-1, 1]
        return torch.cat([torch.sin(projection), torch.cos(projection)], dim=-1)

class ResidualBlock(torch.nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.fc1 = torch.nn.Linear(dim, dim)
        self.fc2 = torch.nn.Linear(dim, dim)
        # SiLU (Swish) provides better C-infinity smoothness than Tanh
        # and avoids vanishing gradients that cause "breaks" in deformations.
        self.act = torch.nn.SiLU()

    def forward(self, x):
        return x + self.act(self.fc2(self.act(self.fc1(x))))

class ImplicitNeuralDeformationField:
    def __init__(self, points, values, epochs=1500, lr=1e-3, weight_decay=1e-4, hidden_dim=128):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Extract dimensions
        self.T, self.N, _ = values.shape
        self.output_dim = 3 # We now output a single 3D deformation vector per (X,T) coordinate

        # 1. Coordinate Normalization (Crucial for MLP convergence)
        self.X_mean = torch.tensor(points.mean(axis=0), dtype=torch.float32, device=self.device)
        self.X_std = torch.tensor(points.std(axis=0) + 1e-8, dtype=torch.float32, device=self.device)

        X = torch.tensor(points, dtype=torch.float32, device=self.device)
        X_norm = (X - self.X_mean) / self.X_std

        # 2. Time Normalization to [-1, 1]
        self.t_vals = torch.linspace(-1, 1, steps=self.T, device=self.device)

        # 3. Create Spatiotemporal Training Pairs (N * T, 4)
        # Expand X and t to match each other: shape (T, N, 3) and (T, N, 1)
        X_expanded = X_norm.unsqueeze(0).expand(self.T, -1, -1)
        T_expanded = self.t_vals.view(self.T, 1, 1).expand(-1, self.N, 1)

        # Combine into (T*N, 4)
        XT_train = torch.cat([X_expanded, T_expanded], dim=-1).reshape(-1, 4)

        # Targets: reshape (T, N, 3) directly to (T*N, 3)
        Y_train = torch.tensor(values, dtype=torch.float32, device=self.device).reshape(-1, 3)

        # 4. Network Architecture
        # Global Affine Baseline (Maps 4D Space-Time to 3D deformation)
        self.affine = torch.nn.Linear(4, self.output_dim).to(self.device)

        # Fourier mapping: Bumps 4 spatiotemporal dimensions up to 128 features
        num_freqs = 64
        self.embedding = FourierEmbedding(in_features=4, num_frequencies=num_freqs, scale=0.3).to(self.device)
        mlp_input_dim = num_freqs * 2

        # Deep Residual MLP Configuration
        num_res_blocks = 4
        mlp_layers = [
            torch.nn.Linear(mlp_input_dim, hidden_dim),
            torch.nn.SiLU()
        ]

        for _ in range(num_res_blocks):
            mlp_layers.append(ResidualBlock(hidden_dim))

        mlp_layers.append(torch.nn.Linear(hidden_dim, self.output_dim))

        self.mlp = torch.nn.Sequential(*mlp_layers).to(self.device)

        # 5. Optimization Setup
        optimizer = torch.optim.Adam([
            {'params': self.affine.parameters(), 'weight_decay': 0.0},
            {'params': self.mlp.parameters(), 'weight_decay': weight_decay}
        ], lr=lr)

        criterion = torch.nn.MSELoss()

        # 6. Training Loop
        self.affine.train()
        self.mlp.train()

        print(f"Training Spatiotemporal Deformation Field on {self.device}...")
        for epoch in range(epochs):
            optimizer.zero_grad()

            # Forward pass using (X, Y, Z, T)
            XT_embedded = self.embedding(XT_train)
            pred = self.affine(XT_train) + self.mlp(XT_embedded)

            loss = criterion(pred, Y_train)

            loss.backward()
            optimizer.step()

            if (epoch + 1) % 50 == 0 or epoch == 0:
                print(f"  Epoch [{epoch+1}/{epochs}] | Loss: {loss.item():.6f}")

        self.affine.eval()
        self.mlp.eval()
        print("Training complete.")

    def __call__(self, query_points, query_times=None, chunk_size=100_000):
        """
        Evaluates the smooth deformation field at arbitrary spatial points and times.

        Parameters:
        -----------
        query_points : np.ndarray or torch.Tensor
            Shape (M, 3) - Arbitrary 3D coordinates.
        query_times : torch.Tensor, optional
            Shape (T_new,) - Normalized time steps to query. If None, uses original training times.
            Pass values > 1.0 to extrapolate into the future.
        chunk_size : int
            Size of chunks to process query points in to avoid OOM.

        Returns:
        --------
        np.ndarray or torch.Tensor (depending on input)
            Shape (T_new, M, 3) - Interpolated/Extrapolated deformations.
        """
        is_numpy = isinstance(query_points, np.ndarray)

        if query_times is None:
            query_times = self.t_vals
        else:
            query_times = query_times.to(self.device)

        T_new = len(query_times)
        M_total = query_points.shape[0]

        predictions = np.zeros((T_new, M_total, 3), dtype=np.float32)
        if not is_numpy:
            predictions = torch.as_tensor(predictions, device=self.device)
            chunker = torch.split(query_points, chunk_size, dim=0)
        else:
            chunker = np.array_split(query_points, np.arange(chunk_size, M_total, chunk_size), axis=0)

        for window_num, sliding_window in enumerate(tqdm(chunker, desc="Evaluating deformation field")):
            if is_numpy:
                X_query = torch.tensor(sliding_window, dtype=torch.float32, device=self.device)
            else:
                X_query = sliding_window.to(self.device)

            M_chunk = X_query.shape[0]

            with torch.no_grad():
                # Normalize query coordinates using training statistics
                X_query_norm = (X_query - self.X_mean) / self.X_std

                chunk_preds = []

                # Evaluate chunk across all requested time steps
                for t_val in query_times:
                    # Create T_expand for this specific time step (M_chunk, 1)
                    t_expanded = t_val.view(1, 1).expand(M_chunk, 1)

                    # Concat into (M_chunk, 4)
                    XT_query = torch.cat([X_query_norm, t_expanded], dim=-1)

                    # Predict 3D output for this time step
                    XT_embedded = self.embedding(XT_query)
                    pred_t = self.affine(XT_query) + self.mlp(XT_embedded)

                    chunk_preds.append(pred_t)

                # Stack to (T_new, M_chunk, 3)
                pred = torch.stack(chunk_preds, dim=0)

            pred_ready = pred.cpu().numpy() if is_numpy else pred

            start_idx = window_num * chunk_size
            end_idx = start_idx + M_chunk
            predictions[:, start_idx:end_idx, :] = pred_ready

        return predictions

if __name__ == '__main__':
    parser = ArgumentParser(description="Use implicit neural representation (INR) interpolation on the GPU to interpolate point cloud data to a grid.")
    parser.add_argument('--patient-dir', required=True, type=Path, help="Directory containing the input .npy files: points.npy, values.npy, etc.")
    parser.add_argument('--fixed-bones', '-f', action='store_true', help="If set, makes bones totally static. Movement around bones has a drop-off, zero at the interfaces, being restored to normal movement at a distance.")
    args = parser.parse_args()

    # Make deterministic.
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.use_deterministic_algorithms(True)
    torch.manual_seed(42)
    torch.cuda.manual_seed_all(42)

    patient_dir = args.patient_dir.resolve()
    points = np.load(patient_dir / "mesh_t0.npy")  # (N, c)
    values = np.load(patient_dir / "relative_deformations.npy")  # (N, c) or (T, N, c)
    original_values_shape = values.shape

    if values.ndim != 3:
        raise RuntimeError("Expected values to be of size (T, N, 3)")

    grid_reference = np.load(patient_dir / "proton_density_volume.npy")  # (X, Y, Z)
    idxs = np.indices(grid_reference.shape)
    query_points = np.transpose(idxs, (1, 2, 3, 0)).reshape(-1, 3)  # (M, 3)

    output_file = patient_dir / "interpolated_deform_fields_inr.npy"
    if not output_file.exists():
        inr = ImplicitNeuralDeformationField(points, values, epochs=500, hidden_dim=128)

        interpolated_grid = inr(query_points, chunk_size=10_000)  # (T, M, c)
        interpolated_grid = interpolated_grid.reshape((values.shape[0], *grid_reference.shape, values.shape[2]))  # (T, X, Y, Z, c)
        interpolated_grid_f16 = interpolated_grid.astype(np.float16) # Save as float16 to reduce file size
        np.save(output_file, interpolated_grid_f16)
    else:
        interpolated_grid = np.load(output_file).astype(np.float32)  # Load existing interpolated grid for density estimation

    if args.fixed_bones:
        print("Applying fixed bones constraint...")
        bone_mask = np.load(patient_dir / "bone_type_volume.npy")  # (X, Y, Z). 0 = non-bone, 1 = rib bone, 2 = spine bone.
        head_to_toe_dir = 1 # Y axis is head-toe in phantom, change as needed.

        spine_bone_mask = (bone_mask == 2).astype(bool)
        interpolated_grid[:, spine_bone_mask, :] = 0  # Set deformation to zero at spine bone locations. Rib cage can still move.

        # Make linear movement drop-off near spine: not much movement near spine.
        spine_bone_finder = FaissNearestNeighbour(points=np.argwhere(spine_bone_mask), values=np.zeros(np.sum(spine_bone_mask)))  # Dummy values since we only care about distances
        distances_spine, _ = spine_bone_finder(query_points)
        distances_spine = np.sqrt(distances_spine) # Turn sum of squared dists -> L2 distance.
        distances_spine = distances_spine.reshape(grid_reference.shape)  # (X, Y, Z)
        dropoff_distance = (np.cbrt(np.prod(grid_reference.shape)) / 8) # Distance at which movement is fully restored (voxels).
        dropoff_distance = np.round(dropoff_distance, 2)
        print('Spine bone movement drop-off distance set to:', dropoff_distance, 'voxels (euclidian distance)')
        distance_kernel = lambda d, w: np.clip(d/w, 0, 1)  # Linear drop-off
        weights = distance_kernel(distances_spine, dropoff_distance)

        interpolated_grid *= weights[None, :, :, :, None]

        # Now tackle rib bones: no vertical movement at rib bones, but objects should "slide" along rib bones, so vertical movement is allowed around it.
        bone_idxs = np.argwhere(bone_mask == 1)
        interpolated_grid[:, bone_mask == 1, head_to_toe_dir] = 0  # Set vertical deformation to zero at rib bone locations. Lung slides along ribs.

        np.save(patient_dir / "interpolated_deform_fields_fixed_bones.npy", interpolated_grid.astype(np.float16))
