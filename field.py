import json
from abc import ABC, abstractmethod
from pathlib import Path
from typing import List, Union

import numpy as np
import torch
from torch import Tensor


class VectorField(ABC):
    """This is an abstract class for the vector field dataset."""

    def __init__(self, data_dir: str, device: str):
        self.data_dir = data_dir
        self.device = device
        self._metadata = json.load(open(Path(f"{data_dir}", "metadata.json"), "r"))
        self._res = self._metadata["res"]
        self._ext = self._metadata["ext"]
        self._dataset_name = self._metadata["dataset_name"]
        if 'toroidal' in self._metadata:
            self.is_toroidal = self._metadata['toroidal']
        else:
            self.is_toroidal = False

    @property
    def res(self) -> List[int]:
        """spatio-temporal resolution of the dataset

        Returns:
            List[int]: List of resolution values [t,x,y(,z)]
        """
        return self._res

    @property
    def size(self) -> int:
        """computes the total spatio-temporal size of the dataset

        Returns:
            int: size of the dataset
        """
        return np.prod(np.array(self._res))

    @property
    def spatial_size(self) -> int:
        """computes the total spatial size of the dataset

        Returns:
            int: spatial size of the dataset
        """
        return np.prod(np.array(self._res[1:]))

    @property
    def ext(self) -> List[List[float]]:
        """The phyiscal domain of the dataset in the format:
            [[t_min, t_max], [x_min, x_max], [y_min, y_max] (,[z_min, z_max])]

        Returns:
            List[List[float]]: returns the physical domain of the dataset as a list of lists.
        """
        return self._ext

    @property
    def diag(self) -> Tensor:
        """Computes the bounding box diagonal of the dataset.

        Returns:
            Tensor: bounding box diagonal
        """
        th_ext = self.to_tensor(self._ext, dtype=torch.float32)
        return (th_ext[1:, 1] - th_ext[1:, 0]).norm()

    @property
    def dataset_name(self) -> str:
        """Dataset Name

        Returns:
            str: name of the dataset
        """
        return self._dataset_name

    @property
    @abstractmethod
    def in_dim():
        pass

    @property
    @abstractmethod
    def out_dim():
        pass

    def to_tensor(self, value: Union[List, np.ndarray], dtype: str) -> Tensor:
        if isinstance(value, list):
            th_value = torch.tensor(value, dtype=dtype, device=self.device)
        if isinstance(value, np.ndarray):
            th_value = torch.from_numpy(value).to(dtype).to(self.device)
        return th_value

    def linear_interpolation(self, physical_coords: Tensor, field: Tensor) -> Tensor:
        """performs trilinear-interpolation at the given position to get the vectors
        for a give 2D time-varying datasets.

        Args:
            physical_coords (Tensor): positional values in the physical domain
            field (Tensor): the entire 2D time varying dataset of shape [2, t_dim, x_dim, y_dim]

        Returns:
            Tensor: interpolated values at the given positions
        """
        coords_t = (physical_coords[:, 0] - self.th_ext[0, 0]) / (self.th_ext[0, 1] - self.th_ext[0, 0])
        coords_x = (physical_coords[:, 1] - self.th_ext[1, 0]) / (self.th_ext[1, 1] - self.th_ext[1, 0])
        coords_y = (physical_coords[:, 2] - self.th_ext[2, 0]) / (self.th_ext[2, 1] - self.th_ext[2, 0])
        coords = torch.stack((coords_t, coords_x, coords_y), dim=1)

        B = coords.shape[0]

        res = torch.tensor([field.shape[1], field.shape[2], field.shape[3]], device=coords.device).unsqueeze(0)  # 1 x 3
        lattice = (res - 1) * coords

        # -> give us a B x 2 x 2 x 2 tensor of control points
        floorpunch = torch.floor(lattice).long()
        t_stencil = torch.tensor([[[0, 0], [0, 0]], [[1, 1], [1, 1]]], dtype=torch.long, device=coords.device)
        x_stencil = torch.tensor([[[0, 0], [1, 1]], [[0, 0], [1, 1]]], dtype=torch.long, device=coords.device)
        y_stencil = torch.tensor([[[0, 1], [0, 1]], [[0, 1], [0, 1]]], dtype=torch.long, device=coords.device)
        t_control = floorpunch[:, 0:1].unsqueeze(2).unsqueeze(3) + t_stencil.unsqueeze(0)
        x_control = floorpunch[:, 1:2].unsqueeze(2).unsqueeze(3) + x_stencil.unsqueeze(0)
        y_control = floorpunch[:, 2:3].unsqueeze(2).unsqueeze(3) + y_stencil.unsqueeze(0)

        # clamp
        t_control[t_control < 0] = 0
        t_control[t_control >= res[0, 0]] = res[0, 0] - 1
        x_control[x_control < 0] = 0
        x_control[x_control >= res[0, 1]] = res[0, 1] - 1
        y_control[y_control < 0] = 0
        y_control[y_control >= res[0, 2]] = res[0, 2] - 1

        # -> access values in field, yielding D x B x 2 x 2 x 2 control points
        F_field = field[:, t_control, x_control, y_control]

        # interpolation matrices
        t_coords = torch.ones(B, 2, device=coords.device)
        t_coords[:, 0] = lattice[:, 0] - floorpunch[:, 0].float()  # vary in position, B x 2
        x_coords = torch.ones(B, 2, device=coords.device)
        x_coords[:, 0] = lattice[:, 1] - floorpunch[:, 1].float()  # vary in position, B x 2
        y_coords = torch.ones(B, 2, device=coords.device)
        y_coords[:, 0] = lattice[:, 2] - floorpunch[:, 2].float()  # vary in position, B x 2

        basis = torch.tensor([[-1, 1], [1, 0]], dtype=coords.dtype, device=coords.device)  # 2 x 2
        t_interpolation = t_coords.mm(basis).unsqueeze(0).unsqueeze(3).unsqueeze(4)
        t_interpolated_field = (t_interpolation * F_field).sum(dim=2)  # over t (2)
        x_interpolation = x_coords.mm(basis).unsqueeze(0).unsqueeze(3)
        x_interpolated_field = (x_interpolation * t_interpolated_field).sum(dim=2)  # over x (2)
        y_interpolation = y_coords.mm(basis).unsqueeze(0)
        interpolated_field = (y_interpolation * x_interpolated_field).sum(dim=2)  # over y (2)

        return interpolated_field.T

    @abstractmethod
    def get_vectors(self, positions: Tensor) -> Tensor:
        pass

    @abstractmethod
    def get_random_coords(self, n_samples: int) -> Tensor:
        pass

    @abstractmethod
    def get_random_compensated_coords(self, n_samples: int, tau: float) -> Tensor:
        pass

    @abstractmethod
    def get_lattice(self) -> Tensor:
        pass

    @abstractmethod
    def map_physical_to_net(self, positions: Tensor) -> Tensor:
        pass


class VectorField2D(VectorField):
    def __init__(self, data_dir: str, device: str):
        super().__init__(data_dir, device)
        self._in_dim = 3
        self._out_dim = 2

        self.spatial_range, self.spacing = list(), list()
        self.th_ext = self.to_tensor(self.ext, dtype=torch.float32)
        for idx in range(self._in_dim - 1):
            self.spatial_range.append(
                torch.linspace(self.th_ext[idx + 1, 0], self.th_ext[idx + 1, 1], self.res[idx + 1])
            )
            self.spacing.append(self.spatial_range[-1][1] - self.spatial_range[-1][0])

        self.temporal_range = torch.linspace(self.th_ext[0, 0], self.th_ext[0, 1], self.res[0]).to(self.device)
        self.field = self.to_tensor(
            np.load(Path(f"{self.data_dir}", "field.npy")),
            dtype=torch.float32,
        ).to(self.device)

    @property
    def dx(self) -> float:
        """spacing along the x-dimenison."""
        return self.spacing[0]

    @property
    def dy(self):
        """spacing along the y-dimenison."""
        return self.spacing[1]

    @property
    def dt(self):
        """spacing along the t-dimenison."""
        return self.temporal_range[1] - self.temporal_range[0]

    @property
    def in_dim(self):
        """Const value of 3 - input dimension correspondin to spatio_temporal position [t,x,y]"""
        return self._in_dim

    @property
    def out_dim(self):
        """Const value of 2 - output dimension correspondin to spatial position [x,y]"""
        return self._out_dim

    def toroidal_map(self, positions: Tensor) -> Tensor:
        """computes the particles position satisfying the periodic boundary condition if
        particle is outside the domain.

        Args:
            positions (Tensor): position of the particles

        Returns:
            Tensor: mapped position
        """
        if not self.is_toroidal:
            return

        for d in range(self.in_dim - 1):
            less_mask = positions[:, d + 1] < self.th_ext[d + 1, 0]
            if less_mask.shape[0] > 0:
                positions[less_mask, d + 1] = (self.th_ext[d + 1, 1] - self.th_ext[d + 1, 0]) + positions[
                    less_mask, d + 1
                ]

            greater_mask = positions[:, d + 1] > self.th_ext[d + 1, 1]
            if greater_mask.shape[0] > 0:
                positions[greater_mask, d + 1] = (self.th_ext[d + 1, 0] - self.th_ext[d + 1, 1]) + positions[
                    greater_mask, d + 1
                ]

            return positions

    def toroidal_subtract(self, p: Tensor, q: Tensor) -> Tensor:
        """Computes difference between two position tensors with periodic boundary condition taken into consideration.

        Args:
            p (Tensor): spatio-temporal position of shape [n,3]
            q (Tensor): spatio-temporal position of shape [n,3]

        Returns:
            Tensor: spatio-temporal position of shape [n,3]
        """
        if not self.is_toroidal:
            return p - q

        main_diff = torch.zeros_like(p)
        for d in range(self.in_dim - 1):
            normal_diff = p[:, d] - q[:, d]

            less_q = q[:, d] < 0.5 * (self.th_ext[d + 1, 0] + self.ext[d + 1, 1])
            greater_q = ~less_q
            reflected_q = q.clone()
            reflected_q[less_q, d] = (self.th_ext[d + 1, 1] - self.th_ext[d + 1, 0]) + q[less_q, d]
            reflected_q[greater_q, d] = (self.th_ext[d + 1, 0] - self.th_ext[d + 1, 1]) + q[greater_q, d]

            reflected_diff = p[:, d] - reflected_q[:, d]
            diff_mask = normal_diff.abs() < reflected_diff.abs()
            main_diff[diff_mask, d] = normal_diff[diff_mask]
            main_diff[~diff_mask, d] = reflected_diff[~diff_mask]

        return main_diff

    def map_random_to_physical(self, values: Tensor) -> Tensor:
        """maps the random values in the range [0,1] to the physical domain of the dataset.

        Args:
            values (Tensor): spatio-temporal position in [0,1]

        Returns:
            Tensor: spatio-temporal position in the dataset physical domain
        """
        return (values * (self.th_ext[:, 1].unsqueeze(0) - self.th_ext[:, 0].unsqueeze(0))) + self.th_ext[
            :, 0
        ].unsqueeze(0)

    def get_vectors(self, positions: Tensor) -> Tensor:
        """performs linear interpolation to get the interpolated values

        Args:
            positions (Tensor): spatio-temporal position in the physical domain of the dataset

        Returns:
            Tensor: vectors at the given positions
        """
        return self.linear_interpolation(positions.to(self.device), self.field.to(self.device))

    def get_random_time_bound_coords(self, n_samples: int) -> Tensor:
        pass

    def get_random_coords(self, n_samples: int) -> Tensor:
        """generate random positions

        Args:
            n_samples (int): number of particle positions

        Returns:
            Tensor: generates random particle positions of shape [n_samples, 3] with values in [0,1]
        """
        coord_samples = torch.rand(n_samples, self._in_dim).to(self.device)
        phyiscal_coords = self.map_random_to_physical(coord_samples)
        return phyiscal_coords

    def get_random_compensated_coords(self, n_samples: int, max_tau: float) -> Tensor:
        """generate random position with compensated temporal position based on the max time-span
        of the integration.

        Args:
            n_samples (int): number of particle positions
            max_tau (float): maximum time-span of integration

        Returns:
            Tensor: random spatio-temporal values in [0,1]
        """
        coord_samples = torch.rand(n_samples, self._in_dim).to(self.device)
        spatial_physical_coords = (
            coord_samples[:, 1:] * ((self.th_ext[1:, 1].unsqueeze(0)) - self.th_ext[1:, 0].unsqueeze(0))
        ) + self.th_ext[1:, 0].unsqueeze(0)
        temporal_physical_coords = (
            coord_samples[:, 0] * ((self.th_ext[0, 1] - max_tau) - self.th_ext[0, 0])
        ) + self.th_ext[0, 0]
        return torch.cat((temporal_physical_coords.unsqueeze(1), spatial_physical_coords), dim=1)

    def get_lattice(self) -> Tensor:
        """generates a dense set of lattice points

        Returns:
            Tensor: tensor of lattice points of shape [n,3]
        """
        return torch.stack(torch.meshgrid(self.spatial_range, indexing="ij"), dim=-1)

    def map_physical_to_net(self, positions: Tensor) -> Tensor:
        """map the spatio-temporal positions in the physical domain to the network domain [-1, 1]

        Args:
            positions (Tensor): position tensor in the physical domain

        Returns:
            Tensor: position tensor in the network domain
        """
        return (2 * (positions - self.th_ext[:, 0].unsqueeze(0))) / (
            self.th_ext[:, 1].unsqueeze(0) - self.th_ext[:, 0].unsqueeze(0)
        ) - 1


class VectorField3D(VectorField):
    def __init__(self, data_dir: str, device: str):
        super().__init__(data_dir, device)
        self._in_dim = 4
        self._out_dim = 3

        self.spatial_range, self.spacing = list(), list()
        self.th_ext = self.to_tensor(self.ext, dtype=torch.float32).to(self.device)
        for idx in range(self._in_dim - 1):
            self.spatial_range.append(
                torch.linspace(self.th_ext[idx + 1, 0], self.th_ext[idx + 1, 1], self.res[idx + 1])
            )
            self.spacing.append(self.spatial_range[-1][1] - self.spatial_range[-1][0])

        self.temporal_range = torch.linspace(self.th_ext[0, 0], self.th_ext[0, 1], self.res[0]).to(self.device)

        MAX_SIZE = 150_000_000
        self.storage_size = 0
        self.n_fields = 0
        while self.storage_size < MAX_SIZE:
            self.storage_size += self.spatial_size
            self.n_fields += 1

    @property
    def dx(self):
        return self.spacing[0]

    @property
    def dy(self):
        return self.spacing[1]

    @property
    def dz(self):
        return self.spacing[2]

    @property
    def dt(self):
        return self.temporal_range[1] - self.temporal_range[0]

    @property
    def in_dim(self):
        return self._in_dim

    @property
    def out_dim(self):
        return self._out_dim

    def linear_interpolation(self, physical_coords, field):
        """performs trilinear-interpolation at the given position to get the vectors
        for a given time-slice of a 3D time-varying datasets.

        Args:
            physical_coords (Tensor): positional values in the physical domain
            field (Tensor): a single time-slice of a 3D time varying dataset of shape [3, x_dim, y_dim, z_dim]

        Returns:
            Tensor: interpolated values at the given positions
        """
        coords_t = (physical_coords[:, 0] - self.th_ext[1, 0]) / (self.th_ext[1, 1] - self.th_ext[1, 0])
        coords_x = (physical_coords[:, 1] - self.th_ext[2, 0]) / (self.th_ext[2, 1] - self.th_ext[2, 0])
        coords_y = (physical_coords[:, 2] - self.th_ext[3, 0]) / (self.th_ext[3, 1] - self.th_ext[3, 0])
        coords = torch.stack((coords_t, coords_x, coords_y), dim=1)

        B = coords.shape[0]

        res = torch.tensor([field.shape[1], field.shape[2], field.shape[3]], device=coords.device).unsqueeze(0)  # 1 x 3
        lattice = (res - 1) * coords

        # -> give us a B x 2 x 2 x 2 tensor of control points
        floorpunch = torch.floor(lattice).long()
        t_stencil = torch.tensor([[[0, 0], [0, 0]], [[1, 1], [1, 1]]], dtype=torch.long, device=coords.device)
        x_stencil = torch.tensor([[[0, 0], [1, 1]], [[0, 0], [1, 1]]], dtype=torch.long, device=coords.device)
        y_stencil = torch.tensor([[[0, 1], [0, 1]], [[0, 1], [0, 1]]], dtype=torch.long, device=coords.device)
        t_control = floorpunch[:, 0:1].unsqueeze(2).unsqueeze(3) + t_stencil.unsqueeze(0)
        x_control = floorpunch[:, 1:2].unsqueeze(2).unsqueeze(3) + x_stencil.unsqueeze(0)
        y_control = floorpunch[:, 2:3].unsqueeze(2).unsqueeze(3) + y_stencil.unsqueeze(0)

        # clamp
        t_control[t_control < 0] = 0
        t_control[t_control >= res[0, 0]] = res[0, 0] - 1
        x_control[x_control < 0] = 0
        x_control[x_control >= res[0, 1]] = res[0, 1] - 1
        y_control[y_control < 0] = 0
        y_control[y_control >= res[0, 2]] = res[0, 2] - 1

        # -> access values in field, yielding D x B x 2 x 2 x 2 control points
        F_field = field[:, t_control, x_control, y_control]

        # interpolation matrices
        t_coords = torch.ones(B, 2, device=coords.device)
        t_coords[:, 0] = lattice[:, 0] - floorpunch[:, 0].float()  # vary in position, B x 2
        x_coords = torch.ones(B, 2, device=coords.device)
        x_coords[:, 0] = lattice[:, 1] - floorpunch[:, 1].float()  # vary in position, B x 2
        y_coords = torch.ones(B, 2, device=coords.device)
        y_coords[:, 0] = lattice[:, 2] - floorpunch[:, 2].float()  # vary in position, B x 2

        basis = torch.tensor([[-1, 1], [1, 0]], dtype=coords.dtype, device=coords.device)  # 2 x 2
        t_interpolation = t_coords.mm(basis).unsqueeze(0).unsqueeze(3).unsqueeze(4)
        t_interpolated_field = (t_interpolation * F_field).sum(dim=2)  # over t (2)
        x_interpolation = x_coords.mm(basis).unsqueeze(0).unsqueeze(3)
        x_interpolated_field = (x_interpolation * t_interpolated_field).sum(dim=2)  # over x (2)
        y_interpolation = y_coords.mm(basis).unsqueeze(0)
        interpolated_field = (y_interpolation * x_interpolated_field).sum(dim=2)  # over y (2)

        return interpolated_field.T

    def map_random_to_physical(self, values: Tensor) -> Tensor:
        return (values * (self.th_ext[:, 1].unsqueeze(0) - self.th_ext[:, 0].unsqueeze(0))) + self.th_ext[
            :, 0
        ].unsqueeze(0)

    def reset_time_idx(self):
        """permute the time indices of the 3D dataset"""
        self.all_tdx = torch.randperm(self.res[0] - 1).to(self.device)

    def load_batch_fields(self):
        """load random time indices into the memory"""
        self.batch_fields = []
        for tdx in range(int(self.n_fields // 2)):
            self.batch_fields.append(
                self.to_tensor(
                    np.load(Path(f"{self.data_dir}", f"field_{self.all_tdx[tdx]}.npy")),
                    dtype=torch.float32,
                ).unsqueeze(0)
            )
            self.batch_fields.append(
                self.to_tensor(
                    np.load(Path(f"{self.data_dir}", f"field_{self.all_tdx[tdx] + 1}.npy")),
                    dtype=torch.float32,
                ).unsqueeze(0)
            )
        self.batch_fields = torch.cat(self.batch_fields, dim=0)

    def set_fields(self, t_dx: int):
        """keeps track of two neighboring time-slices based on the t_dx to perform interpolation

        Args:
            t_dx (int): random time index from the list of time slices loaded into the memory
        """
        if t_dx >= self.res[0] - 1:
            t_dx = t_dx_plus_1 = self.res[0] - 1
        else:
            t_dx_plus_1 = t_dx + 1

        self.field_t = self.to_tensor(np.load(Path(f"{self.data_dir}", f"field_{int(t_dx)}.npy")), dtype=torch.float32)
        self.field_t_plus_1 = self.to_tensor(
            np.load(Path(f"{self.data_dir}", f"field_{int(t_dx_plus_1)}.npy")), dtype=torch.float32
        )

    def get_vectors(self, positions: Tensor) -> Tensor:
        """return the vectors at the provided time-bounded spatio-temporal positions

        Args:
            positions (Tensor): spatio-temporal positions

        Returns:
            Tensor: vectors at the provided positions
        """
        t = positions[:, 0]
        grid_t = self.map_phyiscal_time_to_grid(t)
        floor_grid_t = torch.floor(grid_t)
        vf_t = self.linear_interpolation(positions[:, 1:].to(self.device), self.field_t.to(self.device))
        vf_t_plus_1 = self.linear_interpolation(positions[:, 1:].to(self.device), self.field_t_plus_1.to(self.device))
        alpha = grid_t - floor_grid_t
        alpha = alpha.unsqueeze(1)

        return (1 - alpha) * vf_t + alpha * vf_t_plus_1

    def get_random_coords(self, n_samples: int) -> Tensor:
        """generates random spatial coordinates in between two consecutive grid time slices chosen at random every batch

        Args:
            n_samples (int): corresponds to the batch size for training

        Returns:
            Tensor: space time position in the physical domain bounded by time [t, t+1]
        """
        coord_samples = torch.rand(n_samples, self._in_dim).to(self.device)
        physical_coords = self.map_random_to_physical(coord_samples)
        rand_tdx_map = torch.randint(0, int(self.n_fields // 2), (1,)).to(self.device)
        batch_tdx = self.all_tdx[rand_tdx_map]
        self.field_t = self.batch_fields[2 * rand_tdx_map.item()]
        self.field_t_plus_1 = self.batch_fields[(2 * rand_tdx_map.item()) + 1]
        physical_coords[:, 0] = self.temporal_range[batch_tdx] + torch.rand(n_samples).to(self.device) * self.dt

        return physical_coords

    def get_random_compensated_coords(self, n_samples: int, max_tau: float) -> Tensor:
        coord_samples = torch.rand(n_samples, self._in_dim).to(self.device)
        physical_coords = self.map_random_to_physical(coord_samples)
        rand_tdx_map = torch.randint(0, int(self.n_fields // 2), (1,)).to(self.device)
        batch_tdx = self.all_tdx[rand_tdx_map]
        while self.temporal_range[batch_tdx] + max_tau > self.ext[0][1]:
            rand_tdx_map = torch.randint(0, int(self.n_fields // 2), (1,)).to(self.device)
            batch_tdx = self.all_tdx[rand_tdx_map]
        self.field_t = self.batch_fields[2 * rand_tdx_map]
        self.field_t_plus_1 = self.batch_fields[(2 * rand_tdx_map) + 1]
        physical_coords[:, 0] = self.temporal_range[batch_tdx] + torch.rand(n_samples).to(self.device) * self.dt

        return physical_coords

    def get_lattice(self) -> Tensor:
        return torch.stack(torch.meshgrid(self.spatial_range, indexing="ij"), dim=-1)

    def map_phyiscal_time_to_grid(self, t: Union[Tensor, float]) -> Union[Tensor, float]:
        return (t - self.th_ext[0, 0]) * (self.res[0] - 1) / (self.th_ext[0, 1] - self.th_ext[0, 0])

    def map_physical_to_net(self, positions: Tensor) -> Tensor:
        return (2 * (positions - self.th_ext[:, 0].unsqueeze(0))) / (
            self.th_ext[:, 1].unsqueeze(0) - self.th_ext[:, 0].unsqueeze(0)
        ) - 1
