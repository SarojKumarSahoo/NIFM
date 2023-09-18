from typing import Dict, List

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from loguru import logger
from torch import Tensor


class Swish(nn.Module):
    def __init__(self):
        super(Swish, self).__init__()

    def forward(self, input):
        return F.silu(input)


class Sine(nn.Module):
    def __init__(self, w0=30):
        super(Sine, self).__init__()
        self.w0 = w0

    def forward(self, input):
        return torch.sin(self.w0 * input)


class LinearLayer(nn.Module):
    def __init__(self, in_dim: int, out_dim: int, is_act: bool = True) -> None:
        super(LinearLayer, self).__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.is_act = is_act
        self.act = Swish()

        self.linear_layer = nn.Linear(self.in_dim, self.out_dim, bias=False)
        self.init_hidden_layers(self.linear_layer)

    def init_hidden_layers(self, m: nn.Linear):
        with torch.no_grad():
            m.weight = nn.Parameter(torch.randn(self.out_dim, self.in_dim) / (np.sqrt(self.in_dim)))

    def forward(self, x: torch.Tensor):
        if self.is_act:
            return self.act(self.linear_layer(x))
        else:
            return self.linear_layer(x)


class FourierFeatureMap(nn.Module):
    def __init__(self, in_dim: int, out_dim: int, bias: bool = False) -> None:
        super(FourierFeatureMap, self).__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.act = Sine(w0=1)
        self.ffm = nn.Linear(in_dim, out_dim, bias=bias)

        with torch.no_grad():
            rand_dirs = 2 * torch.rand_like(self.ffm.weight) - 1
            rand_dirs = torch.linspace(0.5, 4, rand_dirs.shape[0]).unsqueeze(1) * (
                rand_dirs / rand_dirs.norm(dim=1, keepdim=True)
            )
            self.ffm.weight = nn.Parameter(rand_dirs)

    def forward(self, input):
        return self.act(self.ffm(input))


class FeatureGrid(nn.Module):
    def __init__(self, f_dim: int, f_size: int, scale: float = 1e-2, n_dim: int = 2) -> None:
        super(FeatureGrid, self).__init__()
        self.f_dim = f_dim
        self.f_size = f_size
        self.scale = scale
        self.n_dim = n_dim
        if n_dim == 2:
            self.f_grid = nn.Parameter(scale * torch.randn(1, f_dim, f_size, f_size, f_size))
        else:
            self.f_grid = nn.Parameter(scale * torch.randn(1, f_dim, f_size, f_size, f_size, f_size))

    def forward(self, input_coords: Tensor):
        n_batch, _ = input_coords.shape
        if self.n_dim == 2:
            input_coords = input_coords.reshape(1, n_batch, 1, 1, 3)
            features = F.grid_sample(self.f_grid, input_coords, align_corners=True, padding_mode="border")[
                0, :, :, 0, 0
            ].T
        else:
            f_t = ((self.f_size - 1) * (input_coords[:, 0] + 1)) / 2
            prev_f_grid_t = min(int(torch.floor(f_t)[0].item()), self.f_size - 1)
            next_f_grid_t = min(int(torch.ceil(f_t)[0].item()), self.f_size - 1)
            alpha = f_t - prev_f_grid_t
            alpha = alpha.unsqueeze(-1)
            prev_features_t = F.grid_sample(
                self.f_grid[:, :, prev_f_grid_t, :, :, :],
                input_coords[:, 1:].reshape(1, n_batch, 1, 1, self.n_dim),
                align_corners=True,
                padding_mode="border",
            )[0, :, :, 0, 0].T
            next_features_t = F.grid_sample(
                self.f_grid[:, :, next_f_grid_t, :, :, :],
                input_coords[:, 1:].reshape(1, n_batch, 1, 1, self.n_dim),
                align_corners=True,
                padding_mode="border",
            )[0, :, :, 0, 0].T
            features = prev_features_t * (1 - alpha) + next_features_t * alpha
        return features


class MultiLevelGrid(nn.Module):
    def __init__(self, f_dims: List[int], f_sizes: List[int], n_levels: int, n_dim: int = 2) -> None:
        super(MultiLevelGrid, self).__init__()

        self.f_dims = f_dims
        self.f_sizes = f_sizes
        self.n_levels = len(f_dims)
        self.n_dim = n_dim
        self.latent_grid = nn.ModuleList()

        for idx in range(self.n_levels):
            self.latent_grid.append(FeatureGrid(f_dims[idx], f_sizes[idx], n_dim=self.n_dim))

        self.act = Swish()

    def forward(self, input_coords: Tensor) -> Tensor:
        out = []
        x = input_coords
        for idx, l_grid in enumerate(self.latent_grid):
            out_f = l_grid(x)
            out.append(out_f)
        return self.act(torch.cat(out, dim=1))


class DecoupledTauNet(nn.Module):
    def __init__(
        self,
        in_dim: int,
        hid_dim: int,
        out_dim: int,
        f_dims: List[int],
        f_sizes: List[int],
        n_layers: int = 3,
        n_vector_layers: int = 2,
        n_dim: int = 2,
    ) -> None:
        super(DecoupledTauNet, self).__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.f_dims = f_dims
        self.f_sizes = f_sizes
        self.n_levels = len(f_dims)
        self.n_layers = n_layers
        self.n_vector_layers = n_vector_layers
        self.hid_dim = hid_dim
        self.n_dim = n_dim

        # vector feature grid
        self.vec_grid = MultiLevelGrid(f_dims, f_sizes, self.n_levels, n_dim=self.n_dim)
        self.vec_pe = FourierFeatureMap(self.in_dim, sum(self.f_dims))
        self.vec_scale_param = nn.Parameter(torch.ones(1, 2 * sum(self.f_dims)) * 0.01)

        self.vec_feature_mapping = nn.ModuleList()
        for _ in range(n_vector_layers):
            self.vec_feature_mapping.append(LinearLayer(self.hid_dim, self.hid_dim))

        # tau feature grid
        self.tau_grid = MultiLevelGrid(f_dims, f_sizes, self.n_levels, n_dim=self.n_dim)
        self.tau_pe = FourierFeatureMap(self.in_dim, sum(self.f_dims), bias=True)

        # shallow mlp
        self.vec_mlp = nn.ModuleList()
        self.tau_params = nn.ParameterList()
        for idx in range(n_layers):
            if idx == 0:
                self.tau_mlp = LinearLayer(self.hid_dim, self.hid_dim, is_act=False)
            self.vec_mlp.append(LinearLayer(self.hid_dim, self.hid_dim, is_act=False))
            self.tau_params.append(nn.Parameter(torch.ones(1, self.hid_dim) * 0.01))
        self.last_layer = nn.Linear(self.hid_dim, self.out_dim, bias=False)
        self.init_last_layers(self.last_layer)

        self.act = Swish()

    def init_last_layers(self, m: nn.Linear):
        with torch.no_grad():
            m.weight = nn.Parameter(torch.randn(self.out_dim, self.hid_dim) / (np.sqrt(self.hid_dim)))

    def forward(self, physical_coords, net_coords, tau) -> Tensor:
        vec_features = self.vec_grid(net_coords)
        vec_ffm = self.vec_pe(net_coords)
        vec_features = torch.cat((vec_features, vec_ffm), dim=1)
        for _, mapping in enumerate(self.vec_feature_mapping):
            vec_features = mapping(vec_features)

        tau_features = self.tau_grid(net_coords)
        tau_ffm = self.tau_pe(net_coords)
        tau_features = torch.cat((tau_features, tau_ffm), dim=1)

        out = torch.tanh(tau * self.vec_scale_param * vec_features)
        for idx, layer in enumerate(self.vec_mlp):
            t_tau = torch.tanh(tau * self.tau_params[idx])
            if idx == 0:
                out = out + t_tau * self.act(layer(out) * self.tau_mlp(tau_features))
            else:
                out = out + t_tau * self.act(layer(out))
        return physical_coords + self.last_layer(out)

    def zero_tau_grad(self, net_coords) -> Tensor:
        vec_features = self.vec_grid(net_coords)
        vec_ffm = self.vec_pe(net_coords)
        vec_features = torch.cat((vec_features, vec_ffm), dim=1)

        for _, mapping in enumerate(self.vec_feature_mapping):
            vec_features = mapping(vec_features)
        out = self.vec_scale_param * vec_features
        return out @ self.last_layer.weight.T


if __name__ == "__main__":
    net = DecoupledTauNet(3, 64, 2, [8, 8, 8, 8], [8, 16, 32, 64], 3, 3)
    print(net)
