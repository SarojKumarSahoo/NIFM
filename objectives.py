from typing import Union

import torch
from torch import Tensor
from vector_field import VectorField2D, VectorField3D

from model import DecoupledTauNet


def vector_objective(opt, vector_field: Union[VectorField2D, VectorField3D], net: DecoupledTauNet) -> Tensor:
    """Computes the vector objective (Sec 3.3 eq. 14 in the paper)

    Args:
        opt (_type_): arguments
        vector_field (Union[VectorField2D, VectorField3D]): Data class
        net (DecoupledTauNet): network

    Returns:
        Tensor: computed vector loss
    """
    physical_coords = vector_field.get_random_coords(opt.batch_size)
    physical_vecs = vector_field.get_vectors(physical_coords)
    physical_vecs = physical_vecs.to(opt.device)
    net_input = vector_field.map_physical_to_net(physical_coords).to(opt.device)
    zero_tau_deriv = net.zero_tau_grad(net_input)
    diff_loss = (zero_tau_deriv - physical_vecs).norm(dim=1).mean()
    return diff_loss


def form_tau_deriv(
    net: DecoupledTauNet, vector_field: Union[VectorField2D, VectorField3D], coords: Tensor, tau: Tensor
) -> Tensor:
    """Computes the derivative of the network output with respect to the time-span

    Args:
        net (DecoupledTauNet): network
        vector_field (Union[VectorField2D, VectorField3D]): data class
        coords (Tensor): flow map seed positions
        tau (Tensor): time-span of integration

    Returns:
        Tensor: Flow map derivative with respect to time-span
    """
    tau.requires_grad = True
    advected_out = net(coords[:, 1:], vector_field.map_physical_to_net(coords), tau)
    advected_deriv = torch.zeros(coords.shape[0], vector_field.out_dim).to(coords.device)
    for d in range(vector_field.out_dim):
        out_coord = advected_out[:, d : d + 1]
        advected_deriv[:, d] = torch.autograd.grad(
            out_coord, tau, grad_outputs=torch.ones_like(out_coord), create_graph=True
        )[0][:, 0]
    return advected_deriv


def advected_objective(opt, vector_field: Union[VectorField2D, VectorField3D], net: DecoupledTauNet) -> Tensor:
    """Computes the self-consistency loss (Sec 3.3 eq. 16 in the paper)

    Args:
        opt (_type_): arguments
        vector_field (Union[VectorField2D, VectorField3D]): data class
        net (DecoupledTauNet): network

    Returns:
        Tensor: self-consistency loss
    """
    time_inc = vector_field.dt
    grid_tau = (opt.min_tau + (opt.max_tau - opt.min_tau) * torch.rand(opt.batch_size, 1)).to(opt.device)
    tau = time_inc * grid_tau

    physical_coords = vector_field.get_random_compensated_coords(opt.batch_size, tau[:, 0].max()).to(opt.device)
    if opt.sampling_strategy == "single":
        time_eps = 1e4 * time_inc * torch.ones_like(grid_tau[:, 0])
    elif opt.sampling_strategy == "full":
        time_eps = time_inc * torch.ones_like(grid_tau[:, 0])
    elif opt.sampling_strategy == "log":
        time_eps = time_inc * torch.maximum(torch.log2(grid_tau[:, 0]), torch.ones_like(grid_tau[:, 0]))
    elif opt.sampling_strategy == "sqrt":
        time_eps = time_inc * torch.maximum(torch.sqrt(grid_tau[:, 0]), torch.ones_like(grid_tau[:, 0]))

    n_steps = (tau[:, 0] // time_eps).long() + 1
    max_step = n_steps.max()
    all_taus = time_eps.unsqueeze(1) * torch.ones(opt.batch_size, max_step).to(opt.device)
    all_taus[torch.arange(opt.batch_size), n_steps - 1] = tau[:, 0] - ((n_steps - 1).float() * time_eps)
    integrated_particles = physical_coords.clone()

    for tdx in range(max_step):
        valid_particles = torch.nonzero(tdx < n_steps).squeeze(dim=1)
        spacetime_particles = integrated_particles[valid_particles]
        particle_tau = all_taus[valid_particles][:, tdx]

        net.eval()
        with torch.no_grad():
            integrated_particles[valid_particles, 1:] = net(
                spacetime_particles[:, 1:],
                vector_field.map_physical_to_net(spacetime_particles),
                particle_tau.view(-1, 1),
            ).detach()
        #
        net.train()

        integrated_particles[valid_particles, 0] += particle_tau

    valid_x = torch.logical_and(
        integrated_particles[:, 1] > vector_field.th_ext[1, 0], integrated_particles[:, 1] < vector_field.th_ext[1, 1]
    )
    valid_y = torch.logical_and(
        integrated_particles[:, 2] > vector_field.th_ext[2, 0], integrated_particles[:, 2] < vector_field.th_ext[2, 1]
    )

    valid_particles = torch.logical_and(valid_x, valid_y)
    if vector_field.out_dim == 3:
        valid_z = torch.logical_and(
            integrated_particles[:, 3] > vector_field.th_ext[3, 0],
            integrated_particles[:, 3] < vector_field.th_ext[3, 1],
        )
        valid_particles = torch.logical_and(valid_particles, valid_z)

    integrated_particles = integrated_particles[valid_particles]
    spacetime_coords = physical_coords[valid_particles]
    tau = tau[valid_particles]

    tau_deriv = form_tau_deriv(net, vector_field, spacetime_coords, tau)
    with torch.no_grad():
        target_tau_deriv = net.zero_tau_grad(vector_field.map_physical_to_net(integrated_particles)).detach()

    diff_loss = (tau_deriv - target_tau_deriv).norm(dim=1).mean()
    return diff_loss
