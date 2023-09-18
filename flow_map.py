import copy
from typing import Tuple, Union

import numpy as np
import torch
from torch import Tensor

from field import VectorField2D, VectorField3D
from model import DecoupledTauNet


class Flowmap:
    """A class to represent the flow map
    Attributes:
        vector_field (Union[VectorField2D, VectorField3D]): data class
        positions (Tensor): seed positions for flow map computation
        tau (Tensor): time-span of integration
    """

    def __init__(self, vector_field: Union[VectorField2D, VectorField3D], positions: Tensor, tau: Tensor):
        self.vector_field = vector_field
        self._positions = positions
        self._tau = tau

    @property
    def tau(self):
        """The tau property."""
        return self._tau

    @tau.setter
    def tau(self, value):
        self._tau = value

    @tau.deleter
    def tau(self):
        del self._tau

    @property
    def positions(self):
        """The positions property."""
        return self._positions

    @positions.setter
    def positions(self, value):
        self._positions = value

    @positions.deleter
    def positions(self):
        del self._positions

    def is_valid(self, positions: Tensor) -> Tensor:
        """Computes wether the given position of the particles are within the physical domain of the dataset

        Args:
            positions (Tensor): positions of all the particles

        Returns:
            Tensor: A mask indicating wether a given position is valid or not
        """
        valid_mask = torch.logical_and(
            torch.le(positions, self.vector_field.th_ext[1:, 1].unsqueeze(0)),
            torch.ge(positions, self.vector_field.th_ext[1:, 0].unsqueeze(0)),
        )
        valid_mask = valid_mask.prod(dim=1).to(torch.bool)
        return valid_mask

    def euler_flowmap(self, step_size: float = 0.25) -> Tuple[Tensor, Tensor]:
        """Computes the flow map using euler integration. Each particle advects 4 times within a single grid-voxel
        and the time-span of integration for all the particles is expected to be the same, if not same then all
        the particles are integrated for the maximum time-span.

        Returns:
            Tuple[Tensor, Tensor]: Flow map and a mask indicating wether a given particle stayed in the domain for
            the entire integration time-span.
        """
        max_tau = self.tau.max() if isinstance(self.tau, Tensor) else self.tau
        ref_step_size = step_size * self.vector_field.dt
        n_ref_steps = int(max_tau // ref_step_size)
        if max_tau != (n_ref_steps * ref_step_size):
            last_step = max_tau - (n_ref_steps * ref_step_size)
            n_ref_steps += 1
        else:
            last_step = ref_step_size
        all_steps = ref_step_size * torch.ones(n_ref_steps).to(self.vector_field.device)
        all_steps[-1] = last_step
        particle_pos = self.positions.clone()
        valid_particles = torch.ones(particle_pos.shape[0], dtype=torch.bool, device=self.vector_field.device)
        const_t = torch.ones((particle_pos.shape[0], 1), dtype=particle_pos.dtype, device=particle_pos.device)

        ic = [self.positions.clone()]
        for _, step in enumerate(all_steps):
            if particle_pos.shape[1] == 4:
                t = particle_pos[0, 0]
                grid_t = self.vector_field.map_phyiscal_time_to_grid(t)
                self.vector_field.set_fields(np.floor(grid_t.item()))
            for idx in range(0, particle_pos.shape[0], 1000000):
                start = idx
                end = min(idx + 1000000, particle_pos.shape[0])
                vectors = self.vector_field.get_vectors(particle_pos[start:end, :])
                particle_pos[start:end, :] += step * torch.cat((const_t[start:end, :], vectors), dim=1)
                if self.vector_field.in_dim == 3:
                    particle_pos = self.vector_field.toroidal_map(particle_pos)
                current_valid_particles = self.is_valid(particle_pos[start:end, 1:])
                valid_particles[start:end] = torch.logical_and(valid_particles[start:end], current_valid_particles)
        ic.append(particle_pos.clone())

        return torch.stack(ic, dim=1), valid_particles

    def rk4_flowmap(self, step_size: float = 0.25) -> Tuple[Tensor, Tensor]:
        """Computes the flow map using rk4 integration. Each particle advects 4 times within a single grid-voxel
        and the time-span of integration for all the particles is expected to be the same, if not same then all
        the particles are integrated for the maximum time-span.

        Returns:
            Tuple[Tensor, Tensor]: Flow map and a mask indicating wether a given particle stayed in the domain for
            the entire integration time-span.
        """
        print("step size : ", step_size)
        max_tau = self.tau.max() if isinstance(self.tau, Tensor) else self.tau
        ref_step_size = step_size * self.vector_field.dt
        n_ref_steps = int(max_tau // ref_step_size)
        if max_tau != (n_ref_steps * ref_step_size):
            last_step = max_tau - (n_ref_steps * ref_step_size)
            n_ref_steps += 1
        else:
            last_step = ref_step_size
        all_steps = ref_step_size * torch.ones(n_ref_steps).to(self.vector_field.device)
        all_steps[-1] = last_step
        particle_pos = self.positions.clone()
        valid_particles = torch.ones(particle_pos.shape[0], dtype=torch.bool, device=self.vector_field.device)
        const_t = torch.ones((particle_pos.shape[0], 1), dtype=particle_pos.dtype, device=particle_pos.device)

        ic = [self.positions.clone()]
        for _, step in enumerate(all_steps):
            if particle_pos.shape[1] == 4:
                t = particle_pos[0, 0]
                grid_t = self.vector_field.map_phyiscal_time_to_grid(t)
                self.vector_field.set_fields(np.floor(grid_t.item()))
            for idx in range(0, particle_pos.shape[0], 1000000):
                start = idx
                end = min(idx + 1000000, particle_pos.shape[0])
                k1 = torch.cat(
                    (const_t[start:end, :], self.vector_field.get_vectors(particle_pos[start:end, :])), dim=1
                )
                y1 = particle_pos[start:end, :] + 0.5 * step * k1
                if self.vector_field.in_dim == 3:
                    y1 = self.vector_field.toroidal_map(y1)
                y1_valid_particles = self.is_valid(y1[start:end, 1:])

                k2 = torch.cat((const_t[start:end, :], self.vector_field.get_vectors(y1)), dim=1)
                y2 = particle_pos[start:end, :] + 0.5 * step * k2
                if self.vector_field.in_dim == 3:
                    y2 = self.vector_field.toroidal_map(y2)
                y2_valid_particles = self.is_valid(y2[start:end, 1:])

                k3 = torch.cat((const_t[start:end, :], self.vector_field.get_vectors(y2)), dim=1)
                y3 = particle_pos[start:end, :] + step * k3
                if self.vector_field.in_dim == 3:
                    y3 = self.vector_field.toroidal_map(y3)
                y3_valid_particles = self.is_valid(y3[start:end, 1:])

                k4 = torch.cat((const_t[start:end, :], self.vector_field.get_vectors(y3)), dim=1)
                particle_pos[start:end, :] += (step / 6) * (k1 + 2 * k2 + 2 * k3 + k4)

                if self.vector_field.in_dim == 3:
                    particle_pos = self.vector_field.toroidal_map(particle_pos)
                current_valid_particles = self.is_valid(particle_pos[start:end, 1:])
                all_valid_particles = torch.stack(
                    (
                        y1_valid_particles,
                        y2_valid_particles,
                        y3_valid_particles,
                        current_valid_particles,
                    )
                ).all(dim=0)
                valid_particles[start:end] = torch.logical_and(valid_particles[start:end], all_valid_particles)
        ic.append(particle_pos.clone())

        return torch.stack(ic, dim=1), valid_particles

    def neural_flowmap(self, net: DecoupledTauNet, grid_steps: int) -> Tensor:
        """Computes the flow map using the network

        Args:
            net (DecoupledTauNet): network with trained weights
            grid_steps (int): number of max steps to be taken (in grid units) to form a composed flow map

        Returns:
            Tensor: flow map
        """
        max_tau = self.tau.max() if isinstance(self.tau, Tensor) else self.tau
        step_size = grid_steps * self.vector_field.dt
        n_steps = int(max_tau // step_size)
        if self.tau != (n_steps * step_size):
            last_step = self.tau - (n_steps * step_size)
            n_steps += 1
        else:
            last_step = step_size

        all_steps = step_size * torch.ones(n_steps).to(self.vector_field.device)
        all_steps[-1] = last_step
        neural_positions = copy.deepcopy(self.positions)

        for idx, step in enumerate(all_steps):
            tau_coord = step * torch.ones(neural_positions.shape[0], 1).to(self.positions.device)
            for idx in range(0, neural_positions.shape[0], 1000000):
                start = idx
                end = min(idx + 1000000, neural_positions.shape[0])
                with torch.inference_mode():
                    neural_spatial_position = net(
                        neural_positions[start:end, 1:],
                        self.vector_field.map_physical_to_net(neural_positions[start:end]),
                        tau_coord[start:end],
                    )
                neural_positions[start:end, 0] += step
                neural_positions[start:end, 1:] = neural_spatial_position.clone()
                if self.vector_field.in_dim == 3:
                    neural_positions = self.vector_field.toroidal_map(neural_positions)

        return neural_positions

    def irregular_euler_integration_flow_map(self) -> Tensor:
        """Computes flow maps using euler integration for varying integration duration

        Returns:
            Tensor: flow map
        """
        n_particles = self.positions.shape[0]
        particle = torch.zeros_like(self.positions)
        particle += self.positions

        time_eps = 0.25 * self.vector_field.dt
        n_steps = (self.tau // time_eps).long() + 1
        max_step = n_steps.max()
        all_taus = time_eps * torch.ones(n_particles, max_step).to(self.positions.device)
        all_taus[torch.arange(n_particles), n_steps - 1] = self.tau - ((n_steps - 1).float() * time_eps)

        valid_particles = torch.ones(self.positions.shape[0], dtype=torch.bool).to(self.positions.device)
        for tdx in range(max_step):
            temporal_mask = torch.nonzero(tdx < n_steps).squeeze(dim=1)
            active_p = particle[temporal_mask]
            active_tau = all_taus[temporal_mask][:, tdx]
            active_n = active_p.shape[0]

            spatial_vf = self.vector_field.get_vectors(active_p)
            constant_t = torch.ones(active_n, 1).to(self.positions.device)
            explicit_euler = torch.cat((constant_t, spatial_vf), dim=1)
            particle[temporal_mask] = particle[temporal_mask] + active_tau.unsqueeze(1) * explicit_euler
            current_valid_particles = self.is_valid(particle[:, 1:])
            valid_particles = torch.logical_and(valid_particles, current_valid_particles)
        return particle, valid_particles

    def neural_irregular_integration_flow_map(self, net: DecoupledTauNet, grid_steps: int) -> Tensor:
        """Computes flow maps using the network for varying integration duration

        Returns:
            Tensor: flow map
        """
        n = self.positions.shape[0]
        particle = torch.zeros_like(self.positions)
        particle += self.positions
        grid_tau = grid_steps * self.vector_field.dt

        n_steps = (self.tau // grid_tau).long() + 1
        max_step = n_steps.max()
        all_taus = grid_tau * torch.ones(n, max_step).to(self.positions.device)
        all_taus[torch.arange(n), n_steps - 1] = self.tau - ((n_steps - 1).float() * grid_tau)

        for tdx in range(max_step):
            temporal_mask = torch.nonzero(tdx < n_steps).squeeze(dim=1)
            active_p = particle[temporal_mask]
            active_tau = all_taus[temporal_mask][:, tdx]

            with torch.no_grad():
                next_out = net(
                    active_p, self.vector_field.map_physical_to_net(active_p), active_tau.unsqueeze(1)
                ).detach()
            particle[temporal_mask, 0] += active_tau
            particle[temporal_mask, 1:] = next_out

        return particle

    def toroidal_subtract(self, p: Tensor, q: Tensor) -> Tensor:
        """Computes the difference between two batches of particles taking periodic boundary conditions into account.
        Domain is hard-coded for fluid simulation dataset.

        Args:
            p (Tensor): position
            q (Tensor): position

        Returns:
            Tensor: Difference between p and q
        """
        main_diff = torch.zeros_like(p)
        normal_diff = p - q

        less_q = q < 0.5
        greater_q = ~less_q
        reflected_q = q.clone()
        reflected_q[less_q] = 1 + q[less_q]
        reflected_q[greater_q] = -1 + q[greater_q]

        reflected_diff = p - reflected_q
        diff_mask = normal_diff.abs() < reflected_diff.abs()
        main_diff[diff_mask] = normal_diff[diff_mask]
        main_diff[~diff_mask] = reflected_diff[~diff_mask]
        return main_diff

    def toroidal_central_difference(self, x: np.ndarray, dx: float, dy: float) -> Union[np.ndarray, np.ndarray]:
        """Computes the gradient using 1st order central differencing scheme taking periodic boundary condition
        into account.

        Args:
            x (np.ndarray): positions
            dx (float): x-axis spacing
            dy (float): y-axis spacing

        Returns:
            Union[np.ndarray, np.ndarray]: gradient with respect to the spatial dimension.
        """
        x = torch.from_numpy(x)
        g_x = torch.zeros_like(x)
        g_y = torch.zeros_like(x)
        for i in range(x.shape[0]):
            if i == 0:
                g_x[i, :] = self.toroidal_subtract(x[i + 1, :], x[i, :]) / dx
                g_y[:, i] = self.toroidal_subtract(x[:, i + 1], x[:, i]) / dy
            elif i == x.shape[0] - 1:
                g_x[i, :] = self.toroidal_subtract(x[i, :], x[i - 1, :]) / dx
                g_y[:, i] = self.toroidal_subtract(x[:, i], x[:, i - 1]) / dy
            else:
                g_x[i, :] = (self.toroidal_subtract(x[i + 1, :], x[i - 1, :]) / dx) / 2
                g_y[:, i] = (self.toroidal_subtract(x[:, i + 1], x[:, i - 1]) / dy) / 2
        return g_x.numpy(), g_y.numpy()

    def ftle(self, flow_map: Union[Tensor, np.ndarray], valid_mask: Tensor, tau: float) -> np.ndarray:
        """Computes the FTLE field for 2D-time-varying datasets

        Args:
            flow_map (Union[Tensor, np.ndarray]): Flow map positions
            valid_mask (Tensor): valid mask for particles indicating if the particles stayed within
                the domain during the integrations.
            tau (float): time-span of integration

        Returns:
            np.ndarray: FTLE field

        Usage:
            # set your device
            device = "cuda"

            # define the data class
            vf = VectorField2D("data/cylinder2d", device)

            # get seed positions for the FTLE
            lattice = vf.get_lattice().view(-1, vf._out_dim)
            t_coords = torch.zeros(lattice.shape[0], 1)
            positions = torch.cat((t_coords, lattice), dim=1).to(device)

            flowmap_obj = Flowmap(vf, positions, 1)

            # compute reference flow map
            ref_flow_map, mask = flowmap_obj.euler_flowmap()

            # compute ftle
            ftle_ = flowmap_obj.ftle(flow_map[:, -1, 1:], mask, 9)
            ftle_ = ftle_.reshape(vf.res[1:], order="F")

            Check ftle_eval.py for detailed usage.
        """
        if isinstance(flow_map, Tensor):
            flow_map = flow_map.view(self.vector_field.res[1], self.vector_field.res[2], self.vector_field._out_dim)
            flow_map = flow_map.detach().cpu().numpy() if flow_map.is_cuda else flow_map.numpy()
        else:
            flow_map = flow_map.reshape(self.vector_field.res[1], self.vector_field.res[2], self.vector_field._out_dim)

        if self.vector_field.is_toroidal:
            dfu_dy, dfu_dx = self.toroidal_central_difference(
                flow_map[:, :, 0], self.vector_field.dy, self.vector_field.dx
            )
            dfv_dy, dfv_dx = self.toroidal_central_difference(
                flow_map[:, :, 1], self.vector_field.dy, self.vector_field.dx
            )
        else:
            dfu_dy, dfu_dx = np.gradient(flow_map[:, :, 0], self.vector_field.dy, self.vector_field.dx)
            dfv_dy, dfv_dx = np.gradient(flow_map[:, :, 1], self.vector_field.dy, self.vector_field.dx)

        du = np.stack((dfu_dx, dfu_dy), axis=0).reshape(2, -1, order="F")
        dv = np.stack((dfv_dx, dfv_dy), axis=0).reshape(2, -1, order="F")
        jacobian = np.stack((du, dv), axis=0)
        jacobian = torch.from_numpy(jacobian).permute(2, 0, 1)
        cauchy_green_tensor = torch.bmm(torch.transpose(jacobian, 1, 2), jacobian)
        eig_val, _ = torch.linalg.eigh(cauchy_green_tensor)
        ftle = torch.log(torch.sqrt(torch.max(eig_val, dim=1).values))
        
        return ftle.numpy()

    def ftle3d(self, flow_map: Union[Tensor, np.ndarray], valid_mask: Tensor, tau: float) -> np.ndarray:
        """Computes the FTLE field for 3D-time-varying datasets

        Args:
            flow_map (Union[Tensor, np.ndarray]): Flow map positions
            valid_mask (Tensor): valid mask for particles indicating if the particles stayed within the
                domain during the integrations.
            tau (float): time-span of integration

        Returns:
            np.ndarray: FTLE field
        """
        if isinstance(flow_map, Tensor):
            flow_map = flow_map.view(
                self.vector_field.res[1], self.vector_field.res[2], self.vector_field.res[3], self.vector_field._out_dim
            )
            flow_map = flow_map.detach().cpu().numpy() if flow_map.is_cuda else flow_map.numpy()
        else:
            flow_map = flow_map.reshape(
                self.vector_field.res[1], self.vector_field.res[2], self.vector_field.res[3], self.vector_field._out_dim
            )

        dfu_dz, dfu_dy, dfu_dx = np.gradient(
            flow_map[:, :, :, 0], self.vector_field.dz, self.vector_field.dy, self.vector_field.dx
        )
        dfv_dz, dfv_dy, dfv_dx = np.gradient(
            flow_map[:, :, :, 1], self.vector_field.dz, self.vector_field.dy, self.vector_field.dx
        )
        dfw_dz, dfw_dy, dfw_dx = np.gradient(
            flow_map[:, :, :, 2], self.vector_field.dz, self.vector_field.dy, self.vector_field.dx
        )

        du = np.stack((dfu_dx, dfu_dy, dfu_dz), axis=0).reshape(3, -1, order="F")
        dv = np.stack((dfv_dx, dfv_dy, dfv_dz), axis=0).reshape(3, -1, order="F")
        dw = np.stack((dfw_dx, dfw_dy, dfw_dz), axis=0).reshape(3, -1, order="F")

        jacobian = np.stack((du, dv, dw), axis=0)
        jacobian = torch.from_numpy(jacobian).permute(2, 0, 1)
        cauchy_green_tensor = torch.bmm(torch.transpose(jacobian, 1, 2), jacobian)
        eig_val, _ = torch.linalg.eigh(cauchy_green_tensor)
        ftle = (1 / tau) * torch.log(torch.sqrt(torch.max(eig_val, dim=1).values))

        return ftle.numpy()

    def reference_streaklines(self, t_res: int) -> Tensor:
        """Computes the streaklines of particles.

        Args:
            t_res (int): resolution of the streakline

        Returns:
            Tensor: streaklines

        Usage:
            # set your device
            device = "cuda"

            # define the data class
            vf = VectorField2D("data/cylinder2d", device)

            # choose the number of particles to compute the streaklines
            n = 150

            # set your start_time
            t = 15

            # compute your seed positions
            my_seeds = torch.rand((2, n), dtype=torch.float32)
            my_seeds = (1 - my_seeds) * vf.th_ext[:, 0] + my_seeds * vf.th_ext[:, 1]
            seeds = torch.ones(my_seeds.shape[1], 3).to(device)
            seeds[:, 0] = t
            seeds[:, 1] = my_seeds[0, :]
            seeds[:, 2] = my_seeds[1, :]

            # set streakline res and the time-plane where all the streaklines would exist
            t_res = 1000
            max_tau = 2

            # compute the streaklines lines
            flowmap_obj = Flowmap(vf, seeds, max_tau)
            ref_streaklines = flowmap_obj.reference_streaklines(t_res)
        """
        self.t_res = t_res
        n = self.positions.shape[0]
        start_time = self.positions[0, 0].item()
        target_tau = self.tau
        all_seeds = torch.zeros(n, t_res, self.vector_field.in_dim)
        all_seeds[:, :, 1:] = self.positions[:, 1:].unsqueeze(1)
        all_seeds[:, :, 0] = (2 * start_time + target_tau) - torch.linspace(
            start_time, start_time + target_tau, t_res
        ).unsqueeze(0).to(
            self.positions.device
        )  # times at which to seed
        bad_seeds = all_seeds.view(-1, self.vector_field.in_dim)  # all seeds to integrate
        all_tau = (start_time + target_tau) - bad_seeds[:, 0]  # integration duration
        self.tau = all_tau.to(self.positions.device)
        self.positions = bad_seeds.to(self.positions.device)
        full_integration, valid_integration = self.irregular_euler_integration_flow_map()
        all_streaklines = full_integration.view(n, t_res, self.vector_field.in_dim)
        all_streaklines[:, :, 0] = (2 * start_time + target_tau) - torch.linspace(
            start_time, start_time + target_tau, t_res
        ).unsqueeze(0).to(self.positions.device)
        valid_seeds = valid_integration.view(n, t_res).all(dim=1)

        return all_streaklines[valid_seeds]

    def neural_streaklines(self, t_res: int) -> Tensor:
        """Computes the streaklines of particles using neural network.

        Args:
            t_res (int): resolution of the streakline

        Returns:
            Tensor: streaklines
        """
        self.t_res = t_res
        n = self.positions.shape[0]
        start_time = self.positions[0, 0].item()
        target_tau = self.tau
        all_seeds = torch.zeros(n, t_res, self.vector_field.in_dim)
        all_seeds[:, :, 1:] = self.positions[:, 1:].unsqueeze(1)
        all_seeds[:, :, 0] = (2 * start_time + target_tau) - torch.linspace(
            start_time, start_time + target_tau, t_res
        ).unsqueeze(0).to(
            self.positions.device
        )  # times at which to seed
        bad_seeds = all_seeds.view(-1, self.vector_field.in_dim)  # all seeds to integrate
        all_tau = (start_time + target_tau) - bad_seeds[:, 0]  # integration duration
        self.tau = all_tau.to(self.positions.device)
        self.positions = bad_seeds.to(self.positions.device)
        full_integration = self.neural_irregular_euler_integration_flow_map()
        all_streaklines = full_integration.view(n, t_res, self.vector_field.in_dim)
        all_streaklines[:, :, 0] = (2 * start_time + target_tau) - torch.linspace(
            start_time, start_time + target_tau, t_res
        ).unsqueeze(0).to(self.positions.device)

        return all_streaklines


if __name__ == "__main__":
    from pathlib import Path

    # import matplotlib.pyplot as plt
    # from pyevtk.hl import imageToVTK

    vf = VectorField2D("F:/Projects/neural_flowmap_representation/flowmap-inr/data/cylinder2d", "cuda")
    # lattice = vf.get_lattice().view(-1, vf._out_dim)
    # t_coords = torch.zeros(lattice.shape[0], 1)
    # positions = torch.cat((t_coords, lattice), dim=1).to("cuda")
    # print(positions.shape)
    # flowmap_obj = Flowmap(vf, positions, 9)
    # print("computing flow map")
    # flow_map, mask = flowmap_obj.euler_flowmap()
    # print("flow map computation is done")

    # ftle_ = flowmap_obj.ftle(flow_map[:, -1, 1:], mask, 9)
    # ftle_ = ftle_.reshape(vf.res[1:], order="F")
    # # imageToVTK("test_3d_ftle", pointData={"u": ftle_})
    # plt.imshow(ftle_.T, cmap='RdYlBu_r')
    # plt.xlim((0, vf.res[1]))
    # plt.ylim((0, vf.res[2]))
    # plt.show()
    min_sim_dim = torch.tensor([[1, -0.4]]).T
    max_sim_dim = torch.tensor([[5, 0.4]]).T
    my_seeds = torch.rand((2, 150), dtype=torch.float32)
    my_seeds = (1 - my_seeds) * min_sim_dim + my_seeds * max_sim_dim
    # torch.save(my_seeds, "final_boss_seeds.pt")
    seeds = torch.ones(my_seeds.shape[1], 3).cuda()
    seeds[:, 0] = 16
    seeds[:, 1] = my_seeds[0, :]
    seeds[:, 2] = my_seeds[1, :]
    flowmap_obj = Flowmap(vf, seeds, 2)

    ref_streaklines = flowmap_obj.reference_streaklines(1000)
    Path("test").mkdir(parents=True, exist_ok=True)
    for idx in range(ref_streaklines.shape[1]):
        file_name = Path("test", f"streakline_{idx}.csv")
        csv_file = open(file_name, "w")
        csv_file.write("x,y,z,id,t_id\n")
        constant_z = 1
        valid_id = torch.arange(ref_streaklines.shape[0])
        for ldx in range(ref_streaklines.shape[0]):
            dist = torch.sqrt(
                (seeds[ldx, 1] - ref_streaklines[ldx, idx, 1]) ** 2
                + (seeds[ldx, 2] - ref_streaklines[ldx, idx, 2]) ** 2
            )
            csv_file.write(
                f"{ref_streaklines[ldx, idx, 1]}, \
                    {ref_streaklines[ldx,idx, 2]}, \
                        {constant_z},{valid_id[ldx].item()},{dist}\n"
            )
    csv_file.close()
