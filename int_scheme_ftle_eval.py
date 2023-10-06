import argparse
import json
import time
from pathlib import Path
import numpy as np

import matplotlib.pyplot as plt
import torch
from loguru import logger
from pyevtk.hl import imageToVTK

from field import VectorField2D, VectorField3D
from flow_map import Flowmap
from model import DecoupledTauNet

def save_fig(fig: np.ndarray, save_filename: str, cmap: str = "RdYlBu_r"):
    plt.figure()
    plt.xlim((0, fig.shape[0]))
    plt.ylim((0, fig.shape[1]))
    plt.tick_params(left=False, right=False, labelleft=False, labelbottom=False, bottom=False)
    plt.imshow(fig.T, cmap=cmap, vmin=0)
    plt.colorbar()
    plt.savefig(f"Experiments/{opt.experiment}/{save_filename}.png", format="png", dpi=300)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", required=True, help="dir to the dataset")
    parser.add_argument("--experiment", required=True, help="name of experiment")
    parser.add_argument("--start_time", required=True, type=float, help="start time of integration")
    parser.add_argument("--tau", required=True, type=float, help="duration of integration")
    parser.add_argument(
        "--grid_steps", required=True, type=float, default=16, help="number steps taken in grid units for the network"
    )
    parser.add_argument("--device", default="cuda", help="device", choices=["cuda", "cpu"])

    opt = parser.parse_args()
    device = opt.device

    net_info = json.load(open(Path("Experiments", opt.experiment, "net_flow_info.json"), "r"))
    vector_field = VectorField2D(opt.data_dir, device=device)

    net = DecoupledTauNet(
        net_info["in_dim"],
        net_info["layer_width"],
        net_info["out_dim"],
        net_info["f_dim"],
        net_info["f_size"],
        net_info["n_layers"],
        n_dim=vector_field.out_dim,
    )
    net.to(device)
    net.load_state_dict(torch.load(Path("Experiments", opt.experiment, net_info["filename"]), map_location="cuda"))
    net.eval()
    lattice = vector_field.get_lattice().view(-1, vector_field.out_dim).to(device)
    t_coords = torch.full((lattice.shape[0], 1), fill_value=opt.start_time).to(device)
    positions = torch.cat((t_coords, lattice), dim=1)

    fp = Flowmap(vector_field, positions.clone(), opt.tau)
    ref_flow_map, mask = fp.euler_flowmap(step_size=0.1)
    del fp

    vector_field = VectorField2D(opt.data_dir, cf=2, device=device)
    fp = Flowmap(vector_field, positions.clone(), opt.tau)
    rk4_flow_map, _ = fp.rk4_flowmap(step_size=opt.grid_steps)
    del fp

    rk_error_img = torch.norm(ref_flow_map[:, -1, 1:] - rk4_flow_map[:, -1, 1:], dim=1)/vector_field.diag
    rk_error_img = rk_error_img.reshape(vector_field.res[1:]).detach().cpu().numpy()
    

    fp = Flowmap(vector_field, positions.clone(), opt.tau)
    euler_flow_map, _ = fp.euler_flowmap(step_size=opt.grid_steps)
    del fp

    euler_error_img = torch.norm(ref_flow_map[:, -1, 1:] - euler_flow_map[:, -1, 1:], dim=1)/vector_field.diag
    euler_error_img = euler_error_img.reshape(vector_field.res[1:]).detach().cpu().numpy()

    vector_field = VectorField2D(opt.data_dir, device=device)
    fp = Flowmap(vector_field, positions.clone(), opt.tau)
    fp.positions = ref_flow_map[:, 0].clone()
    neural_flow_map = fp.neural_flowmap(net, opt.grid_steps)

    neural_error_img = torch.norm(ref_flow_map[:, -1, 1:] - neural_flow_map[:, 1:], dim=1)/vector_field.diag
    neural_error_img = neural_error_img.reshape(vector_field.res[1:]).detach().cpu().numpy()

    ref_ftle_ = fp.ftle(ref_flow_map[:, -1, 1:], mask, opt.tau)
    save_fig(ref_ftle_, "ref_ftle")

    rk4_ftle_ = fp.ftle(rk4_flow_map[:, -1, 1:], mask, opt.tau)
    save_fig(rk4_ftle_, "rk4_ftle")
    save_fig(rk_error_img, "rk_error", cmap="viridis")

    euler_ftle_ = fp.ftle(euler_flow_map[:, -1, 1:], mask, opt.tau)
    save_fig(euler_ftle_, "euler_ftle")
    save_fig(euler_error_img, "euler_error", cmap="viridis")

    neural_ftle_ = fp.ftle(neural_flow_map[:, 1:], mask, opt.tau)
    save_fig(neural_ftle_, "neural_ftle")
    save_fig(neural_error_img, "neural_error", cmap="viridis")

    
