import argparse
import json
import time
from pathlib import Path

import matplotlib.pyplot as plt
import torch
from loguru import logger
from pyevtk.hl import imageToVTK

from field import VectorField2D, VectorField3D
from flow_map import Flowmap
from model import DecoupledTauNet

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
    if net_info["in_dim"] == 3:
        vector_field = VectorField2D(opt.data_dir, device=device)
    else:
        vector_field = VectorField3D(opt.data_dir, device=device)

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
    tick = time.perf_counter()
    ref_flow_map, mask = fp.euler_flowmap()
    tock = time.perf_counter()
    logger.info(f"Referece Flow Map computation Time : {tock-tick}")

    del fp.positions
    fp.positions = ref_flow_map[:, 0].clone()
    tick = time.perf_counter()
    neural_flow_map = fp.neural_flowmap(net, opt.grid_steps)
    tock = time.perf_counter()

    logger.info(f"Neural Flow Map computation Time : {tock-tick}")

    if vector_field.out_dim == 3:
        ref_ftle_ = fp.ftle3d(ref_flow_map[:, -1, 1:], mask, opt.tau)
        ref_ftle_ = ref_ftle_.reshape(vector_field.res[1:], order="F")
        imageToVTK(f"Experiments/{opt.experiment}/ref_ftle", pointData={"u": ref_ftle_})
    else:
        ref_ftle_ = fp.ftle(ref_flow_map[:, -1, 1:], mask, opt.tau)
        ref_ftle_ = ref_ftle_.reshape(vector_field.res[1:], order="F")
        plt.figure()
        plt.xlim((0, ref_ftle_.shape[0]))
        plt.ylim((0, ref_ftle_.shape[1]))
        plt.tick_params(left=False, right=False, labelleft=False, labelbottom=False, bottom=False)
        plt.imshow(ref_ftle_.T, cmap="RdYlBu_r", vmin=0)
        plt.colorbar()
        plt.savefig(f"Experiments/{opt.experiment}/ref_ftle.png", format="png", dpi=300)

    if vector_field.out_dim == 3:
        neural_ftle_ = fp.ftle3d(neural_flow_map[:, 1:], mask, opt.tau)
        neural_ftle_ = neural_ftle_.reshape(vector_field.res[1:], order="F")
        imageToVTK(f"Experiments/{opt.experiment}/neural_ftle", pointData={"u": neural_ftle_})
    else:
        neural_ftle_ = fp.ftle(neural_flow_map[:, 1:], mask, opt.tau)
        neural_ftle_ = neural_ftle_.reshape(vector_field.res[1:], order="F")
        plt.figure()
        plt.xlim((0, neural_ftle_.shape[0]))
        plt.ylim((0, neural_ftle_.shape[1]))
        plt.tick_params(left=False, right=False, labelleft=False, labelbottom=False, bottom=False)
        plt.imshow(neural_ftle_.T, cmap="RdYlBu_r", vmin=0)
        plt.colorbar()
        plt.savefig(f"Experiments/{opt.experiment}/neural_ftle.png", format="png", dpi=300)
