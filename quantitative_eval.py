import argparse
import copy
import json
from pathlib import Path

import numpy as np
import torch
from loguru import logger

from field import VectorField2D, VectorField3D
from flow_map import Flowmap
from model import DecoupledTauNet

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", required=True, help="dir to the dataset")
    parser.add_argument("--experiment", required=True, help="name of experiment")
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
    net.to("cuda")
    net.load_state_dict(torch.load(Path("Experiments", opt.experiment, net_info["filename"]), map_location="cuda"))
    net.eval()
    n = 20000
    tau_steps = np.arange(1, int(net_info["max_tau"]), 2)
    all_tau = [2, 4, 6]  # hard-coded for different datasets
    eval_data = []

    for tau in all_tau:
        start_times = np.linspace(vector_field.ext[0][0], vector_field.ext[0][1] - tau, 8)
        for st in start_times:
            pos = torch.rand(n, vector_field.in_dim, device="cuda")
            pos[:, 0] = st
            for idx in range(vector_field.in_dim - 1):
                pos[:, idx + 1] = vector_field.ext[idx + 1][0] + (
                    vector_field.ext[idx + 1][1] - vector_field.ext[idx + 1][0]
                ) * torch.rand(n).to(device)

            ref_fp = Flowmap(vector_field, copy.deepcopy(pos), tau)
            ref_flowmap, ref_valid_mask = ref_fp.euler_flowmap()
            ref_flowmap = ref_flowmap[ref_valid_mask, :, :]

            for tau_step in tau_steps:
                seeds = ref_flowmap[:, 0].clone()
                seeds = seeds.to(device)
                neural_fp = Flowmap(vector_field, seeds, tau)
                neural_flowmap = neural_fp.neural_flowmap(net, tau_step)

                error = (ref_flowmap[:, -1, 1:] - neural_flowmap[:, 1:]).norm(dim=1) / vector_field.diag
                logger.info(
                    f"Start time : {st}, Duration: {tau}, Grid Steps: {tau_step}, Mean Error: {error.mean().item()}"
                )
                metrics = dict()
                metrics["t"] = float(st)
                metrics["target_tau"] = float(tau)
                metrics["net_tau_units"] = int(tau_step)
                metrics["mean_err"] = float(error.mean().item())
                metrics["experiment"] = net_info
                eval_data.append(metrics)
    json.dump(eval_data, open(Path("Experiments", opt.experiment, f"{opt.experiment}_eval.json"), "w"))
