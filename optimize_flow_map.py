import argparse
import json
import time
from pathlib import Path

import torch
from loguru import logger

from field import VectorField2D, VectorField3D
from model import DecoupledTauNet
from objectives import advected_objective, vector_objective

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", required=True, help="dir to the dataset")
    parser.add_argument("--n_dim", type=int, required=True, choices=[2, 3], help="2d or 3d time-varying dataset")

    parser.add_argument("--experiment", help="name of the experiment")
    parser.add_argument("--min_tau", type=float, default=0.2, help="min tau value for self supervision")
    parser.add_argument("--max_tau", type=float, default=48, help="min tau value for self supervision")
    parser.add_argument(
        "--sampling_strategy",
        default="sqrt",
        choices=["sqrt", "log", "single", "full"],
        help="sampling strategy for tau",
    )
    parser.add_argument("--batch_size", type=int, default=25000, help="batch size")
    parser.add_argument("--n_opt_steps", type=int, default="40000", help="number of optimization steps")
    parser.add_argument(
        "--reset_iter", type=int, default=2000, help="reset the 3d timesteps loaded into the memory after n optim steps"
    )

    parser.add_argument("--device", default="cuda")

    opt = parser.parse_args()
    device = opt.device
    EXP_DIR = Path("Experiments", opt.experiment)
    NET_INFO_PATH = Path(EXP_DIR, "net_info.json")
    NET_PATH = Path(EXP_DIR, f"{opt.experiment}_flow.pth")
    NET_INFO_FILENAME = "net_flow_info.json"

    N_DIM = opt.n_dim

    vector_field = VectorField2D(opt.data_dir, opt.device) if N_DIM == 2 else VectorField3D(opt.data_dir, opt.device)

    net_info = json.load(open(NET_INFO_PATH, "r"))
    net = DecoupledTauNet(
        net_info["in_dim"],
        net_info["layer_width"],
        net_info["out_dim"],
        net_info["f_dim"],
        net_info["f_size"],
        net_info["n_layers"],
        n_dim=N_DIM,
    )
    net.to(device)
    net.load_state_dict(torch.load(NET_PATH, map_location=device))

    net_info["filename"] = net_info["experiment"] + "_flow.pth"

    net_info["min_tau"] = opt.min_tau
    net_info["max_tau"] = opt.max_tau
    net_info["sampling_strategy"] = opt.sampling_strategy

    tau_lr = 1e-2
    tau_optimizer = torch.optim.Adam(
        [
            p
            for name, p in net.named_parameters()
            if "tau_grid" in name
            or "tau_scale_param" in name
            or "tau_pe" in name
            or "tau_feature_mapping" in name
            or "vec_mlp" in name
            or "tau_mlp" in name
            or "tau_params" in name
        ],
        lr=tau_lr,
        weight_decay=0,
    )

    grid_lr = 5e-4
    grid_optimizer = torch.optim.Adam(
        [
            p
            for name, p in net.named_parameters()
            if "vec_grid" in name
            or "vec_pe" in name
            or "vec_scale_param" in name
            or "vec_feature_mapping" in name
            or "last_layer" in name
        ],
        lr=grid_lr,
        weight_decay=0,
    )

    training_start_time = time.perf_counter()
    for idx in range(opt.n_opt_steps):
        alpha = 1 - ((idx + 1) / opt.n_opt_steps)
        tau_optimizer.zero_grad()
        grid_optimizer.zero_grad()

        if idx % opt.reset_iter == 0 and N_DIM == 3:
            if idx > 0:
                vector_field.batch_fields = None
            vector_field.reset_time_idx()
            vector_field.load_batch_fields()
        diff_loss = vector_objective(opt, vector_field, net)
        diff_loss.backward(retain_graph=True)

        advected_loss = advected_objective(opt, vector_field, net)
        advected_loss.backward(retain_graph=True)

        if (idx + 1) % 10 == 0:
            logger.info(f"itr : [{idx}], vector loss : {diff_loss.item()}, advected loss : {advected_loss.item()}")

        tau_optimizer.step()
        grid_optimizer.step()

        for params in grid_optimizer.param_groups:
            params["lr"] = alpha * grid_lr

        for params in tau_optimizer.param_groups:
            params["lr"] = alpha * tau_lr

        if (idx + 1) % 10000 == 0:
            torch.save(net.state_dict(), Path(EXP_DIR, net_info["filename"]))
    torch.save(net.state_dict(), Path(EXP_DIR, net_info["filename"]))
    training_end_time = time.perf_counter()

    net_info["optimization_time"] = int(training_end_time - training_start_time)
    json.dump(net_info, open(Path(EXP_DIR, NET_INFO_FILENAME), "w"))
