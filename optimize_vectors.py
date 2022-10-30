import argparse
import json
import time
from pathlib import Path

import torch
from loguru import logger

from field import VectorField2D, VectorField3D
from model import DecoupledTauNet
from objectives import vector_objective
from utils import compute_all_R

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", required=True, help="dir to the dataset")
    parser.add_argument("--experiment", required=True, help="name of experiment")
    parser.add_argument("--n_dim", type=int, required=True, choices=[2, 3], help="2d or 3d time-varying dataset")

    parser.add_argument("--n_layers", type=int, default=3, help="number of residual layers")
    parser.add_argument("--n_vector_layers", type=int, default=2, help="number of f_v layers")
    parser.add_argument("--n_levels", type=int, default=4, help="number of levels for feature grid")
    parser.add_argument("--layer_width", type=int, default=64, help="layer width")
    parser.add_argument("--split_factor", type=int, default=1.65, help="split factor")

    parser.add_argument("--compression_ratio", type=int, default=10, help="compression ratio")
    parser.add_argument("--batch_size", type=int, default=50000, help="batch size")
    parser.add_argument("--n_opt_steps", type=int, default="40000", help="number of optimization steps")

    parser.add_argument(
        "--reset_iter", type=int, default=2000, help="reset the 3d timesteps loaded into the memory after n optim steps"
    )
    parser.add_argument("--device", default="cuda")

    opt = parser.parse_args()
    device = opt.device

    Path("Experiments", opt.experiment).mkdir(parents=True, exist_ok=True)
    EXP_DIR = Path("Experiments", opt.experiment)
    NET_INFO_FILENAME = "net_info.json"
    N_DIM = opt.n_dim

    vector_field = VectorField2D(opt.data_dir, opt.device) if N_DIM == 2 else VectorField3D(opt.data_dir, opt.device)

    # generate network params
    target_n = int((vector_field.size * opt.n_dim) / opt.compression_ratio)
    f_size = compute_all_R(vector_field.in_dim, target_n, opt.n_levels, opt.layer_width, opt.split_factor)
    grid_d = int(opt.layer_width // (2 * opt.n_levels))
    f_dim = [grid_d for _ in range(opt.n_levels)]
    logger.info(vector_field.in_dim)

    # network
    net = DecoupledTauNet(
        vector_field.in_dim, opt.layer_width, vector_field.out_dim, f_dim, f_size, opt.n_layers, n_dim=N_DIM
    )
    net.to(device)
    n_params = sum(p.numel() for p in net.parameters() if p.requires_grad)
    logger.info(f"Number of Network Parameters : {n_params}")

    network_info = {}
    network_info["filename"] = opt.experiment + "_vector.pth"
    network_info["in_dim"] = vector_field.in_dim
    network_info["out_dim"] = vector_field.out_dim
    network_info["layer_width"] = opt.layer_width
    network_info["f_size"] = f_size
    network_info["f_dim"] = f_dim

    network_info["n_layers"] = net.n_layers
    network_info["n_opt_steps"] = opt.n_opt_steps

    network_info["experiment"] = opt.experiment
    network_info["optimization_time"] = 0

    # learning rate
    grid_lr = 2e-2
    mlp_lr = 1e-2

    # Seperate optimizer for latent grid and other parameters
    grid_optimizer = torch.optim.Adam(
        [p for name, p in net.named_parameters() if "vec_grid" in name], lr=grid_lr, betas=(0.9, 0.99), eps=1e-15
    )
    mlp_optimizer = torch.optim.Adam(
        [
            p
            for name, p in net.named_parameters()
            if "vec_feature_mapping" in name or "vec_pe" in name or "vec_scale_param" in name or "last_layer" in name
        ],
        weight_decay=0,
    )

    training_start_time = time.perf_counter()
    for idx in range(opt.n_opt_steps):
        t1 = time.perf_counter()
        mlp_optimizer.zero_grad()
        grid_optimizer.zero_grad()

        if idx % opt.reset_iter == 0 and N_DIM == 3:
            if idx > 0:
                vector_field.batch_fields = None
            vector_field.reset_time_idx()
            vector_field.load_batch_fields()

        vec_loss = vector_objective(opt, vector_field, net)
        vec_loss.backward()

        grid_optimizer.step()
        mlp_optimizer.step()

        if (idx + 1) % 10 == 0:
            logger.info(f"Training iteration : [{idx}], Vector Loss : {vec_loss.item()}")

        if (idx + 1) % (opt.n_opt_steps // 5) == 0:
            logger.info("LR DECAY")
            for params in mlp_optimizer.param_groups:
                params["lr"] = params["lr"] / 2

            for params in grid_optimizer.param_groups:
                params["lr"] = params["lr"] / 2

        if (idx + 1) % 5000 == 0:
            torch.save(net.state_dict(), Path(EXP_DIR, network_info["filename"]))
    training_end_time = time.perf_counter()

    torch.save(net.state_dict(), Path(EXP_DIR, network_info["filename"]))

    network_info["optimization_time"] = int(training_end_time - training_start_time)
    json.dump(network_info, open(Path(EXP_DIR, NET_INFO_FILENAME), "w"))
