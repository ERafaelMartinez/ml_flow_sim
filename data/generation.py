"""
This module is in charge of generating a dataset
for training a ML flow-simulation accelerator.
For this, it creates a set of input files for the
num-sim program and runs it to generate the output files.
"""
import os

import numpy as np
import torch

import utils
from utils import SimulationParameters

def generate_simulation_datapoint(re: int, u_bound: float) -> (torch.Tensor, torch.Tensor):
    """Generate a single simulation datapoint"""
    print(f"Generating simulation datapoint for re={re}, u_bound={u_bound}")

    # create a working directory
    re: int = int(re)
    work_path = f"./temp/{re}"
    os.makedirs(work_path, exist_ok=True)

    # write the parameters file
    utils.write_params_file(re, u_bound, f"{work_path}/params.txt")

    # call the simulation
    utils.call_simulation(work_path)

    # clean the simulation
    utils.clean_simulation(work_path)

    # ensure sim_out exists
    if not os.path.exists("./sim_out"):
        os.makedirs("./sim_out")
    
    # return the output
    return utils.from_vti_to_pt(f"./sim_out/{re}_in.vti"), utils.from_vti_to_pt(f"./sim_out/{re}_out.vti")


def generate_dataset(sim_params: np.ndarray) -> (torch.Tensor, torch.Tensor):
    """Generate a dataset for training"""
    dataset_in = []
    dataset_out = []
    for re, u_bound in sim_params:
        print(f"Generating training datapoint for re={re}, u_bound={u_bound}")
        in_, out_ = generate_simulation_datapoint(re, u_bound)
        dataset_in.append(in_)
        dataset_out.append(out_)
    return torch.stack(dataset_in), torch.stack(dataset_out)


def generate_dataset_parallel(sim_params: np.ndarray) -> (torch.Tensor, torch.Tensor):
    """Generate a dataset for training in parallel"""
    # import multiprocessing
    from multiprocessing import Pool

    with Pool(os.cpu_count()) as pool:
        # generate the dataset in parallel
        dataset = pool.starmap(generate_simulation_datapoint, sim_params)
    
    return torch.stack([d[0] for d in dataset]), torch.stack([d[1] for d in dataset])
    

if __name__ == "__main__":
    n_data_points = 1
    dataset_name = "bc_ood_left"

    # reynolds number range
    SimulationParameters.re_min = 1000
    SimulationParameters.re_max = 1100
    
    # discretization
    SimulationParameters.n_cells_x = 20
    SimulationParameters.n_cells_y = 20

    # physical size
    SimulationParameters.phisical_size_x = 2
    SimulationParameters.phisical_size_y = 2

    # boundary conditions
    SimulationParameters.u_bound_left = 1
    SimulationParameters.u_bound_right = 0
    SimulationParameters.u_bound_top = 0
    SimulationParameters.u_bound_bottom = 0

    SimulationParameters.v_bound_left = 0
    SimulationParameters.v_bound_right = 0
    
    # create an even distribution of values between min and max re range
    step_size = (
        SimulationParameters.re_max - SimulationParameters.re_min
        ) // n_data_points  # we want to have n data points
    re = np.arange(
        SimulationParameters.re_min, SimulationParameters.re_max, step_size
        ).astype(int)
    u_bound = re / 1000

    # zip the values to use them together
    sim_params = np.column_stack((re, u_bound))

    # generate the dataset
    dataset_in, dataset_out = generate_dataset_parallel(sim_params)

    # remove temp directory
    os.system("rm -rf ./temp")

    # save the dataset
    torch.save(dataset_in, "./resources/dataset_" + dataset_name + "_in.pt")
    torch.save(dataset_out, "./resources/dataset_" + dataset_name + "_out.pt")
