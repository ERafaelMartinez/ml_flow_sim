import os
import time
import subprocess

import torch


class SimulationParameters:
    re_min: int = 500
    re_max: int = 1500
    
    u_bound_left: int = 0
    u_bound_right: int = 0
    u_bound_top: int = 1
    u_bound_bottom: int = 0

    v_bound_left: int = 0
    v_bound_right: int = 0
    v_bound_top: int = 0
    v_bound_bottom: int = 0

    n_cells_x: int = 20
    n_cells_y: int = 20
    phisical_size_x: float = 2
    phisical_size_y: float = 2 

def get_params_text(re: float, u_bound: float) -> str:
    """Get the parameters as a string"""
    return (
f"""
# Settings file for numsim program
# Run ./numsim lid_driven_cavity.txt

# Problem description
physicalSizeX = {SimulationParameters.phisical_size_x}   # physical size of the domain
physicalSizeY = {SimulationParameters.phisical_size_y}
endTime = 10.0        # duration of the simulation
re = {re}
gX = 0.0              # external forces, set to (gX,gY) = (0,-9.81) to account for gravity
gY = 0.0

# Dirichlet boundary conditions
dirichletBottomX = {SimulationParameters.u_bound_bottom * u_bound}
dirichletBottomY = {SimulationParameters.v_bound_bottom * u_bound}
dirichletTopX    = {SimulationParameters.u_bound_top * u_bound}
dirichletTopY    = {SimulationParameters.v_bound_top * u_bound}
dirichletLeftX   = {SimulationParameters.u_bound_left * u_bound}
dirichletLeftY   = {SimulationParameters.v_bound_left * u_bound}
dirichletRightX  = {SimulationParameters.u_bound_right * u_bound}
dirichletRightY  = {SimulationParameters.v_bound_right * u_bound}

# Discretization parameters
nCellsX = {SimulationParameters.n_cells_x}          # number of cells in x and y direction
nCellsY = {SimulationParameters.n_cells_y}
useDonorCell = true   # if donor cell discretization should be used, possible values: true false
alpha = 0.5           # factor for donor-cell scheme, 0 is equivalent to central differences
tau = 0.5             # safety factor for time step width
maximumDt = 0.1       # maximum values for time step width

# Solver parameters
pressureSolver = SOR  # which pressure solver to use, possible values: GaussSeidel SOR CG
omega = 1.6           # overrelaxation factor, only for SOR solver
epsilon = 1e-5        # tolerance for 2-norm of residual
maximumNumberOfIterations = 1e4    # maximum number of iterations in the solver

"""
)

def write_params_file(
    re: float, u_bound: float, filename="params.txt"
    ) -> None:
    """Write the parameters to a file"""
    with open(filename, "w") as f:
        f.write(get_params_text(re, u_bound))

def call_simulation(work_path: str, filename="params.txt") -> None:
    """Call the simulation and await its completion"""
    # create a copy of the executable in the working directory
    os.system(f"cp ./resources/numsim_parallel {work_path}/numsim_parallel")

    # switch the working directory to the work path such that
    # the simulation is called from there
    base_path = os.getcwd()
    os.chdir(work_path)

    # call the simulation using subprocess to track its progress
    process = subprocess.Popen(
        ["mpiexec", "-n", "1", "numsim_parallel", filename],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE
    )

    # await completion
    while process.poll() is None:
        time.sleep(1)

    # check if the process has completed successfully
    if process.poll() is not None and process.returncode != 0:
        # print the error message
        print(process.stderr.read().decode())
        raise Exception("Simulation failed")

    # switch back to the original working directory
    os.chdir(base_path)

def clean_simulation(work_path: str) -> None:
    """Save the output and clean the simulation directory"""
    # copy and rename the first and last output files
    re: int = int(work_path.split("/")[-1])
    os.system(f"cp {work_path}/out/output_0000.vti ./sim_out/{re}_in.vti")
    os.system(f"cp {work_path}/out/output_0010.vti ./sim_out/{re}_out.vti")

    # remove working directory and all its contents
    os.system(f"rm -rf {work_path}")


def from_vti_to_pt(vti_filepath: str) -> torch.Tensor:
    """Convert a vti file to a torch tensor"""
    # import pyvista and pytorch
    import pyvista as pv
    import torch
    
    # read the vti file
    reader = pv.read(vti_filepath)

    # convert to torch tensor
    vel = reader.point_data['velocity']
    return torch.from_numpy(vel.T).float()
    