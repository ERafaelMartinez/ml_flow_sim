import matplotlib.pyplot as plt
import random
import numpy as np

def plot_data_samples(inputs, outputs, sample_idx=None):
    """
    Plots the input and output data for a specific sample index.
    
    Args:   
        inputs: Input data (torch.Tensor or np.ndarray)
        outputs: Output data (torch.Tensor or np.ndarray)
        sample_idx: Index of the sample to plot. If None, a random index is chosen.
    """
    if sample_idx is None:
        sample_idx = random.randint(0, len(inputs) - 1)

    # Helper function to convert to numpy if needed
    def to_numpy(data):
        if hasattr(data, "numpy"):
            return data.numpy()
        return data

    input_sample = to_numpy(inputs[sample_idx])
    output_sample = to_numpy(outputs[sample_idx])

    input_data_u = input_sample[0] # corresponds to a horizontal velocity field
    output_data_u = output_sample[0] # corresponds to a horizontal velocity field
    output_data_v = output_sample[1] # corresponds to a vertical velocity field

    all_data = np.concatenate([input_data_u.flatten(), output_data_u.flatten(), output_data_v.flatten()])
    vmin = all_data.min()
    vmax = all_data.max()

    fig, axes = plt.subplots(1, 3, figsize=(12, 5))

    im1 = axes[0].imshow(input_data_u, origin='lower', cmap="RdYlBu_r", vmin=vmin, vmax=vmax)
    axes[0].set_title(f"Sample Input u (Idx: {sample_idx})")
    fig.colorbar(im1, ax=axes[0], label="Value")

    im2 = axes[1].imshow(output_data_u, origin='lower', cmap="RdYlBu_r", vmin=vmin, vmax=vmax)
    axes[1].set_title(f"Sample Output u (Idx: {sample_idx})")
    #fig.colorbar(im2, ax=axes[1], label="Value")

    im3 = axes[2].imshow(output_data_v, origin='lower', cmap="RdYlBu_r", vmin=vmin, vmax=vmax)
    axes[2].set_title(f"Sample Output v (Idx: {sample_idx})")
    #fig.colorbar(im3, ax=axes[2], label="Value")

    plt.tight_layout()
    plt.show()
