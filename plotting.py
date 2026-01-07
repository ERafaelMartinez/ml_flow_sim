import matplotlib.pyplot as plt
import random
import numpy as np

def plot_data_samples(inputs, outputs, title="", sample_idx=None):
    """
    Plots the input and output data for a specific sample index.
    
    Args:   
        inputs: Input data (torch.Tensor or np.ndarray)
        outputs: Output data (torch.Tensor or np.ndarray)
        title: Title of the plot
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

    # Collect fields to plot
    plots_to_show = []

    # corresponds to a horizontal velocity field (Input u)
    plots_to_show.append((input_sample[0], f"Sample Input u (Idx: {sample_idx})"))

    # corresponds to a vertical velocity field (Input v, if present)
    if input_sample.shape[0] > 1:
        plots_to_show.append((input_sample[1], f"Sample Input v (Idx: {sample_idx})"))

    # corresponds to a horizontal velocity field (Output u)
    plots_to_show.append((output_sample[0], f"Sample Output u (Idx: {sample_idx})"))

    # corresponds to a vertical velocity field (Output v, if present)
    if output_sample.shape[0] > 1:
        plots_to_show.append((output_sample[1], f"Sample Output v (Idx: {sample_idx})"))

    # Compute global vmin/vmax for consistent coloring
    all_data = np.concatenate([p[0].flatten() for p in plots_to_show])
    vmin = all_data.min()
    vmax = all_data.max()

    n_plots = len(plots_to_show)
    # Reduced height to minimize vertical gap, using constrained_layout for better spacing
    # figsize height reduced to 3.5 to fit the horizontal strips better without large gaps
    fig, axes = plt.subplots(1, n_plots, figsize=(12, 3.5), constrained_layout=True)
    
    # Ensure axes is iterable even if only 1 plot
    if n_plots == 1:
        axes = [axes]

    im = None
    for ax, (data, label) in zip(axes, plots_to_show):
        im = ax.imshow(data, origin='lower', cmap="RdYlBu_r", vmin=vmin, vmax=vmax)
        ax.set_title(label)

    # Add single horizontal colorbar below the plots
    fig.colorbar(im, ax=axes, orientation='horizontal', fraction=0.05, aspect=50, label="Value")

    plt.suptitle(title)
    plt.show()
