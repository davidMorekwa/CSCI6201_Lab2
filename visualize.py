from load import load_dataset
import matplotlib.pyplot as plt
from pathlib import Path

def get_figures_directory():
    """
    get path to figures directory
    """
    cwd = Path.cwd()
    figures_dir = cwd / "figures"
    # print(fig_dir)
    return figures_dir

def plot_scatter_plot(dataset_df):
    """
    Plot a scatter plot showing the distribution of median house values against Longitude-Latitude values
    """
    plt.figure()  # start a fresh, blank figure
    scatter = plt.scatter(x=dataset_df["Longitude"], y=dataset_df["Latitude"], c=dataset_df["MedHouseVal"], s=dataset_df["Population"]/1000, cmap="viridis", alpha=0.4)
    plt.xlabel("Longitude")
    plt.ylabel("latitude")
    plt.title("Distribution of median house value")
    cbar = plt.colorbar(scatter)
    cbar.set_label("Median House Value")
    figures_dir = get_figures_directory()
    plt.savefig( figures_dir / "geo-scatter.png")
    plt.close()  # close it so the next chart doesn't draw on top

def plot_histogram(dataset_df):
    """
    Plot a histogram showing the distribution of Median House values
    """
    plt.figure()  # start a fresh, blank figure
    plt.hist(dataset_df["MedHouseVal"], bins=30)
    plt.title("distribution of Median House Values")
    plt.xlabel("Median House Value ($100,000)")
    plt.ylabel("Number")
    figures_dir = get_figures_directory()
    plt.savefig(figures_dir / "house-value-histogram.png")
    plt.close()

# plot_scatter_plot(dataset_df)
# plot_histogram(dataset_df)

    
