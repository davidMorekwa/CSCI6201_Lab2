from load import load_dataset,verify_load
from visualize import plot_scatter_plot, plot_histogram
from model import train_model

if __name__ == "__main__":
    # load dataset
    dataset = load_dataset()
    # verify_load()

    #visualize the dataset
    plot_scatter_plot(dataset.frame)
    plot_histogram(dataset.frame)

    #train and test model
    train_model(dataset)


