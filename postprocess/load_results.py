import os
import numpy as np
from scipy.stats import ks_2samp
import matplotlib.pyplot as plt
import math

def get_results_dir():
    return os.path.join(os.path.dirname(__file__), '../src/models/results/predictions')


class Results:
    def __init__(self, base_model, level, group, predictions, sds, labels):
        self.base_model = base_model
        self.level = level
        self.group = group
        self.predictions = predictions
        self.sds = sds
        self.labels = labels

def load_results(base_model, level, group):
    results_dir = get_results_dir()
    predictions = np.load(f'{results_dir}/lumbar_dataset_model_{base_model}_level{level}_y_p_{group}_original.npy')
    sds = np.load(f'{results_dir}/lumbar_dataset_model_{base_model}_level{level}_logvars_{group}_original.npy')
    labels = np.load(f'{results_dir}/lumbar_dataset_model_{base_model}_level{level}_targets_{group}_original.npy')
    return Results(base_model=base_model, level=level, group=group, predictions=predictions, sds=sds, labels=labels)


def get_x_norm(results):
    x = np.array([i[0] for i in results.labels])
    prediction_x = np.array([i[0] for i in results.predictions])
    sds = np.array([i[0] for i in results.sds])
    return (x - prediction_x) / sds

def get_y_norm(results):
    y = np.array([i[1] for i in results.labels])
    prediction_y = np.array([i[1] for i in results.predictions])
    sds = np.array([i[0] for i in results.sds])
    return (y - prediction_y) / sds

def preform_ks_test(base_model, level, group, norm_values):



    ks_test_x = ks_2samp(norm_values, 'norm')


    # write to file
    with open(f'./testks/{base_model}_lumbar_{group}_{level}_ks_test.txt', 'w') as file:
        file.write(f"KS test : {ks_test_x}\n")

def create_histogram(base_model, level, group, norm_values, line, figsize=(4, 3)):
    """
    Creates a histogram and saves it as a file.
    """
    # Create a figure
    plt.figure(figsize=figsize)

    # Plot the histogram
    plt.hist(norm_values, bins=20, range=(min(norm_values), max(norm_values)))
    # plt.xlim((-3, 3))

    # Save the histogram to a file
    plt.savefig(f'./plots/{base_model}_lumbar_{group}_{level}_histogram_{line}.png')

    # Show the histogram
    plt.show()

    # Close the figure
    plt.close()


import os
import matplotlib.pyplot as plt

def create_histogram(base_model, level, group, norm_values, line, figsize=(4, 3), bins=30, x_range=None, show=True, save_dir='./plots'):
    """
    Creates a histogram with white lines between bins and saves it as a file.

    Parameters:
        base_model (str): Name of the base model.
        level (str): The level of analysis.
        group (str): The group being analyzed.
        norm_values (list or array): The values to plot in the histogram.
        line (str): Line information to include in the filename.
        figsize (tuple): Size of the figure (width, height).
        bins (int): Number of bins for the histogram.
        x_range (tuple): Range for the x-axis (e.g., (-3, 3)).
        show (bool): Whether to display the plot.
        save_dir (str): Directory to save the plot.
    """
    # Ensure the save directory exists
    os.makedirs(save_dir, exist_ok=True)

    # Create a figure
    plt.figure(figsize=figsize)

    # Plot the histogram with white edges
    plt.hist(
        norm_values,
        bins=bins,
        range=x_range or (min(norm_values), max(norm_values)),
        edgecolor='white',  # Add white lines between bins
        linewidth=0.7       # Thickness of the edges
    )

    # Optionally set x-axis range
    # if x_range:
    plt.xlim((-3, 3))

    # Add a bold frame to the plot
    ax = plt.gca()
    ax.spines['top'].set_linewidth(2)  # Top frame
    ax.spines['right'].set_linewidth(2)  # Right frame
    ax.spines['left'].set_linewidth(2)  # Left frame
    ax.spines['bottom'].set_linewidth(2)  # Bottom frame


    # Save the histogram to a file
    filename = f'{base_model}_lumbar_{group}_{level}_histogram_{line}.png'
    filepath = os.path.join(save_dir, filename)
    plt.savefig(filepath, dpi=300)
    print(f'Histogram saved to: {filepath}')

    # Optionally show the histogram
    if show:
        plt.show()

    # Close the figure
    plt.close()


def create_histograms(base_model, level, group, results):
    x_norm = get_x_norm(results)
    y_norm = get_y_norm(results)

    create_histogram(base_model, level, group, x_norm, 'x')
    create_histogram(base_model, level, group, y_norm, 'y')




if __name__ == '__main__':
    # densenet201 efficientnetb4
    base_model = 'densenet201'
    level = 1
    # calib \ test
    group = 'calib'
    results = load_results(base_model, level, group)

    # x = [i[0] for i in results.labels]
    # y = [i[1] for i in results.labels]
    # print std
    # print(f"sd x:{np.std(x)}")
    # print(f"sd y: {np.std(y)}")
    # sum(np.array(predictions) - results.labels)/len(results.labels)

    predictions = [i[0] for i in results.predictions]
    sds = [i[0] for i in results.sds]
    norm_values= [(results.predictions[i][0] - results.labels[i])/math.sqrt(math.exp(results.sds[i][0])) for i in range(len(results.labels))]
    create_histogram(base_model, level, group, norm_values, 'x_y')
    # create_histograms(base_model, level, group, results)
    preform_ks_test(base_model, level, group, norm_values)


