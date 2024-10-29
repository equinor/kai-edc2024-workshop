from ipywidgets import interact, FloatSlider
import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
sns.set_theme()


def perceptron(x):
    """
    Returns the output of a perceptron with 6 input signals.

    Parameters:
        x (List of length 6) Input signals to the perceptron.
    """
    weights = [0.2, -0.3, 0.9, 0.4, -0.9, -0.5]

    # TODO: Calculate y
    z = np.sum(np.array(weights) * np.array(x))

    # Calculate activation
    if z > 0:
        return 1

    return 0


def plot_perceptron(func, x0=[1, 1, 1, 1, 1, 1]):
    if len(x0) != 6:
        raise ValueError('The input has to be of length 6.')
    x0 = np.array(x0)

    # Plot perceptron by varying each input dimension at a time
    steps = np.linspace(-3, 3, 200)

    # Visualise data
    # Create layout
    my_dpi = 192
    fig = plt.figure(figsize=(2250 // my_dpi, 1200 // my_dpi), dpi=150)
    outer = gridspec.GridSpec(2, 3, hspace=0.35)

    # Create axes
    axes = []
    axes.append(plt.Subplot(fig, outer[0, 0]))
    axes.append(plt.Subplot(fig, outer[0, 1]))
    axes.append(plt.Subplot(fig, outer[0, 2]))
    axes.append(plt.Subplot(fig, outer[1, 0]))
    axes.append(plt.Subplot(fig, outer[1, 1]))
    axes.append(plt.Subplot(fig, outer[1, 2]))

    # Add axes to figure
    for ax in axes:
        fig.add_subplot(ax)

    for ida, ax in enumerate(axes):
        out = []
        for step in steps:
            x = np.copy(x0)
            x[ida] += step
            o = func(x)
            out.append(o)
        ax.plot(x0[ida]+steps, out, linewidth=2, label='Perceptron', color=sns.color_palette()[0])

    for ida, ax in enumerate(axes):
        out = []
        for step in steps:
            x = np.copy(x0)
            x[ida] += step
            o = perceptron(x)
            out.append(o)
        ax.plot(x0[ida]+steps, out, linewidth=2, label='Solution', color=sns.color_palette()[3], linestyle='--')

    for ida, ax in enumerate(axes):
        ax.set_xlabel(f'Input x{ida+1}')
        if (ida == 0) or (ida == 3):
            ax.set_ylabel('Output y')

    axes[0].legend()
    plt.show()


def perceptron2(inputs, weights):
    """
    Returns the output of a perceptron with 3 input signals.

    Parameters:
        inputs (np.ndarray of shape (2, n)) Input data.
        weights (np.ndarray of shape (3,)) Weights of the network.
    """
    # Calculate linear output
    z = weights[0] + weights[1] * inputs[0] + weights[2] * inputs[1]

    # Calculate activation
    y = np.array(z > 0, dtype=float)

    return y


def neural_network(inputs, weights):
    """
    Returns the output of the simple neural network.

    Parameters:
        inputs (np.ndarray of shape (2, n)) Input data.
        weights (np.ndarray of shape (12,)) Weights of the network.
    """
    inputs = np.array(inputs)
    weights = np.array(weights)

    # Parse weights
    weights1 = weights[:3]
    weights2 = weights[3:6]
    weights3 = weights[6:9]
    weights4 = weights[9:12]

    # Process data
    n = inputs.shape[1]
    out = np.empty(shape=(4, n))
    for idw, w in enumerate([weights1, weights2, weights3, weights4]):
        out[idw] = perceptron2(inputs, w)

    # If all perceptrons predict a 0, the label is a zero. Otherwise 1
    y = np.sum(out, axis=0)
    y[y > 0] = 1

    return y


def plot_training_results(n_iterations, losses, weights, data):
    # Visualise loss over time
    iterations = np.arange(1, n_iterations+1)
    plt.plot(iterations, losses, linewidth=2)
    plt.xlabel('Iterations')
    plt.ylabel('Loss (number of misclassifications)')
    plt.show()

    # Visualise the learned classification
    X, y = data
    plt.scatter(X[y==0, 0], X[y==0, 1])
    plt.scatter(X[y==1, 0], X[y==1, 1])

    x1 = np.linspace(-3, 3, 100)
    x2 = -(weights[1] * x1 + weights[0]) / weights[2]
    plt.plot(x1, x2, linestyle='-', color='red')
    x2 = -(weights[4] * x1 + weights[3]) / weights[5]
    plt.plot(x1, x2, linestyle='-', color='orange')
    x2 = -(weights[7] * x1 + weights[6]) / weights[8]
    plt.plot(x1, x2, linestyle='-', color='lightblue')
    x2 = -(weights[10] * x1 + weights[9]) / weights[11]
    plt.plot(x1, x2, linestyle='-', color='darkcyan')

    plt.xlim(-3, 3)
    plt.ylim(-6, 6)
    plt.show()


def plot_step_activation():
    def step_function(x):
        return np.where(x > 0, 1, 0)

    def step_function_gradient(x):
        return 0

    # Define the function to plot
    def plot_sine(delta_z):
        my_dpi = 192
        fig = plt.figure(figsize=(2250 // my_dpi, 900 // my_dpi), dpi=150)
        outer = gridspec.GridSpec(1, 2, hspace=0.6)

        # Create axes
        axes = []
        axes.append(plt.Subplot(fig, outer[0]))
        axes.append(plt.Subplot(fig, outer[1]))

        # Add axes to figure
        for ax in axes:
            fig.add_subplot(ax)

        # plot step function
        x = [-0.55, 0, 0.45]
        y = [0, 0, 1]
        axes[0].step(x, y, color='black', linewidth=3)
        axes[1].plot(x, [0, 0, 0], color='black', linewidth=3)

        # plot the starting point
        z0 = -0.05
        z = z0 + delta_z
        axes[0].plot(z0, step_function(z0), 'ro', markerfacecolor='none')
        axes[0].text(z0, step_function(z0) - 0.2, rf'$z_0 = {z0}$', fontsize=12, ha='center')
        axes[1].plot(z0, step_function_gradient(z0), 'ro', markerfacecolor='none')
        axes[1].text(z0, step_function_gradient(z0) - 0.2, rf'$z_0 = {z0}$', fontsize=12, ha='center')

        # plot the moving point
        axes[0].plot(z, step_function(z), 'ro')
        axes[0].text(z, step_function(z) + 0.2, rf'$z = z_0 + \Delta z = {z:.2f}$', fontsize=12, ha='right')
        axes[1].plot(z, step_function_gradient(z), 'ro')
        axes[1].text(z, step_function_gradient(z) + 0.2, rf'$z = z_0 + \Delta z = {z:.2f}$', fontsize=12, ha='right')

        axes[0].set_ylim([-1, 2])
        axes[0].set_xlabel(r'$z = \mathbf{w \cdot x}$')
        axes[0].set_ylabel(r'$f(z)$')
        axes[0].set_title('Output of perceptron')

        axes[1].set_ylim([-1, 2])
        axes[1].set_xlabel(r'$z = \mathbf{w \cdot x}$')
        axes[1].set_ylabel(r'$\partial f(z) / \partial w_j$')
        axes[1].set_title('Gradient of perceptron')
        plt.show()

    # Create the slider
    slider = FloatSlider(min=-0.15, max=0.5, step=0.01, value=0)

    # Create the interactive plot
    func = interact(plot_sine, delta_z=slider)


def plot_sigmoid_activation():
    def sigmoid(x):
        return 1 / (1 + np.exp(-x))

    def sigmoid_gradient(x):
        return np.exp(-x) / (1 + np.exp(-x))**2

    # Define the function to plot
    def plot_sine(delta_z):
        my_dpi = 192
        fig = plt.figure(figsize=(2250 // my_dpi, 900 // my_dpi), dpi=150)
        outer = gridspec.GridSpec(1, 2, hspace=0.6)

        # Create axes
        axes = []
        axes.append(plt.Subplot(fig, outer[0]))
        axes.append(plt.Subplot(fig, outer[1]))

        # Add axes to figure
        for ax in axes:
            fig.add_subplot(ax)

        # Plot reference
        x = np.linspace(-20, 20, 100)
        axes[0].plot(x, sigmoid(x), color='black', linewidth=3)
        axes[1].plot(x, sigmoid_gradient(x), color='black', linewidth=3)

        # plot the starting point
        z0 = -5
        z = z0 + delta_z
        axes[0].plot(z0, sigmoid(z0), 'ro', markerfacecolor='none')
        axes[0].text(z0, sigmoid(z0) - 0.2, rf'$z_0 = {z0}$', fontsize=12, ha='center')
        axes[1].plot(z0, sigmoid_gradient(z0), 'ro', markerfacecolor='none')
        axes[1].text(z0, sigmoid_gradient(z0) + 0.02, rf'$z_0 = {z0}$', fontsize=12, ha='center')

        # plot the moving point
        axes[0].plot(z, sigmoid(z), 'ro')
        axes[0].text(z, sigmoid(z) + 0.2, rf'$z = z_0 + \Delta z = {z:.2f}$', fontsize=12, ha='right')
        axes[1].plot(z, sigmoid_gradient(z), 'ro')
        axes[1].text(z, sigmoid_gradient(z) + 0.02, rf'$z = z_0 + \Delta z = {z:.2f}$', fontsize=12, ha='right')

        axes[0].set_ylim([-0.5, 1.5])
        axes[0].set_xlabel(r'$z = \mathbf{w \cdot x}$')
        axes[0].set_ylabel(r'$f(z)$')
        axes[0].set_title('Output of perceptron')

        axes[1].set_ylim([-0.05, 0.3])
        axes[1].set_xlabel(r'$z = \mathbf{w \cdot x}$')
        axes[1].set_ylabel(r'$\partial f(z) / \partial w_j$')
        axes[1].set_title('Gradient of perceptron')
        plt.show()

    # Create the slider
    slider = FloatSlider(min=-7, max=15, step=0.1, value=0)

    # Create the interactive plot
    func = interact(plot_sine, delta_z=slider)


def plot_results2(n_iterations, losses, model, data):
    # Visualise loss over time
    iterations = np.arange(1, n_iterations+1)
    plt.plot(iterations, losses, linewidth=2)
    plt.xlabel('Iterations')
    plt.ylabel('Loss (cross-entropy)')
    plt.show()

    # Visualise the learned classification
    X, y = data
    plt.scatter(X[y==0, 0], X[y==0, 1])
    plt.scatter(X[y==1, 0], X[y==1, 1])

    # Reshape model weights
    weights = np.empty(12)
    bias = model.layer.bias.detach().numpy()
    weights = model.layer.weight.detach().numpy()

    x1 = np.linspace(-3, 3, 100)
    for idx in range(len(bias)):
        x2 = -(weights[idx, 0] * x1 + bias[idx]) / weights[idx, 1]
        plt.plot(x1, x2, linestyle='-')

    plt.xlim(-3, 3)
    plt.ylim(-6, 6)
    plt.xlabel(r'$x_1$')
    plt.ylabel(r'$x_2$')
    plt.show()


def plot_four_images(images, labels):
    # Visualise data
    # Create layout
    fontsize = 14
    my_dpi = 192
    fig = plt.figure(figsize=(2250 // my_dpi, 1200 // my_dpi), dpi=150)
    outer = gridspec.GridSpec(1, 4)

    # Create axes
    axes = []
    axes.append(plt.Subplot(fig, outer[0]))
    axes.append(plt.Subplot(fig, outer[1]))
    axes.append(plt.Subplot(fig, outer[2]))
    axes.append(plt.Subplot(fig, outer[3]))

    # Add axes to figure
    for ax in axes:
        fig.add_subplot(ax)

    for ida, ax in enumerate(axes):
        ax.imshow(images[ida][0], cmap='gray')
        ax.set_title('Label: %d' % labels[ida])
    plt.show()