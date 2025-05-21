import matplotlib.pyplot as plt


def plot_model(mean, std_dev, xplot, yplot, train_X, train_Y, iter=0, output_dim=0):

    plt.figure(figsize=(8, 6))
    plt.plot(xplot, yplot, label='True Loss Function', color='blue')
    plt.plot(xplot, mean, label='GP Mean', color='orange')
    plt.fill_between(
        xplot,
        mean - 1 * std_dev,
        mean + 1 * std_dev,
        color='orange',
        alpha=0.2,
        label='Confidence Interval (±σ)'
    )
    plt.scatter(train_X.numpy(), train_Y[:,output_dim].numpy(), color='red', label='Training Data')
    # plt.axhline(y=0.5, color='green', linestyle='--', label='y=0.5')
    plt.legend()
    plt.title('Gaussian Process Model with Uncertainty')
    plt.xlabel('x')
    plt.ylabel('f(x)')

