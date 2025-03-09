import matplotlib.pyplot as plt

def plot_history(train_history, val_history, metric, title) -> None:
    """Plots the loss history"""
    plt.figure()
    epoch_idxs = range(len(train_history))
    plt.xticks(epoch_idxs[::5], epoch_idxs[::5])
    plt.plot(epoch_idxs, train_history, "-b", label="training")
    plt.plot(epoch_idxs, val_history, "-r", label="validation")
    plt.title(title)
    plt.legend()
    plt.ylabel(metric)
    plt.xlabel("Epochs")
    plt.show()