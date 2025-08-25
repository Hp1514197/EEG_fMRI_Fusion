import matplotlib.pyplot as plt

def plot_losses(losses, out_path=None):
    plt.figure()
    plt.plot(range(1, len(losses)+1), losses, marker='o')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.title('Training Loss')
    if out_path:
        plt.savefig(out_path)
    else:
        plt.show()
