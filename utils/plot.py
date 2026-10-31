import matplotlib.pyplot as plt

def plot_history(history: dict):
    epochs = range(1, len(history['train_losses']) + 1)

    plt.figure(figsize=(8, 5))
    plt.plot(epochs, history['train_losses'], label='Train Loss')
    plt.plot(epochs, history['val_losses'], label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training vs Validation Loss')
    plt.legend()
    plt.grid(True)
    plt.savefig('loss.png')

    plt.figure(figsize=(8, 5))
    plt.plot(epochs, history['train_accs'], label='Train Accuracy')
    plt.plot(epochs, history['val_accs'], label='Validation Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.title('Training vs Validation Accuracy')
    plt.legend()
    plt.grid(True)
    plt.savefig('accuracy.png')