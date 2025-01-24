import wandb
import matplotlib.pyplot as plt

def log_custom_metrics(epoch: int, train_loss: float, val_loss: float):
    wandb.log({"epoch": epoch, "train_loss": train_loss, "val_loss": val_loss})

def plot_loss(train_losses, val_losses):
    plt.figure(figsize=(10, 6))
    plt.plot(train_losses, label="Train Loss")
    plt.plot(val_losses, label="Validation Loss", linestyle="--")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Loss Curves")
    plt.legend()
    plt.grid()
    plt.savefig("outputs/loss_curve.png")
    plt.show()

    # Log to W&B
    wandb.log({"Loss Curve": wandb.Image("outputs/loss_curve.png")})


