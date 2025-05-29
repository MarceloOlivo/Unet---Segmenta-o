import torch
import albumentations as A
from albumentations.pytorch import ToTensorV2
from tqdm import tqdm
import torch.nn as nn
import torch.optim as optim
from model import UNET
import matplotlib.pyplot as plt
from utils import (
    load_checkpoint,
    save_checkpoint,
    get_loaders,
    check_accuracy,
    save_predictions_as_imgs,
)


LEARNING_RATE = 1e-4
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
BATCH_SIZE = 16
NUM_EPOCHS = 150
NUM_WORKERS = 2
IMAGE_HEIGHT = 160  
IMAGE_WIDTH = 240  
PIN_MEMORY = True
LOAD_MODEL = False
TRAIN_IMG_DIR = "C:/Users/User/Desktop/Programacao/UNET/data/train_images/"
TRAIN_MASK_DIR = "C:/Users/User/Desktop/Programacao/UNET/data/train_masks/"
VAL_IMG_DIR = "C:/Users/User/Desktop/Programacao/UNET/data/val_images/"
VAL_MASK_DIR = "C:/Users/User/Desktop/Programacao/UNET/data/val_masks/"

def train_fn(loader, model, optimizer, loss_fn, scaler):
    loop = tqdm(loader)

    for batch_idx, (data, targets) in enumerate(loop):
        data = data.to(device=DEVICE)
        targets = targets.float().unsqueeze(1).to(device=DEVICE)

      
        with torch.cuda.amp.autocast():
            predictions = model(data)
            loss = loss_fn(predictions, targets)

  
        optimizer.zero_grad()
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

    
        loop.set_postfix(loss=loss.item())


def main():
    train_transform = A.Compose(
        [
            A.Resize(height=IMAGE_HEIGHT, width=IMAGE_WIDTH),
            A.Rotate(limit=35, p=1.0),
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.1),
            A.Normalize(
                mean=[0.0, 0.0, 0.0],
                std=[1.0, 1.0, 1.0],
                max_pixel_value=255.0,
            ),
            ToTensorV2(),
        ],
    )

    val_transforms = A.Compose(
        [
            A.Resize(height=IMAGE_HEIGHT, width=IMAGE_WIDTH),
            A.Normalize(
                mean=[0.0, 0.0, 0.0],
                std=[1.0, 1.0, 1.0],
                max_pixel_value=255.0,
            ),
            ToTensorV2(),
        ],
    )

    model = UNET(in_channels=3, out_channels=1).to(DEVICE)
    loss_fn = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

    train_loader, val_loader = get_loaders(
        TRAIN_IMG_DIR,
        TRAIN_MASK_DIR,
        VAL_IMG_DIR,
        VAL_MASK_DIR,
        BATCH_SIZE,
        train_transform,
        val_transforms,
        NUM_WORKERS,
        PIN_MEMORY,
    )

    if LOAD_MODEL:
        load_checkpoint(torch.load("C:/Users/User/Desktop/Programacao/UNET/my_checkpoint.pth.tar"), model)


    check_accuracy(val_loader, model, device=DEVICE)
    scaler = torch.cuda.amp.GradScaler()

    for epoch in range(NUM_EPOCHS):
        print(f"\nEpoch [{epoch+1}/{NUM_EPOCHS}]")

        train_fn(train_loader, model, optimizer, loss_fn, scaler)

       
        checkpoint = {
            "state_dict": model.state_dict(),
            "optimizer": optimizer.state_dict(),
        }
        save_checkpoint(checkpoint)

       
        print("Train Evaluation:")
        check_accuracy(train_loader, model, device=DEVICE)

        
        print("Validation Evaluation:")
        check_accuracy(val_loader, model, device=DEVICE)

        
        save_predictions_as_imgs(
            val_loader, model, folder="C:/Users/User/Desktop/Programacao/UNET/saved_images/", device=DEVICE
        )

        train_acc_list = []
        train_dice_list = []
        val_acc_list = []
        val_dice_list = []

        scaler = torch.cuda.amp.GradScaler()

        for epoch in range(NUM_EPOCHS):
            print(f"\nEpoch [{epoch+1}/{NUM_EPOCHS}]")

            train_fn(train_loader, model, optimizer, loss_fn, scaler)

          
            checkpoint = {
                "state_dict": model.state_dict(),
                "optimizer": optimizer.state_dict(),
            }
            save_checkpoint(checkpoint)

        
            print("Train Set Evaluation:")
            train_acc, train_dice = check_accuracy(train_loader, model, device=DEVICE)
            train_acc_list.append(train_acc)
            train_dice_list.append(train_dice)

            
            print("Validation Set Evaluation:")
            val_acc, val_dice = check_accuracy(val_loader, model, device=DEVICE)
            val_acc_list.append(val_acc)
            val_dice_list.append(val_dice)

            
            save_predictions_as_imgs(
                val_loader, model, folder="C:/Users/User/Desktop/Programacao/UNET/saved_images/", device=DEVICE
            )

        epochs = range(1, NUM_EPOCHS + 1)

        plt.figure(figsize=(12, 5))

        plt.subplot(1, 2, 1)
        plt.plot(epochs, train_acc_list, label="Train Accuracy")
        plt.plot(epochs, val_acc_list, label="Val Accuracy")
        plt.xlabel("Epoch")
        plt.ylabel("Accuracy (%)")
        plt.title("Accuracy X Epochs")
        plt.legend()

        plt.subplot(1, 2, 2)
        plt.plot(epochs, train_dice_list, label="Train Dice Score")
        plt.plot(epochs, val_dice_list, label="Val Dice Score")
        plt.xlabel("Epoch")
        plt.ylabel("Dice Score")
        plt.title("Dice Score X Epochs")
        plt.legend()

        plt.tight_layout()
        plt.savefig("training_metrics.png")
        plt.show()
        
            

if __name__ == "__main__":
    main()