import torch
from torch.utils.data import DataLoader
import torch.optim as optim
from network.with_pytorch.network import Network
from network.with_pytorch.transforms import preview_transformation, get_base_transform, get_train_transform, get_test_transform
from network.with_pytorch.data_fetching import get_emoji_data

if __name__ == "__main__":
    # for i in range(3):
    #     preview_transformation("dataset/dataset-data/training-data/4/6 (11, 0).png")
    # exit()

    # Load and Split (Ensures different transforms for train/test)
    train_data, test_data = get_emoji_data()

    train_dataloader = DataLoader(train_data, batch_size=32, shuffle=True)
    test_dataloader = DataLoader(test_data, batch_size=32, shuffle=False)

    # Model Setup
    device = torch.accelerator.current_accelerator().type if torch.accelerator.is_available() else "cpu"
    model = Network(device)

    loss_fn = torch.nn.CrossEntropyLoss()
    # Using SGD with momentum as requested
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01, momentum=0.9)

    # The Scheduler (Watches test_loss to adjust LR)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', patience=3, factor=0.5)

    # Training Loop
    epochs = 30
    for t in range(epochs):
        print(f"Epoch {t+1}\n-------------------------------")
        model.train_model(train_dataloader, loss_fn, optimizer)
        
        # This now captures the average loss returned by the network
        current_test_loss, _ = model.test_model(test_dataloader, loss_fn)
        
        # Update the learning rate based on performance
        scheduler.step(current_test_loss)
        
        print(f"Current LR: {optimizer.param_groups[0]['lr']}")

    print("Done!")
    torch.save(model.state_dict(), "network/saved_models/new_model.pth")