import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from model import Autoencoder
from plot import plot_images

if __name__ == '__main__':
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])
    mnist_data = datasets.MNIST(root="./data",
                                train=True,
                                transform=transform)
    data_loader = DataLoader(dataset=mnist_data,
                             batch_size=32,
                             shuffle=True)

    ## Use the following to check if sigmoid or tanh needed
    ## as last activation func in model decoder
    # data = iter(data_loader)
    # imgs, targets = data.next()
    # print(torch.min(imgs))
    # print(torch.max(imgs))

    model = Autoencoder()
    criterion = nn.MSELoss()
    optimizer = optim.SGD(model.parameters(), lr=0.002, momentum=0.9, weight_decay=1e-4)

    NUM_EPOCHS = 10
    total_loss = 0
    outputs = []
    for epoch in range(NUM_EPOCHS):
        total_loss = 0
        for data, _ in data_loader:
            optimizer.zero_grad()
            rebuilt_data = model(data)
            loss = criterion(rebuilt_data, data)
            loss.backward()
            optimizer.step()
            outputs.append((epoch, data, rebuilt_data))
            total_loss += loss.detach().item()
        print(f"Epoch: {epoch + 1} | Loss: {total_loss:.2f}")

    plot_images(outputs, NUM_EPOCHS)
