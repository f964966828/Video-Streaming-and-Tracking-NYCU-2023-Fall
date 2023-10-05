import numpy as np
import pandas as pd

import torch
import torch.nn as nn
from torchsummary import summary
from torchvision import transforms
from torch.utils.data import DataLoader

from net import my_network, my_dataset

'''
    You can add any other package, class and function if you need.
    You should read the .jpg from "./dataset/train/" and save your weight to "./w_{student_id}.pth"
'''

def train():
    # Read data from csv
    df = pd.read_csv("./dataset/train.csv")
    file_paths = ["./dataset/train/" + file_name for file_name in df['name']]
    labels = df['label']
    
    # Get shuffled indices
    indices = np.arange(len(labels))
    np.random.shuffle(indices)
    file_paths = np.array(file_paths)[indices]
    labels = np.array(labels)[indices]

    # Split into training and validation dataset
    length = len(file_paths)
    train_file_paths, valid_file_paths = file_paths[:int(length * 0.75)], file_paths[int(length * 0.75):]
    train_labels, valid_labels = labels[:int(length * 0.75)], labels[int(length * 0.75):]

    # Build training loader
    train_transform = transforms.Compose([
        transforms.RandomHorizontalFlip(),            
        transforms.RandomRotation(30),                
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
        transforms.RandomResizedCrop(64, scale=(0.8, 1.0)),
        transforms.Resize((64, 64)),                         
        transforms.ToTensor(),                                
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)) 
    ])
    train_dataset = my_dataset(file_paths=train_file_paths, labels=train_labels, transform=train_transform)
    train_loader = DataLoader(train_dataset, batch_size=128, shuffle=True)

    # Build validation loader
    valid_transform = transforms.Compose([
        transforms.Resize((64, 64)),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
    valid_dataset = my_dataset(file_paths=valid_file_paths, labels=valid_labels, transform=valid_transform)
    valid_loader = DataLoader(valid_dataset, batch_size=128, shuffle=False)

    # Define model and optimizer
    device = ('cuda' if torch.cuda.is_available() else 'cpu')
    model = my_network()
    model = model.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    summary(model, input_size=(3, 64, 64))

    # Strat training
    num_epoch = 200    
    best_accuracy = 0.0
    for epoch in range(num_epoch):
        train_loss = 0.0
        correct, total = 0, 0
        model.train()
        for images, labels in train_loader:
            images = images.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()
            outputs = model(images)

            batch_loss = criterion(outputs, labels)
            batch_loss.backward()
            optimizer.step()

            train_loss += batch_loss.item()
            correct += sum(torch.argmax(outputs, dim=1) == labels).item()
            total += images.shape[0]
        train_accuracy = correct / total * 100

        valid_loss = 0.0
        correct, total = 0, 0
        model.eval()
        for images, labels in valid_loader:
            images = images.to(device)
            labels = labels.to(device)

            with torch.no_grad():
                outputs = model(images)
            
            batch_loss = criterion(outputs, labels)
            valid_loss += batch_loss.item()
            correct += sum(torch.argmax(outputs, dim=1) == labels).item()
            total += images.shape[0]
        valid_accuracy = correct / total * 100

        if valid_accuracy > best_accuracy:
            best_accuracy = valid_accuracy
            torch.save(model.state_dict(), "w_312552014.pth")

        print(f"Epoch [{epoch+1}/{num_epoch}] - Train Loss: {train_loss/len(train_loader):.4f}, Train Accuracy: {train_accuracy:.2f}%, Valid Loss: {valid_loss/len(valid_loader):.4f}, Valid Accuracy: {valid_accuracy:.2f}%")

    print(f"Best Validation Accuracy: {best_accuracy}")

if __name__ == "__main__":
    train()
