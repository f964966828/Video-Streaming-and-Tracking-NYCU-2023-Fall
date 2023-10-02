import os
import numpy as np
import pandas as pd
from natsort import index_natsorted
from sklearn.metrics import accuracy_score

import torch
from torchvision import transforms
from torch.utils.data import DataLoader

from net import my_network, my_dataset

'''
    You can add any other package, class and function if you need.
    You should read the .jpg files located in "./dataset/test/", make predictions based on the weight file "./w_{student_id}.pth", and save the results to "./pred_{student_id}.csv".
'''

def test():
    test_file_paths = [f"./dataset/test/{i}.jpg" for i in range(120)]
    test_trasform = transforms.Compose([
        transforms.Resize((64, 64)),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
    test_dataset = my_dataset(file_paths=test_file_paths, transform=test_trasform)
    test_loader = DataLoader(test_dataset, batch_size=128, shuffle=False)

    device = ('cuda' if torch.cuda.is_available() else 'cpu')
    model = my_network()
    model = model.to(device)
    model.load_state_dict(torch.load("w_312552014.pth"))
    
    model.eval()
    pred_all = []
    for images in test_loader:
        images = images.to(device)
        
        with torch.no_grad():
            outputs = model(images)
        
        pred = torch.argmax(outputs, dim=1).cpu().detach().tolist()
        pred_all.extend(pred)
    
    df = pd.DataFrame({
        'name': [f"{i}.jpg" for i in range(120)],
        'label': pred_all
    }).to_csv("pred_312552014.csv", index=False)

    if os.path.exists("./dataset/test.csv"):
        gt_df = pd.read_csv("./dataset/test.csv")
        gt_df = gt_df.sort_values(by='name', key=lambda x: np.argsort(index_natsorted(x)))
        
        acc = accuracy_score(pred_all, gt_df['label'])
        print("Accuracy:", acc)

if __name__ == "__main__":
    test()
