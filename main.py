from data import get_dataloader
from train import MultiEngagementPredictor
from torch.optim.lr_scheduler import CosineAnnealingLR
import pandas as pd
import torch.optim as optim
import torch
import numpy as np

label_df = pd.read_csv("label_split.csv")
video_root = "/home/sliuau/video_classification/OTR/cropped_video"
backbone="Vivit"
loss_type = "ldam"
exp = backbone + "_" + loss_type
second_loss_type = None
CUDA = torch.device("cpu")
total_epochs = 500

train_dataloader, val_dataloader, test_dataloader = get_dataloader(2,label_df, video_root, "Vivit")

model = MultiEngagementPredictor(backbone, loss_type, second_loss_type, CUDA = CUDA, labels = train_dataloader.dataset.get_labels())
model = model.to(CUDA)
optimizer = optim.AdamW(model.parameters(), lr=5e-4, weight_decay=1e-1) #1e-1
scheduler = CosineAnnealingLR(optimizer, T_max = 501, eta_min=5e-7) #5e-7

running_loss = []
running_loss_add = []
best_acc, class_acc = 0,0
for epoch in range(1,total_epochs):
    epoch_loss = 0
    epoch_loss_add = 0
    for batch_idx, data in enumerate(train_dataloader):
        video_tensor, video, label = data
        loss, loss_add = model.training_step(video_tensor, label, video)
        total_loss = loss + loss_add
        total_loss.backward()
        optimizer.step()

        print(loss)
        if batch_idx == 1:
            break

        epoch_loss += loss.item()
        epoch_loss_add += loss_add

    running_loss.append(epoch_loss/len(train_dataloader))
    running_loss_add.append(epoch_loss_add/len(train_dataloader))

    if epoch % 1 ==0:
        model.eval()
        prediction = {0:[], 1:[], 2:[], 3:[]}
        for batch_idx, data in enumerate(val_dataloader):
            video_tensor, video, label = data
            truth, score = model.validation_step(video_tensor, label, video)
            for idx in range(len(truth)):
                prediction[truth[idx].item()].append(score[idx].item())

            if batch_idx == 1:
                break

        print("Validation:")
        correct = 0
        class_avg = 0
        disengage_correct, engage_correct = 0,0
        disengage_data, engage_data = 0,0
        for key in prediction:
            temp = np.array(prediction[key])
            if loss_type in ["mocorank", "mse"]:
                dist = np.histogram(temp, bins=(-1.0,-0.5, 0.0, 0.5, 1.0))[0]
            else:
                dist = np.histogram(temp, bins=(-0.1, 0.5, 1.5,2.5,3.1))[0]
            print(dist)
            correct += dist[key]
            class_avg += dist[key]/len(prediction[key])
            if key in [0,1]:
                disengage_correct += dist[0] + dist[1]
                disengage_data += len(prediction[key])
            else:
                engage_correct += dist[2] + dist[3]
                engage_data += len(prediction[key])


        print("Overall", correct/(disengage_data+engage_data))
        print("class avg", class_avg/4)
        print("binary", (disengage_correct + engage_correct)/(disengage_data+engage_data))

        if correct/(disengage_data+engage_data) > best_acc:
            best_acc = correct/(disengage_data+engage_data) 
            class_acc = class_avg/4
            torch.save(model.state_dict(), "ckpt/" + exp + str(epoch) + ".pth")
        elif correct/(disengage_data+engage_data) == best_acc and class_avg/4 > class_acc:
            best_acc = correct/(disengage_data+engage_data) 
            class_acc = class_avg/4
            torch.save(model.state_dict(), "ckpt/" + exp + str(epoch) + ".pth")

        model.train()
        
    
