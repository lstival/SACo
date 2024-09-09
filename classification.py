#author: L. Stival

import torch
import torch.nn as nn
from torchvision.models import resnet18, ResNet18_Weights, vit_b_16, ViT_B_16_Weights, mobilenet_v2, MobileNet_V2_Weights
from tqdm import tqdm
from eval_metrics import Eval
from utils import get_model_time
from modules.sentinel_bands import EuroSATGroupBands
from modules.senti_encoder import load_model_and_weights, SentinelEncoder
from sklearn.metrics import balanced_accuracy_score
# from modules.resnet import ResNet_enconder_LBP

class ClassificationNetwork(nn.Module):
    def __init__(self, encoder=None) -> None:
        super().__init__()

        if encoder is None:
            self.resnet = resnet18(weights=ResNet18_Weights.DEFAULT)
            self.resnet.conv1 = nn.Conv2d(13, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
            self.resnet.fc =  nn.Linear(512, 10)

        else:
            self.resnet = encoder
            # self.resnet.model.model[0] = nn.Conv2d(13, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
            self.resnet.model.fc =  nn.Linear(512, 10)

        # Change input channels to 13
        
        # Freeze model layers
        for param in self.resnet.parameters():
            param.requires_grad = False

        # unfreeze last layer
        if encoder is None:
            for param in self.resnet.fc.parameters():
                param.requires_grad = True
        else:
            for param in self.resnet.model.fc.parameters():
                param.requires_grad = True

    def forward(self, x):
        x = self.resnet(x)
        x = nn.functional.softmax(x, dim=1)
        
        return x
    
class EvalClassification():
    def __init__(self, model_name=None, model=None) -> None:
        if model:
            self.model = model

        if model_name:
            # encoder = SentinelEncoder(bands_selector=EuroSATGroupBandsLBP(), model=ResNet_enconder_LBP())
            encoder = SentinelEncoder(bands_selector=EuroSATGroupBands())
            self.model = ClassificationNetwork(encoder)
            # self.model = ClassificationNetwork()
            try:
                self.model.load_state_dict(torch.load(f"./trained_models/classification/{model_name}/best_model.pth"))
                print("Loaded best model")
            except:
                self.model.load_state_dict(torch.load(f"./trained_models/classification/{model_name}/model.pth"))
                print("Loaded model")
        
        self.model.eval()
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model.to(self.device)

    def predict(self, x):
        return self.model(x)
    
    def __accuracy__(self, outputs, labels):
        predicted = outputs.argmax(dim=1)
        return (predicted == labels).sum().item() / len(labels)
    
    def evaluate(self, dataloader):
        all_outs = []
        all_labels = []

        acc = 0.
        pbar = tqdm(dataloader)
        with torch.no_grad():
            for i, (x, labels) in enumerate(pbar):
                # img = torch.cat((x[:,5:6], x[:,4:5], x[:,3:4]), dim=1)
                img = x.to(self.device)
                labels = labels.to(self.device)
                all_labels.append(labels)

                outputs = self.predict(img)
                all_outs.append(outputs)
                
                acc += self.__accuracy__(outputs, labels)
            return acc / len(dataloader), all_outs, all_labels
    
class TrainClassification():
    def __init__(self, model_name=None, model=None) -> None:
        if model:
            self.model = model
        else:
            self.model = ClassificationNetwork()
        if model_name:
            enconder = SentinelEncoder(bands_selector=EuroSATGroupBands())
            self.model = ClassificationNetwork(enconder)
            try:
                self.model.load_state_dict(torch.load(f"./trained_models/classification/{model_name}/best_model.pth"))
                print("Loaded best model")
            except:
                self.model.load_state_dict(torch.load(f"./trained_models/classification/{model_name}/model.pth"))
                print("Loaded model")

        # Train parameters
        self.best_acc = 0.0
        self.model.train()
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model.to(self.device)

        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = torch.optim.Adam(self.model.resnet.model.fc.parameters(), lr=2e-4, weight_decay=2e-5)
        self.scheduler = torch.optim.lr_scheduler.MultiStepLR(self.optimizer, milestones=[60, 80], gamma=0.1)
        
    
    def save(self, path):
        os.makedirs(path, exist_ok=True)
        path_to_save = f"{path}/model.pth"
        torch.save(self.model.state_dict(), path_to_save)

    def train(self, dataloader, val_dataloader, epochs=200):
        pbar = tqdm(range(epochs))
        loss = 0.
        val_acc = 0.
        acc = 0.
        ba_acc = 0.
        ba_acc_val = 0.
        total_val_acc = 0.
        pbar.set_description(f"Loss: {loss:.4f}, Acc: {acc:.4f}, Val Acc: {total_val_acc:.4f}, Balanced Acc Val: {ba_acc_val:.4f}, Balanced Acc: {ba_acc:.4f}")
        for epoch in pbar:
            total_acc = 0.
            all_outs = []
            all_labels = []
            # Train the network
            for i, (x, labels) in enumerate(dataloader):
                acc = 0.
                self.model.train()
                self.optimizer.zero_grad()
                # img = torch.cat((x[:,5:6], x[:,4:5], x[:,3:4]), dim=1)
                img = x.to(self.device)
                outputs = self.model(img)
                labels = labels.to(self.device)

                predicted = outputs.argmax(dim=1)
                all_outs.append(predicted)
                all_labels.append(labels)
                acc += (predicted == labels).sum().item()
                loss = self.criterion(outputs, labels)
                loss.backward()
                self.optimizer.step()

                acc = acc / len(labels)
                total_acc += acc

            ba_acc = balanced_accuracy_score(torch.cat(all_labels).cpu().numpy(), torch.cat(all_outs).cpu().numpy())
            acc = total_acc / len(dataloader)
            writer.add_scalar('Loss/train', loss.item(), epoch)
            writer.add_scalar('Acc/train', acc, epoch)
            writer.add_scalar('Balanced_Acc/train', ba_acc, epoch)
            pbar.set_description(f"Loss: {loss:.4f}, Acc: {acc:.4f}, Val Acc: {total_val_acc:.4f}, Balanced Acc Val: {ba_acc_val:.4f}, Balanced Acc: {ba_acc:.4f}")
            pbar.update(1)

            # Validate the network
            all_outs_val = []
            all_labels_val = []
            with torch.no_grad():
                self.model.eval()
                total_val_acc = 0.
                for i, (x, labels) in enumerate(val_dataloader):
                    val_acc = 0.
                    # img = torch.cat((x[:,5:6], x[:,4:5], x[:,3:4]), dim=1)
                    img = x.to(self.device)
                    outputs = self.model(img)
                    labels = labels.to(self.device)

                    predicted = outputs.argmax(dim=1)
                    all_outs_val.append(predicted)
                    all_labels_val.append(labels)
                    val_acc += (predicted == labels).sum().item()
                    val_acc /= len(labels)
                    total_val_acc += val_acc

                ba_acc_val = balanced_accuracy_score(torch.cat(all_labels_val).cpu().numpy(), torch.cat(all_outs_val).cpu().numpy())
                total_val_acc = total_val_acc / len(val_dataloader)
                writer.add_scalar('Acc/val', total_val_acc, epoch)
                writer.add_scalar('Balanced_Acc/val', ba_acc_val, epoch)
                pbar.set_description(f"Loss: {loss:.4f}, Acc: {acc:.4f}, Val Acc: {total_val_acc:.4f}, Balanced Acc Val: {ba_acc_val:.4f}, Balanced Acc: {ba_acc:.4f}")
                pbar.update(1)
            self.save(f"./trained_models/classification/{name_to_save}")

            if ba_acc_val > self.best_acc:
                torch.save(self.model.state_dict(), f"./trained_models/classification_lbp/{name_to_save}/best_model.pth")
                self.best_acc = ba_acc_val
            else:
                self.model.load_state_dict(torch.load(f"./trained_models/classification_lbp/{name_to_save}/best_model.pth"))
            
            # Scheduler
            self.scheduler.step()

            # checkpoint model and optimizer
            OrderedDict = {
                'model': self.model.state_dict(),
                'optimizer': self.optimizer.state_dict(),
                'scheduler': self.scheduler.state_dict(),
                'epoch': epoch,
                'loss': loss,
                'acc': acc,
            }
            torch.save(OrderedDict, f"./trained_models/classification_lbp/{name_to_save}/checkpoint_{epoch}.pth")

        print('Finished Training')

if __name__ == "__main__":
    
    from torch.utils.tensorboard import SummaryWriter
    import os
    from data_reader.EuroSAT_ms_reader import MSReadData
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Logs
    name_to_save = get_model_time()
    path = f"./logs/classification/{name_to_save}"
    path_lpb = f"./trained_models/classification_lbp/{name_to_save}"
    os.makedirs(path, exist_ok=True)
    os.makedirs(path_lpb, exist_ok=True)
    writer = SummaryWriter(path)
    
    num_classes = 10

    dataset = MSReadData()
    bath_size = 128
    train = True
    img_size = 64

    if train:
        # Dataloader
        path = "./data/EuroSAT"
        dataloader, val_dataloader = dataset.create_dataLoader(path, img_size, bath_size, train=train, num_workers=12)
        path = "./data/EuroSAT_test"
        _, val_dataloader = dataset.create_dataLoader(path, img_size, bath_size, train=True, num_workers=12)
    else:
        bath_size = 64
        path = "./data/EuroSAT"
        dataloader = dataset.create_dataLoader(path, img_size, bath_size, train=train, num_workers=12)
        path = "./data/EuroSAT_test"
        val_dataloader = dataset.create_dataLoader(path, img_size, bath_size, train=train, num_workers=12)

    ## SaCo Loader
    from modules.senti_encoder import load_model_and_weights, SentinelEncoder

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Fine tunning training
    name = "20240422_053009"
    model = TrainClassification(model_name=name)
    model.train(dataloader, val_dataloader, 20)
