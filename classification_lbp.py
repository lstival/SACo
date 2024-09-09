from classification import ClassificationNetwork, TrainClassification
import torch
import torch.nn as nn
from lbp import lbp_histogram
from modules.senti_encoder import SentinelEncoderLBP
from modules.sentinel_bands import EuroSATGroupBands
from tqdm import tqdm
from modules.resnet import ResNet_enconder_LBP
import os 

class ClassificationNetworkLBP(ClassificationNetwork):
    def __init__(self, encoder=None) -> None:
        super().__init__(encoder)

    def forward(self, x, hist):
        x = self.resnet(x, hist)
        return x
    
class TrainClassificationLBP(TrainClassification):

    def __init__(self, model=None, model_name=None) -> None:
        if model:
            self.model = model
        else:
            self.model = ClassificationNetworkLBP()

        if model_name:
            enconder = SentinelEncoderLBP(bands_selector=EuroSATGroupBands(), model=ResNet_enconder_LBP())
            self.model = ClassificationNetworkLBP(enconder)
            try:
                self.model.load_state_dict(torch.load(f"./trained_models/classification_lbp/{model_name}/best_model.pth"))
                print("Loaded best model")
            except:
                self.model.load_state_dict(torch.load(f"./trained_models/classification_lbp/{model_name}/model.pth"))
                print("Loaded model")

        # self.model = ClassificationNetwork(enconder)
        self.model.resnet.model.fc =  nn.Linear(512+(13*16), 10)
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=0.001)
        self.scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer, step_size=7, gamma=0.1)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)

        self.best_acc = 0.

    def save(self, path):
        os.makedirs(path, exist_ok=True)
        path_to_save = f"{path}/model.pth"
        torch.save(self.model.state_dict(), path_to_save)

    def train(self, dataloader, val_dataloader, epochs=10):
        pbar = tqdm(range(epochs))
        loss = 0.
        val_acc = 0.
        acc = 0.
        pbar.set_description(f"Loss: {loss:.4f}, Acc: {acc:.4f}, Val Acc: {val_acc:.4f}")
        for epoch in pbar:
            total_acc = 0.
            # Train the network
            for i, (x, labels) in enumerate(dataloader):
                acc = 0.
                self.model.train()
                self.optimizer.zero_grad()
                # img = torch.cat((x[:,5:6], x[:,4:5], x[:,3:4]), dim=1)

                hist = lbp_histogram((x*255).type(torch.int)).view(x.shape[0], -1).type(torch.float32)
                img = x.to(self.device)
                hist = hist.to(self.device)

                outputs = self.model(img, hist)
                labels = labels.to(self.device)

                predicted = outputs.argmax(dim=1)
                acc += (predicted == labels).sum().item()
                loss = self.criterion(outputs, labels)
                loss.backward()
                self.optimizer.step()

                acc = acc / len(labels)
                total_acc += acc

            acc = total_acc / len(dataloader)
            writer.add_scalar('Loss/train', loss.item(), epoch)
            writer.add_scalar('Acc/train', acc, epoch)
            pbar.set_description(f"Loss: {loss:.4f}, Acc: {acc:.4f}, Val Acc: {val_acc:.4f}")
            pbar.update(1)

            # Validate the network
            with torch.no_grad():
                self.model.eval()
                total_val_acc = 0.
                for i, (x, labels) in enumerate(val_dataloader):
                    val_acc = 0.
                    # img = torch.cat((x[:,5:6], x[:,4:5], x[:,3:4]), dim=1)
                    hist = lbp_histogram((x*255).type(torch.int)).view(x.shape[0], -1).type(torch.float32)
                    img = x.to(self.device)
                    hist = hist.to(self.device)

                    outputs = self.model(img, hist)
                    labels = labels.to(self.device)

                    predicted = outputs.argmax(dim=1)
                    val_acc += (predicted == labels).sum().item()
                    val_acc /= len(labels)
                    total_val_acc += val_acc

                total_val_acc = total_val_acc / len(val_dataloader)
                writer.add_scalar('Acc/val', total_val_acc, epoch)
                pbar.set_description(f"Loss: {loss:.4f}, Acc: {acc:.4f}, Val Acc: {total_val_acc:.4f}")
                pbar.update(1)
            self.save(f"./trained_models/classification_lbp/{name_to_save}")

            if total_val_acc > self.best_acc:
                torch.save(self.model.state_dict(), f"./trained_models/classification_lbp/{name_to_save}/best_model.pth")
                self.best_acc = total_val_acc
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
    from data_reader.EuroSAT_ms_reader import MSReadData
    from utils import get_model_time
    import os
    from torch.utils.tensorboard import SummaryWriter
    from modules.senti_encoder import load_model_and_weights

    dataset = MSReadData()

    img_size = 64
    bath_size = 64
    train = True

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Logs
    name_to_save = get_model_time()
    path = f"./logs/classification_lbp/{name_to_save}"
    os.makedirs(path, exist_ok=True)
    writer = SummaryWriter(path)

    dataloader, val_dataloader = dataset.create_dataLoader("./data/EuroSAT", img_size, bath_size, train=train, num_workers=12)
    # Train False
    # dataloader = dataset.create_dataLoader("./data/EuroSAT_test", img_size, bath_size, train=False, num_workers=12)

    # Train
    model_name = "20240420_095820"   
    model_path = f"./trained_models/encoder/{model_name}/q_weights.pt"

    # Create the models
    encoder = load_model_and_weights(model_path, bands_selector=EuroSATGroupBands(), model_resnet=ResNet_enconder_LBP(), lbp=True)
    encoder = encoder.to(device)

    network = ClassificationNetworkLBP(encoder).to(device)
    model = TrainClassificationLBP(model=network)
    model.train(dataloader, val_dataloader, epochs=100)
