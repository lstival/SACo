# datalaoder
from data_reader.caco_ms_reader import MSReadData

# model
from modules import resnet
from modules.senti_encoder import SentinelEncoder
from modules.sentinel_bands import SentinelGroupBands, SentinelGroupBandsLBP

# Troch modules
import torch
import torch.nn as nn

# Other modules
import copy
import torch.nn.functional as F
import time
from tqdm import tqdm

from torch.utils.tensorboard import SummaryWriter

class TrainingEncoder():
    def __init__(self, q_encoder, k_encoder, num_epochs, opt, data_dl, val_dl, checkpoint_name=None) -> None:
        """
        Initialize the TrainingEncoder class.

        Parameters:
        - q_encoder: The query encoder model.
        - k_encoder: The key encoder model.
        - num_epochs: The number of training epochs.
        - opt: The optimizer for the model.
        - data_dl: The data loader for training data.
        """

        self.best_loss = 999.

        # init the training parameters
        self.q_encoder = q_encoder

        if checkpoint_name is not None:
            self.q_encoder, opt, _, _ = self.load_checkpoint(q_encoder, opt, f"./trained_models/encoder/{checkpoint_name}/checkpoint_10.pt")
        
        self.k_encoder = k_encoder
        self.num_epochs = num_epochs
        self.opt = opt
        # Training data
        self.data_dl = data_dl
        # Validation data
        self.val_dl = val_dl
        # Create the first queue (with negative samples)
        self.queue = self.init_queue(data_dl, K=K)
        self.val_queue = self.init_queue(val_dl, K=K)

        # Loss for bands
        self.criterion = nn.CosineEmbeddingLoss(reduction='sum', margin=-1.)
        
    def load_checkpoint(self, model, optimizer, path):
        """
        Load a checkpoint for the model.

        Parameters:
        - model: The model to load the checkpoint.
        - optimizer: The optimizer for the model.
        - path: The path to the checkpoint.

        Returns:
        - The model and optimizer with the checkpoint loaded.
        """
        checkpoint = torch.load(path)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        epoch = checkpoint['epoch']
        loss = checkpoint['loss']
        return model, optimizer, epoch, loss
    
    def simclr_loss(self, q, k, t=0.5):
        """
        Compute the SimCLR loss.

        Parameters:
        - q: The query matrix.
        - k: The key matrix.
        - t: The temperature value.

        Returns:
        - The loss value.
        """
        N = q.shape[0]  # batch_size
        C = q.shape[1]  # channel

        # Normalize the query and key vectors
        q = torch.div(q, torch.norm(q, dim=1).reshape(-1, 1))
        k = torch.div(k, torch.norm(k, dim=1).reshape(-1, 1))

        # Compute the cosine similarity matrix
        similarity_matrix = torch.mm(q, k.t())

        # Compute the contrastive loss
        loss = -torch.log(torch.exp(similarity_matrix / t).sum(dim=1) / torch.exp(similarity_matrix / t).diag()).mean()

        return loss

    def contrastive_loss(self, matrix1, matrix2, margin):
        """
        Compute the contrastive loss between two matrices.

        Parameters:
        - matrix1: The first matrix.
        - matrix2: The second matrix.
        - margin: The margin value for the loss.

        Returns:
        - The contrastive loss.
        """
        euclidean_distance = F.pairwise_distance(matrix1, matrix2)
        loss_contrastive = torch.mean((margin - euclidean_distance) ** 2)
        return loss_contrastive
    
    def init_queue(self, dataloader, K=8192):
        """
        Initialize the SeCo queue.

        Parameters:
        - dataloader: The data loader for training data.

        Returns:
        - The initialized queue.
        """
        #initialize the queue
        queue = None
        # K = 8192 # K: number of negatives to store in queue
        # K = 192 # K: number of negatives to store in queue
        flag = 0
        if queue is None:
            while True:
                with torch.no_grad():
                    for img, _, _, _ in dataloader:
                        # extract key samples
                        # xk = img[1].to(device)
                        xk = img.to(device)
                        k = self.k_encoder(xk)
                        k = k.detach()

                        if queue is None:
                            queue = k
                        else:
                            if queue.shape[0] < K: # queue < 8192
                                queue = torch.cat((queue,k),0)
                            else:
                                flag = 1 # stop filling the queue

                        if flag == 1:
                            break 

                if flag == 1:
                    break

        queue = queue[:K]
        return queue
        
    def loss_function(self, q, k, queue, t=0.05):
        """
        Compute the loss function for MoCo V2.

        Parameters:
        - q: The query matrix.
        - k: The key matrix.
        - queue: The Moco queue.
        - t: The temperature value.

        Returns:
        - The loss value.
        """
        # Loss for cluster (model output)
        # t: temperature

        N = q.shape[0]  # batch_size
        C = q.shape[1]  # channel

        # bmm: batch matrix multiplication
        pos = torch.exp(torch.div(torch.bmm(q.view(N, 1, C), k.view(N, C, 1)).view(N, 1), t))
        neg = torch.sum(torch.exp(torch.div(torch.mm(q.view(N, C), torch.t(queue.view(-1, C))), t)), dim=1)

        # Avoid division by zero and inf numbers
        pos = torch.where(torch.isinf(pos), torch.zeros_like(pos), pos)
        neg = torch.where(torch.isinf(neg), torch.zeros_like(neg), neg)
        pos = torch.where(torch.isnan(pos), torch.zeros_like(pos), pos)
        neg = torch.where(torch.isnan(neg), torch.zeros_like(neg), neg)

        # pos = torch.exp(torch.div(torch.bmm(q.view(N, 1, C), k.view(N, C, 1)).view(N, 1), t))
        # neg = torch.sum(torch.exp(torch.div(torch.mm(q.view(N, C), torch.t(queue.view(-1, C))), t)), dim=1)

        # denominator is sum over pos and neg
        denominator = pos + neg

        return torch.mean(-torch.log(torch.div(pos, denominator)))

    def bands_loss(self, out_q):
        """
        Compute the loss for bands.

        Parameters:
        - out_q: The output of the query encoder.

        Returns:
        - The loss value.
        """
        # Adjust to keep close to the cluster value
        loss = 0.0
        for i in range(len(self.q_encoder.outs)):
            # loss += self.criterion(self.q_encoder.outs[i], out_q, torch.tensor([1.0]).to(device))
            loss += self.contrastive_loss(self.q_encoder.outs[i], out_q, 1.0)
            # loss += self.simclr_loss(self.q_encoder.outs[i], out_q, 0.5)
        loss /= len(self.q_encoder.outs)
        return loss
    
    def validate(self, val_dataloader):
        """
        Validate the model on the validation dataset.

        Parameters:
        - val_dataloader: The data loader for the validation dataset.

        Returns:
        - The validation loss.
        """
        self.q_encoder.eval()
        val_loss = 0

        with torch.no_grad():
            for img, arg_positive, temp_positive, _ in val_dataloader:
                xq = img.to(device)
                xk = arg_positive.to(device)
                xk2 = temp_positive.to(device)

                q = self.q_encoder(xq)
                k = self.k_encoder(xk)
                k = k.detach()
                k2 = self.k_encoder(xk2)
                k2 = k2.detach()

                q = torch.div(q, torch.norm(q, dim=1).reshape(-1, 1))
                k = torch.div(k, torch.norm(k, dim=1).reshape(-1, 1))
                k2 = torch.div(k2, torch.norm(k2, dim=1).reshape(-1, 1))

                loss = self.loss_function(q, k, self.val_queue)
                loss += self.loss_function(q, k2, self.val_queue)

                loss_bands = self.bands_loss(q)
                loss += loss_bands

                val_loss += loss.item()

        return val_loss / len(val_dataloader.dataset), loss, loss_bands

    def train(self):
        """
        Train the model using the SeCo method.

        Returns:
        - The trained query encoder, key encoder, and loss history.
        """
        loss_history = []
        momentum = 0.999
        path2weights = f'./trained_models/encoder/{model_name}/q_weights.pt'

        len_data = len(self.data_dl.dataset)

        self.q_encoder.train()
        val_loss = 0
        epoch_loss = 0

        for epoch in range(num_epochs):
            # print('Epoch {}/{}'.format(epoch, num_epochs-1))

            self.q_encoder.train()
            running_loss = 0
            pbar = tqdm(enumerate(self.data_dl), total=len(self.data_dl))

            pbar.set_description(f"Epoch: {epoch}, Loss: {epoch_loss:.6f}, Val Loss: {val_loss:.6f}")
            for i, (img, arg_positive, temp_positive, _) in pbar:
                """
                img: image in time 0
                positive: image in time 1
                """
                # retrieve query and key
                xq = img.to(device) # orignal image
                xk = arg_positive.to(device) # argumented version of original image
                xk2 = temp_positive.to(device) # original image in other time moment

                # get model outputs
                q = self.q_encoder(xq)
                k = self.k_encoder(xk)
                k = k.detach()
                k2 = self.k_encoder(xk2)
                k2 = k2.detach()

                # normalize representations
                q = torch.div(q, torch.norm(q,dim=1).reshape(-1,1))
                k = torch.div(k, torch.norm(k,dim=1).reshape(-1,1))
                k2 = torch.div(k2, torch.norm(k2,dim=1).reshape(-1,1))

                # Loss for clusters (original and argumented)
                loss = self.loss_function(q, k, self.queue)
                # Loss for clusters (original and temporal)
                loss += self.loss_function(q, k2, self.queue)

                # Loss for bands
                loss_bands = self.bands_loss(q)
                loss += loss_bands

                running_loss += loss

                # log the loss
                writter.add_scalar('Loss/train', loss, epoch * len_data + i)
                writter.add_scalar('Cluster Loss/train', loss, epoch * len_data + i)
                writter.add_scalar('Bands Loss/train', loss_bands, epoch * len_data + i)

                opt.zero_grad()
                loss.backward()
                opt.step()

                # update the queue
                self.queue = torch.cat((self.queue, k), 0)

                if self.queue.shape[0] > K:
                    self.queue = self.queue[256:,:]
                
                # update k_encoder
                for q_params, k_params in zip(self.q_encoder.parameters(), self.k_encoder.parameters()):
                    k_params.data.copy_(momentum*k_params + q_params*(1.0-momentum))

                # pbar.set_description(f"Loss: {loss.item():.6f}")
                pbar.set_description(f"Epoch: {epoch}, Loss: {epoch_loss:.6f}, Val Loss: {val_loss:.6f}")

            # store loss history
            epoch_loss = running_loss / len(self.data_dl.dataset)
            loss_history.append(epoch_loss)

            # validate the model
            val_loss, loss, loss_bands = self.validate(self.val_dl)

            if val_loss < self.best_loss:
                torch.save(self.q_encoder.state_dict(), f"./trained_models/encoder/{model_name}/best_model.pth")
                self.best_loss = val_loss
            else:
                self.q_encoder.load_state_dict(torch.load(f"./trained_models/encoder/{model_name}/best_model.pth"))


            writter.add_scalar('Cluster Loss/val', loss, epoch)
            writter.add_scalar('Bands Loss/val', loss_bands, epoch)
            writter.add_scalar('Loss/val', val_loss, epoch)

            # save weights
            torch.save(self.q_encoder.state_dict(), path2weights)
            # Save a checkpoint
            torch.save({
                'epoch': epoch,
                'model_state_dict': self.q_encoder.state_dict(),
                'optimizer_state_dict': opt.state_dict(),
                'loss': epoch_loss,
                }, f'./trained_models/encoder/{model_name}/checkpoint_{epoch}.pt')

        return self.q_encoder, self.k_encoder, loss_history

class TrainingEncoderMoCo(TrainingEncoder):

    def __init__(self, q_encoder, k_encoder, num_epochs, opt, data_dl, val_dl, checkpoint_name=None) -> None:
        self.best_loss = 999.

        # init the training parameters
        self.q_encoder = q_encoder

        if checkpoint_name is not None:
            self.q_encoder, opt, _, _ = self.load_checkpoint(q_encoder, opt, f"./trained_models/encoder/{checkpoint_name}/checkpoint_10.pt")
        
        self.k_encoder = k_encoder
        self.num_epochs = num_epochs
        self.opt = opt
        # Training data
        self.data_dl = data_dl
        # Validation data
        self.val_dl = val_dl
        # Create the first queue (with negative samples)
        self.queue = self.init_queue(data_dl, K=K)
        self.val_queue = self.init_queue(val_dl, K=K)

        # Loss for bands
        self.criterion = nn.CosineEmbeddingLoss(reduction='sum', margin=-1.)

    def init_queue(self, dataloader, K=8192):
        """
        Initialize the Moco queue.

        Parameters:
        - dataloader: The data loader for training data.

        Returns:
        - The initialized queue.
        """
        #initialize the queue
        queue = None
        # K = 8192 # K: number of negatives to store in queue
        # K = 192 # K: number of negatives to store in queue
        flag = 0
        if queue is None:
            while True:
                with torch.no_grad():
                    for img, _, _ in dataloader:
                        # extract key samples
                        # xk = img[1].to(device)
                        xk = img.to(device)
                        k = self.k_encoder(xk)
                        k = k.detach()

                        if queue is None:
                            queue = k
                        else:
                            if queue.shape[0] < K: # queue < 8192
                                queue = torch.cat((queue,k),0)
                            else:
                                flag = 1 # stop filling the queue

                        if flag == 1:
                            break 

                if flag == 1:
                    break

        queue = queue[:K]
        return queue
    
    def validate(self, val_dataloader):
        """
        Validate the model on the validation dataset.

        Parameters:
        - val_dataloader: The data loader for the validation dataset.

        Returns:
        - The validation loss.
        """
        self.q_encoder.eval()
        val_loss = 0

        with torch.no_grad():
            for img, positive, _ in val_dataloader:
                xq = img.to(device)
                xk = positive.to(device)

                q = self.q_encoder(xq)
                k = self.k_encoder(xk)
                k = k.detach()

                q = torch.div(q, torch.norm(q, dim=1).reshape(-1, 1))
                k = torch.div(k, torch.norm(k, dim=1).reshape(-1, 1))

                loss = self.loss_function(q, k, self.val_queue)
                loss_bands = self.bands_loss(q)
                loss += loss_bands

                val_loss += loss.item()

        return val_loss / len(val_dataloader.dataset), loss, loss_bands
        return val_loss / len(val_dataloader.dataset), loss, loss_bands

    def train(self):
        """
        Train the model.

        Returns:
        - The trained query encoder, key encoder, and loss history.
        """
        loss_history = []
        momentum = 0.999
        start_time = time.time()
        path2weights = f'./trained_models/encoder/{model_name}/q_weights.pt'

        len_data = len(self.data_dl.dataset)

        self.q_encoder.train()
        val_loss = 0
        epoch_loss = 0

        for epoch in range(num_epochs):
            # print('Epoch {}/{}'.format(epoch, num_epochs-1))

            self.q_encoder.train()
            running_loss = 0
            pbar = tqdm(enumerate(self.data_dl), total=len(self.data_dl))

            pbar.set_description(f"Epoch: {epoch}, Loss: {epoch_loss:.6f}, Val Loss: {val_loss:.6f}")
            for i, (img, positive, _) in pbar:
                """
                img: image in time 0
                positive: image in time 1
                """
                # retrieve query and key
                xq = img.to(device)
                xk = positive.to(device)

                # get model outputs
                q = self.q_encoder(xq)
                k = self.k_encoder(xk)
                k = k.detach()

                # normalize representations
                q = torch.div(q, torch.norm(q,dim=1).reshape(-1,1))
                k = torch.div(k, torch.norm(k,dim=1).reshape(-1,1))

                # Loss for clusters
                loss = self.loss_function(q, k, self.queue)
                loss = self.loss_function(q, k, self.queue)

                # Loss for bands
                loss_bands = self.bands_loss(q)
                loss += loss_bands

                running_loss += loss

                # log the loss
                writter.add_scalar('Loss/train', loss, epoch * len_data + i)

                opt.zero_grad()
                loss.backward()
                opt.step()

                # update the queue
                self.queue = torch.cat((self.queue, k), 0)

                if self.queue.shape[0] > K:
                    self.queue = self.queue[256:,:]
                
                # update k_encoder
                for q_params, k_params in zip(self.q_encoder.parameters(), self.k_encoder.parameters()):
                    k_params.data.copy_(momentum*k_params + q_params*(1.0-momentum))

                # pbar.set_description(f"Loss: {loss.item():.6f}")
                pbar.set_description(f"Epoch: {epoch}, Loss: {epoch_loss:.6f}, Val Loss: {val_loss:.6f}")

            # store loss history
            epoch_loss = running_loss / len(self.data_dl.dataset)
            loss_history.append(epoch_loss)

            # validate the model
            val_loss, loss, loss_bands = self.validate(self.val_dl)
            writter.add_scalar('Cluster Loss/val', loss, epoch)
            writter.add_scalar('Bands Loss/val', loss_bands, epoch)
            writter.add_scalar('Loss/val', val_loss, epoch)

            # save weights
            torch.save(self.q_encoder.state_dict(), path2weights)
            # Save a checkpoint
            torch.save({
                'epoch': epoch,
                'model_state_dict': self.q_encoder.state_dict(),
                'optimizer_state_dict': opt.state_dict(),
                'loss': epoch_loss,
                }, f'./trained_models/encoder/{model_name}/checkpoint_{epoch}.pt')

        return self.q_encoder, self.k_encoder, loss_history

if __name__ == "__main__":
    print("Main Training...")
    import os
    from utils.utils import get_model_time
    import argparse
    import random
    random.seed(42)

    model_name = get_model_time()
    log_path = f'./runs/encoder/{model_name}'

    os.makedirs(f'./trained_models/encoder/{model_name}', exist_ok=True)
    os.makedirs(log_path, exist_ok=True)

    writter = SummaryWriter(log_path)

    # parser = argparse.ArgumentParser()
    # parser.add_argument("--batch_size", type=int, default=256, help="Batch size for training")
    # parser.add_argument("--img_size", type=int, default=64, help="Image size for training")
    # parser.add_argument("--path", type=str, default=r"../clean_100k_geography", help="Path to the dataset")
    # parser.add_argument("--num_epochs", type=int, default=20, help="Number of epochs for training")
    # parser.add_argument("--train_mode", type=str, default="seco", help="Training mode ['SeCo', 'MoCo', 'Temporal']")
    # args = parser.parse_args()

    # batch_size = args.batch_size
    # img_size = args.img_size
    # path = args.path
    # num_epochs = args.num_epochs
    # mode = args.train_mode.upper()  # SeCo, MoCo, Temporal
    # train=True

    batch_size = 64
    img_size = 64
    mode = "SECO"
    path = r".\data\mini_caco10k\clean_10k_geography"
    # path = "./data/caco100k/clean_100k_geography"
    # path = "E:/seasonal_contrast_1m"

    num_epochs = 10
    train=True

    K = 384
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model_resnet = resnet.ResNet_encoder(pretrained=True).to(device)

    # Create the models
    bands_selector = SentinelGroupBandsLBP(ndvi_bands=False)
    model_q = SentinelEncoder(model_resnet, bands_selector).to(device)
    model_k = copy.deepcopy(model_q).to(device)

    # Define the optimizer
    opt = torch.optim.Adam(model_q.parameters(), lr=1e-4, weight_decay=1e-5)

    # Train data
    train = True
    dataset = MSReadData()
    dataloader, val_dataloader = dataset.create_dataLoader(path, img_size, batch_size, train=train, mode=mode)

    # Load checkpoint
    chekpoint_name = None
    # chekpoint_name = "20240423_091732"
    # path = f"./trained_models/encoder/{chekpoint_name}/checkpoint_10.pt"
    
    # Trainer with Checkpoint
    # Trainer = TrainingEncoder(model_q, model_k, num_epochs, opt, dataloader, val_dataloader, chekpoint_name)

    if mode == "SECO":
        print("Training with SeCo method...")
        if chekpoint_name is not None:
                Trainer = TrainingEncoder(model_q, model_k, num_epochs, opt, dataloader, val_dataloader, chekpoint_name)
        else:
            Trainer = TrainingEncoder(model_q, model_k, num_epochs, opt, dataloader, val_dataloader)
    elif mode == "MOCO":
        print("Training with MoCo method...")
        if chekpoint_name is not None:
            Trainer = TrainingEncoderMoCo(model_q, model_k, num_epochs, opt, dataloader, val_dataloader, chekpoint_name)
        else:
            Trainer = TrainingEncoderMoCo(model_q, model_k, num_epochs, opt, dataloader, val_dataloader)
    else:
        raise ValueError("Invalid training mode")
    
    # Train the model
    model_q, model_k, loss_history = Trainer.train()
    
    print("End of training...")