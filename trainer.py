import os
import time
import copy
import torch
import torch.nn.functional as F
import torchvision
from torchvision import datasets, transforms
from torch.utils.data.dataloader import DataLoader
from torch.utils.tensorboard import SummaryWriter
from zmq import device
import torch.nn
from utils import _create_model_training_folder
from data.transforms import target_transforms


class BYOLTrainer:
    def __init__(self, online_network, target_network, predictor, optimizer, device, **params):
        self.online_network = online_network
        self.target_network = target_network
        self.optimizer = optimizer
        self.device = device
        self.predictor = predictor
        self.max_epochs = params['max_epochs']
        self.writer = SummaryWriter()
        self.m = params['m']
        self.batch_size = params['batch_size']
        self.num_workers = params['num_workers']
        self.checkpoint_interval = params['checkpoint_interval']
        _create_model_training_folder(self.writer, files_to_same=["./config/config.yaml", "main.py", 'trainer.py'])

    @torch.no_grad()
    def _update_target_network_parameters(self):
        """
        Momentum update of the key encoder.
        """
        for param_q, param_k in zip(self.online_network.parameters(), self.target_network.parameters()):
            param_k.data = param_k.data * self.m + param_q.data * (1. - self.m)

    @staticmethod
    def regression_loss(x, y):
        x = F.normalize(x, dim=1)
        y = F.normalize(y, dim=1)
        return 2 - 2 * (x * y).sum(dim=-1)

    def initializes_target_network(self):
        # init momentum network as encoder net
        for param_q, param_k in zip(self.online_network.parameters(), self.target_network.parameters()):
            param_k.data.copy_(param_q.data)  # initialize
            param_k.requires_grad = False  # not update by gradient

    def train(self, train_dataset):

        train_loader = DataLoader(train_dataset, batch_size=self.batch_size,
                                  num_workers=self.num_workers, drop_last=False, shuffle=True)

        niter = 0
        model_checkpoints_folder = os.path.join(self.writer.log_dir, 'checkpoints')

        self.initializes_target_network()

        for epoch_counter in range(self.max_epochs):

            for (batch_view_1, batch_view_2), _ in train_loader:

                batch_view_1 = batch_view_1.to(self.device)
                batch_view_2 = batch_view_2.to(self.device)

                if niter == 0:
                    grid = torchvision.utils.make_grid(batch_view_1[:32])
                    self.writer.add_image('views_1', grid, global_step=niter)

                    grid = torchvision.utils.make_grid(batch_view_2[:32])
                    self.writer.add_image('views_2', grid, global_step=niter)

                loss = self.update(batch_view_1, batch_view_2)
                self.writer.add_scalar('loss', loss, global_step=niter)

                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

                self._update_target_network_parameters()  # update the key encoder
                niter += 1

            print("End of epoch {}".format(epoch_counter))

        # save checkpoints
        self.save_model(os.path.join(model_checkpoints_folder, 'model.pth'))

    def update(self, batch_view_1, batch_view_2):
        # compute query feature
        predictions_from_view_1 = self.predictor(self.online_network(batch_view_1))
        predictions_from_view_2 = self.predictor(self.online_network(batch_view_2))

        # compute key features
        with torch.no_grad():
            targets_to_view_2 = self.target_network(batch_view_1)
            targets_to_view_1 = self.target_network(batch_view_2)

        loss = self.regression_loss(predictions_from_view_1, targets_to_view_1)
        loss += self.regression_loss(predictions_from_view_2, targets_to_view_2)
        return loss.mean()

    def save_model(self, PATH):

        torch.save({
            'online_network_state_dict': self.online_network.state_dict(),
            'target_network_state_dict': self.target_network.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
        }, PATH)


class BYOLTransferTrainer:
    def __init__(self, online_network, target_network, predictor, optimizer, scheduler, device, **params):
        self.online_network = online_network
        #self.target_network = target_network
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.device = device
        self.predictor = predictor
        self.max_epochs = params['max_epochs']
        self.writer = SummaryWriter()
        self.m = params['m']
        self.batch_size = params['batch_size']
        self.num_workers = params['num_workers']
        self.checkpoint_interval = params['checkpoint_interval']
        _create_model_training_folder(self.writer, files_to_same=["./config/config.yaml", "main.py", 'trainer.py'])
        
        #self.criterion = torch.nn.CrossEntropyLoss()
        self.criterion = torch.nn.BCEWithLogitsLoss()

    @staticmethod
    def crossEntropy_loss(criterion,x,y):
        loss = criterion(x,y)
        return loss
    
    @torch.no_grad()
    def _update_target_network_parameters(self):
        """
        Momentum update of the key encoder.
        FN: param_q is for online_network, param_k is for target_network
        """
        for param_q, param_k in zip(self.online_network.parameters(), self.target_network.parameters()):
            param_k.data = param_k.data * self.m + param_q.data * (1. - self.m)

    @staticmethod
    def regression_loss(x, y):
        x = F.normalize(x, dim=1)
        y = F.normalize(y, dim=1)
        return 2 - 2 * (x * y).sum(dim=-1)
    
    '''
    def initializes_target_network(self):
        # init momentum network as encoder net
        for param_q, param_k in zip(self.online_network.parameters(), self.target_network.parameters()):
            param_k.data.copy_(param_q.data)  # initialize
            param_k.requires_grad = False  # not update by gradient
    '''
    def train(self, train_dataset):
        since = time.time()
        #self.online_network.eval()
        best_model_wts = copy.deepcopy(self.online_network.state_dict())
        best_acc = 0.0

        train_loader = {x: DataLoader(train_dataset[x], batch_size=self.batch_size,
                                  num_workers=self.num_workers, drop_last=False, shuffle=True) 
                             for x in ['train','val']}
        
        dataset_sizes = {x: len(train_dataset[x]) for x in ['train', 'val']}

        niter = 0
        model_checkpoints_folder = os.path.join(self.writer.log_dir, 'checkpoints')

        '''
        if self.target_network:
            self.initializes_target_network()
        '''
        
        for epoch_counter in range(self.max_epochs):
            
            print(f'Epoch {epoch_counter}/{self.max_epochs - 1}')
            print('-' * 10)
            
            for phase in ['train', 'val']:
                if phase == 'train':
                    self.online_network.train()  # Set model to training mode
                else:
                    self.online_network.eval()   # Set model to evaluate mode
                
                running_loss = 0.0
                running_corrects = 0
                
                for (batch_view_1, targets) in train_loader[phase]:
                    batch_view_1 = batch_view_1.to(self.device)
                    targets = targets.to(self.device)

                    if niter == 0:
                        grid = torchvision.utils.make_grid(batch_view_1[:2])
                        self.writer.add_image('views_1', grid, global_step=niter)

                        #grid = torchvision.utils.make_grid(batch_view_2[:32])
                        #self.writer.add_image('views_2', grid, global_step=niter)

                    with torch.set_grad_enabled(phase == 'train'):
                        loss, preds = self.update(batch_view_1, targets)
                        #self.writer.add_scalar('loss', loss, global_step=niter)
                        #self.writer.flush()

                        if phase == 'train':
                            self.optimizer.zero_grad()
                            loss.backward()
                            self.optimizer.step()

                    # statistics
                    running_loss += loss.item() * batch_view_1.size(0)
                    running_corrects += torch.div(torch.sum(preds.to(self.device) == targets.data), targets.data.shape[1], rounding_mode='trunc')
                    
                    #self._update_target_network_parameters()  # update the key encoder
                    niter += 1
                    # self.resnet.layer1[0].conv1.weight  Första=5.7593e-02
                if phase == 'train':
                    #self.scheduler.step() # NOT READY
                    pass

                epoch_loss = running_loss / dataset_sizes[phase]
                epoch_acc = running_corrects.double() / dataset_sizes[phase]

                print(f'{phase} Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}')

                # deep copy the model
                if phase == 'val' and epoch_acc > best_acc:
                    best_acc = epoch_acc
                    best_model_wts = copy.deepcopy(self.online_network.state_dict())

            print("End of epoch {}, Loss={}".format(epoch_counter, loss))

        time_elapsed = time.time() - since
        print(f'Training complete in {time_elapsed // 60:.0f}m {time_elapsed % 60:.0f}s')
        print(f'Best val Acc: {best_acc:4f}')

        # load best model weights
        self.online_network.load_state_dict(best_model_wts)

        # save checkpoints
        self.save_model(os.path.join(model_checkpoints_folder, 'model.pth'))
        self.save_model(os.path.join('./runs/pretrained_network/checkpoints', 'model.pth'))

        return self.online_network

    def prediction_transform(self,l):
        y =[]
        for x in l:
            y.append([1.,0.] if x==0 else [0.,1.])
        return torch.FloatTensor(y)
    
    def update(self, batch_view_1, targets):
        #loss = 0
        # compute query feature
        #predictions_from_view_1 = self.predictor(self.online_network(batch_view_1))
        #predictions_from_view_2 = self.predictor(self.online_network(batch_view_2))
        predictions_from_view_1 = self.online_network(batch_view_1)
        #predictions_from_view_2 = self.online_network(batch_view_2)

        # compute key features
        #with torch.no_grad():
            #pass
            #targets_to_view_1 = self.target_network(batch_view_1)
            #targets_to_view_1 = self.target_network(batch_view_2)

        #loss = self.regression_loss(predictions_from_view_1, targets)
        loss = self.crossEntropy_loss(self.criterion, predictions_from_view_1, targets)
        _, preds = torch.max(predictions_from_view_1, 1)
        preds = self.prediction_transform(preds)
        #loss += self.regression_loss(predictions_from_view_2, targets_to_view_2)
        return loss, preds
        #return loss.mean(), preds

    def save_model(self, PATH):

        torch.save({
            'online_network_state_dict': self.online_network.state_dict(),
            #'target_network_state_dict': self.target_network.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
        }, PATH)
