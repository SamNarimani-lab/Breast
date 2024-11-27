import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import torch.optim as optim
import numpy as np
from sklearn.model_selection import KFold
from torch.utils.data import DataLoader, Dataset, Subset
from MONAI import *
import copy
import time
from DenseNetme import DenseNet
from torch.optim.lr_scheduler import *
from monai.networks.nets import AttentionUnet


def reset_weights(m):
  '''
    Try resetting model weights to avoid
    weight leakage.
  '''
  for layer in m.children():
   if hasattr(layer, 'reset_parameters'):
    # print(f'Reset trainable parameters of layer = {layer}')
    layer.reset_parameters()


class BreastTrainer:
    def __init__(self, model, criterion1, criterion2, optimizer, lr_scheduler_params , alpha, beta ,k_fold  ):
        self.model = model
        self.criterion1 = criterion1
        self.criterion2 = criterion2
        self.optimizer = optimizer
        self.alpha = alpha
        self.beta = beta
        self.k_fold = k_fold
        self.lr_scheduler_params  = lr_scheduler_params
        


    def train_step(self, inputs, mask ):
        self.model.train()
        self.optimizer.zero_grad()
        
        outputs = self.model(inputs)
        if isinstance(outputs, dict):
            outputs = outputs['out']
        loss1 = self.criterion1(outputs, mask)
        loss2 = self.criterion2(outputs, mask)
        loss = self.alpha * loss1 + self.beta * loss2
        loss.backward()
        self.optimizer.step()

        return loss.item()
    
    def validation(self, val_loader):
        val_loss = 0
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.eval()
        with torch.no_grad():
            for inputs, masks in val_loader:
                inputs, masks = inputs.to(device).float(), masks.to(device).float()
                for i in range(inputs.size(-1)):
                    outputs = self.model(inputs[..., i ])
                    
                    if isinstance(outputs, dict):
                        outputs = outputs['out']


                    loss1 = self.criterion1(outputs, masks[..., i ])
                    loss2 = self.criterion2(outputs, masks[..., i ])
                    loss = self.alpha * loss1 + self.beta * loss2
                    val_loss += loss.item()
        val_loss /= (len(val_loader) * inputs.size(-1) )

        return val_loss
    
    def test(self, test_loader):
        test_loss = 0
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.eval()
        with torch.no_grad():
            for inputs, masks in test_loader:
                inputs, masks = inputs.to(device).float(), masks.to(device).float()
                for i in range(inputs.size(-1)):
                    outputs = self.model(inputs[..., i ])

                    if isinstance(outputs, dict):
                        outputs = outputs['out']


                    loss1 = self.criterion1(outputs, masks[..., i ])
                    loss2 = self.criterion2(outputs, masks[..., i ])
                    loss = self.alpha * loss1 + self.beta * loss2
                    test_loss += loss.item()

        test_loss /= ( len(test_loader) * inputs.size(-1) )

        return test_loss
    
    def train(self, train_loader, val_loader, test_loader, epochs, patience , model_name ):

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        best_val_loss = float('inf')
        no_improvement = 0 
        results = []
        all_results = []
        self.model = DoubleUNet (spatial_dims1 = 2 , spatial_dims2=2 , in_channels1 = 2 , out_channels1 =64 , features1 = (64 ,128 ,256, 512 ,1024, 64), dropout1 =0.05
                     , in_channels2 = 64 , out_channels2 = 1 , features2 =(64 ,128 ,256, 512 ,1024, 64) , dropout2 = 0.05).to(device=device)
        self.model.apply(reset_weights)
            
        # Initialize optimizer
        self.optimizer = torch.optim.RAdam(self.model.parameters(), lr=0.00001 , weight_decay= 0.00003)
        self.scheduler = ReduceLROnPlateau(self.optimizer, mode='min', factor=0.1, patience=2, threshold=0.0001, threshold_mode='rel', cooldown=0, min_lr=0, eps=1e-08, verbose='True')
        start_time = time.time()
        for epoch in range(epochs):
            train_loss = 0
            self.model.train()
            for inputs, masks in train_loader:
                inputs, masks = inputs.to(device).float(), masks.to(device).float()
                for i in range(inputs.size(-1)):
                    loss = self.train_step(inputs[..., i ], masks[..., i ])
                    train_loss += loss
                
            train_loss /= (len(train_loader) * inputs.size(-1))
            val_loss = self.validation(val_loader)
            test_loss = self.test(test_loader)
            results.append((train_loss , val_loss , test_loss))
            self.lr_scheduler_params.step(val_loss)
            # Early stopping 
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                no_improvement = 0
                best_epoch = epoch
            else:
                no_improvement += 1
                if no_improvement > patience:
                    print('Early stopping')
                    break
            print(f'Epoch {epoch+1}/{epochs}, Train loss: {train_loss:.4f}, Validation Loss: {val_loss:.4f}, Test Loss: {test_loss:.4f}')
        training_time = time.time( ) - start_time
        all_results.append({'results': results,'best_epoch': best_epoch,'last_epoch': epoch  ,'training_time': training_time })
        torch.save({'all_results': all_results}, f'{model_name}_results.pt')
        print('Training complete')
        


    
    def train_cv (self,  dataset , epochs , patience_model, batch_size , model_name ):
        device= torch.device('cuda' if torch.cuda.is_available() else 'cpu')   
        kfold = KFold(n_splits=self.k_fold, shuffle=True)
        

        fold_results = []
        for fold, (train_ids, val_ids) in enumerate(kfold.split(dataset)):
           
            result = []
            print(f'FOLD {fold}')
            print('--------------------------------')   


            train_subsampler = Subset(dataset, train_ids)
            val_subsampler = Subset(dataset, val_ids)

            train_loader = DataLoader(train_subsampler, batch_size=batch_size, shuffle=True)
            val_loader = DataLoader(val_subsampler, batch_size=batch_size, shuffle=True)  

            best_val_loss = float('inf')
            epochs_no_improve = 0
            train_loss = 0

            # Init the neural network
            self.model = CustomFCN(weight = None ,  version = 'resnet50' , num_input_channels =  1 , num_classes = 1  ).to(device=device)
            self.model.apply(reset_weights)
            
            # Initialize optimizer
            self.optimizer = torch.optim.RAdam(self.model.parameters(), lr=0.00001 , weight_decay= 0.00003)
            self.scheduler = ReduceLROnPlateau(self.optimizer, mode='min', factor=0.1, patience=2, threshold=0.0001, threshold_mode='rel', cooldown=0, min_lr=0, eps=1e-08, verbose='True')
                # Track the training time for each fold
            start_time = time.time()
            for epoch in range(epochs):
                for inputs, masks in train_loader:
                    inputs, masks = inputs.to(device).float(), masks.to(device).float()
                    for i in range(inputs.size(-1)):
                        loss = self.train_step(inputs[... , i ], masks[... , i ])
                        train_loss += loss

                train_loss /= (len(train_loader) * inputs.size(-1))
                val_loss = self.test(val_loader)  

                result.append((train_loss , val_loss))

                print(f'Epoch {epoch+1}/{epochs}, Train loss: {train_loss:.4f} , Test Loss: {val_loss:.4f}')

                self.lr_scheduler_params.step(val_loss)

                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    epochs_no_improve = 0
                    best_epoch = epoch
                else:
                    epochs_no_improve += 1
                
                if epochs_no_improve >= patience_model:
                    print(f'Early stopping at epoch {epoch+1}')
                    break
                
            # Calculate training time
            training_time = time.time() - start_time
            
            # Save model state and optimizer state
            model_state = self.model.state_dict()
            optimizer_state = self.optimizer.state_dict()

            fold_results.append({'fold': fold,'results': result,'best_epoch': best_epoch,'last_epoch': epoch - 1 ,
                                 'model_state': model_state,  'optimizer_state': optimizer_state,'training_time': training_time })
            
        torch.save({'fold_results': fold_results}, f'{model_name}_kfold_results.pt')
        print('Training complete')






# import torch
# from torch._six import int_classes as _int_classes
# from torch import Tensor

# from typing import Iterator, Optional, Sequence, List, TypeVar, Generic, Sized

# T_co = TypeVar('T_co', covariant=True)

# class Sampler(Generic[T_co]):
#     r"""Base class for all Samplers.

#     Every Sampler subclass has to provide an :meth:`__iter__` method, providing a
#     way to iterate over indices of dataset elements, and a :meth:`__len__` method
#     that returns the length of the returned iterators.

#     .. note:: The :meth:`__len__` method isn't strictly required by
#               :class:`~torch.utils.data.DataLoader`, but is expected in any
#               calculation involving the length of a :class:`~torch.utils.data.DataLoader`.
#     """

#     def __init__(self, data_source: Optional[Sized]) -> None:
#         pass

#     def __iter__(self) -> Iterator[T_co]:
#         raise NotImplementedError
        
# class SubsetRandomSampler(Sampler[int]):
#     r"""Samples elements randomly from a given list of indices, without replacement.

#     Args:
#         indices (sequence): a sequence of indices
#         generator (Generator): Generator used in sampling.
#     """
#     indices: Sequence[int]

#     def __init__(self, indices: Sequence[int], generator=None) -> None:
#         self.indices = indices
#         self.generator = generator

#     def __iter__(self):
#         return (self.indices[i] for i in torch.randperm(len(self.indices), generator=self.generator))

#     def __len__(self):
#         return len(self.indices) 


# train_dataset = CustomDataset(data_dir=train_path, mode='train') )
# val_dataset = CustomDataset(data_dir=train_path, mode='val') )

#     fold = KFold(5, shuffle=True, random_state=random_seed)
#     for fold,(tr_idx, val_idx) in enumerate(fold.split(dataset)):
#         # initialize the model
#         model = smp.FPN(encoder_name='efficientnet-b4', classes=12 , encoder_weights=None, activation='softmax2d')
    
 
     
#         loss = BCEDiceLoss()
#         optimizer = torch.optim.AdamW([
#             {'params': model.decoder.parameters(), 'lr': 1e-07/2}, 
#             {'params': model.encoder.parameters(), 'lr': 5e-07},  
#         ])
#         scheduler = ReduceLROnPlateau(optimizer, factor=0.15, patience=2)
    
  
    
#         print('#'*35); print('############ FOLD ',fold+1,' #############'); print('#'*35);
#         train_loader = torch.utils.data.DataLoader(dataset=train_dataset, 
#                                                batch_size=batch_size,
#                                                num_workers=1,
#                                                sampler = SubsetRandomSampler(tr_idx)
#                                             )
#         val_loader = torch.utils.data.DataLoader(dataset=val_dataset, 
#                                                batch_size=batch_size,
#                                                num_workers=1,
#                                                sampler = SubsetRandomSampler(val_idx)
#                                             )