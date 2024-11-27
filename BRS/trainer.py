import torch
from torch.utils.data import DataLoader
from sklearn.model_selection import KFold
from torch.utils.data import DataLoader, Subset
from seg_models import *
import time
from torch.optim.lr_scheduler import *
from monai.networks.nets import AttentionUnet


class BreastTrainer:
    def __init__(self, model, criterion1, criterion2, optimizer, 
                 lr_scheduler_params , alpha, beta ,k_fold  ):

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

    
    def train_cv (self,  dataset , epochs , patience_model, batch_size , model_name ):

        device= torch.device('cuda' if torch.cuda.is_available() else 'cpu')   
        kfold = KFold(n_splits=self.k_fold, shuffle=True)
        fold_results = []
        for fold, (train_ids, val_ids) in enumerate(kfold.split(dataset)):
           
            result = []
            print(f'FOLD {fold}')
            print('------------------------------------------------------')   

            train_subsampler = Subset(dataset, train_ids)
            val_subsampler = Subset(dataset, val_ids)

            train_loader = DataLoader(train_subsampler, batch_size=batch_size, shuffle=True)
            val_loader = DataLoader(val_subsampler, batch_size=batch_size, shuffle=True)  

            best_val_loss = float('inf')
            epochs_no_improve = 0
            train_loss = 0

            # Initialize the model for this fold
            self.model = CustomFCN(weight = None ,  version = 'resnet50' , num_input_channels =  1 ,
                                    num_classes = 1  ).to(device=device)
            self.model.apply(reset_weights)
            
            # Initialize optimizer for this fold
            self.optimizer = torch.optim.RAdam(self.model.parameters(), lr=0.00001 , weight_decay= 0.00003)
            self.scheduler = ReduceLROnPlateau(self.optimizer, mode='min', factor=0.1, patience=2,
                                                threshold=0.0001, threshold_mode='rel', cooldown=0,
                                                min_lr=0, eps=1e-08, verbose='True')
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
                
            # Calculate training time for cfp
            training_time = time.time() - start_time
            
            # Save model state and optimizer state 
            model_state = self.model.state_dict()
            optimizer_state = self.optimizer.state_dict()

            fold_results.append({'fold': fold,'results': result,'best_epoch': best_epoch,'last_epoch': epoch - 1 ,
                                 'model_state': model_state,  'optimizer_state': optimizer_state,'training_time': training_time })
            
        torch.save({'fold_results': fold_results}, f'{model_name}_kfold_results.pt')
        print('Training complete')

def reset_weights(m):

  for layer in m.children():
   if hasattr(layer, 'reset_parameters'):
    layer.reset_parameters()