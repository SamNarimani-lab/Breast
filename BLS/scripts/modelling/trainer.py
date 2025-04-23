import torch
from torch.utils.data import DataLoader
from sklearn.model_selection import KFold
from torch.utils.data import DataLoader, Subset
from .segmentation_models import *
from monai.losses import *
from torch.nn import BCELoss
import time
from torch.optim.lr_scheduler import *
from monai.networks.nets import *
import config.config as config


device= torch.device('cuda' if torch.cuda.is_available() else 'cpu')   

# Function to select model
def get_model(model_name , input_channels , output_channels):
    """Function to return a model based on the selected name."""
    if model_name == 'CustomFCN':
        return CustomFCN(weight=None, version='resnet50', num_input_channels= input_channels, num_classes= output_channels).to(device=device)
    elif model_name == 'DeepLabv3':
        return CustomDeepLabv3(weight=None, version='resnet50', num_input_channels = input_channels , num_classes = output_channels).to(device=device)
    elif model_name == 'AttentionUnet':
        return AttentionUnet(spatial_dims = 2 , in_channels = input_channels , out_channels =  output_channels, channels = (64, 128 ,256 ,512, 1024 ), 
                             strides =(2,2,2,2,2) , kernel_size=3, up_kernel_size=3, dropout=0.05).to(device=device)
    elif model_name == 'DoubleUNet':
        return DoubleUNet(spatial_dims1=2, spatial_dims2=2, in_channels1 = input_channels, out_channels1=2,
                           features1=(32, 64 , 128 , 256, 512 , 32),dropout1=0.05, in_channels2=2,
                            out_channels2= output_channels, features2=(16 ,32, 64 , 128 , 256, 16), dropout2=0.05).to(device=device)
    elif model_name == 'ModifiedBasicUNet':
        return ModefiedBasicUNet (spatial_dims= 2 , in_channels = input_channels  , out_channels = output_channels ,
                                  features = (  64, 128, 256 ,512 ,1024, 64 ) , dropout = 0.0).to(device=device)
    elif model_name == 'ModifiedBasicUNetPlusPlus':
        return ModefiedBasicUNetPlusPlus (spatial_dims = 2 , in_channels = input_channels  , out_channels = output_channels ,
                                            features = ( 32, 32, 64, 128, 256, 32) , dropout = 0.0).to(device=device)
    elif model_name == 'UNetPlusPlusASPP':
        return ModefiedBasicUNetPlusPlus (spatial_dims = 2 , in_channels = input_channels  , out_channels = output_channels ,
                                            features = ( 32, 32, 64 , 128 , 256 , 32) , dropout = 0.00).to(device=device)
    elif model_name =='AHUNet':
        return AHNet(layers=(3, 4, 6, 3), spatial_dims=2, in_channels=input_channels, out_channels=output_channels, 
                     psp_block_num=2, upsample_mode='transpose', pretrained=False, progress=True)
    else:
        raise ValueError(f"Unknown model: {model_name}")
    
# Function to select loss functions (criterions)
def get_criterion(criterion_name , alpha_t , beta_t):
    """Function to return a loss function (criterion) based on the selected name."""
    if criterion_name == 'MaskedDiceLoss':
        return MaskedDiceLoss(include_background=False, sigmoid=True)
    elif criterion_name == 'DiceCELoss':
        return DiceCELoss(include_background=True, sigmoid=True, jaccard=False, lambda_dice=0.1, lambda_ce=0.9)
    elif criterion_name == 'BCELoss':
        return BCELoss()
    elif criterion_name == 'DiceFocalLoss':
        return DiceFocalLoss(include_background=True, sigmoid=True, gamma=2, lambda_dice=0.1, lambda_focal=0.9)
    elif criterion_name == 'TverskyLoss':
        return TverskyLoss(include_background=False , to_onehot_y=False, sigmoid=True, softmax=False, 
                           other_act=None, alpha=alpha_t, beta=beta_t)
    
    else:
        raise ValueError(f"Unknown criterion: {criterion_name}")

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
    def __init__(self,  model , model_name, input_channels , output_channels
                  ,  criterion1, criterion2, optimizer, scheduler
                    , alpha_m, beta_m , alpha_t , beta_t ,k_fold   ):
        
        self.model = model
        self.model_name = model_name
        self.input_channels = input_channels
        self.output_channels = output_channels
        self.criterion1 = criterion1
        self.criterion2 = criterion2
        self.optimizer = optimizer
        self.alpha_m = alpha_m
        self.beta_m = beta_m
        self.k_fold = k_fold
        self.scheduler  = scheduler
        self.alpha_t = alpha_t
        self.beta_t = beta_t
        


    def train_step(self, inputs, mask ):
        self.model.train()
        self.optimizer.zero_grad()
        
        outputs = self.model(inputs)
        if isinstance(outputs, dict):
            outputs = outputs['out']
        loss1 = self.criterion1(outputs, mask)
        loss2 = self.criterion2(outputs, mask)
        loss = self.alpha_m * loss1 + self.beta_m * loss2
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
                    loss = self.alpha_m * loss1 + self.beta_m * loss2
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
                    loss = self.alpha_m * loss1 + self.beta_m * loss2
                    test_loss += loss.item()

        test_loss /= ( len(test_loader) * inputs.size(-1) )

        return test_loss
    
    def train(self, train_loader, val_loader, test_loader, epochs, patience , model_name ,seg_type  ):

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        best_val_loss = float('inf')
        no_improvement = 0 
        results = []
        all_results = []
        self.model.apply(reset_weights)
            
        # Initialize optimizer
        # self.optimizer = torch.optim.RAdam(self.model.parameters(), lr=0.00001 , weight_decay= 0.00003)
        # self.scheduler = ReduceLROnPlateau(self.optimizer, mode='min', factor=0.1, patience=2, threshold=0.0001, threshold_mode='rel', cooldown=0, min_lr=0, eps=1e-08, verbose='True')
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
            self.scheduler.step(val_loss)
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
        torch.save({'all_results': all_results}, f'{model_name}-3-subset-{seg_type}_results.pt')
        print('Training complete')
        


    
    def train_cv (self,  dataset , epochs , patience_model, batch_size , model_name,LR ,seg_type  ):
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
            self.model = get_model(self.model_name, self.input_channels, self.output_channels).to(device=device)
            self.model.apply(reset_weights)
            
            # Initialize optimizer
            self.optimizer = torch.optim.RAdam(self.model.parameters(), lr=LR , weight_decay= 0.00003)
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

                self.scheduler.step(val_loss)

                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    epochs_no_improve = 0
                    best_epoch = epoch

                    # Save the model state for the best epoch
                    best_model_state = self.model.state_dict()
                    best_optimizer_state = self.optimizer.state_dict()
                else:
                    epochs_no_improve += 1
                
                if epochs_no_improve >= patience_model:
                    print(f'Early stopping at epoch {epoch+1}')
                    break
                
            # Calculate training time
            training_time = time.time() - start_time
            

            fold_results.append({'fold': fold,'results': result,'best_epoch': best_epoch,'last_epoch': epoch - 1 ,
                                 'model_state': best_model_state,  'optimizer_state': best_optimizer_state,'training_time': training_time })
            
        torch.save({'fold_results': fold_results}, f'{model_name}-KFold-{seg_type}_results.pt')
        print('Training complete')






