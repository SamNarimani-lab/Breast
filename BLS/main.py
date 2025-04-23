import torch
from torch.optim import RAdam
from torch.optim.lr_scheduler import ReduceLROnPlateau
from scripts.data_preprocessing.CD_BE import *
from scripts.modelling.trainer import *
from scripts.modelling.segmentation_models import *
from torchvision.models.segmentation import *
import config.config as config


# Main Training Method

def train_model(
    method, model_name, criterion1_name, criterion2_name,
    input_channels, output_channels, seg_type, data_type, alpha_t=config.ALPHA_T , 
    beta_t=config.BETA_T , lr=config.LR, weight_decay=config.WEIGHT_DECAY, batch_size=config.BATCH_SIZE,
    epochs=config.EPOCHS, patience_m=config.PATIENCE_M, alpha_m=config.ALPHA_M,
    beta_m=config.BETA_M, k_fold=config.K_FOLD, dataset_dir=config.DATASET_DIR,
    mode=config.MODE, factor=config.FACTOR, threshold=config.THRESHOLD,
    threshold_mode=config.THRESHOLD_MODE, cooldown=config.COOLDOWN, patience_s = config.PATIENCE_S,
    min_lr=config.MIN_LR, eps=config.EPS, verbose=config.VERBOSE ):

    # Get the selected model
    model = get_model(model_name, input_channels=input_channels, output_channels=output_channels)
    
    # Optimizer and Scheduler
    optimizer = RAdam(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = ReduceLROnPlateau(
        optimizer, mode, factor, patience_s, threshold,
        threshold_mode, cooldown, min_lr, eps, verbose)

    # Get the selected loss functions (criterions)
    criterion1 = get_criterion(criterion1_name, alpha_t , beta_t)
    criterion2 = get_criterion(criterion2_name , alpha_t , beta_t)

    # Set up the trainer
    trainer = BreastTrainer(
        model=model,
        model_name=model_name, 
        input_channels=input_channels,
        output_channels=output_channels,
        criterion1=criterion1, 
        criterion2=criterion2, 
        optimizer=optimizer, 
        scheduler=scheduler, 
        alpha_m=alpha_m, 
        beta_m=beta_m, 
        alpha_t = alpha_t ,
        beta_t = beta_t,
        k_fold=k_fold if method == 'kfold' else None
    )
    
    # Training depending on the method (3-subset or kfold)
    if method == '3-subset':
        train_loader, val_loader, test_loader = load_3_subset_data(
            seg_type=seg_type,
            batch_size=batch_size,
            data_type=data_type,
            dataset_dir=dataset_dir
        )
        trainer.train(
            train_loader, val_loader, test_loader,
            epochs=epochs, patience=patience_m,
            model_name=model_name,
            seg_type= seg_type
        )
    elif method == 'kfold':
        kfold_dataset = load_kfold_data(
            seg_type=seg_type,
            data_type=data_type,
            dataset_dir=dataset_dir
        )
        trainer.train_cv(
            dataset=kfold_dataset, 
            epochs=epochs, 
            patience_model=patience_m, 
            batch_size=batch_size, 
            model_name=model_name, 
            LR=lr , seg_type=seg_type
        )
    else:
        raise ValueError("Invalid method! Choose between '3-subset' or 'kfold'.")
    
    # Save the model
    torch.save(model.state_dict(), f'{model_name}- {method}-{seg_type}.pt')


# Main function to drive the script
def main():
    train_model(method='kfold',
                model_name='ModifiedBasicUNetPlusPlus',
                input_channels=config.INN_CHANNEL,
                output_channels=config.OUT_CHANNEL,
                criterion1_name='DiceCELoss',
                criterion2_name='DiceFocalLoss',
                seg_type='L',
                data_type='PP1'
                )  
        



if __name__ == "__main__":
    main()








