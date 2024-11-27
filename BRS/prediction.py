import torch
import os
from scripts.data_preprocessing.CD_BE import CustomDataForBreast
from torch.utils.data import  DataLoader
from sklearn.metrics import roc_curve , auc
import numpy as np
from torch import nn
import matplotlib.pyplot as plt
import csv
import seaborn as sns
from monai.metrics import *
import metrics
from MONAI import *
import statistics
import nibabel as nib
import json
import time
import re
from skimage import filters
from matplotlib.patches import Patch, Rectangle
from mpl_toolkits.axes_grid1.inset_locator import zoomed_inset_axes, mark_inset
import seaborn as sns
import pandas as pd


def carbon_footprint_plot(model_results_path):
    cfp_data = []
    model_names = []

    for model_path in model_results_path:
        checkpoint = torch.load(model_path)
        fold_results = checkpoint['fold_results']
        cfp_model = []
        for fold_result in fold_results:
            training_time = fold_result['training_time']
            cfp = training_time * 0.475 / 3600
            cfp_model.append(cfp)
        
        model_name = model_path.split('_')[0]  # Extract model name from the file name
        model_names.extend([model_name] * len(cfp_model))
        cfp_data.extend(cfp_model)

    # Create a DataFrame for easier plotting with Seaborn
    df = pd.DataFrame({
        'Model': model_names,
        'Carbon Footprint (kg CO2)': cfp_data
    })

    # Create the box plot using Seaborn
    f, ax = plt.subplots(figsize=(7, 6))
    sns.boxplot(x='Carbon Footprint (kg CO2)', y='Model', data=df ,  whis=[0, 100] , linewidth = 1)
    
    # plt.xlabel('Models')
    plt.xlabel('Carbon Footprint (kg CO2)'  , fontname = 'Calibri' , fontweight = 'bold')
    plt.title('Carbon Footprint Distribution Across Models', fontsize = 22 , fontname = 'Calibri' , fontweight = 'bold')
    # plt.xticks(rotation=45)
    # plt.grid(True, axis='y')
    # Add in points to show each observation
    sns.stripplot(data=df, x='Carbon Footprint (kg CO2)', y="Model", size=4, color=".3")

    # Tweak the visual presentation
    ax.xaxis.grid(True)
    ax.set(ylabel="")
    #sns.despine(trim=True, left=True)
    plt.show()

    
# model_results_path = ['UNet_kfold_results.pt' , 'UNet++_kfold_results.pt' , 'FCNResNet50_kfold_results.pt' , 'FCNResNet101_kfold_results.pt', 'DenseNet_kfold_results.pt', 'DeepLabv3resnet50_kfold_results.pt', 'DeepLabv3resnet101_kfold_results.pt']

# carbon_footprint_plot(model_results_path)




def plot_and_save_roc(fpr, tpr, roc_auc, model_name):
    plt.figure()
    plt.plot(fpr, tpr, color='darkorange', lw=0.05, linestyle='-', drawstyle='steps-post', label=f'ROC curve (area = {roc_auc:.4f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic')
    plt.legend(loc="lower right")
    plt.savefig(f"{model_name}_roc_curve.pdf")
    plt.close()

def extract_edges(segmentation):
    edges = filters.sobel(segmentation)
    return edges > 0  # Threshold to create a binary edge map
def overlay_plot_diff_models(model_results, model_names, slice_percentage, rows , alpha):
    # Load NIfTI images
    original_img = nib.load('S-PRE-34.nii').get_fdata()
    ground_truth_img = np.rot90(nib.load('S-mask-B-SN-34.nii').get_fdata(), k=3)
    segmentation_files = [model_results[i] for i in range(len(model_results))]
    segmentations = [np.rot90(nib.load(f).get_fdata(), k=3) for f in segmentation_files]

    # Normalize the original image to 0-255
    original_img_normalized = (original_img - np.min(original_img)) / (np.max(original_img) - np.min(original_img)) * 255
    original_img_normalized = original_img_normalized.astype(np.uint8)
    orig_img = np.rot90(original_img_normalized, k=3)

    # Choose a slice to visualize (the middle slice)
    slice_index = int(orig_img.shape[2] * slice_percentage)
    original_slice = orig_img[:, :, slice_index]
    ground_truth_slice = ground_truth_img[:, :, slice_index]
    segmentation_slices = [seg[:, :, slice_index] for seg in segmentations]

    # Define colors and alpha for the segmentations
    colors = [
        [200, 0, 0],       # Red for segmentation 1
        [0, 200, 0],       # Green for segmentation 2
        [247, 129, 191],   # Pink for segmentation 3
        [200, 200, 0],     # Yellow for segmentation 4
        [200, 0, 200],     # Magenta for segmentation 5
        [0, 255, 255],     # Cyan for segmentation 6
        [255, 165, 0],     # Orange for segmentation 7
        [128, 128, 128]    # Gray for segmentation 8
    ]
    # alpha = 0.4  # Transparency level

    # Determine number of columns based on rows
    cols = 8 // rows

    # Create subplots
    fig, axes = plt.subplots(rows, cols, figsize=(18, 9))

    def overlay_segmentation(original_slice, seg_slice, color, alpha):
        overlay = np.stack([original_slice, original_slice, original_slice], axis=-1)
        mask = seg_slice > 0  # Assuming binary masks
        color = np.array(color) / 255.0
        for j in range(3):  # Apply color channel-wise
            overlay[:, :, j] = np.where(mask, (1 - alpha) * overlay[:, :, j] + alpha * color[j] * 255, overlay[:, :, j])
        return overlay

    # Plot original image with ground truth
    overlay_gt = overlay_segmentation(original_slice, ground_truth_slice, colors[7], alpha)
    ax = axes[0] if rows == 1 else axes[0, 0]
    ax.imshow(overlay_gt.astype(np.uint8))
    ax.set_title('Ground Truth', fontname='Calibri', fontweight='bold', fontsize=15)
    ax.axis('off')

    # Plot original image with each prediction
    for i, (seg_slice, model_name) in enumerate(zip(segmentation_slices, model_names)):
        row = (i + 1) // cols if rows > 1 else 0
        col = (i + 1) % cols if rows > 1 else (i + 1)
        ax = axes[row, col] if rows > 1 else axes[col]
        overlay_pred = overlay_segmentation(original_slice, seg_slice, colors[i], alpha)
        ax.imshow(overlay_pred.astype(np.uint8))
        ax.set_title(model_name, fontname='Calibri', fontweight='bold', fontsize=15)
        ax.axis('off')

    # Remove unused subplots
    if rows * cols > len(model_results) + 1:
        for j in range(len(model_results) + 1, rows * cols):
            row = j // cols if rows > 1 else 0
            col = j % cols if rows > 1 else j
            fig.delaxes(axes[row, col] if rows > 1 else axes[col])

    plt.tight_layout()
    plt.show()


# model_results = ['UNet.nii' , 'UNet++.nii' ,'FCNResNet50.nii' , 'FCNResNet101.nii' , 'DenseNet.nii' , 'DeepLabv3resnet50.nii' , 'DeepLabv3resnet101.nii']
# model_name = ['UNet' , 'UNet++' ,'FCNResNet50' , 'FCNResNet101' , 'DenseNet' , 'DeepLabv3Resnet50' , 'DeepLabv3Resnet101']
# overlay_plot_diff_models (model_results , model_name , slice_percentage = 149/150 , rows = 1 , alpha =0.4)



def k_fold_nifti_prediction_save (model_list ,model_results_path, test_path ,threshold):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    for model_name , model_path in zip(model_list,model_results_path):

        dataset_test = CustomDataForBreast(test_path, oversample=True,  min_slices=120, max_slices=160, target_slices=100)
        test_loader = DataLoader(dataset_test, batch_size=1)

        checkpoint = torch.load(model_path)
        model = torch.load(model_name)
        fold_results = checkpoint['fold_results']
    
        with open(test_path , 'r') as f:
            file_paths = json.load(f)
        #numbers = [re.search(r'\d+', path).group() for path in test_path]

        model.eval()
        with torch.no_grad():
            for inputs, masks in test_loader:
                outputs = None
                inputs, masks  = inputs.to(device).float(), masks.to(device).float()     
                f_o = []
                for i in range(inputs.size(-1)):
                    mask_slice = masks[..., i]
                    input_slice = inputs[..., i]

                    predictions = []
                    for fold_result in fold_results:
                        weights = fold_result['model_state']
                        model.load_state_dict(weights)
                        outputs = model(input_slice)              
                        if isinstance(outputs, dict):
                            outputs = outputs['out']
                        predictions.append(outputs)

                    prediction_tensor = torch.stack(predictions, dim=0)
                    kfold_outputs = torch.mean(prediction_tensor, dim=0)
                    outputs = torch.sigmoid(kfold_outputs) 
                    f_o.append(outputs)
                    if outputs.shape != mask_slice.shape:
                        raise ValueError(f"Shape mismatch: outputs shape {outputs.shape}, mask_slice shape {mask_slice.shape}")
                outputs =   torch.stack(f_o , dim=0 )  
                outputs = outputs.permute(1,2,3,4,0)
                outputs = outputs.squeeze().cpu().detach().numpy()
                masks = masks.squeeze().cpu().detach().numpy()
                outputs[outputs >  threshold] = 1
                outputs[outputs <= threshold] = 0

                for file_name in os.listdir(file_paths[0]):
           
                    if file_name.startswith('S-PRE'):
                        pre_contrast_path = os.path.join(file_paths[0], file_name)
                        affine = nib.load(pre_contrast_path).affine
            test_result_nifti = nib.Nifti1Image(outputs, affine= affine)  # Assuming no affine transformation needed
            nib.save(test_result_nifti, f'{model_name}.nii')
model_list = ['FCNResNet50_B_kfold10.pt' , 'FCNResNet101_B_kfold10.pt' ,  'DeepLabv3resnet50_B_kfold10.pt' , 'DeepLabv3resnet101_B_kfold10.pt']
model_results_path = ['FCNResNet50_kfold_results.pt' , 'FCNResNet101_kfold_results.pt',  'DeepLabv3resnet50_kfold_results.pt', 'DeepLabv3resnet101_kfold_results.pt']
k_fold_nifti_prediction_save (model_list= model_list , model_results_path = model_results_path,test_path =r'dataset\processed_data\test1.json'  ,threshold = 0.05 )

def k_fold_results_plot(model_list, model_results_path, test_path, threshold, model_names):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    all_results = []

    for model_file, model_path, model_name in zip(model_list, model_results_path, model_names):
        dataset_test = CustomDataForBreast(test_path, oversample=True, min_slices=120, max_slices=150, target_slices=100)
        test_loader = DataLoader(dataset_test, batch_size=1)

        checkpoint = torch.load(model_path)
        model = torch.load(model_file).to(device)  # Load model according to your saving method
        fold_results = checkpoint['fold_results']

        model.eval()

        model_dice = []
        model_iou = []
        model_hd = []

        with torch.no_grad():
            for inputs, masks in test_loader:
                dice_scores = []
                iou_scores = []
                hd_scores = []

                inputs, masks = inputs.to(device).float(), masks.to(device).float()

                for i in range(inputs.size(-1)):
                    input_slice = inputs[..., i]
                    mask_slice = masks[..., i]

                    predictions = []
                    for fold_result in fold_results:
                        weights = fold_result['model_state']
                        model.load_state_dict(weights)
                        outputs = model(input_slice)
                        if isinstance(outputs, dict):
                            outputs = outputs['out']
                        predictions.append(outputs)

                    prediction_tensor = torch.stack(predictions, dim=0)
                    kfold_outputs = torch.mean(prediction_tensor, dim=0)
                    outputs = torch.sigmoid(kfold_outputs)
                    outputs[outputs > threshold] = 1
                    outputs[outputs <= threshold] = 0
                    tp_m, tp_std, fp_m, fp_std, tn_m, tn_std, fn_m, fn_std = metrics.seg_results(outputs, mask_slice, threshold)
                    accuracy, precision, recall, iou, dice, (fpr, tpr) = metrics.seg_metrics(tp_m, fp_m, tn_m, fn_m)
                    hd = compute_hausdorff_distance (outputs , mask_slice ,distance_metric='euclidean' , percentile=90 )
                    # Calculate metrics for this slice

                    dice_scores.append(dice.item())
                    iou_scores.append(iou.item())
                    hd_scores.append(hd.item())

                # Store metrics for all slices for this model
                model_dice.extend(dice_scores)
                model_iou.extend(iou_scores)
                model_hd.extend(hd_scores)

        # Store results for this model
        all_results.append({
            'Model': model_name,
            'Dice': model_dice,
            'IoU': model_iou,
            'Hausdorff Distance' : model_hd
        })
        print(model_name)

    def plot_scores_violin(all_results, metric):
        sns.set_theme(style="whitegrid")
        plt.figure(figsize=(10, 6))
        metric_data = []
        model_names = []
        
        for result in all_results:
            model_names.extend([result['Model']] * len(result[metric]))
            metric_data.extend(result[metric])
        
        sns.violinplot(x=model_names, y=metric_data , bw_adjust=.5, cut=1, linewidth=1, palette="Set2")   #for pallette Set1,Set2 and Set3 are amazing you can check pastel as well
        plt.title(f'{metric} Scores Across Models' , fontsize = 22 , fontname = 'Calibri' , fontweight = 'bold')
        plt.ylabel(f'{metric} Score' , fontsize = 15 ,fontname = 'Calibri', fontweight = 'bold')
        plt.xticks(rotation = 45)
        plt.tight_layout()
        plt.show()
    def plot_scores_boxen(all_results, metric):
        sns.set_theme(style="whitegrid")
        plt.figure(figsize=(10, 6))
        metric_data = []
        model_names = []
        
        for result in all_results:
            model_names.extend([result['Model']] * len(result[metric]))
            metric_data.extend(result[metric])
        
        sns.boxenplot(x=model_names, y=metric_data , palette="pastel6")   #for pallette Set1,Set2 and Set3 are amazing you can check pastel as well
        plt.title(f'{metric} Scores Across Models' , fontsize = 22 , fontname = 'Calibri' , fontweight = 'bold')
        plt.ylabel(f'{metric} Score' , fontsize = 15 ,fontname = 'Calibri', fontweight = 'bold')
        plt.xticks(rotation = 45)
        plt.tight_layout()
        plt.show()

    # Plotting separately for Dice and IoU scores
    plot_scores_violin(all_results, metric='Dice')
    plot_scores_violin(all_results, metric='IoU')
    plot_scores_boxen(all_results, metric= 'Hausdorff Distance')



# model_names = ['UNet' , 'UNet++' ,'FCNResNet50' , 'FCNResNet101' , 'DenseNet' , 'DeepLabv3Resnet50' , 'DeepLabv3Resnet101']
# model_list = ['UNet_B_kfold10.pt' , 'UNet++_B_kfold10.pt' ,'FCNResNet50_B_kfold10.pt' , 'FCNResNet101_B_kfold10.pt' , 'DenseNet_B_kfold10.pt' , 'DeepLabv3resnet50_B_kfold10.pt' , 'DeepLabv3resnet101_B_kfold10.pt']
# model_results_path = ['UNet_kfold_results.pt' , 'UNet++_kfold_results.pt' , 'FCNResNet50_kfold_results.pt' , 'FCNResNet101_kfold_results.pt', 'DenseNet_kfold_results.pt', 'DeepLabv3resnet50_kfold_results.pt', 'DeepLabv3resnet101_kfold_results.pt']
# k_fold_results_plot (model_list= model_list , model_results_path = model_results_path,test_path =r'dataset\processed_data\test1.json'  ,threshold =  0.99  , model_names= model_names)

def k_fold_prediction (model_path ,model_results_path, test_path ,threshold):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dataset_test = CustomDataForBreast(test_path, oversample=True,  min_slices=120, max_slices=150, target_slices=100)
    test_loader = DataLoader(dataset_test, batch_size=1)

    checkpoint = torch.load(model_results_path)
    model = torch.load(model_path)
    fold_results = checkpoint['fold_results']
    
    model.eval()
    with torch.no_grad():
        

        for inputs, masks in test_loader:
            outputs = None
            inputs, masks  = inputs.to(device).float(), masks.to(device).float()     
            f_o = []
            for i in range(inputs.size(-1)):
                mask_slice = masks[..., i]
                input_slice = inputs[..., i]
                predictions = []

                for fold_result in fold_results:
                    weights = fold_result['model_state']
                    model.load_state_dict(weights)
                    outputs = model(input_slice)              
                    if isinstance(outputs, dict):
                        outputs = outputs['out']
                    predictions.append(outputs)

                prediction_tensor = torch.stack(predictions, dim=0)
                kfold_outputs = torch.mean(prediction_tensor, dim=0)
                outputs = torch.sigmoid(kfold_outputs) 
                outputs[outputs >  threshold] = 1
                outputs[outputs <= threshold] = 0
                f_o.append(outputs)
                if outputs.shape != mask_slice.shape:
                    raise ValueError(f"Shape mismatch: outputs shape {outputs.shape}, mask_slice shape {mask_slice.shape}")
                tp_m , tp_std,  fp_m , fp_std , tn_m , tn_std  , fn_m , fn_std =  metrics.seg_results (outputs , mask_slice , threshold)
                accuracy , precision , recall , iou , dice , (fpr , tpr) = metrics.seg_metrics(tp_m , fp_m , tn_m , fn_m)
                print(dice , compute_hausdorff_distance (outputs , mask_slice))
            outputs =   torch.stack(f_o , dim=0 )  
            outputs = outputs.permute(1,2,3,4,0)
            
            tp_m , tp_std,  fp_m , fp_std , tn_m , tn_std  , fn_m , fn_std =  metrics.seg_results (outputs , masks , threshold)
            accuracy , precision , recall , iou , dice , (fpr , tpr) = metrics.seg_metrics(tp_m , fp_m , tn_m , fn_m)
            #print( dice , compute_hausdorff_distance (outputs , masks))
            # print(compute_hausdorff_distance (outputs , mask_slice))

   

#k_fold_prediction (model_path ='FCNResNet101_B_kfold10.pt'  ,model_results_path = 'FCNResNet101_kfold_results.pt', test_path =r'dataset\processed_data\test1.json'  ,threshold = 0.5)



def inference_time(model_list, model_results_path , test_path ,threshold):
    

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    for model_name , model_path in zip(model_list,model_results_path):
        start_time = time.time()
        checkpoint = torch.load(model_path)
        model = torch.load(model_name).to(device)
        fold_results = checkpoint['fold_results']

        dataset_test = CustomDataForBreast(test_path, oversample=True, min_slices=120, max_slices=150, target_slices=100)
        test_loader = DataLoader(dataset_test, batch_size=1)



        model.eval()
        with torch.no_grad():
            

            for inputs, masks in test_loader:
                outputs = None
                inputs, masks  = inputs.to(device).float(), masks.to(device).float()     
                f_o = []
                for i in range(inputs.size(-1)):
                    mask_slice = masks[..., i]
                    input_slice = inputs[..., i]

                    predictions = []
                    
                    for fold_result in fold_results:
                        weights = fold_result['model_state']
                        model.load_state_dict(weights)

                        outputs = model(input_slice)              
                        if isinstance(outputs, dict):
                            outputs = outputs['out']
                        predictions.append(outputs)

                    prediction_tensor = torch.stack(predictions, dim=0)
                    kfold_outputs = torch.mean(prediction_tensor, dim=0)
                    outputs = torch.sigmoid(kfold_outputs) 
                    outputs[outputs >  threshold] = 1
                    outputs[outputs <= threshold] = 0
                    f_o.append(outputs)
                    if outputs.shape != mask_slice.shape:
                        raise ValueError(f"Shape mismatch: outputs shape {outputs.shape}, mask_slice shape {mask_slice.shape}")

                outputs =   torch.stack(f_o , dim=0 )  
                outputs = outputs.permute(1,2,3,4,0)
        inference_time= time.time() - start_time
        print(f'Inference time for {model_name} is {inference_time}')

# model_list = ['UNet_B_kfold10.pt' , 'UNet++_B_kfold10.pt' ,'FCNResNet50_B_kfold10.pt' , 'FCNResNet101_B_kfold10.pt' , 'DenseNet_B_kfold10.pt' , 'DeepLabv3resnet50_B_kfold10.pt' , 'DeepLabv3resnet101_B_kfold10.pt']
# model_results_path = ['UNet_kfold_results.pt' , 'UNet++_kfold_results.pt' , 'FCNResNet50_kfold_results.pt' , 'FCNResNet101_kfold_results.pt', 'DenseNet_kfold_results.pt', 'DeepLabv3resnet50_kfold_results.pt', 'DeepLabv3resnet101_kfold_results.pt']
# inference_time (model_list= model_list , model_results_path = model_results_path,
#                  test_path =r'dataset\processed_data\test1.json'  ,threshold = 0.5 )

        

def prediction_nifti(model_path,  test_path, threshold):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = torch.load(model_path)
    dataset_test = CustomDataForBreast(test_path, oversample=False, downsample=True, forced_downsample=False,
                                       transform=False, use_subtraction=False, min_slices=120, max_slices=150, target_slices=100)
    test_loader = DataLoader(dataset_test, batch_size=6)

    tp , fp , tn ,fn = [] , [] ,[] , []
    # all_true_labels , all_pred_scores = [] ,[]
    dice_score , IoU= [] , []
    acc ,precision_value , recall_value = [] , [] , []
    fpr_tpr , roc_auc = [] , []
    # confusion_matrix = []

    
    model.eval()
    with torch.no_grad():
        for inputs, masks in test_loader:
            inputs, masks = inputs.to(device).float(), masks.to(device).float()

            for i in range(inputs.size(-1)):
                mask_slice = masks[..., i]
                input_slice = inputs[..., i]


                outputs = model(input_slice)

                if isinstance(outputs, dict):
                    outputs = outputs['out']
                
                # outputs = torch.sigmoid(outputs).squeeze().cpu().detach().numpy()    
                # mask_slice = mask_slice.squeeze().cpu().detach().numpy()
                outputs = torch.sigmoid(outputs) 
                
                if outputs.shape != mask_slice.shape:
                    raise ValueError(f"Shape mismatch: outputs shape {outputs.shape}, mask_slice shape {mask_slice.shape}")

                tp_m , tp_std,  fp_m , fp_std , tn_m , tn_std  , fn_m , fn_std =  metrics.seg_results (outputs , mask_slice , threshold)
                tp.append((tp_m , tp_std)) , fp.append((fp_m , fp_std)) , tn.append((tn_m , tn_std)) , fn.append((fn_m , fn_std))
                accuracy , precision , recall , iou , dice , (fpr , tpr) = metrics.seg_metrics(tp_m , fp_m , tn_m , fn_m)
                acc.append(accuracy) , precision_value.append(precision) , recall_value.append(recall)
                IoU.append(iou) , dice_score.append(dice) , fpr_tpr.append((fpr , tpr))
                
                
                


                # outputs_flat = outputs.flatten()
                # masks_flat = mask_slice.flatten()
                # FPR , TPR , _ = roc_curve(masks_flat , outputs_flat)
                # x = auc(FPR ,TPR)
                # roc_auc.append(x)
                # Dice_score .append( f1_score(masks_flat , outputs_flat > threshold) )
                # acc.append(accuracy_score(masks_flat , outputs_flat > threshold))  , ap_value.append(average_precision_score(masks_flat , outputs_flat > threshold))
                # precision_value.append(precision_score(masks_flat , outputs_flat > threshold)) , recall_value.append(recall_score(masks_flat , outputs_flat > threshold))

                # # Collect true labels and predicted scores for ROC
                # all_true_labels.extend(masks_flat)
                # all_pred_scores.extend(outputs_flat)

    # print(dice_score)
    # mean_Dice = Dice_score.mean()
    # print(Dice_score)
    # print(mean_Dice , len(Dice_score))
    
    # print(IoU)
    # Convert lists to numpy arrays
    # all_true_labels = np.array(all_true_labels)
    # all_pred_scores = np.array(all_pred_scores)
    # # Compute metrics
    # fpr, tpr, _ = roc_curve(all_true_labels, all_pred_scores)
    # roc_auc = auc(fpr, tpr)
    # precision, recall, _ = precision_recall_curve(all_true_labels, all_pred_scores)
    # average_precision = average_precision_score(all_true_labels, all_pred_scores)
    # f1 = f1_score(all_true_labels, all_pred_scores > threshold)
    # accuracy = accuracy_score(all_true_labels, all_pred_scores > threshold)
    # precision_score_value = precision_score(all_true_labels, all_pred_scores > threshold)
    # recall_score_value = recall_score(all_true_labels, all_pred_scores > threshold)
    
    # Create confusion matrix
    # cm = confusion_matrix(all_true_labels, all_pred_scores > threshold)

    # Save metrics to CSV
    # metrics_dict = {
    #     'ROC AUC': roc_auc,
    #     'Average Precision': average_precision,
    #     'Dice Score': f1,
    #     'Accuracy': accuracy,
    #     'Precision': precision_score_value,
    #     'Recall': recall_score_value,
    #     'Confusion Matrix': cm
    # }
    # save_metrics_to_csv(metrics_dict, model_name)

    # # Plot and save ROC curve and PR curve
    # plot_and_save_roc(fpr, tpr, roc_auc, model_name)
    # plot_and_save_pr(precision, recall, average_precision, model_name)
    # plot_confusion_matrix (cm ,model_name)

    # print(f"Metrics saved to {model_name}.csv")
    # print(f"ROC curve saved to {model_name}_roc_curve.pdf")
    # print(f"PR curve saved to {model_name}_pr_curve.pdf")
    # print(f"Confusion Matrix:\n{cm}")

def save_metrics_to_csv(metrics_dict, model_name):
    """
    Save metrics dictionary to a CSV file.
    :param metrics_dict: Dictionary containing metrics as keys and their values.
    :param model_name: Name of the model to save the CSV file.
    """
    csv_file_path = f"{model_name}.csv"
    # Write the metrics to the CSV file
    with open(csv_file_path, 'w', newline='') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=metrics_dict.keys())
        writer.writeheader()
        writer.writerow(metrics_dict)

def calculate_mAP(precision, recall):
    """
    Calculate mean Average Precision (mAP) from Precision-Recall (PR) values.
    :param precision: Precision values for each class
    :param recall: Recall values for each class
    :return: mAP (mean Average Precision)
    """
    # Ensure precision and recall are numpy arrays
    precision = np.array(precision)
    recall = np.array(recall)
    
    # Calculate Average Precision (AP) for each class
    ap = []
    for i in range(len(precision)):
        # Use trapezoidal rule to approximate area under precision-recall curve
        ap.append(np.trapz(precision[i], recall[i]))
    
    # Calculate mean Average Precision (mAP)
    mAP = np.mean(ap)
    
    return mAP



def plot_and_save_pr(precision, recall, average_precision, model_name):
    plt.figure()
    plt.plot(recall, precision, color='blue', lw=0.05, linestyle='-', drawstyle='steps-post', label=f'PR curve (AP = {average_precision:.4f})')
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('Precision-Recall Curve')
    plt.legend(loc="lower left")
    plt.savefig(f"{model_name}_pr_curve.pdf")
    plt.close()

def plot_confusion_matrix(cm , model_name ):
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title('Confusion Matrix')
    plt.savefig(f"confusion_matrix_{model_name}.pdf" , format='pdf')
    plt.close()


# prediction_nifti (model_path = 'UNet_kfold_results.pt' ,
#                   test_path = r'dataset\processed_data\test_files_B.json' , threshold = 0.9)

def model_results (results_path ):
    checkpoint = torch.load(results_path)
    fold_results = checkpoint['fold_results']

    
    
    training_time = []
    train_loss_best = []
    train_loss_last = []
    val_loss_best = []
    val_loss_last = []
    last_epoch = []
    for fold_result in fold_results:
        training_time.append(fold_result['training_time'])
        results = fold_result['results']
        best_epoch = fold_result['best_epoch']
        best_train_loss, best_val_loss = results[best_epoch]
        last_train_loss, last_val_loss = results[-1]
        train_loss_best.append(best_train_loss)
        train_loss_last.append(last_train_loss)
        val_loss_best.append(best_val_loss)
        val_loss_last.append(last_val_loss)

    train_time_mean = int(statistics.mean(training_time)/60) + 1
    train_time_std = int(statistics.stdev(training_time)/60) + 1
    
    train_loss_best_mean = statistics.mean(train_loss_best)
    train_loss_best_std = statistics.stdev(train_loss_best)
    val_loss_best_mean = statistics.mean(val_loss_best)
    val_loss_best_std = statistics.stdev(val_loss_best)
    train_loss_last_mean = statistics.mean(train_loss_last)
    train_loss_last_std = statistics.stdev(train_loss_last)
    val_loss_last_mean = statistics.mean(val_loss_last)
    val_loss_last_std = statistics.stdev(val_loss_last)

    print(f'{train_time_mean} +- {train_time_std} ')
    print('Best epoch results are :',f'Train Loss : {train_loss_best_mean:.4f} +- {train_loss_best_std:.4f}'  ,  f',  Validation loss : {val_loss_best_mean:.4f} +- {val_loss_best_std:.4f}')

    print('Last epoch results are :',f'Train loss : {train_loss_last_mean:.4f} +- {train_loss_last_std:.4f}'  ,  f',  Validation loss : {val_loss_last_mean:.4f} +- {val_loss_last_std:.4f}')





def number_param_layers(model):
    total_params = sum(p.numel() for p in model.parameters())
    total_layers = sum(1 for _ in model.modules())
    return total_params , total_layers

# print(model_results('DeepLabv3ResNet50_kfold_results.pt'))
