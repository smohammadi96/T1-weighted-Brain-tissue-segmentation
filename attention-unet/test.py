from typing import IO
import torch
from loss import dice_coeff, FocalLoss
from model import AttentionUNet
from utils import *
import numpy as np
import os 

os.environ['CUDA_LAUNCH_BLOCKING'] = "1"
batch_size = 4
data_dir = "T1_with_noise"
dataloaders = get_data_loaders(data_dir, batch_size=batch_size)
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model = AttentionUNet()
#state_dict = torch.load("T1_model_attention_unet_MS.pt")
state_dict = torch.load("T1_model_attention_unet_MS.pt", map_location='cpu')
#loaded_state = torch.load(model_path+seq_to_seq_test_model_fname,map_location='cuda:0')

model.load_state_dict(state_dict)
device = 'cpu'
model.to('cpu')

def calculate_metrics(predicted, ground_truth):
    # Convert predicted and ground truth images to binary arrays
    predicted = (predicted != 0).astype(int)
    ground_truth = (ground_truth != 0).astype(int)

    # Calculate True Positives (TP)
    tp = np.sum(np.logical_and(predicted == 1, ground_truth == 1))

    # Calculate True Negatives (TN)
    tn = np.sum(np.logical_and(predicted == 0, ground_truth == 0))

    # Calculate False Positives (FP)
    fp = np.sum(np.logical_and(predicted == 1, ground_truth == 0))

    # Calculate False Negatives (FN)
    fn = np.sum(np.logical_and(predicted == 0, ground_truth == 1))

    return tp, tn, fp, fn


import numpy as np
from scipy.spatial.distance import directed_hausdorff

def calculate_hausdorff_distance(predicted_mask, ground_truth_mask):
    # Convert masks to binary numpy arrays
    predicted_mask_binary = np.asarray(predicted_mask, dtype=bool)
    ground_truth_mask_binary = np.asarray(ground_truth_mask, dtype=bool)

    # Compute Hausdorff distance
    distance = directed_hausdorff(predicted_mask_binary, ground_truth_mask_binary)[0]

    return distance

# Example usage
predicted_mask = [[0, 0, 1, 1, 0],
                  [0, 0, 1, 1, 0],
                  [0, 1, 1, 0, 0]]
ground_truth_mask = [[0, 1, 1, 0, 0],
                     [0, 1, 1, 0, 0],
                     [0, 0, 0, 0, 0]]

distance = calculate_hausdorff_distance(predicted_mask, ground_truth_mask)
print("Hausdorff distance:", distance)

from scipy.spatial.distance import directed_hausdorff

def calculate_hd(ground_truth, predicted):
    import pudb; pu.db
    distance_up = directed_hausdorff(ground_truth, predicted)[0]
    distance_down = directed_hausdorff(predicted, ground_truth)[0]
    hd = max(distance_up, distance_down)
    return hd

import numpy as np
from scipy.ndimage import distance_transform_edt
from scipy.ndimage.morphology import distance_transform_edt

def calculate_assd(ground_truth, predicted):
    gt_dist = distance_transform_edt(1 - ground_truth)
    pred_dist = distance_transform_edt(1 - predicted)
    surface_dist = distance_transform_edt(ground_truth) + distance_transform_edt(predicted)
    assd = np.mean(surface_dist[ground_truth > 0])
    return assd

results_dice = []
results_SENS = []
results_SPEC = []
results_IOU = []
results_EF = []
results_DSC = []
results_PPV = []
results_NPV = []
results_hd = []
results_assd = []
for phase in ['training', 'test']:
    # Iterate over data.
    for sample in iter(dataloaders[phase]):
        inputs = sample['image'].to(device)
        masks = sample['mask'].to(device)
        
        outputs = model(inputs).detach().cpu()
        y_pred = outputs.data.cpu().numpy().ravel()
        y_true = masks.data.cpu().numpy().ravel()
        results, mask, target = dice_coeff(y_pred, y_true)
        tp, tn, fp, fn = calculate_metrics(mask, target)
        SENS = tp / (tp + fn)
        SPEC = tn / (tn + fp)
        EF = fp / (tn + fn)
        IOU = tp / (tp + fn + fp)
        DSC = (2 * tp) / ((2 * tp)+fp+fn)
        PPV = tp / (tp + fp)
        NPV = tn / (tn + fn)
        print(SENS, SPEC, EF, IOU, DSC, PPV, NPV, results)
        #distance = calculate_hausdorff_distance(mask, target)        
        distance1 = calculate_assd(target, mask)
        print(masks.data.cpu().numpy()[0][-1,:,:].shape, y_pred.shape)
        distance2 = calculate_hd(masks.data.cpu().numpy()[0][-1,:,:], outputs.data.cpu().numpy()[0][-1,:,:])
        print(distance1, distance2)
        results_dice.append(results)
        results_SENS.append(SENS)
        results_SPEC.append(SPEC)
        results_assd.append(distance1)
        results_DSC.append(DSC)
        results_EF.append(EF)
        results_hd.append(distance2)
        results_IOU.append(IOU)
        results_NPV.append(NPV)
        results_PPV.append(PPV)

        #print(results)

print("dice:", np.mean(np.array(results_dice)))
print("SPEC:", np.mean(np.array(results_SPEC)))
print("SENS:", np.mean(np.array(results_SENS)))
print("DSC:", np.mean(np.array(results_DSC)))
print("IOU:", np.mean(np.array(results_IOU)))
print("EF:",np.mean(np.array(results_EF)))
print("NPV:", np.mean(np.array(results_NPV)))
print("PPV:", np.mean(np.array(results_PPV)))
print("hd:", np.mean(np.array(results_hd)))
print("assd:", np.mean(np.array(results_assd)))

import pudb; pu.db
inputs = batch['image'].to(device)
prediction = model(inputs).detach().cpu()
