import numpy as np
import torch


class DiceScore(torch.nn.Module):
    def __init__(self, smooth=1):
        super(DiceScore, self).__init__()
        self.smooth = smooth

    def forward(self, logits, targets, sigmoid=True):
        num = targets.size(0)
        probs = torch.sigmoid(logits)
        m1 = probs.view(num, -1) > 0.5
        m2 = targets.view(num, -1) > 0.5
        intersection = m1 * m2

        score = (
            2.0
            * (intersection.sum(1) + self.smooth)
            / (m1.sum(1) + m2.sum(1) + self.smooth)
        )
        score = score.sum() / num
        tp, tn, fp, fn = self.calculate_metrics(probs, targets)
        SENS = tp / (tp + fn)
        SPEC = tn / (tn + fp)
        EF = fp / (tn + fn)
        IOU = tp / (tp + fn + fp)
        DSC = (2 * tp) / ((2 * tp)+fp+fn)
        PPV = tp / (tp + fp)
        NPV = tn / (tn + fn)

        return score, SENS, SPEC, EF, IOU, DSC, PPV, NPV

    def calculate_metrics(self, predicted, ground_truth):
        # Convert predicted and ground truth images to binary arrays
        predicted = predicted.cpu().numpy()#.astype(int) #!= 0).astype(int)
        ground_truth = ground_truth.cpu().numpy()#.astype(int) #!= 0).astype(int)
        
        volume = predicted
        v_max = np.max(volume)
        v_min = np.min(volume)
        volume_norm = (volume - v_min) / (v_max - v_min)
        normalized_predicted = (volume_norm * 255).astype("int")
        predicted = np.where(normalized_predicted>125, 1, 0)

        volume = ground_truth
        v_max = np.max(volume)
        v_min = np.min(volume)
        volume_norm = (volume - v_min) / (v_max - v_min)
        normalized_ground_truth = (volume_norm * 255).astype("int")
        ground_truth = np.where(normalized_ground_truth>125, 1, 0)

        # Calculate True Positives (TP)
        tp = np.sum(np.logical_and(predicted == 1, ground_truth == 1))

        # Calculate True Negatives (TN)
        tn = np.sum(np.logical_and(predicted == 0, ground_truth == 0))

        # Calculate False Positives (FP)
        fp = np.sum(np.logical_and(predicted == 1, ground_truth == 0))

        # Calculate False Negatives (FN)
        fn = np.sum(np.logical_and(predicted == 0, ground_truth == 1))

        return tp, tn, fp, fn

