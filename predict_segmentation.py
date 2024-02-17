import torch
import os
import argparse
import time
import numpy as np
import glob
import cv2

import torch
import torch.nn as nn
from torch.utils.data import ConcatDataset

from Data import dataloaders
from Models import models
from Metrics import performance_metrics


import re 
import math
from pathlib import Path 

file_pattern = re.compile(r'.*?(\d+).*?')
def get_order(file):
    match = file_pattern.match(Path(file).name)
    if not match:
        return math.inf
    return int(match.groups()[0])

def build(args):
    #if torch.cuda.is_available():
     # device = torch.device("cuda")
    #else:
    device = torch.device("cpu")

    img_path = f"{args.dataset}/*.png"
    files = glob.glob(img_path)
    input_paths = sorted(files, key=get_order)
    depth_path = f"{args.labels}/*" # masks/test/GM
    target_paths = sorted(glob.glob(depth_path), key=get_order)
    group1, group2, group3 = dataloaders.get_dataloaders(
        input_paths, target_paths, batch_size=1
    )

    perf = performance_metrics.DiceScore()

    model = models.FCBFormer()

    state_dict = torch.load(
        f"./Trained_models/{args.model_}", map_location=torch.device('cpu')
    )
    model.load_state_dict(state_dict["model_state_dict"])

    model.to(device)

    return device, perf, model, group1, group2, group3


@torch.no_grad()
def predict(args):

    device, perf_measure, model, group1, group2, group3 = build(args)
    folder = args.folder # "Predictions_gm_without_inu"
    if not os.path.exists(f"./{folder}"):
        os.makedirs(f"./{folder}")

    t = time.time()
    model.eval()
    perf_accumulator = []
    results_dice = []
    results_SENS = []
    results_SPEC = []
    results_IOU = []
    results_EF = []
    results_DSC = []
    results_PPV = []
    results_NPV = []
    new_dataset = ConcatDataset([group1.dataset, group2.dataset, group3.dataset])
    for i, (data, target) in enumerate(group1):

        data, target = data.to(device), target.to(device)
        output = model(data)
        score, SENS, SPEC, EF, IOU, DSC, PPV, NPV = perf_measure(output, target)
        print(score, SENS, SPEC, EF, IOU, DSC, PPV, NPV)
        results_SENS.append(SENS)
        results_SPEC.append(SPEC)
        results_DSC.append(DSC)
        results_EF.append(EF)
        results_IOU.append(IOU)
        results_NPV.append(NPV)
        results_PPV.append(PPV)

        perf_accumulator.append(score)
        predicted_map = np.array(output.cpu())
        predicted_map = np.squeeze(predicted_map)
        predicted_map = predicted_map > 0
        
    t = time.time()
    model.eval()
    perf_accumulator = []
    for i, (data, target) in enumerate(group2):
        data, target = data.to(device), target.to(device)
        output = model(data)
        score, SENS, SPEC, EF, IOU, DSC, PPV, NPV = perf_measure(output, target)
        results_SENS.append(SENS)
        results_SPEC.append(SPEC)
        results_DSC.append(DSC)
        results_EF.append(EF)
        results_IOU.append(IOU)
        results_NPV.append(NPV)
        results_PPV.append(PPV)
        perf_accumulator.append(score)
        predicted_map = np.array(output.cpu())
        predicted_map = np.squeeze(predicted_map)
        predicted_map = predicted_map > 0

    t = time.time()
    model.eval()
    perf_accumulator = []
    for i, (data, target) in enumerate(group3):
        data, target = data.to(device), target.to(device)
        output = model(data)
        score, SENS, SPEC, EF, IOU, DSC, PPV, NPV = perf_measure(output, target)
        results_SENS.append(SENS)
        results_SPEC.append(SPEC)
        results_DSC.append(DSC)
        results_EF.append(EF)
        results_IOU.append(IOU)
        results_NPV.append(NPV)
        results_PPV.append(PPV)

        perf_accumulator.append(score)
        predicted_map = np.array(output.cpu())
        predicted_map = np.squeeze(predicted_map)
        predicted_map = predicted_map > 0
  
    print("SPEC:", np.mean(np.array(results_SPEC)))
    print("SENS:", np.mean(np.array(results_SENS)))
    print("DSC:", np.mean(np.array(results_DSC)))
    print("IOU:", np.mean(np.array(results_IOU)))
    print("EF:",np.mean(np.array(results_EF)))
    print("NPV:", np.mean(np.array(results_NPV)))


def get_args():
    parser = argparse.ArgumentParser(
        description="Make predictions on specified dataset"
    )
    parser.add_argument(
        "--dataset", type=str, required=True
    )
    parser.add_argument(
        "--labels", type=str, required=True
    )
    parser.add_argument("--folder", type=str, required=True)
    parser.add_argument("--model_", type=str, required=True)

    return parser.parse_args()


def main():
    args = get_args()
    predict(args)


if __name__ == "__main__":
    main()
