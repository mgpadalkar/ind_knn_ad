import click

from data import MVTecDataset, mvtec_classes
from models import SPADE, PaDiM, PatchCore
from utils import print_and_export_results

from typing import List

# seeds
import torch
import random
import numpy as np
torch.manual_seed(0)
random.seed(0)
np.random.seed(0)

import os
import matplotlib.pyplot as plt
import matplotlib.image as IMAGE
from tqdm import tqdm
from utils import get_tqdm_params

import warnings # for some torch warnings regarding depreciation
warnings.filterwarnings("ignore")

ALL_CLASSES = mvtec_classes()
ALLOWED_METHODS = ["spade", "padim", "patchcore"]

def denormalize(image_tensor):
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    x = image_tensor.squeeze().numpy()
    x = (((x.transpose(1, 2, 0) * std) + mean) * 255.).astype(np.uint8)
    return x

def get_map(img, cmap = plt.get_cmap('jet')):
    if isinstance(img, torch.Tensor):
        img = img.numpy()
    rgba = cmap(img, bytes=True)
    rgb  = np.delete(rgba, 3, 2)
    return rgb

def overlay_heatmap(img, heatmap, prop=0.3):
    out = img.astype(np.float32)*(1-prop) + heatmap.astype(np.float32)*prop
    return np.clip(out.astype(np.uint8), 0, 255)


def visualize_results(img, gt, output):
    bar = np.zeros((img.shape[0],5,3),np.uint8)
    gt_map = overlay_heatmap(img, gt)
    op_map = overlay_heatmap(img, output)
    res = np.hstack((img, bar, gt_map, bar, op_map))
    return res
    

def run_model(method: str, classes: List):
    results = {}
    viz_outputs = {}
    tqdm_params = get_tqdm_params()

    for cls in classes:
        if method == "spade":
            model = SPADE(
                k=50,
                backbone_name="wide_resnet50_2",
            )
        elif method == "padim":
            model = PaDiM(
                d_reduced=350,
                backbone_name="wide_resnet50_2",
            )
        elif method == "patchcore":
            model = PatchCore(
                f_coreset=.10, 
                backbone_name="wide_resnet50_2",
            )

        print(f"\n█│ Running {method} on {cls} dataset.")
        print(  f" ╰{'─'*(len(method)+len(cls)+23)}\n")
        train_ds, test_ds = MVTecDataset(cls).get_dataloaders()

        print("   Training ...")
        model.fit(train_ds)
        print("   Testing ...")
        image_rocauc, pixel_rocauc = model.evaluate(test_ds)

        viz_out = []
        print("   Generating heatmaps ...")
        for img, msk, lbl in tqdm(test_ds, **tqdm_params):
            img_lvl_anom_score, pxl_lvl_anom_score = model.predict(img)
            out = pxl_lvl_anom_score[0]/torch.max(pxl_lvl_anom_score)
            gt  = msk[0][0]
            viz = visualize_results(denormalize(img), get_map(gt), get_map(out))
            viz_out.append(viz)
        
        print(f"\n   ╭{'─'*(len(cls)+15)}┬{'─'*20}┬{'─'*20}╮")
        print(  f"   │ Test results {cls} │ image_rocauc: {image_rocauc:.2f} │ pixel_rocauc: {pixel_rocauc:.2f} │")
        print(  f"   ╰{'─'*(len(cls)+15)}┴{'─'*20}┴{'─'*20}╯")
        results[cls] = [float(image_rocauc), float(pixel_rocauc)]
        viz_outputs[cls] = viz_out
        
    image_results = [v[0] for _, v in results.items()]
    average_image_roc_auc = sum(image_results)/len(image_results)
    image_results = [v[1] for _, v in results.items()]
    average_pixel_roc_auc = sum(image_results)/len(image_results)

    total_results = {
        "per_class_results": results,
        "average image rocauc": average_image_roc_auc,
        "average pixel rocauc": average_pixel_roc_auc,
        "model parameters": model.get_parameters(),
    }
    return total_results, viz_outputs


def save_outputs(viz_dict, method, tqdm_params=get_tqdm_params()):
    root_dir = os.path.join(f"./results/{method}")
    print(f"Saving outputs to '{root_dir}'")
    for cls, viz_outs in tqdm(viz_dict.items(), desc=f"classes", **tqdm_params):
        out_dir = os.path.join(root_dir, cls)
        os.makedirs(out_dir, exist_ok=True)
        for i, img in tqdm(enumerate(viz_outs), desc=f"Image in {cls}", **tqdm_params):
            name = f"{i}.jpg"
            path = os.path.join(out_dir, name)
            IMAGE.imsave(path, img)



@click.command()
@click.argument("method")
@click.option("--dataset", default="all", help="Dataset, defaults to all datasets.")
def cli_interface(method: str, dataset: str): 
    if dataset == "all":
        dataset = ALL_CLASSES
    else:
        dataset = [dataset]

    method = method.lower()
    assert method in ALLOWED_METHODS, f"Select from {ALLOWED_METHODS}."

    total_results, viz_out = run_model(method, dataset)
    
    save_outputs(viz_out, method)

    print_and_export_results(total_results, method)

    
if __name__ == "__main__":
    cli_interface()
