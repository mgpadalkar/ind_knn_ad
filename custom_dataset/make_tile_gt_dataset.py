import os
import argparse
import shutil
from tqdm import tqdm
import cv2


def get_args():
    parser = argparse.ArgumentParser(prog='Dataset for anomaly detection', formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--src", type=str, default="../20200330_for_GANs", help="Source directory containing all images")
    parser.add_argument("--dst", type=str, default="./pairs/combined", help="Destination directory containing images and ground truths")
    parser.add_argument("--img", type=str, default="fused.png", help="Image name suffix")
    parser.add_argument("--img_name", type=str, default="image.png", help="Image name of the saved image")
    parser.add_argument("--gt", type=str, nargs="+", default=["lucid_crack_mask.png", "digital_crack_mask.png"], help="List of ground truth name suffix")
    parser.add_argument("--gt_name", type=str, default="mask.png", help="Ground truth name of the saved image")
    args = parser.parse_args()
    return args


def main(args):
    src = args.src 
    dst = args.dst
    
    shutil.rmtree(dst, ignore_errors=True)
    os.makedirs(dst)
    
    for tile in tqdm(os.listdir(src), desc="tiles"):
        src_img = os.path.join(src, tile, args.img)
        if os.path.exists(src_img):
            img = cv2.imread(src_img, cv2.IMREAD_COLOR)
            if img is not None:
                dst_img = os.path.join(dst, f"{tile}_{args.img_name}")
                cv2.imwrite(dst_img, img)
                gt_img = None
                for gts in args.gt:
                    gt = cv2.imread(os.path.join(src, tile, gts), cv2.IMREAD_GRAYSCALE)
                    assert gt is not None
                    gt_img = gt if gt_img is None else cv2.bitwise_or(gt_img, gt)
                cv2.imwrite(os.path.join(dst, f"{tile}_{args.gt_name}"), gt_img)


if __name__=='__main__':
    args = get_args()
    main(args)
