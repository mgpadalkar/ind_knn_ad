import os
import argparse
import shutil
from tqdm import tqdm


def get_args():
    parser = argparse.ArgumentParser(prog='Dataset for anomaly detection', formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--src", type=str, default="../20200330_for_GANs", help="Source directory containing all images")
    parser.add_argument("--dst", type=str, default="./fused_tiles_2020", help="Destination directory containing images and ground truths")
    parser.add_argument("--img", type=str, default="fused.png", help="Image name suffix")
    parser.add_argument("--gt", type=str, default="lucid_crack_mask.png", help="Ground truth name suffix")
    args = parser.parse_args()
    return args


def main(args):
    src = args.src # "../20200330_for_GANs"
    dst = args.dst # "./fused_tiles_2020"
    cpnames = [args.img, args.gt] # ["fused.png", "lucid_crack_mask.png"]
    
    shutil.rmtree(dst, ignore_errors=True)
    os.makedirs(dst)
    
    for tile in tqdm(os.listdir(src), desc="tiles"):
        for name in cpnames:
            sname = os.path.join(src, tile, name)
            if os.path.exists(sname):
                dname = os.path.join(dst, f"{tile}_{name}")
                shutil.copy2(sname, dname)


if __name__=='__main__':
    args = get_args()
    main(args)
