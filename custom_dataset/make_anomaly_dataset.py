import os
import argparse
import shutil
import random
from tqdm import tqdm
import cv2

def get_args():
    parser = argparse.ArgumentParser(prog='Dataset for anomaly detection', formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--src", type=str, default="./combined", help="Source directory containing images")
    parser.add_argument("--dst", type=str, default="./pavis-aen-ad", help="Destination directory containing image patches in the mv-tec format")
    parser.add_argument("--img", type=str, default="image.png", help="Image name suffix")
    parser.add_argument("--gt", type=str, default="mask.png", help="Ground truth name suffix")
    parser.add_argument("--train_prop", type=float, default=0.5, help="Proportion of images to be used for training")
    parser.add_argument("--patch_size", type=int, default=256, help="Patch size")
    args = parser.parse_args()
    return args


def get_valid(rect, patch_size, img_size):
    x, y, w, h = rect
    # update x and w
    x = x + w//2 - patch_size//2
    x = max(x, 0)
    w = patch_size
    # update y and h
    y = y + h//2 - patch_size//2
    y = max(y, 0)
    h = patch_size
    if x+w > img_size[1]:
        x = x - (img_size[1] - (x+w))
    if y+h > img_size[0]:
        y = y - (img_size[0] - (y+h))
    return x, y, w, h


def get_random(N, img, gt2, patch_size, RNG):
    normal = []
    count  = 0
    while(True):
        y = RNG.randint(0, img.shape[0]-patch_size-1)
        x = RNG.randint(0, img.shape[1]-patch_size-1)
        gt_crop = gt2[y:y+patch_size, x:x+patch_size, ...].copy()
        if gt_crop.sum() == 0:
            img_crop = img[y:y+patch_size, x:x+patch_size, ...].copy()
            normal.append([img_crop, gt_crop])
            count = count + 1
        if count >= N:
            break
    return normal

def show(name, img):
    cv2.namedWindow(name, cv2.WINDOW_NORMAL)
    cv2.imshow(name, img)

def get_patches(img, gt, patch_size, RNG):
    contours, hierarchy = cv2.findContours(gt, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    gt2 = gt*0
    defects = []
    for c in contours:
        x0, y0, w0, h0 = cv2.boundingRect(c)
        x, y, w, h = get_valid([x0, y0, w0, h0], patch_size, img.shape[:2])
        img_crop = img[y:y+h, x:x+w, ...].copy()
        gt_crop  = gt[y:y+h, x:x+w, ...].copy()
        # og_crop  = gt[y0:y0+h0, x0:x0+w0, ...].copy()
        # if gt_crop.sum() == 0:
        #     show('img_crop', img_crop)
        #     show('gt_crop', gt_crop)
        #     show('og_crop', og_crop)
        #     print([x0, y0, w0, h0], [x, y, w, h])
        #     cv2.waitKey(0)
        gt2[y:y+h, x:x+w] = 1
        defects.append([img_crop, gt_crop])
    normal = get_random(len(contours)*2 + 1, img, gt2, patch_size, RNG)
    return normal, defects


def _save_data(dst, tile, data_list):
    for i, (img, gt) in tqdm(enumerate(data_list), leave=False):
        img_path = os.path.join(dst, f"{tile}_{i}_img.png")
        gt_path  = os.path.join(dst, f"{tile}_{i}_gt.png")
        cv2.imwrite(img_path, img)
        cv2.imwrite(gt_path, gt)


def save_data(dst, tile, normal, defective):
    normal_dst = os.path.join(dst, "tmp", "normal")
    defect_dst = os.path.join(dst, "tmp", "defect")
    os.makedirs(normal_dst, exist_ok=True)
    os.makedirs(defect_dst, exist_ok=True)
    _save_data(normal_dst, tile, normal)
    _save_data(defect_dst, tile, defective)


def get_pairs(path, tile):
    img_gt = []
    files  = sorted([f for f in os.listdir(path) if f.split('_')[0] == tile])
    assert len(files)%2 == 0
    img_gt = [[files[i+1], files[i]] for i in range(0, len(files), 2)]
    return img_gt


def _save_split(src, src_name, dst, dst_name):
    os.makedirs(dst, exist_ok=True)
    shutil.copy2(os.path.join(src, src_name), os.path.join(dst, dst_name))


def save_splits(dst, train_tiles, test_tiles):
    for tile in train_tiles:
        src = os.path.join(dst, "tmp", "normal")
        img_gt_pairs = get_pairs(src, tile)
        for img, _ in img_gt_pairs:
            _save_split(src, img, os.path.join(dst, "train", "good"), img)
    for tile in test_tiles:
        good_src = os.path.join(dst, "tmp", "normal")
        img_gt_pairs = get_pairs(good_src, tile)
        for img, _ in img_gt_pairs[:len(img_gt_pairs)//2]:
            _save_split(good_src, img, os.path.join(dst, "test", "good"), img) 
        defect_src = os.path.join(dst, "tmp", "defect")
        img_gt_pairs = get_pairs(defect_src, tile)
        for img, gt in img_gt_pairs:
            _save_split(defect_src, img, os.path.join(dst, "test", "crack"), img)
            _save_split(defect_src, gt,  os.path.join(dst, "ground_truth", "crack"), img[:-4] + '_mask' + img[-4:])
    shutil.rmtree(os.path.join(dst, "tmp"))
            


def main(args):
    src = args.src # "./fused_tiles_2020"
    # dst = args.dst # "./pavis-aen-ad"
    dst = os.path.join(args.dst,  os.path.basename(args.src))
    img_name = args.img # "fused.png"
    gt_name  = args.gt #"lucid_crack_mask.png"
    train_prop = args.train_prop # 0.5
    patch_size = args.patch_size # 256
    RNG = random.Random(0)
    
    shutil.rmtree(dst, ignore_errors=True)
    os.makedirs(dst)
    
    tileset = set([f.split('_')[0] for f in os.listdir(src)])
    tiles   = sorted(list(tileset))
    RNG.shuffle(tiles)
    N_train = int(len(tiles)*train_prop)
    train_tiles = tiles[:N_train]
    test_tiles  = tiles[N_train:]
    
    for tile in tqdm(tiles, desc="tiles"):
        img = cv2.imread(os.path.join(src, f"{tile}_{img_name}"), cv2.IMREAD_COLOR)
        gt  = cv2.imread(os.path.join(src, f"{tile}_{gt_name}"), cv2.IMREAD_GRAYSCALE)
        normal, defective = get_patches(img, gt, patch_size, RNG)
        save_data(dst, tile, normal, defective)
    
    save_splits(dst, train_tiles, test_tiles)


if __name__=='__main__':
    args = get_args()
    main(args)

