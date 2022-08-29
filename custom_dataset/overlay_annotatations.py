import os
import numpy as np
import cv2
import argparse

# parser arguments
def parse_args():
  parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
  parser.add_argument('--data_dir', default='c001', type=str, help='input directory')
  parser.add_argument('--img', default='rad', type=str, help='suffix for image')
  parser.add_argument('--tile', default='tile_mask.png', type=str, help='suffix for tile mask')
  parser.add_argument('--crack', default='crack_mask.png', type=str, help='suffix for crack mask')
  args = parser.parse_args()
  return args


def overlay(img, mask, color):
    contours, hierarchy = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    out = cv2.drawContours(img, contours, -1, color, 3)
    return out



def show(name, img):
    cv2.namedWindow(name, cv2.WINDOW_NORMAL)
    cv2.imshow(name, img)


def main(data_dir, img_name, tile_name, crack_name):
    tile  = cv2.imread(os.path.join(data_dir, tile_name), cv2.IMREAD_GRAYSCALE)
    crack = cv2.imread(os.path.join(data_dir, crack_name), cv2.IMREAD_GRAYSCALE)
    for name in [n for n in os.listdir(data_dir) if img_name in n]:
        img = cv2.imread(os.path.join(data_dir, name), cv2.IMREAD_COLOR)
        out = overlay(img, tile, (255, 0 , 0))
        out = overlay(out, crack, (0, 0 , 255))
        show('img', img)
        show('out', out)
        cv2.waitKey(0)


if __name__ == '__main__':
  args = parse_args()
  main(args.data_dir, args.img, args.tile, args.crack)

