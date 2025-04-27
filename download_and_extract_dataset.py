
import scipy.io
import os
import numpy as np
import urllib.request
import cv2

def download_file(url, filename):
    print(f"Downloading {filename}...")
    urllib.request.urlretrieve(url, filename)
    print("Download complete.")

def extract_mat(mat_path):
    print("Extracting dataset...")
    data = scipy.io.loadmat(mat_path)
    images = data['images']
    depths = data['depths']

    os.makedirs('dataset/rgb', exist_ok=True)
    os.makedirs('dataset/depth', exist_ok=True)

    for i in range(images.shape[0]):
        rgb = np.transpose(images[i], (1, 2, 0))  # (H, W, 3)
        depth = depths[i]

        cv2.imwrite(f'dataset/rgb/{i:04d}.png', cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))
        depth_normalized = (depth / np.max(depth) * 255).astype(np.uint8)
        cv2.imwrite(f'dataset/depth/{i:04d}.png', depth_normalized)

    print("Extraction complete.")

if __name__ == "__main__":
    url = "https://cs.nyu.edu/~silberman/datasets/nyu_depth_v2_labeled.mat"
    mat_path = "nyu_depth_v2_labeled.mat"
    
    if not os.path.exists(mat_path):
        download_file(url, mat_path)
    else:
        print("MAT file already downloaded.")

    extract_mat(mat_path)
