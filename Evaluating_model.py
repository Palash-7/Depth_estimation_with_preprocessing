import os
import numpy as np
import cv2
from skimage.metrics import structural_similarity as ssim

# -------------------------------
# Metric Functions
# -------------------------------

def compute_rmse(pred, gt):
    return np.sqrt(np.mean((pred - gt) ** 2))

def compute_mae(pred, gt):
    return np.mean(np.abs(pred - gt))

def compute_abs_rel(pred, gt):
    return np.mean(np.abs(pred - gt) / (gt + 1e-6))  # Prevent division by zero

def compute_ssim(pred, gt):
    pred_norm = (pred - pred.min()) / (pred.max() - pred.min() + 1e-8)
    gt_norm = (gt - gt.min()) / (gt.max() - gt.min() + 1e-8)
    return ssim(pred_norm, gt_norm, data_range=1.0)

# -------------------------------
# Load Image (Supports PNG, JPG, NPY)
# -------------------------------

def load_image(path):
    ext = os.path.splitext(path)[1].lower()
    if ext == '.npy':
        return np.load(path).astype(np.float32)
    else:
        img = cv2.imread(path, cv2.IMREAD_UNCHANGED)
        return img.astype(np.float32)

# -------------------------------
# Evaluate All Image Pairs
# -------------------------------

def evaluate_dataset(pred_dir, gt_dir):
    metric_list = {
        'RMSE': [],
        'MAE': [],
        'AbsRel': [],
        'SSIM': []
    }

    image_names = sorted(os.listdir(pred_dir))

    for name in image_names:
        pred_path = os.path.join(pred_dir, name)
        gt_path = os.path.join(gt_dir, name)

        if not os.path.exists(gt_path):
            print(f"Missing ground truth for {name}, skipping.")
            continue

        pred = load_image(pred_path)
        gt = load_image(gt_path)

        # Resize if needed
        if pred.shape != gt.shape:
            pred = cv2.resize(pred, (gt.shape[1], gt.shape[0]))

        # Compute metrics
        metric_list['RMSE'].append(compute_rmse(pred, gt))
        metric_list['MAE'].append(compute_mae(pred, gt))
        metric_list['AbsRel'].append(compute_abs_rel(pred, gt))
        metric_list['SSIM'].append(compute_ssim(pred, gt))

    # Average metrics
    print("\n Average Metrics for Dataset:")
    for key, values in metric_list.items():
        avg = np.mean(values)
        print(f"{key}: {avg:.4f}")

    return metric_list

# -------------------------------
# Example Usage
# -------------------------------

if __name__ == "__main__":
    pred_dir = "results/depth_predicted/"   
    gt_dir = "results/ground_truth/"        

    evaluate_dataset(pred_dir, gt_dir)
