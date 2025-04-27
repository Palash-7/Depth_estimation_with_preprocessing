
import os

def download_and_extract_dataset():
    print("\n>>> Step 1: Downloading and extracting NYU Dataset <<<")
    os.system('python download_and_extract_dataset.py')

def run_lcdpnet_exposure_correction():
    print("\n>>> Step 2: Running LCDPNet Exposure Correction <<<")
    os.chdir('LCDPNet/src')
    os.system('python test.py --config config/config.yaml')
    os.chdir('../../')  # Come back to main project folder

def manual_edge_enhancement_instruction():
    print("\n>>> Step 3: Manual Step - Apply SG-WLS Edge Enhancement <<<")
    print("Please manually run SG-WLS filtering on the exposure-corrected images using MATLAB.")
    input("After completing edge enhancement in MATLAB, press ENTER to continue...")

def run_marigold_depth_estimation():
    print("\n>>> Step 4A: Running Marigold Depth Estimation <<<")
    os.chdir('Marigold')
    os.system('python infer.py')
    os.chdir('..')

def run_depth_anything_v2_estimation():
    print("\n>>> Step 4B: Running Depth Anything V2 Estimation <<<")
    os.chdir('Depth-Anything-V2')
    os.system('python demo.py')
    os.chdir('..')

def run_evaluation():
    print("\n>>> Step 5: Evaluating Predicted Depth Maps <<<")
    os.system('python Evaluating_model.py')

def full_pipeline():
    print("\n========== Starting Full Depth Estimation Pipeline ==========")
    download_and_extract_dataset()
    run_lcdpnet_exposure_correction()
    manual_edge_enhancement_instruction()
    run_marigold_depth_estimation()
    run_depth_anything_v2_estimation()
    run_evaluation()
    print("\n========== Pipeline Execution Complete! ==========")

if __name__ == "__main__":
    full_pipeline()
