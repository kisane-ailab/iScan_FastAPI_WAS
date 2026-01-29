import os
import argparse
import glob
import cv2
import shutil
from utils import calibration, compute_relative_pose, save_calibration_results

def get_args():
    parser = argparse.ArgumentParser("")
    parser.add_argument("--left_image_path", type=str, required=True,
                        help="Left images required for Left camera calibration")
    parser.add_argument("--right_image_path", type=str, required=True,
                        help="Right images required for Right camera calibration")
    parser.add_argument("--output_file", type=str, required=True,
                        help="File name to save the calibration results")

    return parser.parse_args()

def prepare_depth_calibration_dir(pathCL, pathCR):
    current_dir = os.getcwd()
    path_depth_offset = os.path.join(current_dir, "resource", "camera", "temp")
    print(f"[prepare_depth_calibration_dir] path_depth_offset: {path_depth_offset}")
    
    # 1) 디렉토리 비우기
    if os.path.exists(path_depth_offset):
        print(f"[prepare_depth_calibration_dir] path_depth_offset: {path_depth_offset}")
        for file in os.listdir(path_depth_offset):
            file_path = os.path.join(path_depth_offset, file)
            if os.path.isfile(file_path):
                os.remove(file_path)
    else:
        print(f"[prepare_depth_calibration_dir] path_depth_offset does not exist: {path_depth_offset}")
        os.makedirs(path_depth_offset, exist_ok=True)

    # 캘리브레이션 이미지 복사
    shutil.copy2(os.path.join(pathCL, "Cal_00.jpg"), os.path.join(path_depth_offset, "Cal_left.jpg"))
    shutil.copy2(os.path.join(pathCR, "Cal_00.jpg"), os.path.join(path_depth_offset, "Cal_right.jpg"))

def main():
    args = get_args()
    pathCL = args.left_image_path
    pathCR = args.right_image_path
    output_file = args.output_file
    
    # Cal_00 이미지를 제외하고 Cal_01부터 참조
    left_images = sorted([f for f in glob.glob(os.path.join(pathCL, "Cal*.jpg")) if not f.endswith("Cal_00.jpg")])
    right_images = sorted([f for f in glob.glob(os.path.join(pathCR, "Cal*.jpg")) if not f.endswith("Cal_00.jpg")])
    
    if len(left_images) == len(right_images):
        NUMBER_OF_CALIBRATION_IMAGES = len(left_images)
    else :
        raise ValueError("The number of images in both directories are not equal")

    print(f"-> Number of calibration images (excluding Cal_00): {NUMBER_OF_CALIBRATION_IMAGES}")

    mtx_l, K_l, R_l, t_l, dist_l, _, image_size = calibration(pathCL, NUMBER_OF_CALIBRATION_IMAGES, 25.0)
    mtx_r, K_r, R_r, t_r, dist_r, _, _ = calibration(pathCR, NUMBER_OF_CALIBRATION_IMAGES, 25.0)

    print(f">>>> Left Camera:\n mtx =\n{mtx_l}\n Intrinsic =\n{K_l}\n Rotation =\n{cv2.Rodrigues(R_l[0])[0]}\n translation = \n{t_l[0]}")
    print(f">>>> Right Camera:\n mtx =\n{mtx_r}\n Intrinsic =\n{K_r}\n Rotation =\n{cv2.Rodrigues(R_r[0])[0]}\n translation = \n{t_r[0]}")

    R, t = compute_relative_pose(R_l[0], t_l[0], R_r[0], t_r[0]) 
    print("============================================================")
    print(f">>>> Relative Pose:\n Rotation =\n{R}\n translation =\n{t}")
    print("============================================================")

    # Depth 보정을 위해 Calibration 이미지 복사
    prepare_depth_calibration_dir(pathCL, pathCR)
    
    # Save calibration results to XML
    save_calibration_results(output_file, image_size, mtx_l, mtx_r, K_l, K_r, dist_l, dist_r, R, t, R_l[0], t_l[0], R_r[0], t_r[0])

if __name__ == "__main__":
    main()
