import numpy as np
import cv2
from tqdm import tqdm
import re
from xml.etree.ElementTree import ElementTree, Element, SubElement, tostring
from xml.dom.minidom import parseString
from matplotlib import pyplot as plt

def compute_relative_pose(Rot1, t_l, Rot2, t_r):
    if Rot1.shape[1] == 1:
        R_l = cv2.Rodrigues(Rot1)[0]
    else:
        R_l = Rot1
    if Rot2.shape[1] == 1:
        R_r = cv2.Rodrigues(Rot2)[0]
    else:
        R_r = Rot2

    Rt_l = np.hstack((R_l, t_l.reshape(3, 1)))
    Rt_l = np.vstack((Rt_l, [0, 0, 0, 1]))
    Rt_r = np.hstack((R_r, t_r.reshape(3, 1)))
    Rt_r = np.vstack((Rt_r, [0, 0, 0, 1]))

    rel_Rt = Rt_r @ np.linalg.inv(Rt_l)

    R = rel_Rt[:3, :3]    
    t = rel_Rt[:, 3][:-1]
    
    return R, t

def create_fundamental_matrix(K_l, K_r, R, t):
    t_cross = np.array([[0, -t[2], t[1]], 
                         [t[2], 0, -t[0]],
                         [-t[1], t[0], 0]])
    E = t_cross @ R 
    
    K_inv = np.linalg.inv(K_l)
    K_inv_trn = np.linalg.inv(K_r.T)

    F = K_inv_trn @ E @ K_inv

    return F

def find_inliers(fundamental_matrix, points1, points2, threshold=0.8):
    """
    Find inliers using the epipolar constraint.

    Args:
        fundamental_matrix (np.array): The fundamental matrix.
        points1 (np.array): Corresponding points in image 1.
        points2 (np.array): Corresponding points in image 2.
        threshold (float): The threshold to consider a point as an inlier.

    Returns:
        np.array: Boolean array indicating which points are inliers.
    """
    # Ensure points are homogeneous
    points1_h = np.c_[points1, np.ones(points1.shape[0])]
    points2_h = np.c_[points2, np.ones(points2.shape[0])]
    # Compute the epipolar constraint
    epipolar_constraint = np.abs(np.sum(points2_h * np.dot(fundamental_matrix, points1_h.T).T, axis=1))
    # Find inliers
    inliers = epipolar_constraint < threshold

    return inliers

def epipolar_line(img1, img2, F):
    sift = cv2.SIFT_create()
    # find the keypoints and descriptors with SIFT
    kp1, des1 = sift.detectAndCompute(img1, None)
    kp2, des2 = sift.detectAndCompute(img2, None)
    # FLANN parameters
    FLANN_INDEX_KDTREE = 1
    index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
    search_params = dict(checks=50)
    flann = cv2.FlannBasedMatcher(index_params, search_params)
    matches = flann.knnMatch(des1, des2, k=2)
    pts1 = []
    pts2 = []
    # ratio test as per Lowe's paper
    for i,(m,n) in enumerate(matches):
        if m.distance < 0.8 * n.distance:
            pts2.append(kp2[m.trainIdx].pt)
            pts1.append(kp1[m.queryIdx].pt)

    pts1 = np.int32(pts1)
    pts2 = np.int32(pts2)
    # F, mask = cv.findFundamentalMat(kps1, pts2, cv.FM_LMEDS)
    mask = find_inliers(F, pts1, pts2, 0.1)
    
    # We select only inlier points
    pts1 = pts1[mask.ravel()==1]
    pts2 = pts2[mask.ravel()==1]
    
    # Drawing
    visualized1 = img1.copy()
    visualized2 = img2.copy()
    
    lines1 = cv2.computeCorrespondEpilines(pts2, 2, F).reshape(-1, 3)
    lines2 = cv2.computeCorrespondEpilines(pts1, 1, F).reshape(-1, 3)

    width = img1.shape[1]

    def draw_point_line(img, point, line, color):
        x0, y0 = map(int, [0, -line[2] / line[1]])
        x1, y1 = map(int, [width, -(line[2] + line[0] * width) / line[1]])
        cv2.line(img, (x0, y0), (x1, y1), color, 2)
        cv2.circle(img, tuple(map(int, point)), 5, color, -1)

    for (kp1, kp2, line1, line2) in zip(pts1[-10:], pts2[-10:], lines1[-10:], lines2[-10:]):
        color = tuple(np.random.randint(0, 255, 3).tolist())
        draw_point_line(visualized1, kp1, line1, color)
        draw_point_line(visualized2, kp2, line2, color)
        
    return visualized1, visualized2

def calibration(pathCL, NUMBER_OF_CALIBRATION_IMAGES, cheker_size):
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

    objp = np.zeros((8 * 6, 3), np.float32)
    objp[:,:2] = np.mgrid[0:8, 0:6].T.reshape(-1, 2) * cheker_size
    
    img_pts = []
    obj_pts = []
    
    for i in tqdm(range(0, NUMBER_OF_CALIBRATION_IMAGES)):
        if i < 9:
            imgCL_gray = cv2.imread(pathCL+f"/Cal_0{i+1}.jpg",0)
        else:
            imgCL_gray = cv2.imread(pathCL + f"/Cal_{i+1}.jpg", 0)
    
        ret, corners = cv2.findChessboardCorners(imgCL_gray, (8, 6), None)
        
        if ret == True:
            obj_pts.append(objp)
            
            corners2 = cv2.cornerSubPix(imgCL_gray,corners, (11, 11), (-1, -1), criteria)
            img_pts.append(corners2)
    
    ret, mtx, dist, R, t = cv2.calibrateCamera(obj_pts, img_pts, imgCL_gray.shape[::-1], None, None)
    print(f"dist : {dist}")

    before_height, before_width = imgCL_gray.shape[:2]

    K, roi = cv2.getOptimalNewCameraMatrix(mtx, dist, (before_width, before_height), 1, (before_width, before_height))

    mean_error = 0
    for i in range(len(obj_pts)):
        imgpoints2, _ = cv2.projectPoints(obj_pts[i], R[i], t[i], mtx, dist)
        error = cv2.norm(img_pts[i], imgpoints2, cv2.NORM_L2) / len(imgpoints2)
        mean_error += error

    print(">>>> [Camera] Reprojection Error: {}".format(mean_error/len(img_pts)))

    return mtx, K, R, t, dist, roi, [before_width, before_height]

def read_pfm(path):
    file = open(path, 'rb')

    color = None
    width = None
    height = None
    scale = None
    endian = None

    header = file.readline().rstrip()
    if header == b'PF':
        color = True
    elif header == b'Pf':
        color = False
    else:
        raise Exception('Not a PFM file.')

    dim_match = re.match(rb'^(\d+)\s(\d+)\s$', file.readline())
    if dim_match:
        width, height = map(int, dim_match.groups())
    else:
        raise Exception('Malformed PFM header.')

    scale = float(file.readline().rstrip())
    if scale < 0: # little-endian
        endian = '<'
        scale = -scale
    else:
        endian = '>' # big-endian

    data = np.fromfile(file, endian + 'f')
    shape = (height, width, 3) if color else (height, width)

    data = np.reshape(data, shape)
    data = np.flipud(data)
    return data

# def writePFM(file, array):
#     import os
#     assert type(file) is str and type(array) is np.ndarray and \
#            os.path.splitext(file)[1] == ".pfm"
#     with open(file, 'wb') as f:
#         H, W = array.shape
#         headers = ["Pf\n", f"{W} {H}\n", "-1\n"]
#         for header in headers:
#             f.write(str.encode(header))
#         array = np.flip(array, axis=0).astype(np.float32)
#         f.write(array.tobytes())

def writePFM(file, image):
    # Open the file in binary mode
    with open(file, 'wb') as image_file:
        if image_file:
            height, width = image.shape[:2]
            number_of_components = image.shape[2] if len(image.shape) == 3 else 1

            # Write the type of the PFM file and end with a newline
            image_file.write(b'PF\n' if number_of_components == 3 else b'Pf\n')

            # Write the width, height, and end with a newline
            image_file.write(f"{width} {height}\n".encode())

            # Assumes little endian storage and ends with a newline
            byte_order = b'-1.000000\n'
            image_file.write(byte_order)

            # Store the floating points RGB color upside down, left to right
            for i in range(height):
                for j in range(width):
                    if number_of_components == 1:
                        value = image[height - 1 - i, j]
                        buffer = bytearray(np.float32(value).tobytes())
                    else:
                        color = image[height - 1 - i, j]

                        # OpenCV stores as BGR
                        buffer = bytearray(np.float32([color[2], color[1], color[0]]).tobytes())

                    # Write the values
                    image_file.write(buffer)
        else:
            print(f"Could not open the file: {file}")
            return False

    return True

def read_raw(path):
    w = 640
    h = 360

    focal_length = 642.320
    base_line = 17.738

    dtype = np.dtype(np.float32)

    with open(path, 'rb') as file:
        raw_data = np.fromfile(file, dtype=dtype) / 16
    image_array = np.reshape(raw_data, (h, w))

    depth = focal_length * base_line / (image_array + 1e-8)
    depth[depth > 1200] = 1200
    depth[depth < 200] = 200
    depth[image_array == 0] = 0
    depth[:, :10] = 0
    depth = cv2.resize(depth, (1280, 720), interpolation=cv2.INTER_LINEAR)

    return depth

def compare_images(img1, img2):
    if img1.shape != img2.shape:
        return "이미지 크기가 다릅니다."

    diff = cv2.absdiff(img1, img2)
    diff_sum = diff.sum()
    similarity = 1 - (diff_sum / (img1.shape[0] * img1.shape[1] * 3 * 255))

    return similarity

def save_calibration_results(filename, image_size, mtx_l, mtx_r, K_l, K_r, dist_l, dist_r, R, t, R_l, t_l, R_r, t_r):
    root = Element('opencv_storage')

    camera_mtx_l = SubElement(root, 'mtx_l')
    write_matrix(camera_mtx_l, mtx_l)

    camera_mtx_r = SubElement(root, 'mtx_r')
    write_matrix(camera_mtx_r, mtx_r)

    intrinsic_l = SubElement(root, 'K_l')
    write_matrix(intrinsic_l, K_l)

    intrinsic_r = SubElement(root, 'K_r')
    write_matrix(intrinsic_r, K_r)

    distortion_l = SubElement(root, 'dist_l')
    write_matrix(distortion_l, dist_l)

    distortion_r = SubElement(root, 'dist_r')
    write_matrix(distortion_r, dist_r)

    rotation = SubElement(root, 'R')
    write_matrix(rotation, R)

    translation = SubElement(root, 't')
    write_matrix(translation, t)

    rotation_l = SubElement(root, 'R_l')
    write_matrix(rotation_l, R_l)

    translation_l = SubElement(root, 't_l')
    write_matrix(translation_l, t_l)

    rotation_r = SubElement(root, 'R_r')
    write_matrix(rotation_r, R_r)

    translation_r = SubElement(root, 't_r')
    write_matrix(translation_r, t_r)

    img_width = SubElement(root, 'img_width')
    img_width.text = str(image_size[0])

    img_height = SubElement(root, 'img_height')
    img_height.text = str(image_size[1])

    xml_string = tostring(root)
    parsed_xml = parseString(xml_string)
    with open(filename, 'w') as f:
        f.write(parsed_xml.toprettyxml(indent="  "))

def save_calibration_results_single(filename, image_size, mtx_l, mtx_s, K_l, K_s, dist_l, dist_s, R_s, t_s, R_l, t_l, R_s_orig, t_s_orig):
    """싱글 캘리브레이션 결과를 XML 파일로 저장"""
    root = Element('opencv_storage')

    camera_mtx_l = SubElement(root, 'mtx_l')
    write_matrix(camera_mtx_l, mtx_l)

    camera_mtx_s = SubElement(root, 'mtx_s')
    write_matrix(camera_mtx_s, mtx_s)

    intrinsic_l = SubElement(root, 'K_l')
    write_matrix(intrinsic_l, K_l)

    intrinsic_s = SubElement(root, 'K_s')
    write_matrix(intrinsic_s, K_s)

    distortion_l = SubElement(root, 'dist_l')
    write_matrix(distortion_l, dist_l)

    distortion_s = SubElement(root, 'dist_s')
    write_matrix(distortion_s, dist_s)

    rotation = SubElement(root, 'R_s')
    write_matrix(rotation, R_s)

    translation = SubElement(root, 't_s')
    write_matrix(translation, t_s)

    rotation_l = SubElement(root, 'R_l')
    write_matrix(rotation_l, R_l)

    translation_l = SubElement(root, 't_l')
    write_matrix(translation_l, t_l)

    rotation_s = SubElement(root, 'R_s_orig')
    write_matrix(rotation_s, R_s_orig)

    translation_s = SubElement(root, 't_s_orig')
    write_matrix(translation_s, t_s_orig)

    img_width = SubElement(root, 'img_width')
    img_width.text = str(image_size[0])

    img_height = SubElement(root, 'img_height')
    img_height.text = str(image_size[1])

    xml_string = tostring(root)
    parsed_xml = parseString(xml_string)
    with open(filename, 'w') as f:
        f.write(parsed_xml.toprettyxml(indent="  "))

def write_matrix(parent, matrix):
    if len(matrix.shape) == 2:
        rows, cols = matrix.shape
        rows_elem = SubElement(parent, 'rows')
        rows_elem.text = str(rows)
        cols_elem = SubElement(parent, 'cols')
        cols_elem.text = str(cols)
        dt_elem = SubElement(parent, 'dt')
        dt_elem.text = 'd'  # assuming matrix type is double
        data_elem = SubElement(parent, 'data')
        data_str = ' '.join([str(value) for value in matrix.flatten()])
        data_elem.text = data_str
    elif len(matrix.shape) == 1:
        rows_elem = SubElement(parent, 'rows')
        rows_elem.text = '1'
        cols_elem = SubElement(parent, 'cols')
        cols_elem.text = str(len(matrix))
        dt_elem = SubElement(parent, 'dt')
        dt_elem.text = 'd'  # assuming matrix type is double
        data_elem = SubElement(parent, 'data')
        data_str = ' '.join([str(value) for value in matrix])
        data_elem.text = data_str
    else:
        raise ValueError("Invalid matrix shape")

def read_matrix(matrix_elem):
    rows = int(matrix_elem.find('rows').text)
    cols = int(matrix_elem.find('cols').text)
    data_str = matrix_elem.find('data').text
    data = [float(x) for x in data_str.split()]

    return np.array(data).reshape(rows, cols)

def load_calibration_results(filename):
    tree = ElementTree()
    tree.parse(filename)
    root = tree.getroot()

    mtx_l = read_matrix(root.find('mtx_l'))
    mtx_r = read_matrix(root.find('mtx_r'))

    K_l = read_matrix(root.find('K_l'))
    K_r = read_matrix(root.find('K_r'))

    dist_l = np.squeeze(read_matrix(root.find('dist_l')))
    dist_r = np.squeeze(read_matrix(root.find('dist_r')))

    R = read_matrix(root.find('R'))
    t = read_matrix(root.find('t'))
    R_l = read_matrix(root.find('R_l'))
    t_l = read_matrix(root.find('t_l'))
    R_r = read_matrix(root.find('R_r'))
    t_r = read_matrix(root.find('t_r'))

    return mtx_l, mtx_r, K_l, K_r, dist_l, dist_r, R, t, R_l, t_l, R_r, t_r

def load_calibration_results_single(filename):
    """싱글 캘리브레이션 파일 로드 (calibration_results_single.xml)
    
    Returns:
        mtx_s, K_s, dist_s, R_s, t_s: 싱글 카메라 캘리브레이션 파라미터
    """
    tree = ElementTree()
    tree.parse(filename)
    root = tree.getroot()

    mtx_s = read_matrix(root.find('mtx_s'))
    K_s = read_matrix(root.find('K_s'))
    dist_s = np.squeeze(read_matrix(root.find('dist_s')))
    R_s = read_matrix(root.find('R_s'))
    t_s = read_matrix(root.find('t_s'))

    return mtx_s, K_s, dist_s, R_s, t_s

def checkerboard_epiline(img1, img2, F):
    def draw_point_line(img, point, line, color):
        width = img1.shape[1]
        x0, y0 = map(int, [0, -line[2] / line[1]])
        x1, y1 = map(int, [width-1, -(line[2] + line[0] * width) / line[1]])
        cv2.line(img, (x0, y0), (x1, y1), color, 2)
        cv2.circle(img, (int(point[0][0]), int(point[0][1])), 5, color, -1)

    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
    gray_l = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
    gray_r = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)

    ret_l, corners_l = cv2.findChessboardCorners(gray_l, (8, 6), None)
    ret_r, corners_r = cv2.findChessboardCorners(gray_r, (8, 6), None)

    if ret_l and ret_r:
        corners_l = cv2.cornerSubPix(gray_l, corners_l, (3, 3), (-1, -1), criteria)
        corners_r = cv2.cornerSubPix(gray_r, corners_r, (3, 3), (-1, -1), criteria)
        corners_l = corners_l.reshape(-1, 1, 2)
        corners_r = corners_r.reshape(-1, 1, 2)
        # Drawing
        visualized1 = img1.copy()
        visualized2 = img2.copy()

        lines_l = cv2.computeCorrespondEpilines(corners_r, 2, F).reshape(-1, 3)
        lines_r = cv2.computeCorrespondEpilines(corners_l, 1, F).reshape(-1, 3)

        for (kp1, kp2, line1, line2) in zip(corners_l, corners_r, lines_l, lines_r):
            color = tuple(np.random.randint(0, 255, 3).tolist())
            draw_point_line(visualized1, kp1, line1, color)
            draw_point_line(visualized2, kp2, line2, color)

        return visualized1, visualized2, corners_l, corners_r, lines_l, lines_r
    else:
        return None

def get_distance(corners_l, corners_r, lines_l, lines_r, threshold=25):
    distance = []
    # left
    for corner, line in zip(corners_l, lines_l):
        a, b, c = line.ravel()
        x0, y0 = corner.ravel()
        d = abs(a*x0 + b*y0 + c) / np.sqrt(a**2 + b**2)
        if d < threshold:
            distance.append(d)
    # right
    for corner, line in zip(corners_r, lines_r):
        a, b, c = line.ravel()
        x0, y0 = corner.ravel()
        d = abs(a*x0 + b*y0 + c) / np.sqrt(a**2 + b**2)
        if d < threshold:
            distance.append(d)

    return np.array(distance, dtype=np.float32)

def get_y_diff(img1, img2, threshold=25):
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
    gray_l = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
    gray_r = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)

    ret_l, corners_l = cv2.findChessboardCorners(gray_l, (8, 6), None)
    ret_r, corners_r = cv2.findChessboardCorners(gray_r, (8, 6), None)

    y_diff_filtered = np.array([])

    if ret_l and ret_r:
        corners_l = cv2.cornerSubPix(gray_l, corners_l, (11, 11), (-1, -1), criteria)
        corners_r = cv2.cornerSubPix(gray_r, corners_r, (11, 11), (-1, -1), criteria)
        corners_l = corners_l.reshape(-1, 2)
        corners_r = corners_r.reshape(-1, 2)

    y_diff = corners_l[:, 1] - corners_r[:, 1]
    # filter outliers
    y_diff_filtered = y_diff[abs(y_diff) < threshold]

    return y_diff_filtered

def draw_histogram(y_diff, point_line_distance):
    fig, axs = plt.subplots(2, 1, figsize=(10, 8))

    mean_y_diff = np.mean(y_diff)
    std_y_diff = np.std(y_diff)
    axs[0].hist(y_diff, bins=30, color='blue', edgecolor='black', range=(-3, 3))
    axs[0].set_title('Y-Axis Difference')
    axs[0].set_xlabel(r'$\Delta y$ (px)')
    axs[0].set_ylabel('frequency')
    axs[0].text(0.95, 0.95, f'Mean: {mean_y_diff:.3f}\nStd: {std_y_diff:.3f}',
                horizontalalignment='right', verticalalignment='top', transform=axs[0].transAxes,
                color='black', fontsize=12)

    mean_distance = np.mean(point_line_distance)
    std_distance = np.std(point_line_distance)
    axs[1].hist(point_line_distance, bins=30, color='red', edgecolor='black', range=(0, 6))
    axs[1].set_title('Point-Epiline Distance')
    axs[1].set_xlabel('distance (px)')
    axs[1].set_ylabel('frequency')
    axs[1].text(0.95, 0.95, f'Mean: {mean_distance:.3f}\nStd: {std_distance:.3f}',
                horizontalalignment='right', verticalalignment='top', transform=axs[1].transAxes,
                color='black', fontsize=12)

    plt.tight_layout()
    plt.savefig('eval_histogram.png')
    plt.close(fig)

def get_new_extrinsic(checkerboard_path, obj_pts, mtx, dist):
    CRITERIA = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

    img_gray = cv2.imread(checkerboard_path, 0)
    ret, corners = cv2.findChessboardCorners(img_gray, (8, 6), None)

    if ret:
        corners_subpix = cv2.cornerSubPix(img_gray, corners, (11, 11), (-1, -1), CRITERIA)
        ret_pnp, new_R, new_t = cv2.solvePnP(obj_pts, corners_subpix, mtx, dist)
        # new_R = cv2.Rodrigues(new_R)

        if ret_pnp:
            # projected_pts, _ = cv2.projectPoints(obj_pts, new_R, new_t, mtx, dist)
            # reprojection_err = cv2.norm(corners, projected_pts, cv2.NORM_L2) / len(projected_pts)
            # print(">>>> Reprojection Error: {}".format(reprojection_err))

            return new_R, new_t
        else:
            raise ValueError("solvePnP failed to find a solution")
    else:
        raise ValueError("Chessboard corners not found")
    
def save_results(file_name, new_R, new_t, new_min_depth, new_max_depth):
    root = Element('opencv_storage')

    rotation = SubElement(root, 'R')
    write_matrix(rotation, new_R)

    translation = SubElement(root, 't')
    write_matrix(translation, new_t)

    min_depth = SubElement(root, 'min_depth')
    min_depth.text = str(new_min_depth)

    max_depth = SubElement(root, 'max_depth')
    max_depth.text = str(new_max_depth)

    xml_string = tostring(root)
    parsed_xml = parseString(xml_string)
    with open(file_name, 'w') as f:
        f.write(parsed_xml.toprettyxml(indent="  "))
