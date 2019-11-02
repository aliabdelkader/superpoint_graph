import numpy as np
import pandas as pd
import cv2
import glob
import argparse
from pathlib import Path
from tqdm import tqdm
from sklearn.model_selection import train_test_split

FOLDERS = { 
    "image_folder": "chunk_demosaic",
    "lidar_scan_folder": "lidar_scans",
    "lidar_project_folder":"lidar_projections",
    "labels_folder":"lidar_labels"
}
FOLDERS_EXT = { 
    "image_folder": ".png",
    "lidar_scan_folder": ".bin",
    "lidar_project_folder":".txt",
    "labels_folder":".csv"
}
label_map = {

    0: 1,  # 'road': [128, 64, 128]
    1: 2,  # 'sidewalk': [244, 35, 232]
    2: 3,  # 'building': [70, 70, 70],
    3: 4,  # 'wall': [102, 102, 156],
    4: 5,  # 'fence': [190, 153, 153],
    5: 6,  # 'pole': [153, 153, 153],
    6: 7,  # 'traffic light': [250, 170, 30],
    7: 8,  # 'traffic sign': [220, 220, 0],
    8: 9,  # 'vegetation': [107, 142, 35],
    9: 10,  # 'terrain': [152, 251, 152],
    10: 11,  # 'sky': [70, 130, 180],
    11: 12,  # 'person': [220, 20, 60],
    12: 13,  # 'rider': [255, 0, 0],
    13: 14,  # 'car': [0, 0, 142],
    14: 15,  # 'truck': [0, 0, 70],
    15: 16,  # 'bus': [0, 60, 100],
    16: 17,  # 'train': [0, 80, 100],
    17: 18,  # 'motorcycle': [0, 0, 230],
    18: 19,  # 'bicycle': [119, 11, 32],
    19: 0,  # 'void': [0, 0, 0],
    20: 0,  # 'outside camera': [255, 255, 0],
    21: 0, #'egocar': [123, 88, 4],
}
def convert_ground_truth(original_label):
    """
    function changes ground truth label according to lidar map
    Args:
        original_label: label from annotation files
    return:
        new label
    """

    return label_map[original_label]

def find_all_lidar_scans(dataset_dir: Path) -> list:
    """
    function finds all scans in a folder

    dataset_dir: Pathlib, root directory of chunk

    returns
        list of scans found
    """

    images_list = dataset_dir.glob("*"+FOLDERS_EXT["lidar_scan_folder"])
    images_list = sorted([i.stem for i in images_list])

    if len(images_list) <= 0:
        print("no images found in dir {dir}".format(dir=str(dataset_dir)))
        return None
    else:
        print("number of images found is {num}".format(num=len(images_list)))
        return images_list

def read_lidar_scan(scan_path: Path) -> np.numarray:
    """
    function reads lidar scan 

    scan_path: Pathlib, path to scan

    returns
        numpy array with shape [number of points, 3]
    """

    lidar_scan_data = np.fromfile(str(scan_path),dtype='float64')
    lidar_scan_data = lidar_scan_data.reshape(-1,3)  

    assert lidar_scan_data.shape[1] == 3, "invalid scan shape"

    return lidar_scan_data

def read_lidar_projection(projection_path: Path) -> np.numarray:
    """
    function reads lidar projection path

    projection_path: Pathlib, path to lidar projection file

    returns
        numpy array with shape [number of points, 2]
    """

    lidar_projections_data = pd.read_csv(str(projection_path),header=None).to_numpy()
    lidar_projections_data = lidar_projections_data.astype(np.uint8)

    assert lidar_projections_data.shape[1] == 2, "invalid lidar projection shape"

    return lidar_projections_data

def read_lidar_labels(labels_path: Path) -> np.numarray:
    """
    function reads lidar labels path

    labels_path: Pathlib, path to lidar labels file

    returns
        numpy array with shape [number of points, 1]
    """

    lidar_labels_data = pd.read_csv(str(labels_path),header=None).to_numpy()
    lidar_labels_data = lidar_labels_data.astype(np.uint8)

    assert lidar_labels_data.shape[1] == 1, "invalid lidar labels shape"

    return lidar_labels_data

def read_image(image_path: Path) -> np.numarray:
    """
    function reads image 

    image_path: Pathlib, path to image file

    returns
        numpy array with shape [H,W, 3]
    """

    image = cv2.imread(str(image_path))
    image = cv2.cvtColor(image,cv2.COLOR_BGR2RGB)
    image = image.astype(np.uint8)

    assert image.shape[2] == 3, "invalid image shape"

    return image

def filter_invalid_scans(root_dataset_dir: Path, scans_names: list) -> list:
    """
    function removes invalid scans from list that have nan values

    Args:
        root_dataset_dir: root directory of dataset
        scans_names: list of found scans in dataset
    return 
        filtered list of scans
    """
    filtered_scans = []
    for scan_num in tqdm(scans_names):

        # read lidar scan
        lidar_scan_path = root_dataset_dir / FOLDERS["lidar_scan_folder"] / ( scan_num + FOLDERS_EXT["lidar_scan_folder"] )
        lidar_scan_data = read_lidar_scan(lidar_scan_path) 

        # skip lidar scans that have nan values
        nn = lidar_scan_data[np.isnan(lidar_scan_data)]
        if not (nn.size == 0):
            continue
        else:
            filtered_scans.append(scan_num)
    
    return filtered_scans


def split_scans(root_dataset_dir: Path, scans_names: list, valset_size: float, testset_size: float) -> [list, list ,list]:
    """
    function splits scans into train, val, test

    Args:
        root_dataset_dir: root directory of dataset
        scans_names: list of found scans in dataset
        valset_size: size of validation set from 0 to 1
        testset_size: size of test set from 0 to 1
    return:
        train, val,  test list of scans

    """
    filtered_scans_names = filter_invalid_scans(root_dataset_dir, scans_names)
    train, test = train_test_split(filtered_scans_names, test_size=testset_size, random_state=42)
    train, val = train_test_split(train, test_size=valset_size, random_state=42)

    return train, val, test

def process_scans(scans_list: list, root_dataset_dir: Path, output_dir: Path):
    """
    function reads lidar scans, projects them to image to get rgb, save scan in disk

    Args:
        scans_list: list of scans to be read
        root_dataset_dir: root directory of original dataset
        output_dir: path to directory to write output to
    
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    for scan_num in tqdm(scans_list, "processing scans for {} set".format(output_dir.stem)):

        # read lidar scan
        lidar_scan_path = root_dataset_dir / FOLDERS["lidar_scan_folder"] / ( scan_num + FOLDERS_EXT["lidar_scan_folder"] )
        lidar_scan_data = read_lidar_scan(lidar_scan_path) 

        # skip lidar scans that have nan values
        nn = lidar_scan_data[np.isnan(lidar_scan_data)]
        if not (nn.size == 0):
            continue

        # read lidar projection
        lidar_projection_path = root_dataset_dir / FOLDERS["lidar_project_folder"] / ( scan_num + FOLDERS_EXT["lidar_project_folder"] )
        lidar_projection_data = read_lidar_projection(lidar_projection_path)

        # read lidar label
        lidar_labels_path = root_dataset_dir / FOLDERS["labels_folder"] / ( scan_num + FOLDERS_EXT["labels_folder"] )
        lidar_labels_data = read_lidar_labels(lidar_labels_path)

        # read image
        image_path = root_dataset_dir / FOLDERS["image_folder"] / ( scan_num + FOLDERS_EXT["image_folder"] )
        image = read_image(image_path)

        # index of points inside image
        projected_idx = np.logical_and(lidar_projection_data[:,0]>0,lidar_projection_data[:,1]>0 )

        # filter points outside image frame
        lidar_scan_data = lidar_scan_data[projected_idx]
        lidar_projection_data  = lidar_projection_data[projected_idx]

        # result array to be saved
        # result shape [ number of points, x,y,z, r,g,b, label]
        result = np.zeros((lidar_scan_data.shape[0],7))

        # loop over every point
        for i in range(lidar_scan_data.shape[0]):
            # set values of result 
            x,y = lidar_projection_data[i,:]
            result[i,3:6] = image[y,x].astype(np.uint8)
            result[i,:3] = lidar_scan_data[i,:].astype(np.float32)
            
            result[i,6] = convert_ground_truth(lidar_labels_data[i].astype(np.uint8)[0])
        
        #save file
        outfile = str(output_dir / scan_num )+ ".npy"

        np.save(outfile, result)
        np.savetxt(outfile[-3] + ".csv", result, delimiter=",")

def main():
    parser = argparse.ArgumentParser(description='prepare oxford')
    
     #parameters
    parser.add_argument('--dataset_files', default='datasets/s3dis',help='path to dataset orignal files')
    parser.add_argument('--include', help='starting index to include')
    parser.add_argument('--output_dir', default='data',help='path to output dir')
    parser.add_argument('--testset_size', default='0.1',help='size of testset from 0 to 1')
    parser.add_argument('--valset_size', default='0.1',help='size of valset from 0 to 1')

    args = parser.parse_args()
    
    root_dataset_dir = Path(args.dataset_files)
    start_index = int(args.include)
    output_dir = Path(args.output_dir)
    testset_size = float(args.testset_size)
    valset_size = float(args.valset_size)

    if not root_dataset_dir.exists():
        print("dataset dir does not exit")
        return None
    
    if not output_dir.exists():
        print("create output dir")
        output_dir.mkdir(parents=True, exist_ok=True)
    
    # find all images
    scans_names = find_all_lidar_scans(dataset_dir=root_dataset_dir / FOLDERS["lidar_scan_folder"] )


    if scans_names is not None:
        # remove glare images
        # images_names = images_names[start_index:]

        # print("number of images after removing glare is {num}".format(num=len(images_names)))
        trainset, valset, testset = split_scans(root_dataset_dir, scans_names, valset_size=valset_size, testset_size=testset_size )

        process_scans(trainset, root_dataset_dir, output_dir / "trainset")
        process_scans(valset, root_dataset_dir, output_dir  / "valset")
        process_scans(testset, root_dataset_dir, output_dir / "testset")


if __name__ == "__main__": 
    main()