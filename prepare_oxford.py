import numpy as np
import pandas as pd
import cv2
import glob
import argparse
from pathlib import Path
from tqdm import tqdm

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
def find_all_images(dataset_dir: Path) -> list:
    """
    function finds all images in a folder

    dataset_dir: Pathlib, root directory of chunk

    returns
        list of images found
    """

    images_list = dataset_dir.glob("*"+FOLDERS_EXT["image_folder"])
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

def main():
    parser = argparse.ArgumentParser(description='prepare oxford')
    
     #parameters
    parser.add_argument('--dataset_files', default='datasets/s3dis',help='path to dataset orignal files')
    parser.add_argument('--include', help='starting index to include')
    parser.add_argument('--output_dir', default='data',help='path to output dir')

    args = parser.parse_args()
    
    root_dataset_dir = Path(args.dataset_files)
    start_index = int(args.include)
    output_dir = Path(args.output_dir)

    if not root_dataset_dir.exists():
        print("dataset dir does not exit")
        return None
    
    if not output_dir.exists():
        print("create output dir")
        output_dir.mkdir(parents=True, exist_ok=True)
    
    # find all images
    images_names = find_all_images(dataset_dir=root_dataset_dir / FOLDERS["image_folder"] )

    if images_names is not None:
        # remove glare images
        images_names = images_names[start_index:]

        print("number of images after removing glare is {num}".format(num=len(images_names)))

        for image_num in tqdm(images_names):

            # read lidar scan
            lidar_scan_path = root_dataset_dir / FOLDERS["lidar_scan_folder"] / ( image_num + FOLDERS_EXT["lidar_scan_folder"] )
            lidar_scan_data = read_lidar_scan(lidar_scan_path) 

            # skip lidar scans that have nan values
            nn = lidar_scan_data[np.isnan(lidar_scan_data)]
            if not (nn.size == 0):
                continue

            # read lidar projection
            lidar_projection_path = root_dataset_dir / FOLDERS["lidar_project_folder"] / ( image_num + FOLDERS_EXT["lidar_project_folder"] )
            lidar_projection_data = read_lidar_projection(lidar_projection_path)

            # read lidar label
            lidar_labels_path = root_dataset_dir / FOLDERS["labels_folder"] / ( image_num + FOLDERS_EXT["labels_folder"] )
            lidar_labels_data = read_lidar_labels(lidar_labels_path)

            # read image
            image_path = root_dataset_dir / FOLDERS["image_folder"] / ( image_num + FOLDERS_EXT["image_folder"] )
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
                result[i,6] = lidar_labels_data[i].astype(np.uint8)
            
            #save file
            outfile = str(output_dir / image_num )+ ".npy"
            np.save(outfile, result)




if __name__ == "__main__": 
    main()