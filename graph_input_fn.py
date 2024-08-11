import numpy as np
import cv2 as cv
import os

input_node_name = "conv2d_input"
calib_batch_size = 16  # Ensure this is an integer

def input_fn(iter=None):
    global calib_batch_size
    dir_path = "./Tumor_Sample"
    img_list = []

    # Check if the directory exists
    if not os.path.isdir(dir_path):
        raise ValueError(f"The directory {dir_path} does not exist.")

    # Loop through each file in the directory
    for img_name in os.listdir(dir_path):
        path = os.path.join(dir_path, img_name)
        
        # Read image using OpenCV
        image = cv.imread(path)
        if image is not None:
            # Convert image to grayscale
            image = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
            # Resize the image to (256, 256)
            image = cv.resize(image, (256, 256))
            # Normalize the image
            image = image.astype(np.float32) / 255.0
            # Add a channel dimension
            image = np.expand_dims(image, axis=-1)
            img_list.append(image)
        else:
            print(f"Warning: Unable to read image {path}. Skipping.")

    # Ensure we have images to return
    if len(img_list) == 0:
        raise ValueError("No valid images found in the specified path.")
    
    # Convert list to numpy array
    img_array = np.array(img_list)
    
    # Check the batch size
    if img_array.shape[0] < calib_batch_size:
        raise ValueError(f"Not enough images to form a batch of size {calib_batch_size}.")

    return {input_node_name: img_array[:calib_batch_size]}

def main():
    data = input_fn()
    print(data)

if __name__ == "__main__":
    main()
