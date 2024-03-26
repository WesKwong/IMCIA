import os
import numpy as np

def rgb_to_yuv(rgb):
    """
    Transform RGB color space to YUV color space
    """
    rgb = rgb.reshape(-1, 3).astype(np.float64)
    rgb[:, [0, 2]] = rgb[:, [2, 0]]
    convert_matrix = np.array([[0.299, 0.587, 0.114],
                               [-0.148, -0.289, 0.437],
                               [0.615, -0.515, -0.100]], dtype=np.float64)
    bias_matrix = np.array([0, 128, 128], dtype=np.float64)
    yuv = (((convert_matrix @ rgb.T).T) + bias_matrix)
    yuv = np.maximum(0, np.minimum(255, yuv)).astype(np.uint8)
    return yuv

def process_images(input_dir, output_dir):
    """
    Transform RGB images in input_dir to YUV images and save them in output_dir
    """
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    for filename in os.listdir(input_dir):
        if filename.endswith(".bmp"):
            # Read the image
            with open(os.path.join(input_dir, filename), "rb") as f:
                header = f.read(54)
                data = np.fromfile(f, dtype=np.uint8)
            # Get the width and height of the image
            width = int.from_bytes(header[18:22], byteorder='little')
            height = int.from_bytes(header[22:26], byteorder='little')
            # Transform the image to YUV color space
            rgb_data = data.reshape(height, -1)[::-1,:width*3].reshape(-1)
            yuv_data = rgb_to_yuv(rgb_data)
            # Save the image
            yuv_filename = os.path.splitext(filename)[0] + ".yuv"
            with open(os.path.join(output_dir, yuv_filename), "wb") as f:
                yuv_data.tofile(f)

if __name__ == "__main__":
    process_images("data/", "results/")