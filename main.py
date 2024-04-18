import cv2
import zipfile
import numpy as np
import matplotlib.pyplot as plt
from numpy.testing import assert_array_equal
import unittest

def extract_files(filename):
    with zipfile.ZipFile(filename, "r") as zip:
        zip.extractall("./")

def load_images(left_img_path, right_img_path):
    left_img = cv2.imread(left_img_path, cv2.IMREAD_GRAYSCALE)
    right_img = cv2.imread(right_img_path, cv2.IMREAD_GRAYSCALE)
    return left_img, right_img

def rank_transform(image, window_size=5):
    pad_size = window_size // 2
    pad_image = cv2.copyMakeBorder(image, pad_size, pad_size, pad_size, pad_size, cv2.BORDER_CONSTANT, value=0)
    transformed_image = np.zeros_like(image)

    for i in range(pad_size, pad_size + image.shape[0]):
        for j in range(pad_size, pad_size + image.shape[1]):
            center_pixel = pad_image[i, j]
            neighbors = pad_image[i-pad_size: i+pad_size+1, j-pad_size: j+pad_size+1].flatten()
            rank = np.sum(neighbors < center_pixel)
            transformed_image[i-pad_size, j-pad_size] = rank

    return transformed_image

class TestRankTransform(unittest.TestCase):

    def do_test(self, image, expected, window_size):
        image = np.array(image)
        expected = np.array(expected)
        result = rank_transform(image, window_size)
        assert_array_equal(result, expected)

    def test_single_tile(self):
        self.do_test(
            image = [
                [0]
            ],
            expected = [
                [0]
            ],
            window_size = 1
        )
        
    def test_single_pixel_window(self):
        self.do_test(
            image = [
                [1],
                [2],
                [3]
            ],
            expected = [
                [0],
                [0],
                [0]
            ],
            window_size = 1
        )

    def test_skinny_array(self):
        self.do_test(
            image = [
                [1],
                [2],
                [3]
            ],
            expected = [
                [0],
                [1],
                [2]
            ],
            window_size = 3
        )

    def test_uniform_window(self):
        image = np.ones((5, 5))
        result = rank_transform(image, window_size=3)
        expected = 4 * np.ones((5, 5))  # Each pixel is greater than 4 others in the window
        self.assertTrue(np.array_equal(result, expected))

    def test_simple_gradient(self):
        image = np.tile(np.arange(0, 256, 256/5), (5, 1))
        result = rank_transform(image, window_size=3)
        expected = np.array([
            [0, 1, 2, 3, 4],
            [0, 1, 2, 3, 4],
            [0, 1, 2, 3, 4],
            [0, 1, 2, 3, 4],
            [0, 1, 2, 3, 4]
        ])
        self.assertTrue(np.array_equal(result, expected))

    def test_edge_handling(self):
        image = np.ones((5, 5))
        result = rank_transform(image, window_size=3)
        expected = 4 * np.ones((5, 5))
        # Edges should be handled correctly, and in this case, they should have the same rank as the center pixels
        self.assertTrue(np.array_equal(result, expected))

def disparity_maps(rank_left, rank_right, window_size, disparity_range=64):
    pad_size = window_size // 2

    # Based on online research, in order to do a fair comparison between the images
    # we solely need to pad the right image.
    pad_right_image = cv2.copyMakeBorder(rank_right, pad_size, pad_size, pad_size, pad_size, cv2.BORDER_CONSTANT, value=0)
    disparity_map = np.zeros_like(rank_left)

    for i in range(pad_size, pad_size + rank_left.shape[0]):
        for j in range(pad_size, pad_size+rank_left.shape[1]):
            left_patch = rank_left[i-pad_size:i+pad_size+1, j-pad_size: j+pad_size+1]
            best_match = None
            min_diff = float('inf')

            for d in range(disparity_range):
                right_patch = pad_right_image[i-pad_size:i+pad_size+1, j-pad_size: j+pad_size+1]
                abs_diff = np.sum(np.abs(left_patch - right_patch))
                if abs_diff < min_diff:
                    min_diff = abs_diff
                    best_match = d
            
            disparity_map[i-pad_size, j-pad_size] = best_match

    return disparity_map

# Don't believe we have to pad the images since the disparity_map, left_image, and right_image should be all of the same size
def sum_absolute_differences(disparity_map, left_image, right_image):
    sad = 0

    for i in range(disparity_map.shape[0]):
        for j in range(disparity_map.shape[1]):
            disparity = disparity_map[i, j]
            left_pixel = left_image[i, j]
            right_pixel = right_image[i, j - disparity]
            sad += abs(left_pixel - right_pixel)

    return sad

def read_ground_truth(ground_truth_map_path):
    ground_truth_map = cv2.imread(ground_truth_map_path, cv2.IMREAD_GRAYSCALE)
    ground_truth_map = np.round(ground_truth_map_path / 4).astype(np.uint8)
    return ground_truth_map

def compute_error_rate(disparity_map, ground_truth_map):
    error_count = np.sum(np.abs(disparity_map - ground_truth_map) > 1)
    total_pixels = disparity_map.shape[0] * disparity_map.shape[1]
    return error_count / total_pixels

def main():
    extract_files("teddy.zip")

    left_image, right_image = load_images('teddyL.pgm', 'teddyR.pgm')
    if left_image is None or right_image is None:
        print("Failed to load images for stereo algorithm")
        return

    left_5_transform = rank_transform(left_image)
    right_5_transform = rank_transform(right_image)

    disparity_map_3 = disparity_maps(left_5_transform, right_5_transform, 3)
    disparity_map_15 = disparity_maps(left_5_transform, right_5_transform, 15)

    sad3 = sum_absolute_differences(disparity_map_3, left_image, right_image)
    sad15 = sum_absolute_differences(disparity_map_15, left_image, right_image)

    print("Sum of absolute differences (3x3 window):", sad3)
    print("Sum of absolute differences (15x15 window):", sad15)

    ground_truth_disparity_map = read_ground_truth('disp2.pgm')
    error_rate_3 = compute_error_rate(disparity_map_3, ground_truth_disparity_map)
    error_rate_15 = compute_error_rate(disparity_map_15, ground_truth_disparity_map)

    print("Error rate (3x3 window):", error_rate_3)
    print("Error rate (15x15 window):", error_rate_15)

    # Display disparity maps
    plt.figure(figsize=(12, 6))

    # 3x3 window disparity map
    plt.subplot(1, 2, 1)
    plt.imshow(disparity_map_3, cmap='gray')
    plt.title('Disparity Map (3x3 Window)')
    plt.colorbar()

    # 15x15 window disparity map
    plt.subplot(1, 2, 2)
    plt.imshow(disparity_map_15, cmap='gray')
    plt.title('Disparity Map (15x15 Window)')
    plt.colorbar()

    plt.show()

if __name__ == "__main__":
    main()