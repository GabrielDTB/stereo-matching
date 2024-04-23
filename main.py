import cv2
import numpy as np

def process_bytes(left_image, right_image):
    cv2.imshow("Loaded Left Image", left_image)
    cv2.imshow("Loaded Right Image", right_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

def get_window_values(image, center_x, center_y, window_size):
    window_values = []
    half_size = window_size // 2

    for i in range(-half_size, half_size + 1):
        for j in range(-half_size, half_size + 1):
            window_values.append(image[center_y + i, center_x + j])

    return window_values

def vector_subtract(v1, v2):
    return [x - y for x, y in zip(v1, v2)]

def vector_abs(v):
    return [abs(x) for x in v]

def vec_sum(v):
    return sum(v)

def main():
    left_image = cv2.imread("teddyL.pgm", cv2.IMREAD_GRAYSCALE)
    right_image = cv2.imread("teddyR.pgm", cv2.IMREAD_GRAYSCALE)
    if left_image is None or right_image is None:
        print("Could not open or find the two images")
        return

    process_bytes(left_image, right_image)

    window_size = 5
    half_size = 2
    max_disparity = 64

    padded_left_image = cv2.copyMakeBorder(left_image, half_size, half_size, half_size, half_size, cv2.BORDER_CONSTANT, value=-1)
    padded_right_image = cv2.copyMakeBorder(right_image, half_size, half_size, half_size, half_size, cv2.BORDER_CONSTANT, value=-1)

    disparity_map = np.zeros_like(left_image, dtype=np.uint8)
    for y in range(half_size, left_image.shape[0] - half_size):
        for x in range(half_size, left_image.shape[1] - half_size):
            min_sad = float('inf')
            best_disparity = 0
            left_neighbors = get_window_values(padded_left_image, x, y, window_size)

            for r in range(half_size, right_image.shape[1] - half_size):
                right_neighbors = get_window_values(padded_right_image, r, y, window_size)
                sub = vector_subtract(left_neighbors, right_neighbors)
                sub = vector_abs(sub)
                sad = vec_sum(sub)

                if sad < min_sad:
                    min_sad = sad
                    best_disparity = r - x

            best_disparity = int((best_disparity * 64) / max_disparity)

            disparity_map[y - half_size, x - half_size] = best_disparity

    cv2.imshow("Disparity Maps", disparity_map)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()

