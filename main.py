import cv2
import numpy as np
import unittest

def show_images(left_image, right_image):
    cv2.imshow("Left Image", left_image)
    cv2.imshow("Right Image", right_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

def neighbors(image, center_x, center_y, padding):
    return image[
        center_y - padding : center_y + padding + 1,
        center_x - padding : center_x + padding + 1
    ].flatten()

def abs_dif(x, y):
    if x > y:
        return x - y
    else:
        return y - x

def vec_abs_dif(x, y):
    return np.vectorize(abs_dif)(x, y)

def main():
    left_image_name = "teddyL.pgm"
    right_image_name = "teddyR.pgm"

    left_image = cv2.imread(left_image_name, cv2.IMREAD_GRAYSCALE)
    right_image = cv2.imread(right_image_name, cv2.IMREAD_GRAYSCALE)

    if left_image is None:
        print("Couldn't find", left_image_name)
        return
    if right_image is None:
        print("Couldn't find", right_image_name)
        return
    if left_image.shape != right_image.shape:
        print("Images have different sizes", left_image.shape, right_image.shape)
        return

    # show_images(left_image, right_image)

    window_size = 5
    padding = window_size // 2
    max_disparity = 64

    padded_left_image = cv2.copyMakeBorder(
        left_image,
        padding,
        padding,
        padding,
        padding,
        cv2.BORDER_CONSTANT,
        value=-1
    )
    padded_right_image = cv2.copyMakeBorder(
        right_image,
        padding,
        padding,
        padding,
        padding,
        cv2.BORDER_CONSTANT,
        value=-1
    )

    print(left_image.shape)

    sad = sum(vec_abs_dif(
        neighbors(padded_left_image, padding, padding, padding),
        neighbors(padded_right_image, padding, padding, padding)
    ))
    print(sad)


    disparity_map = np.zeros_like(left_image, dtype=np.uint16)
    counter = 0

    for y in range(padding, padded_left_image.shape[0] - padding):
        for x in range(padding, padded_left_image.shape[1] - padding):
            min_sad = float('inf')
            best_disparity = 0
            left_neighbors = neighbors(padded_left_image, x, y, padding)

            for r in range(padding, padded_right_image.shape[1] - padding):
                right_neighbors = neighbors(padded_right_image, r, y, padding)
                sad = sum(vec_abs_dif(
                    left_neighbors, right_neighbors
                ))
                counter += 1
                if counter % 100000 == 0:
                    print(counter)

                if sad < min_sad:
                    min_sad = sad
                    best_disparity = abs(int(r) - x)

            disparity_map[y - padding, x - padding] = best_disparity

    # cv2.imshow("Disparity Maps", disparity_map)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()

if __name__ == "__main__":
    main()