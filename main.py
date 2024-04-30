import cv2
import numpy as np
from numpy.lib.stride_tricks import sliding_window_view
import unittest
import time

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

def sad(window1, window2):
    if window1.shape != window2.shape:
        print("sad: window1 shape does not match window2 shape:", window1.shape, "vs", window2.shape)
        exit(1)
    if window1.shape[0] != window1.shape[1]:
        print("sad: window height not equal to width:", window1.shape[0], "vs", window1.shape[1])
        exit(1)

    # flat1, flat2 = window1.ravel(), window2.ravel()
    # abs_dif = vec_abs_dif(flat1, flat2)
    # return np.sum(abs_dif)
    return np.sum(np.absolute(window1 - window2))

class SadTest(unittest.TestCase):
    def do_test(self, arr1, arr2, expected):
        result = sad(arr1, arr2)
        self.assertEqual(result, expected)

    def test_1x1(self):
        self.do_test(
            np.array([
                [0]
            ], dtype='int32'),
            np.array([
                [0]
            ], dtype='int32'),
            0
        )
        self.do_test(
            np.array([
                [1]
            ], dtype='int32'),
            np.array([
                [0]
            ], dtype='int32'),
            1
        )
        self.do_test(
            np.array([
                [1]
            ], dtype='int32'),
            np.array([
                [2]
            ], dtype='int32'),
            1
        )
        
    def test_3x3(self):
        self.do_test(
            np.array([
                [0,0,0],
                [0,0,0],
                [0,0,0]
            ], dtype='int32'),
            np.array([
                [0,0,0],
                [0,0,0],
                [0,0,0]
            ], dtype='int32'),
            0
        )
        self.do_test(
            np.array([
                [1,0,1],
                [0,1,0],
                [1,0,1]
            ], dtype='int32'),
            np.array([
                [1,0,1],
                [0,1,0],
                [1,0,1]
            ], dtype='int32'),
            0
        )
        self.do_test(
            np.array([
                [1,0,1],
                [0,1,0],
                [1,0,1]
            ], dtype='int32'),
            np.array([
                [0,1,0],
                [1,0,1],
                [0,1,0]
            ], dtype='int32'),
            9
        )

    def test_big_nums(self):
        self.do_test(
            np.array([
                [255]
            ], dtype='int32'),
            np.array([
                [0]
            ], dtype='int32'),
            255
        )
        self.do_test(
            np.array([
                [255, 127],
                [0, 50]
            ], dtype='int32'),
            np.array([
                [0, 50],
                [127, 255]
            ], dtype='int32'),
            255+127-50+127+205
        )

    def test_big_arrays(self):
        self.do_test(
            np.ones((50,50), dtype='int32'),
            np.zeros((50,50), dtype='int32'),
            50*50
        )
        # self.do_test(
        #     np.ones((5000,5000), dtype='int32'),
        #     np.zeros((5000,5000), dtype='int32'),
        #     5000*5000
        # )
    def test_benchmark(self):
        n_iters = 10000
        input1 = np.zeros((5,5), dtype='int32')
        input2 = np.zeros((5,5), dtype='int32')
        expected = 0
        start = time.perf_counter()
        for i in range(n_iters):
            self.do_test(
                input1,
                input2,
                expected,
            )
        end = time.perf_counter()
        print("SAD TIME:", (end - start)/n_iters)


def sad_across_y(left_window, right_strip, output_array):
    if (left_window.shape[0] != right_strip.shape[0]):
        print("sad_across_y: left_window height not equal to right_strip height: ", left_window.shape[0], "vs", right_strip.shape[0])
        exit(1)
    if left_window.shape[0] != left_window.shape[1]:
        print("sad_across_y: left_window height not equal to width:", left_window.shape[0], "vs", left_window.shape[1])
        exit(1)
    if right_strip.shape[0] > right_strip.shape[1]:
        print("sad_across_y: right_strip height is greater than width:", right_strip.shape[0], "vs", right_strip.shape[1])
        exit(1)
    if output_array.shape[0] != 1:
        print("sad_across_y: output array does not have height 1:", output_array.shape[0])
        exit(1)

    right_windows = sliding_window_view(right_strip, (right_strip.shape[0], right_strip.shape[0]))
    # print(right_windows.shape)
    # print(right_windows)

    if output_array.shape[1] != right_windows.shape[1]:
        print("sad_across_y: number of outputs is not equal to windows:", output_array.shape[1], "vs", right_windows.shape[1])
        exit(1)

    for i in range(output_array.shape[1]):
        output_array[0, i] = sad(left_window, right_windows[0, i].reshape(left_window.shape))

class SadAcrossYTest(unittest.TestCase):
    def do_test(self, left_window, right_strip, expected):
        result = np.zeros_like(expected)
        sad_across_y(left_window, right_strip, result)
        np.testing.assert_equal(result, expected)

    def test_1x1(self):
        self.do_test(
            np.array([
                [0]
            ], dtype='int32'),
            np.array([
                [0]
            ], dtype='int32'),
            np.array([[0]])
        )

    def test_1x1_longer(self):
        self.do_test(
            np.array([
                [0]
            ]),
            np.array([
                [0,0,0]
            ]),
            np.array([
                [0,0,0]
            ])
        )
        self.do_test(
            np.array([
                [1]
            ]),
            np.array([
                [1,2,3]
            ]),
            np.array([
                [0,1,2]
            ])
        )

    def test_2x2(self):
        self.do_test(
            np.array([
                [0,0],
                [0,0]
            ]),
            np.array([
                [0,0,0,0,0,0],
                [0,0,0,0,0,0]
            ]),
            np.array([
                [0,0,0,0,0]
            ])
        )
        self.do_test(
            np.array([
                [0,1],
                [1,0]
            ]),
            np.array([
                [0,0,0,0,0,0],
                [0,0,0,0,0,0]
            ]),
            np.array([
                [2,2,2,2,2]
            ])
        )
        self.do_test(
            np.array([
                [0,1],
                [1,0]
            ]),
            np.array([
                [0,1,0,1,0,1],
                [1,0,1,0,1,0]
            ]),
            np.array([
                [0,4,0,4,0]
            ])
        )

    def test_big(self):
        self.do_test(
            np.zeros((5,5)),
            np.zeros((5,10000)),
            np.zeros((1,9996))
        )

    def test_benchmark(self):
        n_iters = 100
        input1 = np.zeros((5,5))
        input2 = np.zeros((5,450))
        expected = np.zeros((1,446))
        start = time.perf_counter()
        for i in range(n_iters):
            self.do_test(
                input1,
                input2,
                expected
            )
        end = time.perf_counter()
        print("SAD_Y_TIME:", (end - start)/n_iters)

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

    show_images(left_image, right_image)

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

    cv2.imshow("Disparity Maps", disparity_map)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

if __name__ == "__main__":
    unittest.main(exit=True)
    main()