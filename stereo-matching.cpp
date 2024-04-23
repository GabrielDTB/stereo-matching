#include <vector>
#include <iostream>
#include <opencv2/opencv.hpp>
using namespace cv;
using namespace std;

void process_bytes(Mat left_image, Mat right_image) {
    imshow("Loaded Left Image", left_image);
    imshow("Loaded Right Image", right_image);
    waitKey(0);
    destroyAllWindows();

    return;
}

vector<int> get_window_values(const Mat &image, int center_x, int center_y, int window_size) {
    vector<int> window_values;
    int half_size = window_size / 2;

    for (int i = -half_size; i <= half_size; ++i) {
        for (int j = -half_size; j <= half_size; ++j) {
            window_values.push_back(image.at<uchar>(center_y + i, center_x + j));
        }
    }

    return window_values;
}

vector<int> vector_subtract(vector<int> &v1, vector<int> &v2) {
    vector<int> v;
    for (int i = 0; i < v1.size(); ++i) {
        v.push_back((v1[i] - v2[i]));
    }
    return v;
}

void vector_abs(vector<int> &v) {
    for (int i = 0; i < v1.size(); ++i) {
        if (v[i] < 0) v[i] = (0 - v[i]);
    }  
    return;
}

int vec_sum(vector<int> &v) {
    int sum = 0;
    for (int i = 0; i < v.size(); ++i) {
        sum += v[i];
    }
    return sum;
}

int main() {
    Mat left_image = imread("teddyL.pgm", IMREAD_GRAYSCALE);
    Mat right_image = imread("teddyR.pgm", IMREAD_GRAYSCALE);
    if (left_image.empty() || right_image.empty()) {
        std::cerr << "Could not open or find the two images" << std::endl;
        return -1;
    }

    process_bytes(left_image, right_image);

    int window_size = 5, half_size = 2, max_disparity = 64;    
    Mat padded_left_image;
    Mat padded_right_image;
    copyMakeBorder(left_image, padded_left_image, half_size, half_size, half_size, half_size, BORDER_CONSTANT, Scalar(-1));
    copyMakeBorder(right_image, padded_right_image, half_size, half_size, half_size, half_size, BORDER_CONSTANT, Scalar(-1));

    Mat disparity_map(left_image.size(), CV_8UC1, Scalar(0));
    for (int y = half_size; y < left_image.rows - half_size; ++y) {
        for (int x = half_size; x < left_image.cols - half_size; ++x) {
            int min_sad = INT_MAX, best_disparity = 0, sad = 0;
            vector<int> left_neighbors = get_window_values(&padded_left_image, x, y, window_size);

            for (int r = half_size; r < right_image.cols - half_size; ++r) {
                vector<int> right_neighbors = get_window_values(&padded_right_image, r, y, window_size);
                vector<int> sub = vector_subtract(left_neighbors, right_neighbors);
                vector_abs(sub);
                sad = vec_sum(sub);

                if (sad < min_sad) {
                    min_sad = sad;
                    best_disparity = r - x;
                }
            }

            disparity_map.at<uchar>(y-half_size, x_half_size) = best_disparity * (64 / left_image.cols);
        }
    }
        
    imshow("Disparity Maps", disparity_map);

}

/*
 * Ptr<StreeoSGBM> stereo = StereoSGBM::create(0, num_dis, block_size);
 * stereo->compute(left_image, right_image, disparity_map);
 * normalize(disparity_map, disparity_map, 0, 255, NORM_MINMAX, CV_8UC1);
 * imshow("Disparity Map", disparity_map);
 * waitKey(0);
 */
