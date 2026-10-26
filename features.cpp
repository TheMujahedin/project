#include "features.h"
#include <iostream>
#include <cmath>
#include <random>
#include <fstream>
#include <algorithm>

cv::Mat extract_sift_descriptors(const std::string& image_path) {
    cv::Mat img = cv::imread(image_path, cv::IMREAD_GRAYSCALE);
    if (img.empty()) {
        return cv::Mat();
    }

    cv::Ptr<cv::SIFT> detector = cv::SIFT::create();
    std::vector<cv::KeyPoint> keypoints;
    cv::Mat descriptors;
    detector->detectAndCompute(img, cv::noArray(), keypoints, descriptors);

    return descriptors;
}

void VisualVocabulary::train(const std::vector<cv::Mat>& descriptors_list, int max_iters, unsigned int seed) {
    std::vector<cv::Mat> valid_descriptors;
    int total_descriptors = 0;

    for (const auto& mat : descriptors_list) {
        if (!mat.empty() && mat.cols == 128) {
            cv::Mat float_mat;
            if (mat.type() != CV_32F) {
                mat.convertTo(float_mat, CV_32F);
            } else {
                float_mat = mat;
            }
            valid_descriptors.push_back(float_mat);
            total_descriptors += float_mat.rows;
        }
    }

    if (total_descriptors == 0) {
        std::cerr << "Σφάλμα: Δεν βρέθηκαν έγκυροι SIFT descriptors για εκπαίδευση!" << std::endl;
        return;
    }

    cv::Mat all_descriptors(total_descriptors, 128, CV_32F);
    int current_row = 0;
    for (const auto& mat : valid_descriptors) {
        mat.copyTo(all_descriptors.rowRange(current_row, current_row + mat.rows));
        current_row += mat.rows;
    }

    if (total_descriptors < K) {
        K = total_descriptors;
    }

    cv::TermCriteria criteria(cv::TermCriteria::EPS + cv::TermCriteria::COUNT, max_iters, 1e-4);
    cv::Mat labels;
    cv::kmeans(all_descriptors, K, labels, criteria, 3, cv::KMEANS_PP_CENTERS, centroids);
}

std::vector<float> VisualVocabulary::compute_bow_vector(const cv::Mat& descriptors) const {
    std::vector<float> bow(K, 0.0f);
    if (descriptors.empty() || centroids.empty()) {
        return bow;
    }

    cv::Mat float_desc;
    descriptors.convertTo(float_desc, CV_32F);

    for (int i = 0; i < float_desc.rows; ++i) {
        cv::Mat query_row = float_desc.row(i);
        double min_dist = 1e18;
        int best_cluster = 0;

        for (int k = 0; k < centroids.rows; ++k) {
            double dist = cv::norm(query_row, centroids.row(k), cv::NORM_L2SQR);
            if (dist < min_dist) {
                min_dist = dist;
                best_cluster = k;
            }
        }
        bow[best_cluster] += 1.0f;
    }

    // L2 Κανονικοποίηση (Unit Norm)
    float sum_sq = 0.0f;
    for (float val : bow) {
        sum_sq += val * val;
    }
    if (sum_sq > 0.0f) {
        float norm = std::sqrt(sum_sq);
        for (float& val : bow) {
            val /= norm;
        }
    }

    return bow;
}

bool VisualVocabulary::save(const std::string& filepath) const {
    std::ofstream out(filepath, std::ios::binary);
    if (!out.is_open()) return false;

    out.write(reinterpret_cast<const char*>(&K), sizeof(K));
    int rows = centroids.rows;
    int cols = centroids.cols;
    int type = centroids.type();

    out.write(reinterpret_cast<const char*>(&rows), sizeof(rows));
    out.write(reinterpret_cast<const char*>(&cols), sizeof(cols));
    out.write(reinterpret_cast<const char*>(&type), sizeof(type));
    out.write(reinterpret_cast<const char*>(centroids.data), centroids.total() * centroids.elemSize());
    return true;
}

bool VisualVocabulary::load(const std::string& filepath) {
    std::ifstream in(filepath, std::ios::binary);
    if (!in.is_open()) return false;

    in.read(reinterpret_cast<char*>(&K), sizeof(K));
    int rows, cols, type;
    in.read(reinterpret_cast<char*>(&rows), sizeof(rows));
    in.read(reinterpret_cast<char*>(&cols), sizeof(cols));
    in.read(reinterpret_cast<char*>(&type), sizeof(type));

    centroids.create(rows, cols, type);
    in.read(reinterpret_cast<char*>(centroids.data), centroids.total() * centroids.elemSize());
    return true;
}