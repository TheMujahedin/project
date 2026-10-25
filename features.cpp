#include "features.h"
#include <opencv2/features2d.hpp>
#include <cmath>
#include <limits>
#include <random>
#include <algorithm>
#include <iostream>

// Εξαγωγή SIFT descriptors από εικόνα σε κλίμακα του γκρι
cv::Mat extract_sift_descriptors(const std::string& filepath) {
    cv::Mat image = cv::imread(filepath, cv::IMREAD_GRAYSCALE);
    if (image.empty()) {
        return cv::Mat();
    }

    static cv::Ptr<cv::SIFT> detector = cv::SIFT::create();
    std::vector<cv::KeyPoint> keypoints;
    cv::Mat descriptors;
    detector->detectAndCompute(image, cv::noArray(), keypoints, descriptors);
    return descriptors;
}

// Υπολογισμός τετραγώνου Ευκλείδειας απόστασης μεταξύ δύο διανυσμάτων μήκους d
static float euclidean_dist_sq(const float* a, const float* b, int d) {
    float dist = 0.0f;
    for (int i = 0; i < d; ++i) {
        float diff = a[i] - b[i];
        dist += diff * diff;
    }
    return dist;
}

// Εκπαίδευση του Visual Vocabulary (k-means στα SIFT descriptors)
void VisualVocabulary::train(const std::vector<cv::Mat>& descriptors_list, int max_iters, unsigned int seed) {
    std::vector<std::vector<float>> all_descriptors;
    for (const auto& mat : descriptors_list) {
        if (mat.empty()) continue;
        for (int r = 0; r < mat.rows; ++r) {
            std::vector<float> desc(128);
            for (int c = 0; c < 128; ++c) {
                desc[c] = mat.at<float>(r, c);
            }
            all_descriptors.push_back(desc);
        }
    }

    int N = static_cast<int>(all_descriptors.size());
    if (N < K) {
        std::cerr << "Σφάλμα: Λιγότεροι descriptors (" << N << ") από τα κέντρα K (" << K << ")\n";
        return;
    }

    // Τυχαία αρχικοποίηση των K κέντρων
    std::mt19937 rng(seed);
    std::vector<int> indices(N);
    for (int i = 0; i < N; ++i) indices[i] = i;
    std::shuffle(indices.begin(), indices.end(), rng);

    centroids.assign(K, std::vector<float>(128, 0.0f));
    for (int k = 0; k < K; ++k) {
        centroids[k] = all_descriptors[indices[k]];
    }

    // k-means iterations
    std::vector<int> assignments(N, -1);
    for (int iter = 0; iter < max_iters; ++iter) {
        bool changed = false;

        // Βήμα 1: Ανάθεση στο πλησιέστερο κέντρο
        for (int i = 0; i < N; ++i) {
            float min_d = std::numeric_limits<float>::max();
            int best_k = 0;
            for (int k = 0; k < K; ++k) {
                float d = euclidean_dist_sq(all_descriptors[i].data(), centroids[k].data(), 128);
                if (d < min_d) {
                    min_d = d;
                    best_k = k;
                }
            }
            if (assignments[i] != best_k) {
                assignments[i] = best_k;
                changed = true;
            }
        }

        if (!changed && iter > 0) break;

        // Βήμα 2: Επαναϋπολογισμός κέντρων
        std::vector<std::vector<float>> new_centroids(K, std::vector<float>(128, 0.0f));
        std::vector<int> counts(K, 0);

        for (int i = 0; i < N; ++i) {
            int k = assignments[i];
            counts[k]++;
            for (int d = 0; d < 128; ++d) {
                new_centroids[k][d] += all_descriptors[i][d];
            }
        }

        for (int k = 0; k < K; ++k) {
            if (counts[k] > 0) {
                for (int d = 0; d < 128; ++d) {
                    centroids[k][d] = new_centroids[k][d] / counts[k];
                }
            }
        }
    }
}

// Μετατροπή των SIFT descriptors σε κανονικοποιημένο διάνυσμα BoVW (άθροισμα = 1)
std::vector<float> VisualVocabulary::compute_bow_vector(const cv::Mat& descriptors) const {
    std::vector<float> bow(K, 0.0f);
    int num_descriptors = descriptors.rows;

    if (num_descriptors == 0) {
        return bow;
    }

    for (int r = 0; r < num_descriptors; ++r) {
        std::vector<float> desc(128);
        for (int c = 0; c < 128; ++c) {
            desc[c] = descriptors.at<float>(r, c);
        }

        float min_d = std::numeric_limits<float>::max();
        int best_k = 0;
        for (int k = 0; k < K; ++k) {
            float d = euclidean_dist_sq(desc.data(), centroids[k].data(), 128);
            if (d < min_d) {
                min_d = d;
                best_k = k;
            }
        }
        bow[best_k] += 1.0f;
    }

    for (int k = 0; k < K; ++k) {
        bow[k] /= static_cast<float>(num_descriptors);
    }

    return bow;
}
