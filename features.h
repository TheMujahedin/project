#ifndef FEATURES_H
#define FEATURES_H

#include <string>
#include <vector>
#include <opencv2/opencv.hpp>

struct ImageRecord {
    std::string id;              // π.χ. "v_bird/1" ή "im10.jpg"
    std::string sequence_id;     // π.χ. "v_bird" (κενό για distractors)
    int img_index;               // 1 έως 6 (0 για distractors)
    bool is_query;               // true μόνο για το 1.ppm των validation/test
    std::vector<float> bow_vec;  // κανονικοποιημένο BoVW διάνυσμα
};

cv::Mat extract_sift_descriptors(const std::string& image_path);

class VisualVocabulary {
public:
    int K;
    cv::Mat centroids;

    VisualVocabulary(int k = 256) : K(k) {}

    void train(const std::vector<cv::Mat>& descriptors_list, int max_iters = 20, unsigned int seed = 42);
    std::vector<float> compute_bow_vector(const cv::Mat& descriptors) const;

    bool save(const std::string& filepath) const;
    bool load(const std::string& filepath);
};

#endif // FEATURES_H