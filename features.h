#ifndef FEATURES_H
#define FEATURES_H

#include <vector>
#include <string>
#include <opencv2/opencv.hpp>

// Αναπαράσταση μιας εικόνας στη βάση ή ως query
struct ImageRecord {
    std::string id;              // π.χ. "v_bird/1" ή "im102.jpg"
    std::string sequence_id;     // π.χ. "v_bird" (για HPatches)
    int img_index;               // 1 έως 6 για HPatches, 0 για distractors
    bool is_query;               // true αν είναι η εικόνα 1
    std::vector<float> bow_vec;  // Διάνυσμα διάστασης K (BoVW)
};

// Εξαγωγή όλων των SIFT descriptors μιας εικόνας
cv::Mat extract_sift_descriptors(const std::string& filepath);

// Κλάση δημιουργίας Visual Vocabulary (k-means) και εξαγωγής BoVW διανυσμάτων
class VisualVocabulary {
private:
    int K;                       // Πλήθος κέντρων λεξικού
    std::vector<std::vector<float>> centroids; // K κέντρα διάστασης 128

public:
    VisualVocabulary(int k_clusters) : K(k_clusters) {}

    // Εκπαίδευση του k-means πάνω σε SIFT descriptors (με σταθερό seed)
    void train(const std::vector<cv::Mat>& training_descriptors, int max_iters, unsigned int seed);

    // Μετατροπή των SIFT μιας εικόνας σε κανονικοποιημένο διάνυσμα x_I μήκους K
    std::vector<float> compute_bow_vector(const cv::Mat& descriptors) const;

    int getK() const { return K; }
    const std::vector<std::vector<float>>& getCentroids() const { return centroids; }
};

#endif // FEATURES_H
