//Δηλώσεις δομών δεδομένων και κλάσεων για την εξαγωγή χαρακτηριστικών και το Bag-of-Visual-Words (BoVW).
//Τη δομή ImageRecord που συγκεντρώνει τα μεταδεδομένα και το τελικό BoVW διάνυσμα κάθε εικόνας.
//Τη συνάρτηση εξαγωγής τοπικών περιγραφέων SIFT (128-D).
//Την κλάση VisualVocabulary που διαχειρίζεται την εκπαίδευση, την κβάντιση και την αποθήκευση/φόρτωση
//του οπτικού λεξικού K-Means.

#ifndef FEATURES_H
#define FEATURES_H

#include <string>
#include <vector>
#include <opencv2/opencv.hpp>
//Ενιαία δομή αποθήκευσης μεταδεδομένων και διανυσματικής αναπαράστασης μιας εικόνας.
//Χρησιμοποιείται ομοιόμορφα τόσο για τις εικόνες του συνόλου HPatches (βάση και queries)
// όσο και για τις εικόνες περισπασμού (distractors) από το σύνολο MIRFlickr.

struct ImageRecord {
    std::string id;              // π.χ. "v_bird/1" ή "im10.jpg"
    std::string sequence_id;     // π.χ. "v_bird" (κενό για distractors)
    int img_index;               // 1 έως 6 (0 για distractors)
    bool is_query;               // true μόνο για το 1.ppm των validation/test
    std::vector<float> bow_vec;  // κανονικοποιημένο BoVW διάνυσμα
};

cv::Mat extract_sift_descriptors(const std::string& image_path);
// Διαχειριστής του οπτικού λεξικο

class VisualVocabulary {
public:
    int K;  // Το μέγεθος του οπτικού λεξικού
    cv::Mat centroids;  // Πίνακας (K x 128) τύπου CV_32F που περιέχει τα κέντρα των συστάδων (visual words).

    VisualVocabulary(int k = 256) : K(k) {}
    //Συγχωνεύει όλους τους έγκυρους descriptors των εικόνων εκπαίδευσης και εκτελεί συσταδοποίηση
    // k-means++ για τον υπολογισμό των K κέντρων.

    void train(const std::vector<cv::Mat>& descriptors_list, int max_iters = 20, unsigned int seed = 42);
    //Αντιστοιχίζει κάθε descriptor (128-D) στο πλησιέστερο centroid βάσει Ευκλείδειας απόστασης,
    // κατασκευάζει το ιστόγραμμα συχνοτήτων και το κανονικοποιεί σε μοναδιαίο μέτρο (||v||_2 = 1.0).
    
    std::vector<float> compute_bow_vector(const cv::Mat& descriptors) const;
    //Aποθήκευση του λεξικού σε δυαδικό αρχείο για caching
    
    bool save(const std::string& filepath) const;//Φόρτωση προ-εκπαιδευμένου λεξικού από δυαδικό αρχείο.
    bool load(const std::string& filepath);
};

#endif // FEATURES_H