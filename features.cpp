// Υλοποίηση της εξαγωγής χαρακτηριστικών SIFT 
// και της κατασκευής Bag-of-Visual-Words (BoVW).
//
//Το αρχείο περιλαμβάνει:
//1. Την εξαγωγή τοπικών περιγραφέων SIFT από εικόνες (128 διαστάσεων).
//2. Την εκπαίδευση του οπτικού λεξικού (Visual Vocabulary) μεγέθους K=256 μέσω K-Means.
//3. Τη σειριοποίηση και φόρτωση του λεξικού και του cache (vocab_256.bin, mir_cache_256.bin).
//4. Την κβάντιση των descriptors και την κατασκευή L2-κανονικοποιημένων ιστογραμμάτων BoVW.
//
#include "features.h"
#include <iostream>
#include <cmath>
#include <random>
#include <fstream>
#include <algorithm>

//Μετατρέπει την εικόνα σε κλίμακα του γκρι (grayscale) και εντοπίζει σημεία ενδιαφέροντος (keypoints),
// υπολογίζοντας έναν περιγραφέα 128 διαστάσεων για κάθε σημείο.


cv::Mat extract_sift_descriptors(const std::string& image_path) {
    // Φόρτωση της εικόνας απευθείας σε μονοκαναλική μορφή (κλίμακα του γκρι)
    cv::Mat img = cv::imread(image_path, cv::IMREAD_GRAYSCALE);
    if (img.empty()) {
        return cv::Mat();// Επιστροφή κενού πίνακα αν η διαδρομή δεν είναι έγκυρη
    }
    // Δημιουργία αντικειμένου SIFT με τις προεπιλεγμένες παραμέτρους του OpenCV
    cv::Ptr<cv::SIFT> detector = cv::SIFT::create();
    std::vector<cv::KeyPoint> keypoints;
    cv::Mat descriptors;
    // Ταυτόχρονη ανίχνευση σημείων (detect) και εξαγωγή των 128-D descriptors (compute)
    detector->detectAndCompute(img, cv::noArray(), keypoints, descriptors);

    return descriptors;
}

//  Εκπαίδευση του οπτικού λεξικού (Visual Vocabulary) με χρήση K-Means clustering.
// Συλλέγει, ελέγχει και συγχωνεύει όλους τους descriptors των εικόνων εκπαίδευσης σε έναν 
// ενιαίο πίνακα τύπου CV_32F. Στη συνέχεια, ομαδοποιεί τα διανύσματα σε K συστάδες
// χρησιμοποιώντας k-means++ για τη βέλτιστη τοποθέτηση των αρχικών κέντρων.


void VisualVocabulary::train(const std::vector<cv::Mat>& descriptors_list, int max_iters, unsigned int seed) {
    std::vector<cv::Mat> valid_descriptors;
    int total_descriptors = 0;
    
    //Φιλτράρισμα και επικύρωση διαστάσεων (εξασφάλιση ότι d = 128)
    for (const auto& mat : descriptors_list) {
        if (!mat.empty() && mat.cols == 128) {
            cv::Mat float_mat;
            // Μετατροπή των τιμών σε float 32-bit αν δεν είναι ήδη, όπως απαιτεί ο k-means
            if (mat.type() != CV_32F) {
                mat.convertTo(float_mat, CV_32F);
            } else {
                float_mat = mat;
            }
            valid_descriptors.push_back(float_mat);
            total_descriptors += float_mat.rows;// Άθροιση του συνολικού πλήθους descriptors
        }
        }
    }

    // Έλεγχος ύπαρξης επαρκών δεδομένων για εκπαίδευση
    if (total_descriptors == 0) {
        std::cerr << "Σφάλμα: Δεν βρέθηκαν έγκυροι SIFT descriptors για εκπαίδευση!" << std::endl;
        return;
    }
    //Δέσμευση ενιαίου συνεχούς πίνακα μνήμης για όλα τα διανύσματα (total_descriptors x 128)
    cv::Mat all_descriptors(total_descriptors, 128, CV_32F);
    int current_row = 0;
    for (const auto& mat : valid_descriptors) {
        mat.copyTo(all_descriptors.rowRange(current_row, current_row + mat.rows));
        current_row += mat.rows;
    }

    // Ασφαλιστική δικλίδα: Αν το σύνολο των διανυσμάτων είναι μικρότερο από το K,
    // προσαρμόζουμε το K ώστε να μην αποτύχει ο αλγόριθμος clustering
    if (total_descriptors < K) {
        K = total_descriptors;
    }
    
    //Ορισμός κριτηρίων τερματισμού του K-Means:
    // Τερματίζει αν επιτευχθούν οι max_iters επαναλήψεις ή αν η μετατόπιση των centroids είναι < 1e-4
    cv::TermCriteria criteria(cv::TermCriteria::EPS + cv::TermCriteria::COUNT, max_iters, 1e-4);
    cv::Mat labels;

    // Εκτέλεση K-Means:    
    // - 3 επαναλήψεις με διαφορετικές αρχικοποιήσεις για επιλογή του καλύτερου inertia
    // - cv::KMEANS_PP_CENTERS για αποφυγή τοπικών ελαχίστων
    // - Τα κέντρα (visual words) αποθηκεύονται απευθείας στο μέλος centroids (K x 128)
    cv::kmeans(all_descriptors, K, labels, criteria, 3, cv::KMEANS_PP_CENTERS, centroids);
}

// Κατασκευή του κανονικοποιημένου διανύσματος Bag-of-Visual-Words (BoVW) για μία εικόνα.
//
//Για κάθε SIFT descriptor της εικόνας εντοπίζεται η πλησιέστερη οπτική λέξη (centroid)
//βάσει της τετραγωνικής Ευκλείδειας απόστασης L2 και ενημερώνεται το ιστόγραμμα συχνοτήτων.
//Στο τέλος, εφαρμόζεται L2 κανονικοποίηση ώστε το διάνυσμα να έχει μοναδιαίο μέτρο.
std::vector<float> VisualVocabulary::compute_bow_vector(const cv::Mat& descriptors) const {
    // Αρχικοποίηση του ιστογράμματος μεγέθους K με μηδενικά
    std::vector<float> bow(K, 0.0f);
    if (descriptors.empty() || centroids.empty()) {
        return bow;
    }
    // Εξασφάλιση ότι οι descriptors είναι σε μορφή CV_32F για υπολογισμό αποστάσεων
    cv::Mat float_desc;
    descriptors.convertTo(float_desc, CV_32F);

    //Vector Quantization 
    // Αντιστοίχιση κάθε τοπικού descriptor στο πλησιέστερο οπτικό κέντρο του λεξικού    
    for (int i = 0; i < float_desc.rows; ++i) {
        cv::Mat query_row = float_desc.row(i);
        double min_dist = 1e18;
        int best_cluster = 0;

        // Σύγκριση με όλα τα K centroids του λεξικού
        for (int k = 0; k < centroids.rows; ++k) {
            // Υπολογισμός τετραγωνικής Ευκλείδειας απόστασης (αποφυγή sqrt για ταχύτητα)
            double dist = cv::norm(query_row, centroids.row(k), cv::NORM_L2SQR);
            if (dist < min_dist) {
                min_dist = dist;
                best_cluster = k;
            }
        }
        // Αύξηση του μετρητή της αντίστοιχης οπτικής λέξης στο ιστόγραμμα
        bow[best_cluster] += 1.0f;
    }

    // L2 Κανονικοποίηση (Unit Norm: ||bow||_2 = 1.0)
    // Εξασφαλίζει ότι το μέτρο του διανύσματος είναι ανεξάρτητο από το πλήθος των εντοπισμένων
    // keypoints της εικόνας, διασφαλίζοντας ισότιμη σύγκριση στην Ευκλείδεια αναζήτηση    float sum_sq = 0.0f;
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

//Αποθήκευση του εκπαιδευμένου λεξικού σε δυαδική μορφή (binary serialization).
//Αποθηκεύει το K, τις διαστάσεις (γραμμές, στήλες), τον τύπο δεδομένων OpenCV 
//και τα ανεπεξέργαστα bytes των κέντρων, επιτρέποντας άμεση επαναφόρτωση (caching).

bool VisualVocabulary::save(const std::string& filepath) const {
    std::ofstream out(filepath, std::ios::binary);
    if (!out.is_open()) return false;
    
    // Εγγραφή του μεγέθους του λεξικού K
    out.write(reinterpret_cast<const char*>(&K), sizeof(K));

    // Εγγραφή των μεταδεδομένων του πίνακα centroids
    int rows = centroids.rows;
    int cols = centroids.cols;
    int type = centroids.type();

    out.write(reinterpret_cast<const char*>(&rows), sizeof(rows));
    out.write(reinterpret_cast<const char*>(&cols), sizeof(cols));
    out.write(reinterpret_cast<const char*>(&type), sizeof(type));

    // Εγγραφή των δυαδικών δεδομένων των centroids (K x 128 x sizeof(float))
    out.write(reinterpret_cast<const char*>(centroids.data), centroids.total() * centroids.elemSize());
    return true;
}



// Φόρτωση προ-εκπαιδευμένου λεξικού από δυαδικό αρχείο (binary deserialization).
//
//Ανακτά το K, τις διαστάσεις και τα centroids από το αποθηκευμένο αρχείο,
//παρακάμπτοντας την υπολογιστικά ακριβή διαδικασία του K-Means clustering.

bool VisualVocabulary::load(const std::string& filepath) {
    std::ifstream in(filepath, std::ios::binary);
    if (!in.is_open()) return false;

    // Ανάγνωση του μεγέθους K
    in.read(reinterpret_cast<char*>(&K), sizeof(K));

    // Ανάγνωση των διαστάσεων και του τύπου του πίνακα
    int rows, cols, type;
    in.read(reinterpret_cast<char*>(&rows), sizeof(rows));
    in.read(reinterpret_cast<char*>(&cols), sizeof(cols));
    in.read(reinterpret_cast<char*>(&type), sizeof(type));

    // Δέσμευση χώρου στον πίνακα centroids και άμεση αντιγραφή των δεδομένων
    centroids.create(rows, cols, type);
    in.read(reinterpret_cast<char*>(centroids.data), centroids.total() * centroids.elemSize());
    return true;
}