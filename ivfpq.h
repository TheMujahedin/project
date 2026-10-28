
 // Δηλώσεις κλάσης και δομών για το ευρετήριο Inverted File with Product Quantization (IVFPQ).
 //
 //Ο αλγόριθμος IVFPQ συνδυάζει χονδρική διαμέριση χώρου (coarse quantization) με λεπτή 
 //συμπίεση διανυσμάτων υπολοίπου (fine product quantization) μέσω των εξής σταδίων:
 //
 //1. Coarse Quantization (IVF): Διαχωρισμός του χώρου σε k_clusters περιοχές Voronoi.
 //2. Υπολογισμός Residuals: r(x) = x - c(x), όπου c(x) είναι το πλησιέστερο coarse centroid.
 //3. Product Quantization (PQ): Τεμαχισμός κάθε residual σε M_subvectors υποδιανύσματα
 //   μήκους d_sub = dim / M, και ανεξάρτητη κβάντιση καθενός σε k_subclusters κέντρα.
 //4. Συμπιεσμένη Αποθήκευση: Κάθε διάνυσμα κωδικοποιείται σε M bytes (uint8_t) στις Inverted Lists.
 //5. Ασύμμετρη Αναζήτηση (ADC): Ταχύτατος υπολογισμός προσεγγιστικής απόστασης μέσω lookup tables.
 //

#ifndef IVFPQ_H
#define IVFPQ_H

#include "features.h"
#include <vector>
#include <string>
#include <cstdint>

class IVFPQIndex {
private:
    int k_clusters;     // Ο αριθμός των coarse συστάδων Voronoi (coarse quantizer).
    int n_probe;        // Ο αριθμός των πλησιέστερων coarse συστάδων που εξετάζονται ανά query.
    int M_subvectors;   // Ο αριθμός των υποδιανυσμάτων στα οποία διασπάται κάθε διάνυσμα (π.χ. M = 16).
    int n_bits;         // Ο αριθμός των bits κωδικοποίησης ανά υποχώρο (τυπικά n_bits = 8).
    int k_subclusters;  // Ο αριθμός των centroids ανά υποχώρο: 2^n_bits (για n_bits=8 είναι 256).
    int dim;            // Η αρχική διάσταση των διανυσμάτων BoVW (d = 256).
    int d_sub;          // Η διάσταση κάθε επιμέρους υποδιανύσματος (d_sub = dim / M_subvectors).

    /// Αναφορά στη βάση δεδομένων εικόνων (HPatches βάση + MIRFlickr distractors).
    const std::vector<ImageRecord>& database;

    /// Κέντρα χονδρικής κβάντισης (Coarse Centroids) διάστασης (k_clusters x dim).
    std::vector<std::vector<float>> coarse_centroids;

    //
    // Κωδικολόγια Product Quantization (Codebooks).
    //
    //Περιλαμβάνει M_subvectors πίνακες (έναν ανά υποχώρο).
    //Κάθε πίνακας περιέχει k_subclusters κέντρα (sub-centroids) διάστασης d_sub.
    //
    std::vector<std::vector<std::vector<float>>> pq_codebooks;

    //
    // EncodedVector
    // Συμπιεσμένη αναπαράσταση ενός διανύσματος της βάσης.
    //
    struct EncodedVector {
        int original_idx;            // Ο αρχικός δείκτης (index) της εικόνας στον πίνακα database.
        std::vector<uint8_t> codes;  // Οι M συμπιεσμένοι κωδικοί κβάντισης (1 byte ανά υποδιάνυσμα).
    };

    /// Ανεστραμμένοι κατάλογοι (Inverted Lists): cluster_id -> λίστα συμπιεσμένων διανυσμάτων (EncodedVector).
    std::vector<std::vector<EncodedVector>> inverted_lists;

    //
    // Εκπαίδευση του K-Means για τον m-οστό υποχώρο Product Quantization.
    //
    // m Ο δείκτης του υποδιανύσματος / υποχώρου (0 έως M_subvectors - 1).
    // residuals Πίνακας με όλα τα διανύσματα υπολοίπου r(x) της βάσης.
    // max_iters Μέγιστος αριθμός επαναλήψεων του k-means.
    // seed Σπόρος γεννήτριας τυχαίων αριθμών.
    //
    void train_subspace_kmeans(int m, const std::vector<std::vector<float>>& residuals, int max_iters, unsigned int seed);

public:
    //
    // Κατασκευαστής του ευρετηρίου IVFPQ.
    //
    // db Αναφορά στη βάση δεδομένων εικόνων.
    // kclusters Πλήθος συστάδων Voronoi (coarse quantizer).
    // nprobe Πλήθος συστάδων προς εξέταση ανά query.
    // M Πλήθος υποδιανυσμάτων Product Quantization (π.χ. 16).
    // nbits Πλήθος bits κωδικοποίησης ανά υποχώρο (π.χ. 8 bits -> 256 sub-centroids).
    // dimension Αρχική διάσταση των BoVW διανυσμάτων (256).
    //
    IVFPQIndex(const std::vector<ImageRecord>& db, int kclusters, int nprobe, int M, int nbits, int dimension);

    //
    // Πλήρης εκπαίδευση και κατασκευή του ευρετηρίου IVFPQ.
    //
    //1. Εκπαιδεύει τον coarse quantizer (k-means στα 256-D διανύσματα).
    //2. Υπολογίζει τα διανύσματα υπολοίπου (residuals).
    //3. Εκπαιδεύει τα M ανεξάρτητα υπο-λεξικά (PQ codebooks).
    //4. Κβαντίζει και αποθηκεύει τα συμπιεσμένα διανύσματα (M bytes έκαστο) στις inverted lists.
    //
    // max_iters Μέγιστος αριθμός επαναλήψεων ανά K-Means εκπαίδευση.
    // seed Σπόρος γεννήτριας τυχαίων αριθμών.
    //
    void train_and_build(int max_iters, unsigned int seed);

    //
    //@ Εκτέλεση προσεγγιστικής αναζήτησης με Asymmetric Distance Computation (ADC).
    //
    //Επιλέγει τις n_probe πλησιέστερες coarse συστάδες, προϋπολογίζει πίνακα αποστάσεων (lookup table)
    //για το residual του query, και εκτελεί ταχύτατη αξιολόγηση μέσω ADC table lookups.
    //
    // query_vec Το каνονικοποιημένο BoVW διάνυσμα του ερωτήματος (διάστασης d=256).
    //@ top_k Ο αριθμός των πλησιέστερων γειτόνων προς ανάκτηση (προεπιλογή: 10).
    //return std::vector<std::string> Λίστα με τα αναγνωριστικά (IDs) των top-k εικόνων σε αύξουσα σειρά απόστασης.
    //
    std::vector<std::string> search(const std::vector<float>& query_vec, int top_k = 10) const;
};

#endif // IVFPQ_H