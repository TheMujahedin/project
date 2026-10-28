
// Δηλώσεις κλάσης για το ευρετήριο Inverted File with Flat Vectors (IVFFlat).
//
//Το ευρετήριο IVFFlat οργανώνει τα διανύσματα BoVW σε κελιά Voronoi μέσω K-Means συσταδοποίησης.
//Τα διανύσματα αποθηκεύονται ασυμπίεστα (flat) σε ανεστραμμένους καταλόγους (inverted lists)
//ανάλογα με το πλησιέστερο κέντρο συστάδας (centroid).
//
//Χαρακτηριστικά λειτουργίας:
//- Εκπαίδευση k_clusters συστάδων στον χώρο των διανυσμάτων (d = 256).
//- Κατασκευή ανεστραμμένων καταλόγων: cluster_id -> δείκτες διανυσμάτων βάσης.
//- Προσεγγιστική αναζήτηση εξετάζοντας αποκλειστικά τις n_probe πλησιέστερες συστάδες
//  στο διάνυσμα του ερωτήματος, πετυχαίνοντας υπογραμμική πολυπλοκότητα.
//
#ifndef IVFFLAT_H
#define IVFFLAT_H

#include "features.h"
#include <vector>
#include <string>

class IVFFlatIndex {
private:
    int k_clusters;  // Ο αριθμός των συστάδων (Voronoi cells / centroids) του K-Means.
    int n_probe;     // Ο αριθμός των πλησιέστερων συστάδων που θα διερευνηθούν κατά την αναζήτηση.
    int dim;         // Η διάσταση των διανυσμάτων χαρακτηριστικών (d = 256 για BoVW).

    /// Αναφορά στη βάση δεδομένων των εικόνων (HPatches βάση + MIRFlickr distractors).
    const std::vector<ImageRecord>& database;    
 
 /// Τα κέντρα των συστάδων (centroids) διάστασης (k_clusters x dim).
    std::vector<std::vector<float>> centroids;     

    std::vector<std::vector<int>> inverted_lists;  // Inverted list: cluster_id -> δείκτες στη βάση

public:
//
    // Κατασκευαστής της δομής ευρετηρίου IVFFlat.
    // db Αναφορά στη βάση δεδομένων εικόνων.
    // kclusters Ο αριθμός των συστάδων k (π.χ. k = sqrt(N) ή σταθερή τιμή).
    // nprobe Ο αριθμός των συστάδων προς εξέταση ανά query (π.χ. nprobe = 10).
    // dimension Η διάσταση των διανυσμάτων (d = 256).
    //
    IVFFlatIndex(const std::vector<ImageRecord>& db, int kclusters, int nprobe, int dimension);

    //
    // Εκπαίδευση του K-Means και κατασκευή των ανεστραμμένων καταλόγων.
    //
    //1. Αρχικοποιεί τυχαία k_clusters κέντρα από τα διανύσματα της βάσης.
    //2. Εκτελεί τον αλγόριθμο K-Means για max_iters επαναλήψεις ή μέχρι σύγκλιση.
    //3. Αναθέτει κάθε διάνυσμα στον αντίστοιχο ανεστραμμένο κατάλογο του κέντρου του.
    //
    // max_iters Μέγιστος αριθμός επαναλήψεων του αλγορίθμου K-Means (π.χ. 15-20).
    // seed Σπόρος γεννήτριας τυχαίων αριθμών για αναπαραγωγιμότητα.
    //
    void train_and_build(int max_iters, unsigned int seed);

    //
    // Προσεγγιστική αναζήτηση για την εύρεση των top-k πλησιέστερων εικόνων.
    //
    //Εντοπίζει τα n_probe πλησιέστερα κέντρα συστάδων στο διάνυσμα query_vec και
    //υπολογίζει την πραγματική Ευκλείδεια απόσταση L2 μόνο για τα διανύσματα των αντίστοιχων
    //ανεστραμμένων καταλόγων.
    //
    // query_vec Το κανονικοποιημένο BoVW διάνυσμα του ερωτήματος (query).
    // top_k Ο αριθμός των κορυφαίων αποτελεσμάτων προς επιστροφή (προεπιλογή: 10).
    //return std::vector<std::string> Λίστα με τα IDs των εικόνων ταξινομημένα σε αύξουσα σειρά απόστασης.
    //
    std::vector<std::string> search(const std::vector<float>& query_vec, int top_k = 10) const;
};

#endif // IVFFLAT_H