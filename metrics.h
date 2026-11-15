
 //f Συναρτήσεις υπολογισμού μετρικών αξιολόγησης ανάκτησης (Recall@5, Recall@10, AP@10).
 //
 //Το αρχείο ορίζει συναρτήσεις επικύρωσης σχετικότητας (ground truth relevance) και
 //αξιολόγησης της ποιότητας ανάκτησης εικόνων βάσει του συνόλου HPatches:
 //
 //1. Ground Truth συνθήκη: Για κάθε query (εικόνα 1 μιας ακολουθίας, π.χ. "v_bird/1"),
 //   υπάρχουν ακριβώς 5 πραγματικά σχετικές εικόνες στη βάση (οι εικόνες 2 έως 6
 //   της ίδιας ακολουθίας, π.χ. "v_bird/2" έως "v_bird/6"). Οι εικόνες από άλλες
 //   ακολουθίες ή από το σύνολο MIRFlickr θεωρούνται μη σχετικές (distractors).
 //
 //2. Recall@N: Το κλάσμα των σχετικών εικόνων που ανακτήθηκαν μέσα στις πρώτες N θέσεις.
 //
 //3. AP@10 (Average Precision at 10): Μέση ακρίβεια που ενσωματώνει τη σειρά κατάταξης
 //   (ranking) των σχετικών εικόνων στην πρώτη δεκάδα. Ο μέσος όρος του AP@10 για όλα τα
 //   queries παράγει το τελικό mAP@10 (Mean Average Precision).
 //

#ifndef METRICS_H
#define METRICS_H

#include <vector>
#include <string>
#include <algorithm>

//
// Έλεγχος εάν ένα ανακτηθέν αποτέλεσμα είναι πραγματικά σχετικό (ground truth match).
//
//Εξετάζει αν το μοναδικό αναγνωριστικό της ανακτηθείσας εικόνας (result_id) ξεκινά
//με το πρόθεμα της ακολουθίας του query (query_seq + "/").
//
//Πχ:
//- Query: "v_bird"
//- result_id: "v_bird/3" -> Σχετικό (true)
//- result_id: "i_contruction/2" -> Μη σχετικό (false)
//- result_id: "im1042.jpg" (MIRFlickr) -> Μη σχετικό (false)
//
// query_seq Το αναγνωριστικό της ακολουθίας του ερωτήματος (π.χ. "v_bird").
// result_id Το αναγνωριστικό της εικόνας που επιστράφηκε από τον αλγόριθμο.
//return true Αν η εικόνα προέρχεται από την ίδια ακολουθία HPatches, false διαφορετικά.
//
inline bool is_relevant(const std::string& query_seq, const std::string& result_id) {
    std::string prefix = query_seq + "/";
    // Έλεγχος αν το prefix βρίσκεται στη θέση 0 του result_id (C++20 starts_with ισοδύναμο)
    return (result_id.rfind(prefix, 0) == 0);
}

///
 // Υπολογισμός της μετρικής ανάκλησης Recall@N (τυπικά N = 5 ή N = 10).
 //
 //Μαθηματικός τύπος:
 //\f[
 //\text{Recall@N} = \frac{|\text{Σχετικές εικόνες στα πρώτα } N \text{ αποτελέσματα}|}{|\text{Συνολικές σχετικές εικόνες στη βάση}|} = \frac{\text{relevant\_found}}{5.0}
 //\f]
 //
 // query_seq Το όνομα της ακολουθίας του query (π.χ. "v_dog").
 // top_results Διάνυσμα με τα IDs των εικόνων που ανέκτησε ο αλγόριθμος, ταξινομημένα κατά αύξουσα απόσταση.
 // n Το πλήθος των κορυφαίων αποτελεσμάτων που αξιολογούνται (5 ή 10).
 //return double Η τιμή του Recall@N στο διάστημα [0.0, 1.0].
 //
inline double compute_recall_at_n(const std::string& query_seq, const std::vector<std::string>& top_results, int n) {
    int relevant_found = 0;
    int limit = std::min(n, static_cast<int>(top_results.size()));
    
    // Καταμέτρηση των σχετικών εικόνων εντός του παραθύρου N
    for (int i = 0; i < limit; ++i) {
        if (is_relevant(query_seq, top_results[i])) {
            relevant_found++;
        }
    }
    
    // Στο HPatches υπάρχουν πάντα ακριβώς 5 ground truth σχετικές εικόνες (εικόνες 2-6)
    return static_cast<double>(relevant_found) / 5.0;
}

//
// Υπολογισμός του Average Precision στα πρώτα 10 αποτελέσματα (AP@10) για ένα ερώτημα.
//
//Υπολογίζει την ακρίβεια Precision@r σε κάθε θέση r όπου εντοπίζεται σχετική εικόνα
//και κανονικοποιεί με το σύνολο των πραγματικά σχετικών εικόνων (5):
//\f[
//\text{AP@10} = \frac{1}{5} \sum_{r=1}^{\min(10, |results|)} \left( \text{Precision@r} \times \text{rel}(r) \right)
//\f]
//όπου \f$\text{rel}(r) = 1\f$ αν το στοιχείο στη θέση r είναι σχετικό, αλλιώς 0.
//
//Η μετρική επιβραβεύει τις μεθόδους που τοποθετούν τις σχετικές εικόνες στις
//πρώτες θέσεις της κατάταξης (υψηλότερο ranking precision).
//
// query_seq Το όνομα της ακολουθίας του query.
// top_results Διάνυσμα με τα ανακτηθέντα IDs σε αύξουσα σειρά απόστασης.
//return double Η τιμή του AP@10 για το συγκεκριμένο query στο διάστημα [0.0, 1.0].
//
inline double compute_ap_at_10(const std::string& query_seq, const std::vector<std::string>& top_results) {
    double sum_precision = 0.0;
    int relevant_count = 0;
    int limit = std::min(10, static_cast<int>(top_results.size()));

    // Σάρωση των θέσεων κατάταξης (1-based index)
    for (int r = 1; r <= limit; ++r) {
        if (is_relevant(query_seq, top_results[r - 1])) {
            relevant_count++;
            // Precision@r = (σχετικές εικόνες μέχρι τη θέση r) / r
            sum_precision += static_cast<double>(relevant_count) / static_cast<double>(r);
        }
    }
    
    // Κανονικοποίηση διά του συνολικού πλήθους των σχετικών εικόνων (5)
    return sum_precision / 5.0;
}

#endif // METRICS_H