// Η κλάση ExactSearch εκτελεί γραμμική σάρωση O(N * d) 
//στη βάση δεδομένων και λειτουργεί
//ως το απόλυτο σημείο αναφοράς (Ground Truth). 
// Χρησιμοποιείται για την αξιολόγηση των 
//προσεγγιστικών αλγορίθμων (LSH, Hypercube, IVFFlat, IVFPQ) ως προς τις μετρικές 
//Recall@5, Recall@10 και mAP@10.
 
#ifndef EXACT_SEARCH_H
#define EXACT_SEARCH_H

#include "features.h"
#include <vector>
#include <string>
#include <queue>
#include <cmath>

class ExactSearch {
private:
    // Αναφορά στη βάση δεδομένων εικόνων (HPatches βάση + MIRFlickr distractors)
    // Χρήση const reference για αποφυγή περιττής αντιγραφής δεδομένων στη μνήμη
    const std::vector<ImageRecord>& database;

public:
    ExactSearch(const std::vector<ImageRecord>& db) : database(db) {}

    std::vector<std::string> search(const std::vector<float>& query_vec, int top_k = 10) {
        // Max-heap για τη διατήρηση των top_k μικρότερων αποστάσεων
        // Το ζεύγος αποθηκεύει (απόσταση, αναγνωριστικό_εικόνας)
        typedef std::pair<float, std::string> Pair;
        std::priority_queue<Pair> pq;

        int d_len = static_cast<int>(query_vec.size());
        
        // Γραμμική σάρωση όλων των εγγραφών της βάσης δεδομένων
        for (const auto& item : database) {
            float dist = 0.0f;
            // Υπολογισμός τετραγωνικής Ευκλείδειας απόστασης L2
            for (int i = 0; i < d_len; ++i) {
                float diff = query_vec[i] - item.bow_vec[i];
                dist += diff * diff;
            }
            dist = std::sqrt(dist);
            // Εισαγωγή στο heap και διατήρηση μεγέθους το πολύ ίσο με top_k
            pq.push({dist, item.id});
            if (pq.size() > static_cast<size_t>(top_k)) {
                pq.pop();// Αφαίρεση του στοιχείου με τη μεγαλύτερη απόσταση
            }
        }
        // Εξαγωγή αποτελεσμάτων από το heap
        // Γεμίζουμε τον πίνακα από το τέλος προς την αρχή ώστε τα αποτελέσματα
        // να ταξινομηθούν σε αύξουσα σειρά απόστασης 
        // (από το πιο όμοιο στο λιγότερο όμοιο)

        std::vector<std::string> results(pq.size());
        for (int i = static_cast<int>(pq.size()) - 1; i >= 0; --i) {
            results[i] = pq.top().second;
            pq.pop();
        }
        return results;
    }
};

#endif // EXACT_SEARCH_H
