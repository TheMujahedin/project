
// Υλοποίηση του ευρετηρίου Inverted File with Flat Vectors (IVFFlat).
//
//Ο αλγόριθμος IVFFlat χωρίζει τον χώρο των διανυσμάτων σε k_clusters περιοχές Voronoi 
//μέσω k-means clustering. Κάθε διάνυσμα της βάσης αποθηκεύεται ασυμπίεστο (flat) 
//στον ανεστραμμένο κατάλογο του πλησιέστερου κέντρου (centroid).
//
//Κατά την αναζήτηση:
//1. Εντοπίζονται τα n_probe πλησιέστερα κέντρα συστάδων ως προς το ερώτημα.
//2. Εξετάζονται αποκλειστικά τα διανύσματα που ανήκουν σε αυτές τις n_probe συστάδες,
//   μειώνοντας δραστικά τον αριθμό των υπολογισμών απόστασης σε σχέση με την εξαντλητική αναζήτηση.
//
 #include "ivfflat.h"
#include <random>
#include <cmath>
#include <queue>
#include <algorithm>
#include <limits>

//
// Βοηθητική συνάρτηση υπολογισμού της τετραγωνικής Ευκλείδειας απόστασης L2 μεταξύ δύο διανυσμάτων.
//
//Χρησιμοποιεί δείκτες για μέγιστη υπολογιστική ταχύτητα και αποφεύγει τον υπολογισμό τετραγωνικής ρίζας (sqrt).
//
// a Δείκτης στις συντεταγμένες του πρώτου διανύσματος.
// b Δείκτης στις συντεταγμένες του δεύτερου διανύσματος.
// d Η διάσταση των διανυσμάτων (d = 256).
//return float Η τετραγωνική Ευκλείδεια απόσταση ||a - b||^2.
//

 static float dist_sq(const float* a, const float* b, int d) {
    float sum = 0.0f;
    for (int i = 0; i < d; ++i) {
        float diff = a[i] - b[i];
        sum += diff * diff;
    }
    return sum;
}

//
// Κατασκευαστής του ευρετηρίου IVFFlat.
//
// db Αναφορά στη βάση δεδομένων των εικόνων (HPatches + distractors).
// kclusters Ο αριθμός των συστάδων (Voronoi cells / centroids).
// nprobe Ο αριθμός των πλησιέστερων συστάδων που θα εξερευνηθούν κατά την αναζήτηση.
// dimension Η διάσταση των διανυσμάτων (d = 256 για BoVW).
//

IVFFlatIndex::IVFFlatIndex(const std::vector<ImageRecord>& db, int kclusters, int nprobe, int dimension)
    : k_clusters(kclusters), n_probe(nprobe), dim(dimension), database(db) {}

//
// Εκπαίδευση του K-Means και κατασκευή των ανεστραμμένων καταλόγων (Inverted Lists).
//
//1. Αρχικοποιεί τα k_clusters κέντρα επιλέγοντας τυχαία διανύσματα από τη βάση δεδομένων.
//2. Εκτελεί επαναληπτική βελτιστοποίηση K-Means (Lloyd's algorithm):
//   - Assignment step: Κάθε διάνυσμα ανατίθεται στο πλησιέστερο κέντρο.
//   - Update step: Τα κέντρα επανυπολογίζονται ως ο μέσος όρος των ανατεθειμένων διανυσμάτων.
//3. Δημιουργεί τους ανεστραμμένους καταλόγους καταγράφοντας τους δείκτες των διανυσμάτων ανά συστάδα.
//
// max_iters Ο μέγιστος αριθμός επαναλήψεων του k-means (κριτήριο τερματισμού).
// seed Ο σπόρος τυχαιότητας για την αρχικοποίηση των κέντρων.
//

void IVFFlatIndex::train_and_build(int max_iters, unsigned int seed) {
    int N = static_cast<int>(database.size());
    // Ασφαλιστική δικλίδα: τα clusters δεν μπορούν να υπερβαίνουν το πλήθος των σημείων
    if (k_clusters > N) k_clusters = std::max(1, N);

    
    centroids.assign(k_clusters, std::vector<float>(dim, 0.0f));
    inverted_lists.assign(k_clusters, std::vector<int>());
//  Τυχαία αρχικοποίηση κέντρων K-Means (τυχαία δειγματοληψία χωρίς επανατοποθέτηση)    std::mt19937 rng(seed);
    std::vector<int> perm(N);
    for (int i = 0; i < N; ++i) perm[i] = i;
    std::shuffle(perm.begin(), perm.end(), rng);

    for (int k = 0; k < k_clusters; ++k) {
        centroids[k] = database[perm[k]].bow_vec;
    }
// Πίνακας καταγραφής της ανάθεσης κάθε διανύσματος σε cluster (cluster ID)
    std::vector<int> assignments(N, 0);

    // Βρόχος εκπαίδευσης K-Means
    for (int iter = 0; iter < max_iters; ++iter) {
        bool changed = false;

        //Φάση Ανάθεσης (Expectation Step): Εύρεση πλησιέστερου κέντρου για κάθε διάνυσμα
        for (int i = 0; i < N; ++i) {
            float min_d = std::numeric_limits<float>::max();
            int best_k = 0;
            const float* vec = database[i].bow_vec.data();
            for (int k = 0; k < k_clusters; ++k) {
                float d = dist_sq(vec, centroids[k].data(), dim);
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

        // Πρώιμος τερματισμός: αν δεν άλλαξε καμία ανάθεση, τα κέντρα συνέκλιναν
        if (!changed && iter > 0) break;

        //Φάση Ενημέρωσης (Maximization Step): Επανυπολογισμός των κέντρων
        std::vector<std::vector<float>> new_centroids(k_clusters, std::vector<float>(dim, 0.0f));
        std::vector<int> counts(k_clusters, 0);

        /// Υπολογισμός μέσου όρου για κάθε συστάδα
        for (int i = 0; i < N; ++i) {
            int k = assignments[i];
            counts[k]++;
            const float* vec = database[i].bow_vec.data();
            for (int d = 0; d < dim; ++d) {
                new_centroids[k][d] += vec[d];
            }
        }

        for (int k = 0; k < k_clusters; ++k) {
            if (counts[k] > 0) {
                for (int d = 0; d < dim; ++d) {
                    centroids[k][d] = new_centroids[k][d] / counts[k];
                }
            }
        }
    }

    // Κατασκευή ανεστραμμένων καταλόγων
    //// Αντιστοίχιση των δεικτών των διανυσμάτων στη λίστα του αντίστοιχου cluster
    for (int i = 0; i < N; ++i) {
        inverted_lists[assignments[i]].push_back(i);
    }
}

//
// Εκτέλεση προσεγγιστικής αναζήτησης για την εύρεση των top-k πλησιέστερων εικόνων.
//
//1. Υπολογίζει την απόσταση του query από όλα τα κέντρα συστάδων (centroids).
//2. Ταξινομεί τα κέντρα και επιλέγει τα probes = min(n_probe, k_clusters) πλησιέστερα.
//3. Σαρώνει σειριακά μόνο τα διανύσματα που βρίσκονται στους ανεστραμμένους καταλόγους αυτών των συστάδων.
//4. Διατηρεί τα top_k πλησιέστερα στοιχεία με Max-Heap.
//
// query_vec Το κανονικοποιημένο BoVW διάνυσμα του ερωτήματος (query).
// top_k Ο αριθμός των πλησιέστερων γειτόνων προς επιστροφή (προεπιλογή: 10).
//return std::vector<std::string> Λίστα με τα IDs των εικόνων σε αύξουσα σειρά απόστασης.
//
std::vector<std::string> IVFFlatIndex::search(const std::vector<float>& query_vec, int top_k) const {
    const float* q = query_vec.data();


    //Υπολογισμός απόστασης του query από όλα τα κέντρα συστάδων
    std::vector<std::pair<float, int>> cluster_dists(k_clusters);
    for (int k = 0; k < k_clusters; ++k) {
        cluster_dists[k] = {dist_sq(q, centroids[k].data(), dim), k};
    }
    //// Ταξινόμηση ώστε να βρούμε τις n_probe πλησιέστερες συστάδες
    std::sort(cluster_dists.begin(), cluster_dists.end());

    int probes = std::min(n_probe, k_clusters);

    // Max-heap για τη διατήρηση των k πλησιέστερων γειτόνων (απόσταση, id)
    typedef std::pair<float, std::string> Pair;
    std::priority_queue<Pair> pq;

//    Σάρωση των διανυσμάτων μόνο εντός των n_probe επιλεγμένων συστάδων
    for (int p = 0; p < probes; ++p) {
        int c_id = cluster_dists[p].second;
        for (int idx : inverted_lists[c_id]) {
            const auto& item = database[idx];
            /// Υπολογισμός πραγματικής Ευκλείδειας απόστασης L2
            float d = std::sqrt(dist_sq(q, item.bow_vec.data(), dim));

            // Ενημέρωση του max-heap
            pq.push({d, item.id});
            if (pq.size() > static_cast<size_t>(top_k)) {
                pq.pop();// Αφαίρεση του υποψηφίου με τη μεγαλύτερη απόσταση
            }
        }
    }

    //Εξαγωγή των αποτελεσμάτων από το heap σε αύξουσα σειρά απόστασης
    std::vector<std::string> results(pq.size());
    for (int i = static_cast<int>(pq.size()) - 1; i >= 0; --i) {
        results[i] = pq.top().second;
        pq.pop();
    }
    return results;
}