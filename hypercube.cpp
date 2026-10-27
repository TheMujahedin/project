//Υλοποίηση του αλγορίθμου προσεγγιστικής αναζήτησης Random Hypercube Projection.

#include "hypercube.h"
#include <random>
#include <cmath>
#include <queue>
#include <algorithm>

//Κατασκευαστής και αρχικοποίηση των παραμέτρων του Hypercube Index.
//Δημιουργεί τις k_proj συναρτήσεις προβολής αντλώντας τυχαία διανύσματα v από κανονική
// κατανομή N(0, 1) και μετατοπίσεις t από ομοιόμορφη κατανομή U(0, w).
HypercubeIndex::HypercubeIndex(const std::vector<ImageRecord>& db, int kproj, double w, int M, int probes, int dimension, unsigned int seed)
    : k_proj(kproj), w_width(w), max_M(M), max_probes(probes), dim(dimension), database(db) {
    
    std::mt19937 rng(seed);
    std::normal_distribution<float> norm_dist(0.0f, 1.0f);
    std::uniform_real_distribution<float> uniform_t(0.0f, static_cast<float>(w));

    functions.resize(k_proj);
    f_mappings.resize(k_proj);

    // Αρχικοποίηση των k_proj τυχαίων υπερεπιπέδων v και των μετατοπίσεων t
    for (int i = 0; i < k_proj; ++i) {
        functions[i].v.resize(dim);
        for (int d = 0; d < dim; ++d) {
            functions[i].v[d] = norm_dist(rng);
        }
        functions[i].t = uniform_t(rng);
    }
}

//Υπολογισμός της συνάρτησης f_i(h) που αντιστοιχίζει έναν ακέραιο κάδο h σε ένα bit {0, 1}.
//Ελέγχει αν η τιμή h έχει ήδη αντιστοιχιστεί σε προηγούμενο βήμα (caching μέσω unordered_map).
// Αν εμφανίζεται για πρώτη φορά, παράγει ντετερμινιστικά το bit {0, 1} μέσω ψευδοτυχαίου κατακερματισμού.
//func_idx Ο δείκτης της συνάρτησης προβολής (0 έως k_proj - 1).
//param h_val Η ακέραια τιμή του LSH bucket: floor((x * v + t) / w).

int HypercubeIndex::get_f_bit(int func_idx, int h_val) const {
    auto it = f_mappings[func_idx].find(h_val);
    if (it != f_mappings[func_idx].end()) {
        return it->second;// Επιστροφή αποθηκευμένης τιμής αν έχει ήδη υπολογιστεί
    }
    // Ψευδοτυχαία αντιστοίχιση f(h) -> {0, 1}
    int bit = (std::abs(h_val * 2654435761LL) % 2);
    f_mappings[func_idx][h_val] = bit;
    return bit;
}


//Προβολή ενός σημείου x στις συντεταγμένες μιας κορυφής του υπερκύβου.
//Υπολογίζει τις k_proj συναρτήσεις h_i(x), μετατρέπει κάθε μία σε bit μέσω του get_f_bit(),
// και συνθέτει τον ακέραιο αριθμό της κορυφής (bitstring μήκους k_proj).
//x Δείκτης στον πίνακα των συντεταγμένων του BoVW διανύσματος (float[256]).
// return int Ο αριθμός της κορυφής του υπερκύβου (από 0 έως 2^{k_proj} - 1).
int HypercubeIndex::project_to_cube(const float* x) const {
    int vertex = 0;
    for (int i = 0; i < k_proj; ++i) {
        float dot = 0.0f;
        // Εσωτερικό γινόμενο x * v_i
        for (int d = 0; d < dim; ++d) {
            dot += x[d] * functions[i].v[d];
        }
        // Ευκλείδειος LSH κάδος h_i(x)
        int h = static_cast<int>(std::floor((dot + functions[i].t) / w_width));
        // Χαρτογράφηση στο αντίστοιχο bit
        int bit = get_f_bit(i, h);
        // Τοποθέτηση του bit στην i-οστή θέση του bitstring της κορυφής
        vertex |= (bit << i);
    }
    return vertex;
}

//Κατασκευή της δομής ευρετηρίου του υπερκύβου (indexing).
// Προβάλλει κάθε διάνυσμα της βάσης δεδομένων σε μία κορυφή του υπερκύβου
// και καταχωρεί το index του στον αντίστοιχο κάδο (inverted bucket list).
void HypercubeIndex::build() {
    for (size_t idx = 0; idx < database.size(); ++idx) {
        int vertex = project_to_cube(database[idx].bow_vec.data());
        cube_vertices[vertex].push_back(static_cast<int>(idx));
    }
}
// Εκτέλεση προσεγγιστικής αναζήτησης για την εύρεση των top-k πλησιέστερων γειτόνων.
//. Προβάλλει το ερώτημα q στην κορυφή q_vertex.
//. Εξερευνά κορυφές κατά αύξουσα απόσταση Hamming (dist_hamm = 0, 1, 2, ...).
//. Σταματά μόλις ελεγχθούν probes_checked == max_probes κορυφές ή points_checked == max_M σημεία.
//. Διατηρεί τα top-k κοντινότερα σημεία χρησιμοποιώντας Max-Heap.
//  query_vec Το κανονικοποιημένο BoVW διάνυσμα του ερωτήματος.
//  top_k Ο αριθμός των πλησιέστερων γειτόνων προς ανάκτηση (προεπιλογή: 10).
// return std::vector<std::string> Λίστα με τα αναγνωριστικά (IDs) των top-k εικόνων, ταξινομημένα σε αύξουσα σειρά απόστασης.
//


std::vector<std::string> HypercubeIndex::search(const std::vector<float>& query_vec, int top_k) const {
    // Προβολή του query στον υπερκύβο
    int q_vertex = project_to_cube(query_vec.data());
    const float* q = query_vec.data();

    int points_checked = 0; // Μετρητής εξετασθέντων διανυσμάτων (όριο: max_M)
    int probes_checked = 0; // Μετρητής εξετασθεισών κορυφών (όριο: max_probes)

    // Max-heap για τη διατήρηση των k πλησιέστερων στοιχείων (απόσταση, id)
    typedef std::pair<float, std::string> Pair;
    std::priority_queue<Pair> pq;

    // Εξερεύνηση κορυφών με κριτήριο την απόσταση Hamming: από dist_hamm = 0 έως k_proj
    for (int dist_hamm = 0; dist_hamm <= k_proj && probes_checked < max_probes && points_checked < max_M; ++dist_hamm) {
        
        // Έλεγχος όλων των πιθανών κορυφών v του υπερκύβου (2^{k_proj})
        for (int v = 0; v < (1 << k_proj); ++v) {
            // Υπολογισμός απόστασης Hamming: μέτρηση των άσσων στο XOR (v ^ q_vertex)
            int diff = v ^ q_vertex;
            int bit_count = 0;
            for (int b = diff; b > 0; b >>= 1) {
                if (b & 1) bit_count++;
            }
            // Αν η κορυφή v βρίσκεται στην τρέχουσα απόσταση Hamming
            if (bit_count == dist_hamm) {
                auto it = cube_vertices.find(v);

                // Αν η κορυφή περιέχει αποθηκευμένα σημεία
                if (it != cube_vertices.end()) {
                    probes_checked++;// Αύξηση του πλήθους ελεγμένων κορυφών

                    // Εξέταση των υποψήφιων διανυσμάτων που ανήκουν στην κορυφή v
                    for (int idx : it->second) {
                        const auto& item = database[idx];
                        float dist = 0.0f;
                        
                        // Υπολογισμός πραγματικής Ευκλείδειας απόστασης L2
                        for (int d = 0; d < dim; ++d) {
                            float diff_val = q[d] - item.bow_vec[d];
                            dist += diff_val * diff_val;
                        }
                        dist = std::sqrt(dist);

                        // Ενημέρωση του max-heap
                        pq.push({dist, item.id});
                        if (pq.size() > static_cast<size_t>(top_k)) {
                            pq.pop();
                        }

                        points_checked++;// Αύξηση του πλήθους ελεγμένων σημείων

                        // Έλεγχος υπέρβασης μέγιστου ορίου υποψηφίων M
                        if (points_checked >= max_M) break;
                    }
                }// Έλεγχος κριτηρίων πρώιμου τερματισμού (probes ή M)
                if (probes_checked >= max_probes || points_checked >= max_M) break;
            }
        }
    }

    // Εξαγωγή αποτελεσμάτων από το heap σε αύξουσα σειρά απόστασης
    std::vector<std::string> results(pq.size());
    for (int i = static_cast<int>(pq.size()) - 1; i >= 0; --i) {
        results[i] = pq.top().second;
        pq.pop();
    }
    return results;
}