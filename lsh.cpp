// Υλοποίηση του αλγορίθμου Locality Sensitive Hashing (LSH) για Ευκλείδεια απόσταση ($L_2$).
//
//Ο αλγόριθμος LSH χρησιμοποιεί τυχαίες προβολές $p$-stable (Gaussian) κατανομών για να διαμερίσει
//τον συνεχή χώρο των διανυσμάτων BoVW σε κάδους κατακερματισμού (buckets).
//
//Βασικά στάδια λειτουργίας:
//1. Στοιχειακός κατακερματισμός: $h(x) = \lfloor (x \cdot v + t) / w \rfloor$, με $v \sim \mathcal{N}(0, I)$ και $t \sim \mathcal{U}[0, w)$.
//2. Ενίσχυση (Amplification): Συνδυασμός $k$ συναρτήσεων $h_i$ μέσω τυχαίων ακεραίων βαρών $r_i$:
//   $g(x) = \left( \sum_{i=1}^k r_i h_i(x) \pmod{M_{prime}} \right) \pmod{\text{TableSize}}$.
//3. Πολλαπλοί πίνακες: Χρήση $L$ ανεξάρτητων πινάκων για μεγιστοποίηση της πιθανότητας σύγκρουσης των πραγματικών γειτόνων (Recall).
//4. Φιλτράρισμα υποψηφίων: Εξέταση μόνο των διανυσμάτων που συγκρούονται στους κάδους του ερωτήματος και ταξινόμηση μέσω Max-Heap.
//

#include "lsh.h"
#include <random>
#include <cmath>
#include <queue>
#include <unordered_set>
#include <algorithm>

/// Μεγάλος πρώτος αριθμός ($2^{31} - 1$ προσέγγιση / standard modulo) για την αποφυγή υπερχείλισης και ομοιόμορφη κατανομή hash keys
static const int PRIME_MOD = 1000000007;

//
// Κατασκευαστής της δομής ευρετηρίου LSH.
//
//Αρχικοποιεί τους $L$ πίνακες, δεσμεύει μνήμη για τις $k$ συναρτήσεις προβολής ανά πίνακα
//και παράγει τυχαία διανύσματα $v \sim \mathcal{N}(0, 1)$, μετατοπίσεις $t \sim \mathcal{U}[0, w)$
//και ακέραια βάρη $r_i \in [1, 100000]$.
//
// db Αναφορά στη βάση δεδομένων εικόνων (HPatches + distractors).
// k Ο αριθμός των στοιχειωδών συναρτήσεων ανά πίνακα κατακερματισμού.
// L Ο αριθμός των ανεξάρτητων πινάκων κατακερματισμού (hash tables).
// w Το εύρος του παραθύρου προβολής (slot window width).
// dimension Η διάσταση των διανυσμάτων (d = 256 για BoVW).
// seed Ο σπόρος τυχαιότητας για επαναληψιμότητα των πειραμάτων.
//
LSHIndex::LSHIndex(const std::vector<ImageRecord>& db, int k, int L, double w, int dimension, unsigned int seed)
    : k_hash(k), L_tables(L), w_width(w), dim(dimension), database(db) {
    
    std::mt19937 rng(seed);
    std::normal_distribution<float> norm_dist(0.0f, 1.0f);
    std::uniform_real_distribution<float> uniform_t(0.0f, static_cast<float>(w));
    std::uniform_int_distribution<int> rand_r(1, 100000);

    // Δυναμικός καθορισμός μεγέθους πίνακα κατακερματισμού ανάλογα με το μέγεθος της βάσης
    table_size = std::max(16, static_cast<int>(database.size()) / 8);
    tables.resize(L_tables);
    hash_functions.resize(L_tables, std::vector<LSHFunction>(k_hash));
    r_weights.resize(L_tables, std::vector<int>(k_hash));

    // Δημιουργία των παραμέτρων προβολής για κάθε πίνακα l και κάθε συνάρτηση i
    for (int l = 0; l < L_tables; ++l) {
        for (int i = 0; i < k_hash; ++i) {
            hash_functions[l][i].v.resize(dim);
            // Τυχαία προβολή από κανονική κατανομή N(0, 1)
            for (int d = 0; d < dim; ++d) {
                hash_functions[l][i].v[d] = norm_dist(rng);
            }
            // Ομοιόμορφη μετατόπιση t ~ U[0, w)
            hash_functions[l][i].t = uniform_t(rng);
            // Τυχαίος συντελεστής βαρών r_i
            r_weights[l][i] = rand_r(rng);
        }
    }
}

//
// Υπολογισμός της στοιχειώδους Ευκλείδειας συνάρτησης $h(x) = \lfloor (x \cdot v + t) / w \rfloor$.
//
// x Δείκτης στον πίνακα συντεταγμένων του διανύσματος εισόδου (διάστασης dim).
// fn Η δομή που περιέχει το διάνυσμα προβολής $v$ και τη μετατόπιση $t$.
//return int Ο ακέραιος δείκτης του κάδου κατά μήκος της ευθείας προβολής.
//
int LSHIndex::compute_h(const float* x, const LSHFunction& fn) const {
    float dot = 0.0f;
    // Εσωτερικό γινόμενο x * v
    for (int d = 0; d < dim; ++d) {
        dot += x[d] * fn.v[d];
    }
    // Εφαρμογή μετατόπισης και διαμέριση σε διαστήματα εύρους w
    return static_cast<int>(std::floor((dot + fn.t) / w_width));
}

//
// Υπολογισμός της σύνθετης συνάρτησης κατακερματισμού $g(x)$ για τον πίνακα table_idx.
//
//Συνδυάζει τις $k$ ανεξάρτητες συναρτήσεις $h_i(x)$ με τα αντίστοιχα βάρη $r_i$ και εφαρμόζει
//διπλό modulo για ασφαλή αντιστοίχιση σε μη-αρνητικό ακέραιο εντός του εύρους $[0, \text{table\_size}-1]$.
//
// x Δείκτης στις συντεταγμένες του διανύσματος.
// table_idx Ο δείκτης του πίνακα κατακερματισμού (0 έως L_tables - 1).
//return int Το τελικό κλειδί κάδου (bucket index).
//
int LSHIndex::compute_g(const float* x, int table_idx) const {
    long long sum = 0;
    for (int i = 0; i < k_hash; ++i) {
        int h_val = compute_h(x, hash_functions[table_idx][i]);
        // Γραμμικός συνδυασμός με τα τυχαία βάρη modulo PRIME_MOD
        sum = (sum + (1LL * r_weights[table_idx][i] * h_val) % PRIME_MOD) % PRIME_MOD;
    }
    // Τελικό modulo με το μέγεθος του πίνακα κατακερματισμού (με εξασφάλιση θετικού αποτελέσματος)
    int hash_val = static_cast<int>((sum % table_size + table_size) % table_size);
    return hash_val;
}

//
// Κατασκευή του ευρετηρίου LSH (Indexing).
//
//Εισάγει όλα τα διανύσματα της βάσης δεδομένων και στους $L$ πίνακες κατακερματισμού.
//Κάθε διάνυσμα καταχωρείται στον αντίστοιχο κάδο $g_l(x)$ του πίνακα $l$.
//
void LSHIndex::build() {
    for (size_t idx = 0; idx < database.size(); ++idx) {
        const float* x = database[idx].bow_vec.data();
        for (int l = 0; l < L_tables; ++l) {
            int g_val = compute_g(x, l);
            tables[l][g_val].push_back(static_cast<int>(idx));
        }
    }
}

//
// Εκτέλεση προσεγγιστικής αναζήτησης για την εύρεση των top-k πλησιέστερων εικόνων.
//
//1. Εντοπίζει τους κάδους $g_l(q)$ στους οποίους κατακερματίζεται το ερώτημα $q$ στους $L$ πίνακες.
//2. Συλλέγει τους υποψηφίους σε ένα `std::unordered_set` για την αυτόματη εξάλειψη των διπλοτύπων.
//3. Υπολογίζει την πραγματική Ευκλείδεια απόσταση $L_2$ μόνο για το υποσύνολο των υποψηφίων.
//4. Διατηρεί τα top-k πλησιέστερα σημεία με χρήση Max-Heap (`std::priority_queue`).
//
// query_vec Το κανονικοποιημένο BoVW διάνυσμα του ερωτήματος.
// top_k Ο αριθμός των πλησιέστερων γειτόνων προς ανάκτηση (προεπιλογή: 10).
//return std::vector<std::string> Λίστα με τα IDs των εικόνων σε αύξουσα σειρά απόστασης.
//
std::vector<std::string> LSHIndex::search(const std::vector<float>& query_vec, int top_k) const {
    std::unordered_set<int> candidates;
    const float* q = query_vec.data();

    // Συλλογή μοναδικών υποψήφιων σημείων από τους αντίστοιχους κάδους των L πινάκων
    for (int l = 0; l < L_tables; ++l) {
        int g_val = compute_g(q, l);
        auto it = tables[l].find(g_val);
        if (it != tables[l].end()) {
            for (int idx : it->second) {
                candidates.insert(idx); // Αποφυγή διπλοεξέτασης σημείων που συνέπεσαν σε πολλαπλούς πίνακες
            }
        }
    }

    // Αξιολόγηση των υποψηφίων βάσει της πραγματικής Ευκλείδειας απόστασης
    typedef std::pair<float, std::string> Pair;
    std::priority_queue<Pair> pq; // Max-heap για διατήρηση των k ελάχιστων αποστάσεων

    for (int idx : candidates) {
        const auto& item = database[idx];
        float dist = 0.0f;
        
        // Υπολογισμός τετραγωνικής Ευκλείδειας απόστασης L2
        for (int d = 0; d < dim; ++d) {
            float diff = q[d] - item.bow_vec[d];
            dist += diff * diff;
        }
        dist = std::sqrt(dist);

        // Εισαγωγή στο heap και διατήρηση μεγέθους το πολύ ίσο με top_k
        pq.push({dist, item.id});
        if (pq.size() > static_cast<size_t>(top_k)) {
            pq.pop(); // Αφαίρεση του υποψηφίου με τη μεγαλύτερη απόσταση
        }
    }

    // Εξαγωγή αποτελεσμάτων σε αύξουσα σειρά απόστασης
    std::vector<std::string> results(pq.size());
    for (int i = static_cast<int>(pq.size()) - 1; i >= 0; --i) {
        results[i] = pq.top().second;
        pq.pop();
    }
    return results;
}