
#include "lsh.h"
#include <random>
#include <cmath>
#include <queue>
#include <unordered_set>
#include <algorithm>

static const int PRIME_MOD = 1000000007;

LSHIndex::LSHIndex(const std::vector<ImageRecord>& db, int k, int L, double w, int dimension, unsigned int seed)
    : k_hash(k), L_tables(L), w_width(w), dim(dimension), database(db) {
    
    std::mt19937 rng(seed);
    std::normal_distribution<float> norm_dist(0.0f, 1.0f);
    std::uniform_real_distribution<float> uniform_t(0.0f, static_cast<float>(w));
    std::uniform_int_distribution<int> rand_r(1, 100000);

    table_size = std::max(16, static_cast<int>(database.size()) / 8);
    tables.resize(L_tables);
    hash_functions.resize(L_tables, std::vector<LSHFunction>(k_hash));
    r_weights.resize(L_tables, std::vector<int>(k_hash));

    for (int l = 0; l < L_tables; ++l) {
        for (int i = 0; i < k_hash; ++i) {
            hash_functions[l][i].v.resize(dim);
            for (int d = 0; d < dim; ++d) {
                hash_functions[l][i].v[d] = norm_dist(rng);
            }
            hash_functions[l][i].t = uniform_t(rng);
            r_weights[l][i] = rand_r(rng);
        }
    }
}

int LSHIndex::compute_h(const float* x, const LSHFunction& fn) const {
    float dot = 0.0f;
    for (int d = 0; d < dim; ++d) {
        dot += x[d] * fn.v[d];
    }
    return static_cast<int>(std::floor((dot + fn.t) / w_width));
}

int LSHIndex::compute_g(const float* x, int table_idx) const {
    long long sum = 0;
    for (int i = 0; i < k_hash; ++i) {
        int h_val = compute_h(x, hash_functions[table_idx][i]);
        sum = (sum + (1LL * r_weights[table_idx][i] * h_val) % PRIME_MOD) % PRIME_MOD;
    }
    int hash_val = static_cast<int>((sum % table_size + table_size) % table_size);
    return hash_val;
}

void LSHIndex::build() {
    for (size_t idx = 0; idx < database.size(); ++idx) {
        const float* x = database[idx].bow_vec.data();
        for (int l = 0; l < L_tables; ++l) {
            int g_val = compute_g(x, l);
            tables[l][g_val].push_back(static_cast<int>(idx));
        }
    }
}

std::vector<std::string> LSHIndex::search(const std::vector<float>& query_vec, int top_k) const {
    std::unordered_set<int> candidates;
    const float* q = query_vec.data();

    // Συλλογή υποψήφιων σημείων από όλους τους πίνακες L
    for (int l = 0; l < L_tables; ++l) {
        int g_val = compute_g(q, l);
        auto it = tables[l].find(g_val);
        if (it != tables[l].end()) {
            for (int idx : it->second) {
                candidates.insert(idx);
            }
        }
    }

    // Ταξινόμηση υποψηφίων βάσει Ευκλείδειας απόστασης
    typedef std::pair<float, std::string> Pair;
    std::priority_queue<Pair> pq;

    for (int idx : candidates) {
        const auto& item = database[idx];
        float dist = 0.0f;
        for (int d = 0; d < dim; ++d) {
            float diff = q[d] - item.bow_vec[d];
            dist += diff * diff;
        }
        dist = std::sqrt(dist);

        pq.push({dist, item.id});
        if (pq.size() > static_cast<size_t>(top_k)) {
            pq.pop();
        }
    }

    std::vector<std::string> results(pq.size());
    for (int i = static_cast<int>(pq.size()) - 1; i >= 0; --i) {
        results[i] = pq.top().second;
        pq.pop();
    }
    return results;
}
