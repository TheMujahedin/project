#include "ivfflat.h"
#include <random>
#include <cmath>
#include <queue>
#include <algorithm>
#include <limits>

static float dist_sq(const float* a, const float* b, int d) {
    float sum = 0.0f;
    for (int i = 0; i < d; ++i) {
        float diff = a[i] - b[i];
        sum += diff * diff;
    }
    return sum;
}

IVFFlatIndex::IVFFlatIndex(const std::vector<ImageRecord>& db, int kclusters, int nprobe, int dimension)
    : k_clusters(kclusters), n_probe(nprobe), dim(dimension), database(db) {}

void IVFFlatIndex::train_and_build(int max_iters, unsigned int seed) {
    int N = static_cast<int>(database.size());
    if (k_clusters > N) k_clusters = std::max(1, N);

    centroids.assign(k_clusters, std::vector<float>(dim, 0.0f));
    inverted_lists.assign(k_clusters, std::vector<int>());

    // Αρχικοποίηση κέντρων k-means
    std::mt19937 rng(seed);
    std::vector<int> perm(N);
    for (int i = 0; i < N; ++i) perm[i] = i;
    std::shuffle(perm.begin(), perm.end(), rng);

    for (int k = 0; k < k_clusters; ++k) {
        centroids[k] = database[perm[k]].bow_vec;
    }

    std::vector<int> assignments(N, 0);

    // Εκπαίδευση συστάδων
    for (int iter = 0; iter < max_iters; ++iter) {
        bool changed = false;

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

        if (!changed && iter > 0) break;

        std::vector<std::vector<float>> new_centroids(k_clusters, std::vector<float>(dim, 0.0f));
        std::vector<int> counts(k_clusters, 0);

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
    for (int i = 0; i < N; ++i) {
        inverted_lists[assignments[i]].push_back(i);
    }
}

std::vector<std::string> IVFFlatIndex::search(const std::vector<float>& query_vec, int top_k) const {
    const float* q = query_vec.data();

    // Εύρεση των n_probe πλησιέστερων συστάδων
    std::vector<std::pair<float, int>> cluster_dists(k_clusters);
    for (int k = 0; k < k_clusters; ++k) {
        cluster_dists[k] = {dist_sq(q, centroids[k].data(), dim), k};
    }
    std::sort(cluster_dists.begin(), cluster_dists.end());

    int probes = std::min(n_probe, k_clusters);

    typedef std::pair<float, std::string> Pair;
    std::priority_queue<Pair> pq;

    for (int p = 0; p < probes; ++p) {
        int c_id = cluster_dists[p].second;
        for (int idx : inverted_lists[c_id]) {
            const auto& item = database[idx];
            float d = std::sqrt(dist_sq(q, item.bow_vec.data(), dim));

            pq.push({d, item.id});
            if (pq.size() > static_cast<size_t>(top_k)) {
                pq.pop();
            }
        }
    }

    std::vector<std::string> results(pq.size());
    for (int i = static_cast<int>(pq.size()) - 1; i >= 0; --i) {
        results[i] = pq.top().second;
        pq.pop();
    }
    return results;
}