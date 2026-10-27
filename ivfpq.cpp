#include "ivfpq.h"
#include <random>
#include <cmath>
#include <queue>
#include <algorithm>
#include <limits>

static float dist_sq_sub(const float* a, const float* b, int d) {
    float sum = 0.0f;
    for (int i = 0; i < d; ++i) {
        float diff = a[i] - b[i];
        sum += diff * diff;
    }
    return sum;
}

IVFPQIndex::IVFPQIndex(const std::vector<ImageRecord>& db, int kclusters, int nprobe, int M, int nbits, int dimension)
    : k_clusters(kclusters), n_probe(nprobe), M_subvectors(M), n_bits(nbits), dim(dimension), database(db) {
    d_sub = dim / M_subvectors;
    k_subclusters = (1 << n_bits); // 2^nbits
}

void IVFPQIndex::train_subspace_kmeans(int m, const std::vector<std::vector<float>>& residuals, int max_iters, unsigned int seed) {
    int N = static_cast<int>(residuals.size());
    pq_codebooks[m].assign(k_subclusters, std::vector<float>(d_sub, 0.0f));

    std::mt19937 rng(seed + m * 31);
    std::vector<int> perm(N);
    for (int i = 0; i < N; ++i) perm[i] = i;
    std::shuffle(perm.begin(), perm.end(), rng);

    for (int k = 0; k < k_subclusters && k < N; ++k) {
        const float* sub = residuals[perm[k]].data() + m * d_sub;
        for (int d = 0; d < d_sub; ++d) {
            pq_codebooks[m][k][d] = sub[d];
        }
    }

    std::vector<int> assignments(N, 0);
    for (int iter = 0; iter < max_iters; ++iter) {
        bool changed = false;
        for (int i = 0; i < N; ++i) {
            const float* sub = residuals[i].data() + m * d_sub;
            float min_d = std::numeric_limits<float>::max();
            int best_k = 0;
            for (int k = 0; k < k_subclusters; ++k) {
                float d = dist_sq_sub(sub, pq_codebooks[m][k].data(), d_sub);
                if (d < min_d) {
                    min_d = d;
                    best_k = k;
                }
            }
            if (assignments[i] != best_k) {
                assignments[i] != best_k;
                assignments[i] = best_k;
                changed = true;
            }
        }
        if (!changed && iter > 0) break;

        std::vector<std::vector<float>> new_c(k_subclusters, std::vector<float>(d_sub, 0.0f));
        std::vector<int> counts(k_subclusters, 0);
        for (int i = 0; i < N; ++i) {
            int k = assignments[i];
            counts[k]++;
            const float* sub = residuals[i].data() + m * d_sub;
            for (int d = 0; d < d_sub; ++d) {
                new_c[k][d] += sub[d];
            }
        }
        for (int k = 0; k < k_subclusters; ++k) {
            if (counts[k] > 0) {
                for (int d = 0; d < d_sub; ++d) {
                    pq_codebooks[m][k][d] = new_c[k][d] / counts[k];
                }
            }
        }
    }
}

void IVFPQIndex::train_and_build(int max_iters, unsigned int seed) {
    int N = static_cast<int>(database.size());
    if (k_clusters > N) k_clusters = std::max(1, N);

    coarse_centroids.assign(k_clusters, std::vector<float>(dim, 0.0f));
    inverted_lists.assign(k_clusters, std::vector<EncodedVector>());
    pq_codebooks.resize(M_subvectors);

    // 1. Coarse Quantization (k-means στα πλήρη διανύσματα)
    std::mt19937 rng(seed);
    std::vector<int> perm(N);
    for (int i = 0; i < N; ++i) perm[i] = i;
    std::shuffle(perm.begin(), perm.end(), rng);

    for (int k = 0; k < k_clusters; ++k) {
        coarse_centroids[k] = database[perm[k]].bow_vec;
    }

    std::vector<int> coarse_assignments(N, 0);
    for (int iter = 0; iter < max_iters; ++iter) {
        bool changed = false;
        for (int i = 0; i < N; ++i) {
            float min_d = std::numeric_limits<float>::max();
            int best_k = 0;
            const float* vec = database[i].bow_vec.data();
            for (int k = 0; k < k_clusters; ++k) {
                float d = dist_sq_sub(vec, coarse_centroids[k].data(), dim);
                if (d < min_d) {
                    min_d = d;
                    best_k = k;
                }
            }
            if (coarse_assignments[i] != best_k) {
                coarse_assignments[i] != best_k;
                changed = true;
            }
        }
        if (!changed && iter > 0) break;

        std::vector<std::vector<float>> new_c(k_clusters, std::vector<float>(dim, 0.0f));
        std::vector<int> counts(k_clusters, 0);
        for (int i = 0; i < N; ++i) {
            int k = coarse_assignments[i];
            counts[k]++;
            const float* vec = database[i].bow_vec.data();
            for (int d = 0; d < dim; ++d) {
                new_c[k][d] += vec[d];
            }
        }
        for (int k = 0; k < k_clusters; ++k) {
            if (counts[k] > 0) {
                for (int d = 0; d < dim; ++d) {
                    coarse_centroids[k][d] = new_c[k][d] / counts[k];
                }
            }
        }
    }

    // 2. Υπολογισμός residuals: r(x) = x - centroid
    std::vector<std::vector<float>> residuals(N, std::vector<float>(dim));
    for (int i = 0; i < N; ++i) {
        int c_id = coarse_assignments[i];
        for (int d = 0; d < dim; ++d) {
            residuals[i][d] = database[i].bow_vec[d] - coarse_centroids[c_id][d];
        }
    }

    // 3. Εκπαίδευση Product Quantization ανά υποχώρο
    for (int m = 0; m < M_subvectors; ++m) {
        train_subspace_kmeans(m, residuals, max_iters, seed);
    }

    // 4. Κωδικοποίηση residuals και αποθήκευση στις Inverted Lists
    for (int i = 0; i < N; ++i) {
        int c_id = coarse_assignments[i];
        EncodedVector ev;
        ev.original_idx = i;
        ev.codes.resize(M_subvectors);

        for (int m = 0; m < M_subvectors; ++m) {
            const float* sub = residuals[i].data() + m * d_sub;
            float min_d = std::numeric_limits<float>::max();
            int best_code = 0;
            for (int k = 0; k < k_subclusters; ++k) {
                float d = dist_sq_sub(sub, pq_codebooks[m][k].data(), d_sub);
                if (d < min_d) {
                    min_d = d;
                    best_code = k;
                }
            }
            ev.codes[m] = static_cast<uint8_t>(best_code);
        }
        inverted_lists[c_id].push_back(ev);
    }
}

std::vector<std::string> IVFPQIndex::search(const std::vector<float>& query_vec, int top_k) const {
    const float* q = query_vec.data();

    std::vector<std::pair<float, int>> cluster_dists(k_clusters);
    for (int k = 0; k < k_clusters; ++k) {
        cluster_dists[k] = {dist_sq_sub(q, coarse_centroids[k].data(), dim), k};
    }
    std::sort(cluster_dists.begin(), cluster_dists.end());

    int probes = std::min(n_probe, k_clusters);

    typedef std::pair<float, std::string> Pair;
    std::priority_queue<Pair> pq;

    for (int p = 0; p < probes; ++p) {
        int c_id = cluster_dists[p].second;
        const float* c_vec = coarse_centroids[c_id].data();

        std::vector<float> q_residual(dim);
        for (int d = 0; d < dim; ++d) {
            q_residual[d] = q[d] - c_vec[d];
        }

        std::vector<std::vector<float>> dist_table(M_subvectors, std::vector<float>(k_subclusters, 0.0f));
        for (int m = 0; m < M_subvectors; ++m) {
            const float* q_sub = q_residual.data() + m * d_sub;
            for (int code = 0; code < k_subclusters; ++code) {
                dist_table[m][code] = dist_sq_sub(q_sub, pq_codebooks[m][code].data(), d_sub);
            }
        }

        for (const auto& ev : inverted_lists[c_id]) {
            float approx_dist = 0.0f;
            for (int m = 0; m < M_subvectors; ++m) {
                approx_dist += dist_table[m][ev.codes[m]];
            }
            approx_dist = std::sqrt(approx_dist);

            pq.push({approx_dist, database[ev.original_idx].id});
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