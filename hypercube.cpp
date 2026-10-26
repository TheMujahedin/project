#include "hypercube.h"
#include <random>
#include <cmath>
#include <queue>
#include <algorithm>

HypercubeIndex::HypercubeIndex(const std::vector<ImageRecord>& db, int kproj, double w, int M, int probes, int dimension, unsigned int seed)
    : k_proj(kproj), w_width(w), max_M(M), max_probes(probes), dim(dimension), database(db) {
    
    std::mt19937 rng(seed);
    std::normal_distribution<float> norm_dist(0.0f, 1.0f);
    std::uniform_real_distribution<float> uniform_t(0.0f, static_cast<float>(w));

    functions.resize(k_proj);
    f_mappings.resize(k_proj);

    for (int i = 0; i < k_proj; ++i) {
        functions[i].v.resize(dim);
        for (int d = 0; d < dim; ++d) {
            functions[i].v[d] = norm_dist(rng);
        }
        functions[i].t = uniform_t(rng);
    }
}

int HypercubeIndex::get_f_bit(int func_idx, int h_val) const {
    auto it = f_mappings[func_idx].find(h_val);
    if (it != f_mappings[func_idx].end()) {
        return it->second;
    }
    // Ψευδοτυχαία αντιστοίχιση f(h) -> {0, 1}
    int bit = (std::abs(h_val * 2654435761LL) % 2);
    f_mappings[func_idx][h_val] = bit;
    return bit;
}

int HypercubeIndex::project_to_cube(const float* x) const {
    int vertex = 0;
    for (int i = 0; i < k_proj; ++i) {
        float dot = 0.0f;
        for (int d = 0; d < dim; ++d) {
            dot += x[d] * functions[i].v[d];
        }
        int h = static_cast<int>(std::floor((dot + functions[i].t) / w_width));
        int bit = get_f_bit(i, h);
        vertex |= (bit << i);
    }
    return vertex;
}

void HypercubeIndex::build() {
    for (size_t idx = 0; idx < database.size(); ++idx) {
        int vertex = project_to_cube(database[idx].bow_vec.data());
        cube_vertices[vertex].push_back(static_cast<int>(idx));
    }
}

std::vector<std::string> HypercubeIndex::search(const std::vector<float>& query_vec, int top_k) const {
    int q_vertex = project_to_cube(query_vec.data());
    const float* q = query_vec.data();

    int points_checked = 0;
    int probes_checked = 0;

    typedef std::pair<float, std::string> Pair;
    std::priority_queue<Pair> pq;

    for (int dist_hamm = 0; dist_hamm <= k_proj && probes_checked < max_probes && points_checked < max_M; ++dist_hamm) {
        for (int v = 0; v < (1 << k_proj); ++v) {
            int diff = v ^ q_vertex;
            int bit_count = 0;
            for (int b = diff; b > 0; b >>= 1) {
                if (b & 1) bit_count++;
            }

            if (bit_count == dist_hamm) {
                auto it = cube_vertices.find(v);
                if (it != cube_vertices.end()) {
                    probes_checked++;
                    for (int idx : it->second) {
                        const auto& item = database[idx];
                        float dist = 0.0f;
                        for (int d = 0; d < dim; ++d) {
                            float diff_val = q[d] - item.bow_vec[d];
                            dist += diff_val * diff_val;
                        }
                        dist = std::sqrt(dist);

                        pq.push({dist, item.id});
                        if (pq.size() > static_cast<size_t>(top_k)) {
                            pq.pop();
                        }

                        points_checked++;
                        if (points_checked >= max_M) break;
                    }
                }
                if (probes_checked >= max_probes || points_checked >= max_M) break;
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