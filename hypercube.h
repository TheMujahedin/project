#ifndef HYPERCUBE_H
#define HYPERCUBE_H

#include "features.h"
#include <vector>
#include <unordered_map>
#include <string>
#include <map>

class HypercubeIndex {
private:
    int k_proj;
    double w_width;
    int max_M;
    int max_probes;
    int dim;

    struct HashFunc {
        std::vector<float> v;
        float t;
    };

    std::vector<HashFunc> functions;
    // Χαρτογράφηση f_i: h_i -> {0, 1}
    mutable std::vector<std::map<int, int>> f_mappings;

    std::unordered_map<int, std::vector<int>> cube_vertices;
    const std::vector<ImageRecord>& database;

    int get_f_bit(int func_idx, int h_val) const;
    int project_to_cube(const float* x) const;

public:
    HypercubeIndex(const std::vector<ImageRecord>& db, int kproj, double w, int M, int probes, int dimension, unsigned int seed);
    void build();
    std::vector<std::string> search(const std::vector<float>& query_vec, int top_k = 10) const;
};

#endif // HYPERCUBE_H