#ifndef IVFPQ_H
#define IVFPQ_H

#include "features.h"
#include <vector>
#include <string>
#include <cstdint>

class IVFPQIndex {
private:
    int k_clusters;
    int n_probe;
    int M_subvectors;
    int n_bits;
    int k_subclusters; // 2^nbits (συνήθως 256)
    int dim;
    int d_sub;
    const std::vector<ImageRecord>& database;

    std::vector<std::vector<float>> coarse_centroids; // Coarse quantizer (IVF)

    // Codebooks: M_subvectors πίνακες, καθένας με k_subclusters κέντρα διάστασης d_sub
    std::vector<std::vector<std::vector<float>>> pq_codebooks;

    struct EncodedVector {
        int original_idx;
        std::vector<uint8_t> codes; // M bytes
    };

    std::vector<std::vector<EncodedVector>> inverted_lists;

    void train_subspace_kmeans(int m, const std::vector<std::vector<float>>& residuals, int max_iters, unsigned int seed);

public:
    IVFPQIndex(const std::vector<ImageRecord>& db, int kclusters, int nprobe, int M, int nbits, int dimension);
    void train_and_build(int max_iters, unsigned int seed);
    std::vector<std::string> search(const std::vector<float>& query_vec, int top_k = 10) const;
};

#endif // IVFPQ_H