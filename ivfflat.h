#ifndef IVFFLAT_H
#define IVFFLAT_H

#include "features.h"
#include <vector>
#include <string>

class IVFFlatIndex {
private:
    int k_clusters;
    int n_probe;
    int dim;
    const std::vector<ImageRecord>& database;

    std::vector<std::vector<float>> centroids;     // Κέντρα συστάδων
    std::vector<std::vector<int>> inverted_lists;  // Inverted list: cluster_id -> δείκτες στη βάση

public:
    IVFFlatIndex(const std::vector<ImageRecord>& db, int kclusters, int nprobe, int dimension);
    void train_and_build(int max_iters, unsigned int seed);
    std::vector<std::string> search(const std::vector<float>& query_vec, int top_k = 10) const;
};

#endif // IVFFLAT_H