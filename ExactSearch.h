#ifndef EXACT_SEARCH_H
#define EXACT_SEARCH_H

#include "features.h"
#include <vector>
#include <string>
#include <queue>
#include <cmath>

class ExactSearch {
private:
    const std::vector<ImageRecord>& database;

public:
    ExactSearch(const std::vector<ImageRecord>& db) : database(db) {}

    std::vector<std::string> search(const std::vector<float>& query_vec, int top_k = 10) {
        // Max-heap για τη διατήρηση των top_k μικρότερων αποστάσεων
        typedef std::pair<float, std::string> Pair;
        std::priority_queue<Pair> pq;

        int d_len = static_cast<int>(query_vec.size());

        for (const auto& item : database) {
            float dist = 0.0f;
            for (int i = 0; i < d_len; ++i) {
                float diff = query_vec[i] - item.bow_vec[i];
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
};

#endif // EXACT_SEARCH_H
