#ifndef METRICS_H
#define METRICS_H

#include <vector>
#include <string>
#include <algorithm>

// Ελέγχει αν το ανακτηθέν id ανήκει στην ίδια ακολουθία με το query (θεωρούνται 5 σχετικά)
inline bool is_relevant(const std::string& query_seq, const std::string& result_id) {
    std::string prefix = query_seq + "/";
    return (result_id.rfind(prefix, 0) == 0);
}

// Υπολογισμός Recall@n (n = 5 ή 10)
inline double compute_recall_at_n(const std::string& query_seq, const std::vector<std::string>& top_results, int n) {
    int relevant_found = 0;
    int limit = std::min(n, static_cast<int>(top_results.size()));
    for (int i = 0; i < limit; ++i) {
        if (is_relevant(query_seq, top_results[i])) {
            relevant_found++;
        }
    }
    return static_cast<double>(relevant_found) / 5.0; // Υπάρχουν ακριβώς 5 σχετικά
}

// Υπολογισμός AP@10
inline double compute_ap_at_10(const std::string& query_seq, const std::vector<std::string>& top_results) {
    double sum_precision = 0.0;
    int relevant_count = 0;
    int limit = std::min(10, static_cast<int>(top_results.size()));

    for (int r = 1; r <= limit; ++r) {
        if (is_relevant(query_seq, top_results[r - 1])) {
            relevant_count++;
            sum_precision += static_cast<double>(relevant_count) / static_cast<double>(r);
        }
    }
    return sum_precision / 5.0;
}

#endif // METRICS_H
