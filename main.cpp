#include <iostream>
#include <fstream>
#include <sstream>
#include <vector>
#include <string>
#include <chrono>
#include <algorithm>
#include <filesystem>
#include <cmath>

namespace fs = std::filesystem;

#include "features.h"
#include "metrics.h"
#include "ExactSearch.h"
#include "lsh.h"
#include "hypercube.h"
#include "ivfflat.h"
#include "ivfpq.h"

struct SplitInfo {
    std::string seq_name;
    std::string set_type; // "train", "validation", "test"
};

std::vector<SplitInfo> read_split_file(const std::string& path) {
    std::vector<SplitInfo> splits;
    std::ifstream infile(path);
    std::string line;
    while (std::getline(infile, line)) {
        if (line.empty()) continue;
        std::stringstream ss(line);
        std::string seq, type;
        if (ss >> seq >> type) {
            if (!type.empty() && type.back() == '.') type.pop_back();
            splits.push_back({seq, type});
        }
    }
    return splits;
}

// Βοηθητικές συναρτήσεις για Caching των BoVW διανυσμάτων του MIRFlickr
void load_mir_cache(const std::string& cache_file, std::vector<ImageRecord>& cached_mir) {
    std::ifstream in(cache_file, std::ios::binary);
    if (!in.is_open()) return;

    size_t count = 0;
    in.read(reinterpret_cast<char*>(&count), sizeof(count));
    cached_mir.resize(count);

    for (size_t i = 0; i < count; ++i) {
        size_t id_len = 0;
        in.read(reinterpret_cast<char*>(&id_len), sizeof(id_len));
        cached_mir[i].id.resize(id_len);
        in.read(&cached_mir[i].id[0], id_len);

        cached_mir[i].sequence_id = "";
        cached_mir[i].img_index = 0;
        cached_mir[i].is_query = false;

        size_t vec_len = 0;
        in.read(reinterpret_cast<char*>(&vec_len), sizeof(vec_len));
        cached_mir[i].bow_vec.resize(vec_len);
        in.read(reinterpret_cast<char*>(cached_mir[i].bow_vec.data()), vec_len * sizeof(float));
    }
}

void save_mir_cache(const std::string& cache_file, const std::vector<ImageRecord>& cached_mir) {
    std::ofstream out(cache_file, std::ios::binary);
    if (!out.is_open()) return;

    size_t count = cached_mir.size();
    out.write(reinterpret_cast<const char*>(&count), sizeof(count));

    for (size_t i = 0; i < count; ++i) {
        size_t id_len = cached_mir[i].id.size();
        out.write(reinterpret_cast<const char*>(&id_len), sizeof(id_len));
        out.write(cached_mir[i].id.data(), id_len);

        size_t vec_len = cached_mir[i].bow_vec.size();
        out.write(reinterpret_cast<const char*>(&vec_len), sizeof(vec_len));
        out.write(reinterpret_cast<const char*>(cached_mir[i].bow_vec.data()), vec_len * sizeof(float));
    }
}

int main(int argc, char* argv[]) {
    // Κοινές παράμετροι
    std::string hp_dir, mir_dir, split_file, set_type = "validation", output_file = "output.txt";
    int vocab_K = 256;
    int S_max = 500;
    int D_mir = 1000;
    unsigned int seed = 42;

    // Μέθοδος
    std::string method = "exact";

    // Παράμετροι μεθόδων
    int lsh_k = 4, lsh_L = 5;
    double lsh_w = 0.05;

    int cube_kproj = 10, cube_M = 500, cube_probes = 10;
    double cube_w = 0.05;

    int ivf_kclusters = -1, ivf_nprobe = 5;
    int pq_M = 16, pq_nbits = 8;

    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];
        if (arg == "-hp" && i + 1 < argc) hp_dir = argv[++i];
        else if (arg == "-mir" && i + 1 < argc) mir_dir = argv[++i];
        else if (arg == "-split" && i + 1 < argc) split_file = argv[++i];
        else if (arg == "-vocab" && i + 1 < argc) vocab_K = std::stoi(argv[++i]);
        else if (arg == "-S" && i + 1 < argc) S_max = std::stoi(argv[++i]);
        else if (arg == "-D" && i + 1 < argc) D_mir = std::stoi(argv[++i]);
        else if (arg == "-set" && i + 1 < argc) set_type = argv[++i];
        else if (arg == "-o" && i + 1 < argc) output_file = argv[++i];
        else if (arg == "-seed" && i + 1 < argc) seed = static_cast<unsigned int>(std::stoul(argv[++i]));
        else if (arg == "-exact") method = "exact";
        else if (arg == "-lsh") method = "lsh";
        else if (arg == "-hypercube") method = "hypercube";
        else if (arg == "-ivfflat") method = "ivfflat";
        else if (arg == "-ivfpq") method = "ivfpq";
        else if (arg == "-k" && i + 1 < argc) lsh_k = std::stoi(argv[++i]);
        else if (arg == "-L" && i + 1 < argc) lsh_L = std::stoi(argv[++i]);
        else if (arg == "-w" && i + 1 < argc) {
            double w_val = std::stod(argv[++i]);
            lsh_w = w_val;
            cube_w = w_val;
        }
        else if (arg == "-kproj" && i + 1 < argc) cube_kproj = std::stoi(argv[++i]);
        else if (arg == "-M" && i + 1 < argc) {
            int m_val = std::stoi(argv[++i]);
            cube_M = m_val;
            pq_M = m_val;
        }
        else if (arg == "-probes" && i + 1 < argc) cube_probes = std::stoi(argv[++i]);
        else if (arg == "-kclusters" && i + 1 < argc) ivf_kclusters = std::stoi(argv[++i]);
        else if (arg == "-nprobe" && i + 1 < argc) ivf_nprobe = std::stoi(argv[++i]);
        else if (arg == "-nbits" && i + 1 < argc) pq_nbits = std::stoi(argv[++i]);
    }

    std::cout << "[1/5] Φόρτωση split αρχείου..." << std::endl;
    auto splits = read_split_file(split_file);

    // 1. Έλεγχος αν υπάρχει αποθηκευμένο λεξικό
    VisualVocabulary vocab(vocab_K);
    std::string vocab_file = "vocab_" + std::to_string(vocab_K) + ".bin";

    if (vocab.load(vocab_file)) {
        std::cout << "[2/5] Βρέθηκε αποθηκευμένο λεξικό (" << vocab_file << "). Φόρτωση..." << std::endl;
    } else {
        std::cout << "[2/5] Δεν βρέθηκε αποθηκευμένο λεξικό. Εξαγωγή SIFT descriptors για εκπαίδευση..." << std::endl;
        std::vector<cv::Mat> train_descriptors;
        for (const auto& sp : splits) {
            if (sp.set_type == "train") {
                for (int img_idx = 1; img_idx <= 6; ++img_idx) {
                    std::string path = hp_dir + "/" + sp.seq_name + "/" + std::to_string(img_idx) + ".ppm";
                    cv::Mat desc = extract_sift_descriptors(path);
                    if (!desc.empty()) {
                        if (desc.rows > S_max) {
                            desc = desc.rowRange(0, S_max).clone();
                        }
                        train_descriptors.push_back(desc);
                    }
                }
            }
        }
        vocab.train(train_descriptors, 20, seed);
        vocab.save(vocab_file);
        std::cout << "Το λεξικό αποθηκεύτηκε επιτυχώς στο " << vocab_file << std::endl;
        train_descriptors.clear();
    }

    // 2. Κατασκευή βάσης δεδομένων και query vectors
    std::cout << "[3/5] Κατασκευή διανυσμάτων βάσης και queries..." << std::endl;
    std::vector<ImageRecord> queries;
    std::vector<ImageRecord> database;

    for (const auto& sp : splits) {
        if (sp.set_type == set_type) {
            for (int img_idx = 1; img_idx <= 6; ++img_idx) {
                std::string path = hp_dir + "/" + sp.seq_name + "/" + std::to_string(img_idx) + ".ppm";
                cv::Mat desc = extract_sift_descriptors(path);
                auto vec = vocab.compute_bow_vector(desc);

                ImageRecord rec;
                rec.sequence_id = sp.seq_name;
                rec.img_index = img_idx;
                rec.id = sp.seq_name + "/" + std::to_string(img_idx);
                rec.bow_vec = vec;

                if (img_idx == 1) {
                    rec.is_query = true;
                    queries.push_back(rec);
                } else {
                    rec.is_query = false;
                    database.push_back(rec);
                }
            }
        }
    }

    // Caching και φόρτωση για MIRFlickr
    std::vector<ImageRecord> cached_mir;
    std::string mir_cache_file = "mir_cache_" + std::to_string(vocab_K) + ".bin";
    load_mir_cache(mir_cache_file, cached_mir);

    int cur_mir_id = cached_mir.size() + 1;
    bool new_features_computed = false;

    while (static_cast<int>(cached_mir.size()) < D_mir && cur_mir_id <= 25000) {
        std::string filename = "im" + std::to_string(cur_mir_id) + ".jpg";
        std::string path = mir_dir + "/" + filename;
        if (fs::exists(path)) {
            cv::Mat desc = extract_sift_descriptors(path);
            if (!desc.empty()) {
                ImageRecord rec;
                rec.sequence_id = "";
                rec.img_index = 0;
                rec.id = filename;
                rec.is_query = false;
                rec.bow_vec = vocab.compute_bow_vector(desc);
                cached_mir.push_back(rec);
                new_features_computed = true;

                if (cached_mir.size() % 1000 == 0) {
                    std::cout << "Υπολογίστηκαν " << cached_mir.size() << " / " << D_mir << " εικόνες MIR..." << std::endl;
                }
            }
        }
        cur_mir_id++;
    }

    if (new_features_computed) {
        save_mir_cache(mir_cache_file, cached_mir);
    }

    // Προσθήκη των πρώτων D_mir διανυσμάτων στη βάση
    for (int i = 0; i < D_mir && i < static_cast<int>(cached_mir.size()); ++i) {
        database.push_back(cached_mir[i]);
    }

    std::cout << "Μέγεθος βάσης: " << database.size() << " διανύσματα. Αριθμός Queries: " << queries.size() << std::endl;

    if (ivf_kclusters <= 0) {
        ivf_kclusters = static_cast<int>(std::ceil(std::sqrt(database.size())));
    }

    // 3. Δημιουργία δομών ευρετηρίου
    std::cout << "[4/5] Κατασκευή δομής ευρετηρίου (" << method << ")..." << std::endl;
    ExactSearch exact_idx(database);
    LSHIndex lsh_idx(database, lsh_k, lsh_L, lsh_w, vocab_K, seed);
    HypercubeIndex cube_idx(database, cube_kproj, cube_w, cube_M, cube_probes, vocab_K, seed);
    IVFFlatIndex ivfflat_idx(database, ivf_kclusters, ivf_nprobe, vocab_K);
    IVFPQIndex ivfpq_idx(database, ivf_kclusters, ivf_nprobe, pq_M, pq_nbits, vocab_K);

    if (method == "lsh") lsh_idx.build();
    else if (method == "hypercube") cube_idx.build();
    else if (method == "ivfflat") ivfflat_idx.train_and_build(20, seed);
    else if (method == "ivfpq") ivfpq_idx.train_and_build(20, seed);

    // 4. Αναζήτηση
    std::cout << "[5/5] Εκτέλεση αναζήτησης..." << std::endl;
    std::ofstream outfile(output_file);

    std::string method_print = method;
    std::transform(method_print.begin(), method_print.end(), method_print.begin(), ::toupper);
    outfile << method_print << "\n";

    double total_recall5 = 0.0;
    double total_recall10 = 0.0;
    double total_ap10 = 0.0;
    double total_time_ms = 0.0;

    for (const auto& q : queries) {
        outfile << "Query: " << q.sequence_id << "\n";

        auto start = std::chrono::high_resolution_clock::now();
        std::vector<std::string> results;

        if (method == "exact") results = exact_idx.search(q.bow_vec, 10);
        else if (method == "lsh") results = lsh_idx.search(q.bow_vec, 10);
        else if (method == "hypercube") results = cube_idx.search(q.bow_vec, 10);
        else if (method == "ivfflat") results = ivfflat_idx.search(q.bow_vec, 10);
        else if (method == "ivfpq") results = ivfpq_idx.search(q.bow_vec, 10);

        auto finish = std::chrono::high_resolution_clock::now();
        double elapsed_ms = std::chrono::duration<double, std::milli>(finish - start).count();
        total_time_ms += elapsed_ms;

        for (size_t r = 0; r < results.size() && r < 10; ++r) {
            outfile << "Rank-" << (r + 1) << ": " << results[r] << "\n";
        }

        double rec5 = compute_recall_at_n(q.sequence_id, results, 5);
        double rec10 = compute_recall_at_n(q.sequence_id, results, 10);
        double ap10 = compute_ap_at_10(q.sequence_id, results);

        total_recall5 += rec5;
        total_recall10 += rec10;
        total_ap10 += ap10;

        outfile << "Recall@5: " << rec5 << "\n";
        outfile << "Recall@10: " << rec10 << "\n";
        outfile << "AP@10: " << ap10 << "\n";
    }

    int Q = static_cast<int>(queries.size());
    if (Q > 0) {
        outfile << "Mean Recall@5: " << (total_recall5 / Q) << "\n";
        outfile << "Mean Recall@10: " << (total_recall10 / Q) << "\n";
        outfile << "mAP@10: " << (total_ap10 / Q) << "\n";
        outfile << "Average query time: " << (total_time_ms / Q) << "\n";
    }

    std::cout << "Η διαδικασία ολοκληρώθηκε! Έξοδος: " << output_file << std::endl;
    return 0;
}