#include <iostream>
#include <vector>
#include <string>
#include <fstream>
#include <random>
#include <chrono>
#include <iomanip>
#include <cmath>
#include <sstream>
#include <filesystem>
#include <memory>

// Google DP headers
#include "algorithms/bounded-sum.h"
#include "algorithms/count.h"
#include "algorithms/numerical-mechanisms.h"
#include "proto/util.h"
#include "absl/status/statusor.h"

struct DatasetConfig {
    std::string name;
    std::string file;
    int rows;
};

std::vector<double> load_column_from_csv(const std::string& file_path,
                                         const std::string& col_name) {
    std::ifstream file(file_path);
    std::vector<double> data;
    std::string line;

    if (!file.is_open()) {
        std::cerr << "Error: Could not open file " << file_path << "\n";
        return data;
    }

    // Read header to find column index
    std::getline(file, line);
    std::stringstream header_ss(line);
    std::string segment;
    int col_idx = -1;
    int current_idx = 0;
    while (std::getline(header_ss, segment, ',')) {
        if (segment == col_name) {
            col_idx = current_idx;
            break;
        }
        current_idx++;
    }

    if (col_idx == -1) {
        std::cerr << "Error: Column '" << col_name << "' not found.\n";
        return data;
    }

    // Read data rows
    while (std::getline(file, line)) {
        std::stringstream line_ss(line);
        current_idx = 0;
        while (std::getline(line_ss, segment, ',')) {
            if (current_idx == col_idx) {
                try {
                    data.push_back(std::stod(segment));
                } catch (...) {
                    // Ignore parse errors
                }
                break;
            }
            current_idx++;
        }
    }
    return data;
}

struct CsvWriter {
    std::string path;
    bool header_written;

    explicit CsvWriter(const std::string& p)
        : path(p),
          header_written(std::filesystem::exists(p) &&
                         std::filesystem::file_size(p) > 0) {}

    void ensure_header() {
        if (header_written) return;
        std::ofstream out(path, std::ios::app);
        out << "library,language,dataset_name,dataset_size,query,mechanism,"
               "epsilon,mae,avg_time_ms,runs,lower,upper\n";
        header_written = true;
    }

    void write_row(const std::string& library,
                   const std::string& language,
                   const std::string& dataset_name,
                   long long dataset_size,
                   const std::string& query,
                   const std::string& mechanism,
                   double epsilon,
                   double mae,
                   double avg_time_ms,
                   int runs,
                   double lower,
                   double upper) {
        ensure_header();
        std::ofstream out(path, std::ios::app);
        out << library << "," << language << ","
            << '"' << dataset_name << '"' << "," << dataset_size << ","
            << query << "," << mechanism << ","
            << std::fixed << std::setprecision(6) << epsilon << ","
            << std::setprecision(10) << mae << ","
            << std::setprecision(10) << avg_time_ms << ","
            << runs << "," << lower << "," << upper << "\n";
    }
};

int main() {
    const std::vector<DatasetConfig> datasets_to_test = {
        {"Small (10k rows)",  "small_dataset.csv",  10000},
        {"Medium (100k rows)","medium_dataset.csv", 100000},
        {"Large (1M rows)",   "large_dataset.csv",  1000000},
    };

    const std::vector<double> EPSILON_VALUES =
        {0.1, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0};

    const int NUM_RUNS = 30;
    const double SUM_LOWER_BOUND = -100.0;
    const double SUM_UPPER_BOUND = 100.0;
    const std::string SUM_COLUMN = "value_col_1";

    // (ε, δ)-DP delta for Gaussian
    const double DELTA = 1e-5;

    CsvWriter csv("gdp_native_results.csv");

    for (const auto& config : datasets_to_test) {
        std::cout << "\n========================================\n";
        std::cout << "Running Experiments on: " << config.name << "\n";
        std::cout << "========================================\n";

        std::vector<double> numbers =
            load_column_from_csv(config.file, SUM_COLUMN);
        if (numbers.empty()) continue;

        double true_count = static_cast<double>(numbers.size());
        double true_sum = 0.0;
        for (double n : numbers) true_sum += n;

        // -------------------
        // COUNT – Laplace
        // -------------------
        std::cout << "\n--- Query: COUNT (Laplace) ---\n";
        std::cout << "True Count: " << true_count << "\n";

        for (double epsilon : EPSILON_VALUES) {
            std::vector<double> noisy_counts;
            auto start_time = std::chrono::high_resolution_clock::now();

            for (int i = 0; i < NUM_RUNS; ++i) {
                auto count_statusor =
                    differential_privacy::Count<int64_t>::Builder()
                        .SetEpsilon(epsilon)
                        .Build();

                if (!count_statusor.ok()) {
                    std::cerr << "Error building Count: "
                              << count_statusor.status() << "\n";
                    return 1;
                }
                auto count_dp = std::move(count_statusor).value();

                auto result_statusor =
                    count_dp->Result(numbers.begin(), numbers.end());
                if (!result_statusor.ok()) {
                    std::cerr << "Error computing Count result: "
                              << result_statusor.status() << "\n";
                    return 1;
                }
                auto result = std::move(result_statusor).value();
                noisy_counts.push_back(
                    static_cast<double>(
                        differential_privacy::GetValue<int64_t>(result)));
            }

            auto end_time = std::chrono::high_resolution_clock::now();
            double avg_time =
                std::chrono::duration<double>(end_time - start_time).count()
                / NUM_RUNS;

            double avg_noisy_count = 0.0;
            for (double nc : noisy_counts) avg_noisy_count += nc;
            avg_noisy_count /= NUM_RUNS;

            double mae = 0.0;
            for (double nc : noisy_counts) mae += std::abs(nc - true_count);
            mae /= NUM_RUNS;

            std::cout << "[Laplace] Epsilon: " << epsilon
                      << " | Avg Result: " << avg_noisy_count
                      << " | MAE: " << mae
                      << " | Avg Time: " << avg_time * 1000.0 << "ms\n";

            csv.write_row(
                "GoogleDP", "C++",
                config.name, static_cast<long long>(config.rows),
                "Count", "Laplace",
                epsilon,
                mae,
                avg_time * 1000.0,
                NUM_RUNS,
                SUM_LOWER_BOUND, SUM_UPPER_BOUND
            );
        }

        // -------------------
        // COUNT – Gaussian
        // -------------------
        std::cout << "\n--- Query: COUNT (Gaussian) ---\n";
        std::cout << "True Count: " << true_count << "\n";

        for (double epsilon : EPSILON_VALUES) {
            std::vector<double> noisy_counts;
            auto start_time = std::chrono::high_resolution_clock::now();

            for (int i = 0; i < NUM_RUNS; ++i) {
                auto count_statusor =
                    differential_privacy::Count<int64_t>::Builder()
                        .SetEpsilon(epsilon)
                        .SetDelta(DELTA)
                        // Use GaussianMechanism via the generic mechanism builder hook
                        .SetLaplaceMechanism(
                            std::make_unique<
                                differential_privacy::GaussianMechanism::Builder>())
                        .Build();

                if (!count_statusor.ok()) {
                    std::cerr << "Error building Count (Gaussian): "
                              << count_statusor.status() << "\n";
                    return 1;
                }
                auto count_dp = std::move(count_statusor).value();

                auto result_statusor =
                    count_dp->Result(numbers.begin(), numbers.end());
                if (!result_statusor.ok()) {
                    std::cerr << "Error computing Count (Gaussian) result: "
                              << result_statusor.status() << "\n";
                    return 1;
                }
                auto result = std::move(result_statusor).value();
                noisy_counts.push_back(
                    static_cast<double>(
                        differential_privacy::GetValue<int64_t>(result)));
            }

            auto end_time = std::chrono::high_resolution_clock::now();
            double avg_time =
                std::chrono::duration<double>(end_time - start_time).count()
                / NUM_RUNS;

            double avg_noisy_count = 0.0;
            for (double nc : noisy_counts) avg_noisy_count += nc;
            avg_noisy_count /= NUM_RUNS;

            double mae = 0.0;
            for (double nc : noisy_counts) mae += std::abs(nc - true_count);
            mae /= NUM_RUNS;

            std::cout << "[Gaussian] Epsilon: " << epsilon
                      << " | Avg Result: " << avg_noisy_count
                      << " | MAE: " << mae
                      << " | Avg Time: " << avg_time * 1000.0 << "ms\n";

            csv.write_row(
                "GoogleDP", "C++",
                config.name, static_cast<long long>(config.rows),
                "Count", "Gaussian",
                epsilon,
                mae,
                avg_time * 1000.0,
                NUM_RUNS,
                SUM_LOWER_BOUND, SUM_UPPER_BOUND
            );
        }

        // -------------------
        // SUM – Laplace
        // -------------------
        std::cout << "\n--- Query: SUM (Laplace) ---\n";
        std::cout << "True Sum: " << true_sum << "\n";

        for (double epsilon : EPSILON_VALUES) {
            std::vector<double> noisy_sums;
            auto start_time = std::chrono::high_resolution_clock::now();

            for (int i = 0; i < NUM_RUNS; ++i) {
                auto sum_statusor =
                    differential_privacy::BoundedSum<double>::Builder()
                        .SetEpsilon(epsilon)
                        .SetLower(SUM_LOWER_BOUND)
                        .SetUpper(SUM_UPPER_BOUND)
                        .Build();

                if (!sum_statusor.ok()) {
                    std::cerr << "Error building BoundedSum: "
                              << sum_statusor.status() << "\n";
                    return 1;
                }
                auto sum_dp = std::move(sum_statusor).value();

                auto result_statusor =
                    sum_dp->Result(numbers.begin(), numbers.end());
                if (!result_statusor.ok()) {
                    std::cerr << "Error computing Sum result: "
                              << result_statusor.status() << "\n";
                    return 1;
                }
                auto result = std::move(result_statusor).value();

                noisy_sums.push_back(
                    differential_privacy::GetValue<double>(result));
            }

            auto end_time = std::chrono::high_resolution_clock::now();
            double avg_time =
                std::chrono::duration<double>(end_time - start_time).count()
                / NUM_RUNS;

            double avg_noisy_sum = 0.0;
            for (double ns : noisy_sums) avg_noisy_sum += ns;
            avg_noisy_sum /= NUM_RUNS;

            double mae = 0.0;
            for (double ns : noisy_sums) mae += std::abs(ns - true_sum);
            mae /= NUM_RUNS;

            std::cout << "[Laplace] Epsilon: " << epsilon
                      << " | Avg Result: " << avg_noisy_sum
                      << " | MAE: " << mae
                      << " | Avg Time: " << avg_time * 1000.0 << "ms\n";

            csv.write_row(
                "GoogleDP", "C++",
                config.name, static_cast<long long>(config.rows),
                "Sum", "Laplace",
                epsilon,
                mae,
                avg_time * 1000.0,
                NUM_RUNS,
                SUM_LOWER_BOUND, SUM_UPPER_BOUND
            );
        }

        // -------------------
        // SUM – Gaussian
        // -------------------
        std::cout << "\n--- Query: SUM (Gaussian) ---\n";
        std::cout << "True Sum: " << true_sum << "\n";

        for (double epsilon : EPSILON_VALUES) {
            std::vector<double> noisy_sums;
            auto start_time = std::chrono::high_resolution_clock::now();

            for (int i = 0; i < NUM_RUNS; ++i) {
                auto sum_statusor =
                    differential_privacy::BoundedSum<double>::Builder()
                        .SetEpsilon(epsilon)
                        .SetDelta(DELTA)
                        .SetLower(SUM_LOWER_BOUND)
                        .SetUpper(SUM_UPPER_BOUND)
                        .SetLaplaceMechanism(
                            std::make_unique<
                                differential_privacy::GaussianMechanism::Builder>())
                        .Build();

                if (!sum_statusor.ok()) {
                    std::cerr << "Error building BoundedSum (Gaussian): "
                              << sum_statusor.status() << "\n";
                    return 1;
                }
                auto sum_dp = std::move(sum_statusor).value();

                auto result_statusor =
                    sum_dp->Result(numbers.begin(), numbers.end());
                if (!result_statusor.ok()) {
                    std::cerr << "Error computing Sum (Gaussian) result: "
                              << result_statusor.status() << "\n";
                    return 1;
                }
                auto result = std::move(result_statusor).value();

                noisy_sums.push_back(
                    differential_privacy::GetValue<double>(result));
            }

            auto end_time = std::chrono::high_resolution_clock::now();
            double avg_time =
                std::chrono::duration<double>(end_time - start_time).count()
                / NUM_RUNS;

            double avg_noisy_sum = 0.0;
            for (double ns : noisy_sums) avg_noisy_sum += ns;
            avg_noisy_sum /= NUM_RUNS;

            double mae = 0.0;
            for (double ns : noisy_sums) mae += std::abs(ns - true_sum);
            mae /= NUM_RUNS;

            std::cout << "[Gaussian] Epsilon: " << epsilon
                      << " | Avg Result: " << avg_noisy_sum
                      << " | MAE: " << mae
                      << " | Avg Time: " << avg_time * 1000.0 << "ms\n";

            csv.write_row(
                "GoogleDP", "C++",
                config.name, static_cast<long long>(config.rows),
                "Sum", "Gaussian",
                epsilon,
                mae,
                avg_time * 1000.0,
                NUM_RUNS,
                SUM_LOWER_BOUND, SUM_UPPER_BOUND
            );
        }
    }

    return 0;
}
