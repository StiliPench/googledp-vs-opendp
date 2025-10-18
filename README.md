# Differential Privacy: OpenDP (Rust) vs GoogleDP (C++)

This project benchmarks and compares **OpenDP** (Rust) and **Google Differential Privacy** (C++) for `Count` and `Sum` queries on synthetic datasets.

- **OpenDP (Rust):** implemented in `opendp_rust/`
- **GoogleDP (C++):** implemented in `googledp_cpp/` (built with Bazel)
- **Python scripts:** handle dataset generation and result plotting

---

## 1. Prerequisites

Ensure you have the following installed:

- **Python 3.9+**
- **Rust toolchain** (`rustup`, `cargo`)
- **C++ compiler** (e.g., g++, clang++)
- **Bazel** (for GoogleDP C++ build)

---

## 2. Python Setup

Install dependencies in a virtual environment:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```
---

## 3. Generate Synthetic Datasets

Run this once to generate three synthetic datasets (10k, 100k, and 1M rows):
```bash
python generate_databases.py
```

This creates identical CSVs in both project folders:
```pgsql
googledp_cpp/{small,medium,large}_dataset.csv
opendp_rust/{small,medium,large}_dataset.csv
```
Each dataset contains 5 numerical columns with random values in the range [-100, 100].

---

## 4. Build and Run the Experiments

Each run performs 30 executions per dataset $\times$ $\epsilon$ value.

### A. GoogleDP (C++, Bazel)
From the repo root:
```bash
cd googledp_cpp
# Build
bazel build //:dp_comparison
# Run
./bazel-bin/dp_comparison # run
```

This produces:
```bash
googledp_cpp/gdp_native_results.csv
```

### B. OpenDP (Rust, Cargo)
From the repo root:
```bash
cd opendp_rust
# Build and run
cargo run --release
```

This produces:
```bash
opendp_rust/opendp_native_results.csv
```

# 5. Combine Results and Generate Plots

After both native experiments finish:
```bash
python plot_combiner.py      # merges both CSVs → combined_laplace_results.csv
python generate_plots.py     # generates performance and utility plots
```

Output:
```bash
combined_laplace_results.csv
comparison_plots/
```