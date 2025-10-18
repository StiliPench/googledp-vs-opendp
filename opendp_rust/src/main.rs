use opendp::measurements::then_laplace;
use opendp::metrics::SymmetricDistance;
use opendp::transformations::{
    make_split_lines, then_cast_default, then_clamp, then_count, then_impute_constant, then_sum,
};
use std::error::Error;
use std::fs::{OpenOptions, metadata};
use std::time::Instant;
use csv::WriterBuilder;

struct DatasetConfig {
    name: &'static str,
    file: &'static str,
    rows: usize,
}

struct CsvWriter {
    path: &'static str,
    header_written: bool,
}

impl CsvWriter {
    fn new(path: &'static str) -> Self {
        let header_written = metadata(path).map(|m| m.len() > 0).unwrap_or(false);
        Self { path, header_written }
    }

    fn ensure_header(&mut self) -> csv::Result<()> {
        if self.header_written { return Ok(()); }
        let file = OpenOptions::new().create(true).append(true).open(self.path)?;
        let mut wtr = WriterBuilder::new().has_headers(false).from_writer(file);
        wtr.write_record([
            "library","language","dataset_name","dataset_size","query","mechanism",
            "epsilon","mae","avg_time_ms","runs","lower","upper"
        ])?;
        wtr.flush()?;
        self.header_written = true;
        Ok(())
    }

    fn write_row(&mut self,
                 library: &str, language: &str,
                 dataset_name: &str, dataset_size: usize,
                 query: &str, mechanism: &str,
                 epsilon: f64, mae: f64, avg_time_ms: f64, runs: usize,
                 lower: f64, upper: f64) -> csv::Result<()> {
        self.ensure_header()?;
        let file = OpenOptions::new().create(true).append(true).open(self.path)?;
        let mut wtr = WriterBuilder::new().has_headers(false).from_writer(file);
        wtr.write_record(&[
            library, language, dataset_name, &dataset_size.to_string(),
            query, mechanism, &format!("{:.6}", epsilon),
            &format!("{:.10}", mae), &format!("{:.10}", avg_time_ms),
            &runs.to_string(), &lower.to_string(), &upper.to_string()
        ])?;
        wtr.flush()?;
        Ok(())
    }
}

fn main() -> Result<(), Box<dyn Error>> {
    // --- Experiment Configuration ---
    let datasets_to_test = vec![
        DatasetConfig { name: "Small (10k rows)",  file: "small_dataset.csv",  rows: 10_000 },
        DatasetConfig { name: "Medium (100k rows)", file: "medium_dataset.csv", rows: 100_000 },
        DatasetConfig { name: "Large (1M rows)",    file: "large_dataset.csv",  rows: 1_000_000 },
    ];
    const EPSILON_VALUES: &[f64] = &[0.1, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0];
    const NUM_RUNS: usize = 30;
    const SUM_LOWER_BOUND: f64 = -100.0;
    const SUM_UPPER_BOUND: f64 = 100.0;
    const SUM_COLUMN: &str = "value_col_1";

    let mut csv = CsvWriter::new("opendp_native_results.csv");
    let sum_bounds = (SUM_LOWER_BOUND, SUM_UPPER_BOUND);

    // --- Run Experiments ---
    for config in &datasets_to_test {
        println!("\n========================================");
        println!("Running Experiments on: {}", config.name);
        println!("========================================");

        // --- Load data from the specified column in the CSV file ---
        let mut rdr = csv::Reader::from_path(config.file)?;
        let headers = rdr.headers()?.clone();
        let col_index = headers.iter().position(|h| h == SUM_COLUMN)
            .ok_or_else(|| format!("Column '{}' not found in '{}'", SUM_COLUMN, config.file))?;

        let mut numbers = Vec::with_capacity(config.rows);
        for result in rdr.records() {
            let record = result?;
            if let Some(value_str) = record.get(col_index) {
                if let Ok(val) = value_str.parse::<f64>() {
                    numbers.push(val);
                }
            }
        }

        // OpenDP expects newline-separated data for the pipeline below
        let data = numbers.iter().map(|n| n.to_string()).collect::<Vec<String>>().join("\n");

        // --- Calculate True Values from the loaded data ---
        let true_count = numbers.len() as f64;
        let true_sum: f64 = numbers.iter().sum();

        // --- Count Query Experiment ---
        println!("\n--- Query: COUNT ---");
        println!("True Count: {}", true_count);

        for &epsilon in EPSILON_VALUES {
            let count_scale = 1.0 / epsilon;

            let count_measurement = ((make_split_lines()?
                >> then_cast_default::<SymmetricDistance, String, f64>()
                >> then_count::<f64, usize>())?
                >> then_laplace(count_scale, None))?;

            let mut noisy_counts = Vec::with_capacity(NUM_RUNS);
            let start_time = Instant::now();

            for _ in 0..NUM_RUNS {
                let noisy_count = count_measurement.invoke(&data)? as f64;
                noisy_counts.push(noisy_count);
            }

            let elapsed_time = start_time.elapsed();
            let avg_time = elapsed_time.as_secs_f64() / NUM_RUNS as f64;

            let avg_noisy_count: f64 = noisy_counts.iter().sum::<f64>() / NUM_RUNS as f64;
            let mean_absolute_error: f64 = noisy_counts
                .iter()
                .map(|&noisy| (noisy - true_count).abs())
                .sum::<f64>()
                / NUM_RUNS as f64;

            println!(
                "Epsilon: {:<4} | Avg Result: {:<10.2} | MAE: {:<10.2} | Avg Time: {:.4}ms",
                epsilon, avg_noisy_count, mean_absolute_error, avg_time * 1000.0
            );

            csv.write_row(
                "OpenDP", "Rust",
                config.name, config.rows,
                "Count", "Laplace",
                epsilon, mean_absolute_error, avg_time * 1000.0, NUM_RUNS,
                SUM_LOWER_BOUND, SUM_UPPER_BOUND
            )?;
        }

        // --- Sum Query Experiment ---
        println!("\n--- Query: SUM ---");
        println!("True Sum: {:.2}", true_sum);

        for &epsilon in EPSILON_VALUES {
            // (Keep your existing calibration choice)
            let sum_sensitivity = sum_bounds.1 - sum_bounds.0;
            let sum_scale = sum_sensitivity / (epsilon - 1e-9);

            let sum_measurement = ((make_split_lines()?
                >> then_cast_default::<SymmetricDistance, String, f64>()
                >> then_impute_constant(0.0)
                >> then_clamp(sum_bounds)
                >> then_sum::<SymmetricDistance, f64>())?
                >> then_laplace(sum_scale, None))?;

            let mut noisy_sums = Vec::with_capacity(NUM_RUNS);
            let start_time = Instant::now();

            for _ in 0..NUM_RUNS {
                let noisy_sum = sum_measurement.invoke(&data)?;
                noisy_sums.push(noisy_sum);
            }

            let elapsed_time = start_time.elapsed();
            let avg_time = elapsed_time.as_secs_f64() / NUM_RUNS as f64;

            let avg_noisy_sum: f64 = noisy_sums.iter().sum::<f64>() / NUM_RUNS as f64;
            let mean_absolute_error: f64 = noisy_sums
                .iter()
                .map(|&noisy| (noisy - true_sum).abs())
                .sum::<f64>()
                / NUM_RUNS as f64;

            println!(
                "Epsilon: {:<4} | Avg Result: {:<10.2} | MAE: {:<10.2} | Avg Time: {:.4}ms",
                epsilon, avg_noisy_sum, mean_absolute_error, avg_time * 1000.0
            );

            csv.write_row(
                "OpenDP", "Rust",
                config.name, config.rows,
                "Sum", "Laplace",
                epsilon, mean_absolute_error, avg_time * 1000.0, NUM_RUNS,
                SUM_LOWER_BOUND, SUM_UPPER_BOUND
            )?;
        }
    }

    Ok(())
}
