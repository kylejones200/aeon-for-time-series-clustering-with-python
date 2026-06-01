use aeon_for_time_series_clustering_with_python_core::dtw_distance;

fn main() {
    let a: Vec<f64> = (0..200).map(|i| (i as f64 * 0.1).sin()).collect();
    let b: Vec<f64> = (0..180).map(|i| (i as f64 * 0.12).cos()).collect();
    for _ in 0..5000 {
        let _ = dtw_distance(&a, &b);
    }
}
