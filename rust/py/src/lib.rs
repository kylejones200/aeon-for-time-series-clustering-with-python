use aeon_for_time_series_clustering_with_python_core::dtw_distance;
use numpy::PyReadonlyArray1;
use pyo3::prelude::*;

#[pyfunction]
fn dtw_distance_py(a: PyReadonlyArray1<f64>, b: PyReadonlyArray1<f64>) -> PyResult<f64> {
    Ok(dtw_distance(a.as_slice()?, b.as_slice()?))
}

#[pyfunction]
#[pyo3(signature = (a, b, iterations=5000))]
fn bench_kernel_py(a: PyReadonlyArray1<f64>, b: PyReadonlyArray1<f64>, iterations: usize) -> PyResult<f64> {
    let aa = a.as_slice()?.to_vec();
    let bb = b.as_slice()?.to_vec();
    let start = std::time::Instant::now();
    for _ in 0..iterations {
        let _ = dtw_distance(&aa, &bb);
    }
    Ok(start.elapsed().as_secs_f64())
}

#[pymodule]
fn aeon_for_time_series_clustering_with_python_rs(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(dtw_distance_py, m)?)?;
    m.add_function(wrap_pyfunction!(bench_kernel_py, m)?)?;
    Ok(())
}
