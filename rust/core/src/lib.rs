//! Dynamic time warping distance (squared Euclidean local cost).

pub fn dtw_distance(a: &[f64], b: &[f64]) -> f64 {
    let n = a.len();
    let m = b.len();
    if n == 0 || m == 0 {
        return f64::INFINITY;
    }
    let mut prev = vec![f64::INFINITY; m + 1];
    let mut curr = vec![f64::INFINITY; m + 1];
    prev[0] = 0.0;
    for i in 1..=n {
        curr[0] = f64::INFINITY;
        for j in 1..=m {
            let cost = (a[i - 1] - b[j - 1]).powi(2);
            let best = prev[j].min(prev[j - 1]).min(curr[j - 1]);
            curr[j] = cost + best;
        }
        std::mem::swap(&mut prev, &mut curr);
    }
    prev[m].sqrt()
}
