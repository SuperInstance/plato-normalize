use serde::{Deserialize, Serialize};
use std::collections::HashMap;

// ---------------------------------------------------------------------------
// Core types
// ---------------------------------------------------------------------------

/// Supported normalization methods.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum NormalizationMethod {
    MinMax,
    ZScore,
    Robust,
    Log,
    Quantile,
}

/// Fitted parameters for a single series.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct NormalizationParams {
    pub method: NormalizationMethod,
    pub min: f64,
    pub max: f64,
    pub mean: f64,
    pub std: f64,
    pub median: f64,
    pub iqr: f64,
}

/// Wraps a single value with its normalization metadata.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct NormalizedValue {
    pub original: f64,
    pub normalized: f64,
    pub method: NormalizationMethod,
}

/// A named collection of fitted parameters, ready to normalize/denormalize.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct NormalizationPipeline {
    pub params: HashMap<String, NormalizationParams>,
    pub default_method: NormalizationMethod,
}

// ---------------------------------------------------------------------------
// Free functions
// ---------------------------------------------------------------------------

/// Min-max normalize into [0, 1].
pub fn min_max_normalize(value: f64, min: f64, max: f64) -> f64 {
    if (max - min).abs() < f64::EPSILON {
        return 0.0; // constant series → map to 0
    }
    (value - min) / (max - min)
}

/// Inverse of `min_max_normalize`.
pub fn min_max_denormalize(value: f64, min: f64, max: f64) -> f64 {
    if (max - min).abs() < f64::EPSILON {
        return min;
    }
    value * (max - min) + min
}

/// Z-score: (x - mean) / std.
pub fn z_score_normalize(value: f64, mean: f64, std: f64) -> f64 {
    if std.abs() < f64::EPSILON {
        return 0.0;
    }
    (value - mean) / std
}

/// Inverse of `z_score_normalize`.
pub fn z_score_denormalize(value: f64, mean: f64, std: f64) -> f64 {
    value * std + mean
}

/// Robust normalization using median and IQR.
pub fn robust_normalize(value: f64, median: f64, iqr: f64) -> f64 {
    if iqr.abs() < f64::EPSILON {
        return 0.0;
    }
    (value - median) / iqr
}

/// Log normalization: ln(1 + x). Caller must ensure x >= 0.
pub fn log_normalize(value: f64) -> f64 {
    (1.0 + value).ln()
}

/// Inverse of `log_normalize`.
pub fn log_denormalize(value: f64) -> f64 {
    value.exp() - 1.0
}

/// Quantile (rank-based) normalization. Replaces each value with the
/// corresponding quantile of the standard uniform distribution.
/// Returns the normalized vector and sorts the input slice.
pub fn quantile_normalize(values: &mut [f64]) -> Vec<f64> {
    let n = values.len();
    if n == 0 {
        return vec![];
    }
    // Collect original indices, sort by value
    let mut indexed: Vec<(usize, f64)> = values.iter().copied().enumerate().collect();
    indexed.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal));

    let mut result = vec![0.0f64; n];
    for (rank, (orig_idx, _)) in indexed.into_iter().enumerate() {
        result[orig_idx] = (rank as f64 + 0.5) / n as f64;
    }
    // Also sort the input slice (documented behavior)
    values.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    result
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

fn median_of(data: &[f64]) -> f64 {
    let mut s = data.to_vec();
    s.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    let mid = s.len() / 2;
    if s.len() % 2 == 0 {
        (s[mid - 1] + s[mid]) / 2.0
    } else {
        s[mid]
    }
}

fn iqr_of(data: &[f64]) -> f64 {
    let mut s = data.to_vec();
    s.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    let n = s.len();
    if n < 4 {
        return s.last().copied().unwrap_or(0.0) - s.first().copied().unwrap_or(0.0);
    }
    let q1 = median_of(&s[..n / 2]);
    let q3 = median_of(&s[(n + 1) / 2..]);
    q3 - q1
}

fn std_of(data: &[f64]) -> f64 {
    let mean = data.iter().sum::<f64>() / data.len() as f64;
    let variance = data.iter().map(|x| (x - mean).powi(2)).sum::<f64>() / data.len() as f64;
    variance.sqrt()
}

// ---------------------------------------------------------------------------
// NormalizationParams impl
// ---------------------------------------------------------------------------

impl NormalizationParams {
    /// Fit parameters from raw data for the given method.
    pub fn fit(data: &[f64], method: NormalizationMethod) -> Self {
        let min = data.iter().cloned().fold(f64::INFINITY, f64::min);
        let max = data.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
        let mean = data.iter().sum::<f64>() / data.len() as f64;
        let std = std_of(data);
        let median = median_of(data);
        let iqr = iqr_of(data);
        Self { method, min, max, mean, std, median, iqr }
    }

    /// Normalize a single value using the stored parameters.
    pub fn normalize(&self, value: f64) -> f64 {
        match self.method {
            NormalizationMethod::MinMax => min_max_normalize(value, self.min, self.max),
            NormalizationMethod::ZScore => z_score_normalize(value, self.mean, self.std),
            NormalizationMethod::Robust => robust_normalize(value, self.median, self.iqr),
            NormalizationMethod::Log => log_normalize(value),
            NormalizationMethod::Quantile => {
                // Quantile needs the whole dataset; fall back to rank approximation
                if (self.max - self.min).abs() < f64::EPSILON {
                    0.5
                } else {
                    (value - self.min) / (self.max - self.min)
                }
            }
        }
    }

    /// Denormalize a single value using the stored parameters.
    pub fn denormalize(&self, value: f64) -> f64 {
        match self.method {
            NormalizationMethod::MinMax => min_max_denormalize(value, self.min, self.max),
            NormalizationMethod::ZScore => z_score_denormalize(value, self.mean, self.std),
            NormalizationMethod::Robust => {
                if self.iqr.abs() < f64::EPSILON {
                    self.median
                } else {
                    value * self.iqr + self.median
                }
            }
            NormalizationMethod::Log => log_denormalize(value),
            NormalizationMethod::Quantile => {
                if (self.max - self.min).abs() < f64::EPSILON {
                    self.min
                } else {
                    value * (self.max - self.min) + self.min
                }
            }
        }
    }
}

// ---------------------------------------------------------------------------
// NormalizationPipeline impl
// ---------------------------------------------------------------------------

impl NormalizationPipeline {
    pub fn new(default_method: NormalizationMethod) -> Self {
        Self { params: HashMap::new(), default_method }
    }

    /// Fit parameters for a named series.
    pub fn fit(&mut self, name: &str, data: &[f64]) {
        let params = NormalizationParams::fit(data, self.default_method);
        self.params.insert(name.to_string(), params);
    }

    /// Normalize a value for a named series.
    pub fn normalize(&self, name: &str, value: f64) -> f64 {
        self.params.get(name)
            .map(|p| p.normalize(value))
            .unwrap_or(value)
    }

    /// Denormalize a value for a named series.
    pub fn denormalize(&self, name: &str, value: f64) -> f64 {
        self.params.get(name)
            .map(|p| p.denormalize(value))
            .unwrap_or(value)
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    const TOL: f64 = 1e-9;

    // -- MinMax --
    #[test]
    fn min_max_normal_range() {
        let v = min_max_normalize(5.0, 0.0, 10.0);
        assert!((v - 0.5).abs() < TOL);
    }

    #[test]
    fn min_max_edge_min_equals_max() {
        let v = min_max_normalize(42.0, 42.0, 42.0);
        assert_eq!(v, 0.0);
    }

    #[test]
    fn min_max_roundtrip() {
        let (min, max) = (-3.0, 7.0);
        for x in [-3.0, -1.0, 0.0, 4.0, 7.0] {
            let n = min_max_normalize(x, min, max);
            let d = min_max_denormalize(n, min, max);
            assert!((d - x).abs() < TOL, "roundtrip failed for {x}");
        }
    }

    // -- ZScore --
    #[test]
    fn z_score_mean_zero_std_one() {
        let data = [1.0, 2.0, 3.0, 4.0, 5.0];
        let mean = data.iter().sum::<f64>() / 5.0;
        let std = std_of(&data);
        let normed: Vec<f64> = data.iter().map(|&x| z_score_normalize(x, mean, std)).collect();
        let normed_mean = normed.iter().sum::<f64>() / 5.0;
        let normed_std = std_of(&normed);
        assert!(normed_mean.abs() < TOL, "mean should be ~0, got {normed_mean}");
        assert!((normed_std - 1.0).abs() < TOL, "std should be ~1, got {normed_std}");
    }

    #[test]
    fn z_score_roundtrip() {
        let (mean, std) = (10.0, 2.5);
        for x in [5.0, 10.0, 15.0] {
            let d = z_score_denormalize(z_score_normalize(x, mean, std), mean, std);
            assert!((d - x).abs() < TOL);
        }
    }

    // -- Robust --
    #[test]
    fn robust_handles_outliers() {
        let clean = [1.0, 2.0, 3.0, 4.0, 5.0];
        let with_outlier = [1.0, 2.0, 3.0, 4.0, 100.0];
        let med_clean = median_of(&clean);
        let iqr_clean = iqr_of(&clean);
        let med_out = median_of(&with_outlier);
        let iqr_out = iqr_of(&with_outlier);
        // The value 3.0 should be close to 0 in both
        let n_clean = robust_normalize(3.0, med_clean, iqr_clean);
        let n_out = robust_normalize(3.0, med_out, iqr_out);
        assert!(n_clean.abs() < 1.0, "clean: {n_clean}");
        assert!(n_out.abs() < 1.0, "outlier: {n_out}");
    }

    // -- Log --
    #[test]
    fn log_correctness() {
        let v = log_normalize(0.0);
        assert!((v - 0.0).abs() < TOL);
        let v = log_normalize(f64::exp(1.0) - 1.0);
        assert!((v - 1.0).abs() < TOL);
    }

    #[test]
    fn log_roundtrip() {
        for x in [0.0, 1.0, 10.0, 100.0] {
            assert!((log_denormalize(log_normalize(x)) - x).abs() < 1e-6);
        }
    }

    // -- Quantile --
    #[test]
    fn quantile_uniform_output() {
        let mut data = [3.0, 1.0, 4.0, 2.0];
        let result = quantile_normalize(&mut data);
        assert_eq!(result.len(), 4);
        // Should be roughly uniformly spaced in (0, 1)
        for &r in &result {
            assert!(r > 0.0 && r < 1.0, "value {r} out of range");
        }
        let mean = result.iter().sum::<f64>() / 4.0;
        assert!((mean - 0.5).abs() < 0.1);
    }

    // -- Params fit --
    #[test]
    fn params_fit_from_data() {
        let data: Vec<f64> = (0..=100).map(|i| i as f64).collect();
        let p = NormalizationParams::fit(&data, NormalizationMethod::MinMax);
        assert!((p.min - 0.0).abs() < TOL);
        assert!((p.max - 100.0).abs() < TOL);
        assert!((p.mean - 50.0).abs() < TOL);
    }

    // -- Pipeline --
    #[test]
    fn pipeline_roundtrip() {
        let mut pipe = NormalizationPipeline::new(NormalizationMethod::MinMax);
        let data = [10.0, 20.0, 30.0, 40.0, 50.0];
        pipe.fit("temp", &data);
        for &v in &data {
            let n = pipe.normalize("temp", v);
            let d = pipe.denormalize("temp", n);
            assert!((d - v).abs() < TOL, "roundtrip for {v}");
        }
    }

    #[test]
    fn pipeline_multiple_series() {
        let mut pipe = NormalizationPipeline::new(NormalizationMethod::ZScore);
        pipe.fit("a", &[1.0, 2.0, 3.0]);
        pipe.fit("b", &[100.0, 200.0, 300.0]);
        // Different params for different series
        let pa = pipe.params.get("a").unwrap();
        let pb = pipe.params.get("b").unwrap();
        assert!(pa.mean.abs() < pb.mean, "b mean should be larger");
    }

    // -- Edge cases --
    #[test]
    fn constant_data() {
        let data = [5.0; 10];
        let p = NormalizationParams::fit(&data, NormalizationMethod::MinMax);
        assert_eq!(p.normalize(5.0), 0.0);
    }

    #[test]
    fn single_value() {
        let data = [42.0];
        let p = NormalizationParams::fit(&data, NormalizationMethod::ZScore);
        assert_eq!(p.normalize(42.0), 0.0); // std=0 → 0
    }

    #[test]
    fn negative_values() {
        let data = [-10.0, -5.0, 0.0, 5.0, 10.0];
        let p = NormalizationParams::fit(&data, NormalizationMethod::MinMax);
        assert!((p.normalize(-10.0) - 0.0).abs() < TOL);
        assert!((p.normalize(10.0) - 1.0).abs() < TOL);
    }

    #[test]
    fn large_values() {
        let data = [1e15, 2e15, 3e15];
        let p = NormalizationParams::fit(&data, NormalizationMethod::ZScore);
        let n = p.normalize(2e15);
        assert!(n.is_finite(), "should be finite");
    }

    #[test]
    fn serialization() {
        let data = [1.0, 2.0, 3.0, 4.0, 5.0];
        let p = NormalizationParams::fit(&data, NormalizationMethod::MinMax);
        let json = serde_json::to_string(&p).unwrap();
        let p2: NormalizationParams = serde_json::from_str(&json).unwrap();
        assert_eq!(p.method, p2.method);
        assert!((p.min - p2.min).abs() < TOL);
    }

    #[test]
    fn pipeline_serialization() {
        let mut pipe = NormalizationPipeline::new(NormalizationMethod::Robust);
        pipe.fit("x", &[1.0, 2.0, 3.0]);
        let json = serde_json::to_string(&pipe).unwrap();
        let p2: NormalizationPipeline = serde_json::from_str(&json).unwrap();
        assert_eq!(pipe.default_method, p2.default_method);
        assert!(p2.params.contains_key("x"));
    }
}
