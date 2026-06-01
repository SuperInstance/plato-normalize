# plato-normalize

> Normalization methods for PLATO tile data — MinMax, Z-Score, Robust, Log, Quantile

## What This Does

plato-normalize provides five normalization methods for scaling tile data to a common range. Each method can be fit to training data (learning min/max/mean/std/median/IQR), then applied to new data and inverted. A pipeline orchestrates normalization across multiple named series.

## The Key Idea

Different sensors produce values on wildly different scales: temperature in Celsius (15-35), humidity in percent (0-100), CO2 in ppm (400-2000). Machine learning models need all inputs on a similar scale. MinMax squashes to [0,1], Z-Score centers at 0 with unit variance, Robust uses median/IQR (resistant to outliers), Log handles skewed distributions, Quantile maps to uniform distribution.

## Install

```bash
cargo add plato-normalize
```

## Quick Start

```rust
use plato_normalize::*;

// MinMax normalize to [0, 1]
let normalized = min_max_normalize(50.0, 0.0, 100.0); // 0.5

// Z-Score
let z = z_score_normalize(22.5, 20.0, 2.0); // 1.25

// Inverse (denormalize)
let original = min_max_denormalize(0.5, 0.0, 100.0); // 50.0

// Pipeline: fit on data, apply to new values
let mut pipeline = NormalizationPipeline::new(NormalizationMethod::MinMax);
pipeline.fit("temperature", &vec![15.0, 20.0, 25.0, 30.0, 35.0]);
let n = pipeline.normalize("temperature", 22.5);
```

## API Reference

### Methods

| Method | Formula | Best For |
|---|---|---|
| `MinMax` | (x-min)/(max-min) | Known bounded ranges |
| `ZScore` | (x-mean)/std | Normally distributed data |
| `Robust` | (x-median)/IQR | Data with outliers |
| `Log` | ln(x+1) | Right-skewed data |
| `Quantile` | Rank-based mapping | Any distribution |

### Functions

| Function | Description |
|---|---|
| `min_max_normalize(value, min, max)` / `min_max_denormalize(...)` | [0,1] normalization |
| `z_score_normalize(value, mean, std)` / `z_score_denormalize(...)` | Standard score |
| `robust_normalize(value, median, iqr)` | Outlier-resistant |
| `log_normalize(value)` / `log_denormalize(...)` | Log transform |
| `fit_params(data, method)` | Learn parameters from data |

### NormalizationPipeline

```rust
let mut p = NormalizationPipeline::new(NormalizationMethod::ZScore);
p.fit("sensor-a", &training_data);
p.normalize("sensor-a", new_value);
p.denormalize("sensor-a", normalized_value);
```

## Testing

18 tests: each normalization method, inverse operations, edge cases (constant series, zero std), pipeline fit/apply.

## License

Apache-2.0
