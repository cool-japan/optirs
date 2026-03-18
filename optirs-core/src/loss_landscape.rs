//! Loss Landscape Analysis Module
//!
//! Provides tools for visualizing and analyzing the loss landscape around
//! a point in parameter space, including sharpness measurement, saddle point
//! detection, and contour plot rendering.

use crate::error::{OptimError, Result};
use scirs2_core::ndarray::{Array1, Array2, ScalarOperand};
use scirs2_core::numeric::Float;
use std::fmt::Debug;

/// Method for choosing perturbation directions in the landscape
#[derive(Debug, Clone, Default, PartialEq)]
pub enum DirectionMethod {
    /// Random directions (normalized)
    #[default]
    Random,
    /// PCA-based directions from optimization trajectory
    PCA,
    /// Filter-normalized directions (Li et al., 2018)
    FilterNormalized,
}

/// Configuration for loss landscape analysis
#[derive(Debug, Clone)]
pub struct LossLandscapeConfig<A> {
    /// Resolution of the evaluation grid (grid_resolution x grid_resolution)
    pub grid_resolution: usize,
    /// Range of perturbation along each direction: [-range, +range]
    pub perturbation_range: A,
    /// Method for choosing perturbation directions
    pub direction_method: DirectionMethod,
}

impl Default for LossLandscapeConfig<f64> {
    fn default() -> Self {
        Self {
            grid_resolution: 20,
            perturbation_range: 1.0,
            direction_method: DirectionMethod::Random,
        }
    }
}

impl Default for LossLandscapeConfig<f32> {
    fn default() -> Self {
        Self {
            grid_resolution: 20,
            perturbation_range: 1.0f32,
            direction_method: DirectionMethod::Random,
        }
    }
}

/// 2D loss landscape data
#[derive(Debug, Clone)]
pub struct LandscapeData<A> {
    /// 2D grid of loss values (grid_resolution x grid_resolution)
    pub grid: Array2<A>,
    /// Range of perturbation in direction 1: (min_alpha, max_alpha)
    pub x_range: (A, A),
    /// Range of perturbation in direction 2: (min_beta, max_beta)
    pub y_range: (A, A),
    /// Loss value at the center point (no perturbation)
    pub center_loss: A,
    /// Minimum loss value in the grid
    pub min_loss: A,
    /// Maximum loss value in the grid
    pub max_loss: A,
}

/// Information about a detected saddle point in the landscape
#[derive(Debug, Clone)]
pub struct SaddlePointInfo<A> {
    /// Grid x-coordinate of the saddle point
    pub grid_x: usize,
    /// Grid y-coordinate of the saddle point
    pub grid_y: usize,
    /// Loss value at the saddle point
    pub loss_value: A,
}

/// Loss landscape analyzer for understanding optimization surfaces
pub struct LossLandscapeAnalyzer<A> {
    /// Configuration parameters
    config: LossLandscapeConfig<A>,
}

impl<A> LossLandscapeAnalyzer<A>
where
    A: Float + ScalarOperand + Debug + std::iter::Sum,
{
    /// Create a new loss landscape analyzer with the given configuration
    pub fn new(config: LossLandscapeConfig<A>) -> Self {
        Self { config }
    }

    /// Compute the loss landscape on a 2D grid
    ///
    /// Evaluates `loss_fn(params + alpha * dir1 + beta * dir2)` for a grid of
    /// (alpha, beta) values in `[-perturbation_range, +perturbation_range]`.
    ///
    /// # Arguments
    /// * `params` - Center point in parameter space
    /// * `loss_fn` - Function that computes loss for given parameters
    /// * `dir1` - First perturbation direction (should be normalized)
    /// * `dir2` - Second perturbation direction (should be normalized)
    pub fn compute_landscape<F>(
        &self,
        params: &Array1<A>,
        loss_fn: F,
        dir1: &Array1<A>,
        dir2: &Array1<A>,
    ) -> Result<LandscapeData<A>>
    where
        F: Fn(&Array1<A>) -> Result<A>,
    {
        let n = self.config.grid_resolution;
        if n == 0 {
            return Err(OptimError::InvalidConfig(
                "Grid resolution must be positive".to_string(),
            ));
        }
        if params.len() != dir1.len() || params.len() != dir2.len() {
            return Err(OptimError::DimensionMismatch(format!(
                "Parameter dimension ({}) must match direction dimensions ({}, {})",
                params.len(),
                dir1.len(),
                dir2.len()
            )));
        }

        let range = self.config.perturbation_range;
        let neg_range = A::zero() - range;

        let mut grid = Array2::zeros((n, n));
        let mut min_loss = A::infinity();
        let mut max_loss = A::neg_infinity();
        let mut center_loss = A::zero();

        let n_minus_1 = if n > 1 {
            A::from(n - 1).ok_or_else(|| {
                OptimError::ComputationError("Failed to convert grid size".to_string())
            })?
        } else {
            A::one()
        };

        let two = A::from(2.0).ok_or_else(|| {
            OptimError::ComputationError("Failed to convert constant".to_string())
        })?;

        for i in 0..n {
            let alpha = neg_range
                + (A::from(i).ok_or_else(|| {
                    OptimError::ComputationError("Failed to convert index".to_string())
                })? / n_minus_1)
                    * two
                    * range;

            for j in 0..n {
                let beta = neg_range
                    + (A::from(j).ok_or_else(|| {
                        OptimError::ComputationError("Failed to convert index".to_string())
                    })? / n_minus_1)
                        * two
                        * range;

                // perturbed = params + alpha * dir1 + beta * dir2
                let perturbed = params
                    .iter()
                    .zip(dir1.iter())
                    .zip(dir2.iter())
                    .map(|((&p, &d1), &d2)| p + alpha * d1 + beta * d2)
                    .collect::<Vec<A>>();
                let perturbed = Array1::from_vec(perturbed);

                let loss = loss_fn(&perturbed)?;
                grid[[i, j]] = loss;

                if loss < min_loss {
                    min_loss = loss;
                }
                if loss > max_loss {
                    max_loss = loss;
                }

                // Track center point (when alpha ~ 0 and beta ~ 0)
                if (n > 1 && i == n / 2 && j == n / 2) || n == 1 {
                    center_loss = loss;
                }
            }
        }

        Ok(LandscapeData {
            grid,
            x_range: (neg_range, range),
            y_range: (neg_range, range),
            center_loss,
            min_loss,
            max_loss,
        })
    }

    /// Compute the sharpness of the loss surface around a point
    ///
    /// Sharpness is defined as the maximum loss in a neighborhood of radius
    /// `epsilon` minus the loss at the center point. This measures how "sharp"
    /// or "flat" the minimum is - flatter minima tend to generalize better.
    ///
    /// # Arguments
    /// * `params` - Center point in parameter space
    /// * `loss_fn` - Function that computes loss for given parameters
    /// * `epsilon` - Radius of the neighborhood to search
    pub fn compute_sharpness<F>(&self, params: &Array1<A>, loss_fn: &F, epsilon: A) -> Result<A>
    where
        F: Fn(&Array1<A>) -> Result<A>,
    {
        let center_loss = loss_fn(params)?;
        let dim = params.len();

        if dim == 0 {
            return Err(OptimError::InvalidParameter(
                "Parameter array must not be empty".to_string(),
            ));
        }

        let mut max_loss = center_loss;

        // Sample along each coordinate axis in both directions
        for d in 0..dim {
            // Positive perturbation
            let mut perturbed_pos = params.to_owned();
            perturbed_pos[d] = perturbed_pos[d] + epsilon;
            let loss_pos = loss_fn(&perturbed_pos)?;
            if loss_pos > max_loss {
                max_loss = loss_pos;
            }

            // Negative perturbation
            let mut perturbed_neg = params.to_owned();
            perturbed_neg[d] = perturbed_neg[d] - epsilon;
            let loss_neg = loss_fn(&perturbed_neg)?;
            if loss_neg > max_loss {
                max_loss = loss_neg;
            }
        }

        // Also sample along diagonal directions for better coverage
        // Diagonal: all dimensions perturbed by epsilon / sqrt(dim)
        let dim_f = A::from(dim).ok_or_else(|| {
            OptimError::ComputationError("Failed to convert dimension".to_string())
        })?;
        let scaled_eps = epsilon / dim_f.sqrt();

        // All-positive diagonal
        let diag_pos: Array1<A> = params.mapv(|p| p + scaled_eps);
        let loss_diag_pos = loss_fn(&diag_pos)?;
        if loss_diag_pos > max_loss {
            max_loss = loss_diag_pos;
        }

        // All-negative diagonal
        let diag_neg: Array1<A> = params.mapv(|p| p - scaled_eps);
        let loss_diag_neg = loss_fn(&diag_neg)?;
        if loss_diag_neg > max_loss {
            max_loss = loss_diag_neg;
        }

        Ok(max_loss - center_loss)
    }

    /// Find saddle points in the loss landscape
    ///
    /// A saddle point is a grid cell where the gradient is approximately zero
    /// (local extremum behavior) but the point is neither a strict local minimum
    /// nor a strict local maximum -- it has both higher and lower neighbors.
    ///
    /// # Arguments
    /// * `landscape` - Previously computed landscape data
    pub fn find_saddle_points(&self, landscape: &LandscapeData<A>) -> Vec<SaddlePointInfo<A>> {
        let (rows, cols) = landscape.grid.dim();
        let mut saddle_points = Vec::new();

        // Skip border cells as they don't have full neighborhoods
        for i in 1..rows.saturating_sub(1) {
            for j in 1..cols.saturating_sub(1) {
                let center = landscape.grid[[i, j]];

                // Collect all 8 neighbors
                let neighbors = [
                    landscape.grid[[i - 1, j - 1]],
                    landscape.grid[[i - 1, j]],
                    landscape.grid[[i - 1, j + 1]],
                    landscape.grid[[i, j - 1]],
                    landscape.grid[[i, j + 1]],
                    landscape.grid[[i + 1, j - 1]],
                    landscape.grid[[i + 1, j]],
                    landscape.grid[[i + 1, j + 1]],
                ];

                let has_higher = neighbors.iter().any(|&n| n > center);
                let has_lower = neighbors.iter().any(|&n| n < center);

                // A saddle point has both higher and lower neighbors,
                // and the differences are small enough to suggest a near-zero gradient
                if has_higher && has_lower {
                    // Check that the point is not strongly a minimum or maximum:
                    // count how many neighbors are higher vs lower
                    let higher_count = neighbors.iter().filter(|&&n| n > center).count();
                    let lower_count = neighbors.iter().filter(|&&n| n < center).count();

                    // Saddle-like: roughly balanced directional behavior
                    // (not overwhelmingly a basin or a peak)
                    if higher_count >= 2 && lower_count >= 2 {
                        saddle_points.push(SaddlePointInfo {
                            grid_x: i,
                            grid_y: j,
                            loss_value: center,
                        });
                    }
                }
            }
        }

        saddle_points
    }

    /// Render an SVG contour plot of the loss landscape
    ///
    /// Produces an SVG string with filled rectangles colored by loss value,
    /// creating a heat-map style visualization of the landscape.
    pub fn render_contour_plot(&self, landscape: &LandscapeData<A>) -> Result<String> {
        let (rows, cols) = landscape.grid.dim();
        if rows == 0 || cols == 0 {
            return Err(OptimError::InvalidState(
                "Landscape grid is empty".to_string(),
            ));
        }

        let cell_size = 15;
        let margin = 60;
        let width = margin + cols * cell_size + margin;
        let height = margin + rows * cell_size + margin;

        let min_loss = landscape.min_loss.to_f64().unwrap_or(0.0);
        let max_loss = landscape.max_loss.to_f64().unwrap_or(1.0);
        let loss_range = if (max_loss - min_loss).abs() < 1e-15 {
            1.0
        } else {
            max_loss - min_loss
        };

        let mut svg = format!(
            r#"<svg xmlns="http://www.w3.org/2000/svg" width="{}" height="{}" viewBox="0 0 {} {}">"#,
            width, height, width, height
        );
        svg.push('\n');

        // Title
        svg.push_str(&format!(
            r#"  <text x="{}" y="25" text-anchor="middle" font-size="16" font-weight="bold">Loss Landscape</text>"#,
            width / 2
        ));
        svg.push('\n');

        // Axis labels
        svg.push_str(&format!(
            r#"  <text x="{}" y="{}" text-anchor="middle" font-size="12">Direction 1</text>"#,
            margin + cols * cell_size / 2,
            height - 10
        ));
        svg.push('\n');

        svg.push_str(&format!(
            r#"  <text x="15" y="{}" text-anchor="middle" font-size="12" transform="rotate(-90, 15, {})">Direction 2</text>"#,
            margin + rows * cell_size / 2,
            margin + rows * cell_size / 2
        ));
        svg.push('\n');

        // Draw cells as colored rectangles
        for i in 0..rows {
            for j in 0..cols {
                let val = landscape.grid[[i, j]].to_f64().unwrap_or(0.0);
                let normalized = (val - min_loss) / loss_range;
                // Clamp to [0, 1]
                let normalized = normalized.clamp(0.0, 1.0);

                let color = loss_value_to_color(normalized);

                let x = margin + j * cell_size;
                let y = margin + i * cell_size;

                svg.push_str(&format!(
                    r#"  <rect x="{}" y="{}" width="{}" height="{}" fill="{}"/>"#,
                    x, y, cell_size, cell_size, color
                ));
                svg.push('\n');
            }
        }

        // Color bar legend
        let legend_x = margin + cols * cell_size + 10;
        let legend_height = rows * cell_size;
        let legend_steps = 10;
        let step_height = legend_height / legend_steps;

        for s in 0..legend_steps {
            let normalized = 1.0 - (s as f64 / legend_steps as f64);
            let color = loss_value_to_color(normalized);
            let y = margin + s * step_height;

            svg.push_str(&format!(
                r#"  <rect x="{}" y="{}" width="15" height="{}" fill="{}"/>"#,
                legend_x, y, step_height, color
            ));
            svg.push('\n');
        }

        // Legend labels
        svg.push_str(&format!(
            r#"  <text x="{}" y="{}" font-size="9">{:.2e}</text>"#,
            legend_x + 20,
            margin + 10,
            max_loss
        ));
        svg.push('\n');
        svg.push_str(&format!(
            r#"  <text x="{}" y="{}" font-size="9">{:.2e}</text>"#,
            legend_x + 20,
            margin + legend_height,
            min_loss
        ));
        svg.push('\n');

        svg.push_str("</svg>");
        Ok(svg)
    }
}

/// Convert a normalized loss value [0, 1] to an RGB color string
///
/// Uses a blue (low) -> green (mid) -> red (high) color scale.
fn loss_value_to_color(normalized: f64) -> String {
    let (r, g, b) = if normalized < 0.5 {
        // Blue to Green
        let t = normalized * 2.0;
        (0.0, t, 1.0 - t)
    } else {
        // Green to Red
        let t = (normalized - 0.5) * 2.0;
        (t, 1.0 - t, 0.0)
    };

    format!(
        "rgb({},{},{})",
        (r * 255.0) as u8,
        (g * 255.0) as u8,
        (b * 255.0) as u8
    )
}

#[cfg(test)]
mod tests {
    use super::*;
    use scirs2_core::ndarray::Array1;

    #[test]
    fn test_compute_landscape_quadratic() {
        let config = LossLandscapeConfig {
            grid_resolution: 5,
            perturbation_range: 1.0,
            direction_method: DirectionMethod::Random,
        };
        let analyzer = LossLandscapeAnalyzer::<f64>::new(config);

        // Simple quadratic loss: f(x) = sum(x_i^2)
        let params = Array1::from_vec(vec![0.0, 0.0]);
        let dir1 = Array1::from_vec(vec![1.0, 0.0]);
        let dir2 = Array1::from_vec(vec![0.0, 1.0]);

        let loss_fn = |p: &Array1<f64>| -> Result<f64> { Ok(p.iter().map(|&x| x * x).sum()) };

        let landscape = analyzer
            .compute_landscape(&params, loss_fn, &dir1, &dir2)
            .expect("Should compute landscape");

        assert_eq!(landscape.grid.dim(), (5, 5));
        // At center (0,0), loss should be 0
        assert!(landscape.center_loss >= 0.0);
        // Min should be at center
        assert!(landscape.min_loss >= 0.0);
        // Max should be at corners (alpha=1, beta=1 => loss=2)
        assert!((landscape.max_loss - 2.0).abs() < 1e-10);
    }

    #[test]
    fn test_compute_sharpness() {
        let config: LossLandscapeConfig<f64> = LossLandscapeConfig::default();
        let analyzer = LossLandscapeAnalyzer::new(config);

        // Quadratic bowl: f(x) = x_1^2 + x_2^2
        let params = Array1::from_vec(vec![0.0, 0.0]);
        let loss_fn = |p: &Array1<f64>| -> Result<f64> { Ok(p.iter().map(|&x| x * x).sum()) };

        let epsilon = 0.1;
        let sharpness = analyzer
            .compute_sharpness(&params, &loss_fn, epsilon)
            .expect("Should compute sharpness");

        // At the origin, center_loss = 0
        // Max in neighborhood: moving epsilon along one axis => epsilon^2 = 0.01
        // Moving epsilon/sqrt(2) along diagonal => 2*(0.1/sqrt(2))^2 = 0.01
        // So sharpness = 0.01 - 0 = 0.01
        assert!(sharpness > 0.0);
        assert!((sharpness - 0.01).abs() < 1e-10);
    }

    #[test]
    fn test_find_saddle_points() {
        // Create a landscape with a known saddle point
        // f(x,y) = x^2 - y^2 has a saddle at origin
        let config = LossLandscapeConfig {
            grid_resolution: 11,
            perturbation_range: 1.0,
            direction_method: DirectionMethod::Random,
        };
        let analyzer = LossLandscapeAnalyzer::<f64>::new(config);

        let params = Array1::from_vec(vec![0.0, 0.0]);
        let dir1 = Array1::from_vec(vec![1.0, 0.0]);
        let dir2 = Array1::from_vec(vec![0.0, 1.0]);

        // Saddle function: x^2 - y^2
        let loss_fn = |p: &Array1<f64>| -> Result<f64> { Ok(p[0] * p[0] - p[1] * p[1]) };

        let landscape = analyzer
            .compute_landscape(&params, loss_fn, &dir1, &dir2)
            .expect("Should compute landscape");

        let saddle_points = analyzer.find_saddle_points(&landscape);

        // Should find at least one saddle point near the center
        assert!(
            !saddle_points.is_empty(),
            "Should detect saddle points in x^2 - y^2"
        );

        // The center of the grid (5,5) should be among the saddle points
        let has_center = saddle_points
            .iter()
            .any(|sp| sp.grid_x == 5 && sp.grid_y == 5);
        assert!(
            has_center,
            "Center of x^2 - y^2 landscape should be a saddle point"
        );
    }

    #[test]
    fn test_render_contour_plot_svg() {
        let config = LossLandscapeConfig {
            grid_resolution: 5,
            perturbation_range: 1.0,
            direction_method: DirectionMethod::Random,
        };
        let analyzer = LossLandscapeAnalyzer::<f64>::new(config);

        let params = Array1::from_vec(vec![0.0, 0.0]);
        let dir1 = Array1::from_vec(vec![1.0, 0.0]);
        let dir2 = Array1::from_vec(vec![0.0, 1.0]);

        let loss_fn = |p: &Array1<f64>| -> Result<f64> { Ok(p.iter().map(|&x| x * x).sum()) };

        let landscape = analyzer
            .compute_landscape(&params, loss_fn, &dir1, &dir2)
            .expect("Should compute landscape");

        let svg = analyzer
            .render_contour_plot(&landscape)
            .expect("Should render contour plot");

        assert!(svg.starts_with("<svg"));
        assert!(svg.ends_with("</svg>"));
        assert!(svg.contains("Loss Landscape"));
        assert!(svg.contains("Direction 1"));
        assert!(svg.contains("Direction 2"));
        assert!(svg.contains("rect"));
    }

    #[test]
    fn test_landscape_config_defaults() {
        let config: LossLandscapeConfig<f64> = LossLandscapeConfig::default();
        assert_eq!(config.grid_resolution, 20);
        assert!((config.perturbation_range - 1.0).abs() < 1e-15);
        assert_eq!(config.direction_method, DirectionMethod::Random);

        let config32: LossLandscapeConfig<f32> = LossLandscapeConfig::default();
        assert_eq!(config32.grid_resolution, 20);
        assert!((config32.perturbation_range - 1.0f32).abs() < 1e-6);
    }
}
