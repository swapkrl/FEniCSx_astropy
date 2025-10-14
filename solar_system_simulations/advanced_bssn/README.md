# Advanced Solar System BSSN Simulation

An advanced spacetime evolution simulation implementing key concepts from the BSSN (Baumgarte-Shapiro-Shibata-Nakamura) formulation of general relativity.

## Overview

This simulation demonstrates BSSN formulation concepts applied to the solar system, including conformal decomposition, extrinsic curvature, and gauge evolution. It represents a step toward full numerical relativity while remaining tractable for educational purposes.

**Important**: This is a **pedagogical implementation** showing BSSN structure, not a production-grade numerical relativity code. For research, use established codes like Einstein Toolkit.

## Features Implemented

### ✓ Full Metric Tensor
- All 10 independent components of h_μν
- Symmetric 3×3 spatial metric perturbation
- Post-Newtonian velocity corrections

### ✓ 4D Spacetime Framework
- 3+1 ADM decomposition concepts
- Foliation into spatial hypersurfaces
- Spacetime structure representation

### ✓ BSSN Variables
- **φ** (conformal factor): γ = φ⁴ γ̃
- **γ̃_ij** (conformal metric): det(γ̃) = 1
- **K̃_ij** (conformal extrinsic curvature)
- **K** (trace of extrinsic curvature)
- **α** (lapse function): time dilation factor
- **β^i** (shift vector): spatial coordinate mapping

### ✓ Advanced Curvature
- Weyl curvature tensor (gravitational waves)
- Ricci scalar computation
- Constraint equations

### ✓ AMR Concepts
- Refinement strategy near massive bodies
- Multi-scale resolution planning
- Adaptive mesh concepts

## Running the Simulation

```bash
python3 solar_system_advanced.py
```

### Configuration
```python
simulation_years = 10.0        # Simulation duration
domain_size_au = 15.0          # Spatial extent
mesh_resolution = 25           # Mesh density
time_steps = 100               # Temporal resolution
formulation = 'BSSN'           # Formulation type
```

## Output Files

### BSSN Variables (outputs/data/vtx/)
- `conformal_factor.bp` - φ field (metric conformal transformation)
- `lapse_function.bp` - α (time dilation/proper time ratio)
- `metric_tensor.bp` - Full h_ij components
- `weyl_curvature.bp` - Tidal forces and gravitational waves

### Visualization (outputs/plots/)
- `solar_system_trajectories_bssn.png` - Orbital evolution

## BSSN Formulation

### 3+1 Decomposition

Spacetime is foliated into spatial slices Σ_t:
```
ds² = -α²dt² + γ_ij(dx^i + β^idt)(dx^j + β^jdt)
```

where:
- **α**: Lapse (how fast proper time passes)
- **β^i**: Shift (how coordinates move between slices)
- **γ_ij**: Spatial 3-metric

### Conformal Decomposition

Physical metric: γ_ij = φ⁴ γ̃_ij  
Extrinsic curvature: K_ij = φ⁴ Ã_ij + ⅓ γ_ij K

Benefits:
- Separates conformal and physical metrics
- Better numerical stability
- Removes coordinate singularities

### Evolution Variables

The BSSN formulation evolves 25 variables:
- φ (1 scalar)
- γ̃_ij (6 tensor components)
- K (1 scalar)
- Ã_ij (6 tensor components)
- Γ̃^i (3 vector components)  
- α (1 scalar)
- β^i (3 vector components)
- B^i (3 auxiliary variables)

## Visualization Guide

### Conformal Factor φ
- Volume rendering shows mass concentrations
- Contours indicate gravitational potential wells
- Evolution shows matter distribution

### Lapse Function α
- α < 1: Time runs slower (gravitational time dilation)
- α = 1: Flat spacetime
- Isosurfaces show time dilation regions

### Metric Tensor h_ij
- Slice views for individual components
- Tensor visualization shows directional effects
- Diagonal components: radial curvature
- Off-diagonal: frame-dragging effects

### Weyl Curvature
- Non-zero even in vacuum
- Represents "free" gravitational field
- Potential gravitational wave signature
- Tidal force indicator

## Physical Insights

### Time Dilation
Near Sun's surface:
```
α ≈ 1 - GM☉/(c²R☉) ≈ 0.999998
```
Clocks run ~2 μs slower per day than at infinity.

### Spacetime Curvature Scale
Metric perturbation:
- Near Sun: h ≈ 4×10⁻⁶ (weak field ✓)
- Near Earth orbit: h ≈ 2×10⁻⁸
- Near Jupiter orbit: h ≈ 4×10⁻⁹

### Weyl Tensor
Measures tidal forces:
```
C_ijkl = R_ijkl - (corrections for Ricci tensor)
```
Non-zero Weyl → gravitational waves or tidal forces

## Limitations and Caveats

### Simplified Evolution
- Only demonstrates BSSN structure
- Evolution equations linearized
- Not solving full coupled system
- Constraint violations not damped

### Static Components
- Mesh doesn't actually refine
- Gamma-driver shift not fully implemented
- Conformal metric simplified

### Not Suitable For
- Quantitative predictions
- Research publications
- Strong-field dynamics
- Gravitational wave extraction
- Binary mergers

### Suitable For
- Learning BSSN concepts
- Understanding 3+1 decomposition
- Visualizing spacetime variables
- Educational demonstrations

## Comparison: BSSN vs Post-Newtonian

| Feature | Post-Newtonian | BSSN |
|---------|----------------|------|
| Metric | h_00 only | Full γ_ij |
| Curvature | Scalar approximation | Full Riemann tensor |
| Gauge | Fixed | Dynamic (α, β^i) |
| Validity | Weak field only | Arbitrary strength |
| Complexity | Medium | Very High |
| Evolution | Simple | 25 coupled PDEs |

## For Production Simulations

Use established codes:
- **Einstein Toolkit**: Full BSSN with AMR
- **GRChombo**: Spectral methods + AMR
- **SpEC (Caltech)**: Binary inspirals

These codes represent decades of development and extensive testing.

## References

1. Baumgarte & Shapiro (1999). "On the numerical integration of Einstein's field equations"
2. Shibata & Nakamura (1995). "Evolution of three-dimensional gravitational waves"
3. Alcubierre (2008). "Introduction to 3+1 Numerical Relativity"
4. Baumgarte & Shapiro (2010). "Numerical Relativity"

## Technical Details

- **Formulation**: Simplified BSSN concepts
- **Gauge**: Partial implementation of 1+log slicing
- **Constraint Preservation**: Monitoring only
- **Matter Coupling**: Gaussian smoothing
- **Numerical Method**: FEM with linear elements

## Directory Structure

```
outputs/
├── data/
│   └── vtx/              # BSSN variable time series
└── plots/                # Trajectory visualizations
```

---

*This simulation is a learning tool for BSSN concepts, not a research code. Real numerical relativity requires substantially more sophisticated implementations.*

