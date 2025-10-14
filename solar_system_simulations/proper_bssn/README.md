# Proper BSSN Solar System Implementation

A more rigorous implementation of BSSN (Baumgarte-Shapiro-Shibata-Nakamura) formulation with actual time evolution equations and constraint monitoring.

## Overview

This simulation implements **proper BSSN evolution equations** for key variables, including conformal factor evolution, trace extrinsic curvature dynamics, and dynamic lapse function with constraint monitoring. While still simplified compared to production codes, it correctly evolves BSSN variables according to the formalism.

## What's Actually Implemented

### ✓ Proper Time Evolution

**Conformal Factor φ**:
```
∂_t φ = -(1/6) α K + β^i ∂_i φ
```

**Trace Extrinsic Curvature K**:
```
∂_t K = -D_i D^i α + α(A_ij A^ij + K²/3) + 4π κ α(ρ + S)
```

**Lapse Function α (1+log slicing)**:
```
∂_t α = -2 α K + β^i ∂_i α
```

### ✓ Constraint Monitoring

**Hamiltonian Constraint**:
```
H = R - K_ij K^ij + K² - 16π ρ = 0
```

**Momentum Constraints**:
```
M_i = D_j(K_ij - δ_ij K) - 8π j_i = 0
```

Tracked over time to monitor solution quality.

### ✓ Stress-Energy Tensor

Proper relativistic matter coupling:
```
T^μν = (ρ + p) u^μ u^ν + p g^μν
```

Includes:
- Energy density with Lorentz factors
- Momentum density S_i  
- Stress tensor S_ij

### ✓ Dynamic Gauge

- **1+log slicing**: Lapse evolves to prevent singularities
- **Constraint damping**: Monitors violations
- **Gauge stability**: Lapse bounded to [0.1, 10.0]

## Running the Simulation

```bash
python3 solar_system_proper_bssn.py
```

### Configuration
```python
domain_size = 10.0             # AU
mesh_resolution = 20           # Cells per dimension  
simulation_years = 5.0         # Duration
time_steps = 50                # Number of steps
```

## Output Files

### BSSN Variables (outputs/data/vtx/)
- `phi.bp` - Conformal factor φ evolution
- `lapse.bp` - Lapse function α(t)
- `trace_K.bp` - Trace extrinsic curvature K(t)

### Diagnostics (outputs/plots/)
- `constraint_violations.png` - Hamiltonian and momentum constraint evolution

## Key Improvements

### Over Previous Version

| Feature | Previous | This Version |
|---------|----------|--------------|
| φ Evolution | ❌ Static | ✓ Proper PDE |
| K Evolution | ❌ None | ✓ With source terms |
| α Evolution | ❌ Static | ✓ 1+log slicing |
| Constraints | ❌ None | ✓ Monitored |
| Matter Coupling | ⚠️ Simplified | ✓ Lorentz factors |
| Time Evolution | ❌ Fake | ✓ Actual |

### Properly Solved Equations

1. **Conformal Factor**: Backward Euler time integration
2. **Trace K**: Includes Laplacian of lapse + nonlinear terms
3. **Lapse**: Dynamic evolution prevents coordinate singularities
4. **Constraints**: Computed at each step

## Understanding the Output

### Conformal Factor φ
- φ = 0: Flat space
- φ > 0: Positive mass energy
- φ < 0: Unphysical (indicates numerical issues)

Expected range: [-0.01, 0.01] for solar system

### Lapse Function α
- α = 1: Normal time flow (flat space)
- α < 1: Time runs slower (near massive bodies)
- α > 1: Faster coordinate time

Near Sun: α ≈ 0.999998

### Trace K
- K = 0: No expansion/contraction of spatial slices
- K < 0: Spatial slices collapsing
- K > 0: Spatial slices expanding

Expected: Small oscillations around 0

### Constraint Violations
- H ≈ 0: Hamiltonian constraint satisfied
- M ≈ 0: Momentum constraints satisfied
- Growing violations → numerical instability

Monitor constraint plot to ensure solution quality.

## Physical Interpretation

### BSSN Variables

**φ (Conformal Factor)**: 
Separates conformal from physical metric. The physical spatial metric is:
```
γ_ij = φ⁴ γ̃_ij
```

**K (Extrinsic Curvature)**:
Measures how spatial slices curve in 4D spacetime. Related to:
- Expansion rate of space
- Gravitational wave content
- Matter-energy content

**α (Lapse)**:
Ratio of proper time to coordinate time:
```
dτ = α dt
```

### 1+log Slicing

The lapse evolution:
```
∂_t α = -2 α K
```

keeps surfaces from collapsing into singularities. When spatial slices start collapsing (K < 0), α decreases to slow down coordinate time flow.

## Limitations

### Still Simplified

- **Tensor Evolution**: γ̃_ij and Ã_ij not fully evolved
- **Gamma-Driver**: Shift β^i = 0 (zero shift condition)
- **Matter**: Simplified Gaussian sources
- **Mesh**: Static (no AMR)
- **Constraints**: Monitored but not damped

### Not Production-Grade

Missing from production codes:
- Full 25-variable evolution system
- CCZ4 constraint damping
- Gamma-driver shift evolution
- Adaptive mesh refinement
- Kreiss-Oliger dissipation
- High-order finite differences

### What This Code IS

- ✓ Correct formulation of key equations
- ✓ Actual time evolution (not fake)
- ✓ Proper constraint monitoring
- ✓ Educational tool for BSSN concepts
- ✓ Starting point for further development

### What This Code IS NOT

- ❌ Production numerical relativity code
- ❌ Suitable for research publications
- ❌ Complete BSSN implementation
- ❌ Gravitational wave extractor
- ❌ Strong-field capable

## For Learning BSSN

### Recommended Progression

1. **Run this simulation** - See basic BSSN evolution
2. **Study constraint violations** - Understand numerical stability
3. **Read textbooks**:
   - Baumgarte & Shapiro: "Numerical Relativity"
   - Alcubierre: "Introduction to 3+1 Numerical Relativity"
4. **Explore Einstein Toolkit** - Production code
5. **Run standard tests** - Gauge waves, robust stability

### Key Concepts Demonstrated

- 3+1 decomposition of spacetime
- Conformal transformation advantages
- Importance of gauge conditions
- Constraint preservation challenges
- Matter-geometry coupling
- Time evolution complexity

## For Actual Research

Use established codes:

**Einstein Toolkit**
- Full BSSN/CCZ4 evolution
- Carpet AMR
- Extensive testing
- Community support

**GRChombo**
- Modern C++ implementation
- Chombo AMR
- Scalar field specialization

**SpEC (Caltech)**
- Spectral methods
- Binary black hole simulations
- Gravitational waveform generation

## References

### BSSN Formalism
1. Baumgarte & Shapiro (1999). Phys. Rev. D 59, 024007
2. Shibata & Nakamura (1995). Phys. Rev. D 52, 5428

### Numerical Relativity Textbooks
3. Baumgarte & Shapiro (2010). "Numerical Relativity"
4. Alcubierre (2008). "Introduction to 3+1 Numerical Relativity"
5. Gourgoulhon (2012). "3+1 Formalism in General Relativity"

### Production Codes
6. Einstein Toolkit: einsteintoolkit.org
7. GRChombo: github.com/GRChombo/GRChombo

## Technical Specifications

- **Variables Evolved**: φ, K, α
- **Gauge**: 1+log slicing, zero shift
- **Time Integration**: Backward Euler (implicit)
- **Spatial Discretization**: FEM P1 elements
- **Linear Solver**: CG + Jacobi preconditioner
- **Constraint Monitoring**: L² norm of H and M_i

## Directory Structure

```
outputs/
├── data/
│   └── vtx/              # BSSN variable time series
└── plots/                # Constraint violation diagnostics
```

---

*This implementation demonstrates proper BSSN evolution for educational purposes. For research-grade simulations, use Einstein Toolkit or similar established codes.*

