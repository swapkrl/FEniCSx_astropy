# Solar System Post-Newtonian Spacetime Evolution

A FEniCS-based simulation of solar system dynamics with curved spacetime geometry using Post-Newtonian approximations of General Relativity.

## Overview

This simulation models the solar system (Sun + 7 planets) as a dynamical spacetime where gravity manifests as curvature. It goes beyond Newtonian gravity by including relativistic corrections that account for effects like Mercury's perihelion precession.

## Key Features

- **Real Astronomical Data**: Accurate masses, orbits, eccentricities, and inclinations
- **Post-Newtonian Corrections**: Includes v²/c² relativistic effects
- **Curved Spacetime**: Computes metric perturbations h_μν
- **Time Evolution**: Tracks geometry changes as planets orbit
- **Multiple Fields**:
  - Metric perturbation (spacetime curvature)
  - Ricci curvature scalar (matter distribution)
  - Gravitational field vectors
- **Kepler Orbits**: Accurate orbital mechanics with eccentric orbits

## Included Bodies

| Body    | Mass (kg)   | Semi-Major Axis | Orbital Period | Eccentricity |
|---------|-------------|-----------------|----------------|--------------|
| Sun     | 1.989×10³⁰  | 0 AU            | -              | -            |
| Mercury | 3.301×10²³  | 0.387 AU        | 87.97 days     | 0.2056       |
| Venus   | 4.867×10²⁴  | 0.723 AU        | 224.70 days    | 0.0068       |
| Earth   | 5.972×10²⁴  | 1.000 AU        | 365.26 days    | 0.0167       |
| Mars    | 6.417×10²³  | 1.524 AU        | 686.98 days    | 0.0934       |
| Jupiter | 1.898×10²⁷  | 5.203 AU        | 11.86 years    | 0.0484       |
| Saturn  | 5.683×10²⁶  | 9.537 AU        | 29.46 years    | 0.0539       |

## Running the Simulation

```bash
python3 solar_system_spacetime.py
```

### Configuration
```python
simulation_years = 10.0        # Duration in Earth years
domain_size_au = 15.0          # Domain extent (AU)
mesh_resolution = 25           # Cells per dimension
time_steps = 100               # Number of timesteps
```

## Output Files

### Data Files (outputs/data/vtx/)
- `metric_perturbation.bp` - Spacetime curvature field h_00
- `ricci_curvature.bp` - Curvature scalar R
- `gravitational_field.bp` - Gravitational acceleration vectors

### Plots (outputs/plots/)
- `solar_system_trajectories.png` - 3D view of all planetary orbits
- `inner_planets_orbits.png` - Detailed view of Mercury, Venus, Earth, Mars

## Physical Principles

### From Newton to Einstein

**Newtonian**: Gravity is a force F = GMm/r²

**Einsteinian**: Gravity is curved spacetime geometry

### Post-Newtonian Approximation

The metric is written as:
```
g_μν = η_μν + h_μν
```

where η_μν is flat Minkowski spacetime and h_μν represents curvature.

For weak fields:
```
h_00 ≈ 2Φ/c²
```

where Φ is the Newtonian potential with corrections:
```
Φ_PN = Φ_Newton + (v²/2c²)Φ_Newton + (GM/rc²)Φ_Newton
```

### Observable Effects

**Mercury's Perihelion Precession**:  
Predicted: 43 arcseconds per century beyond Newtonian  
Observed: 43.1 ± 0.5 arcsec/century ✓

**Gravitational Time Dilation**:
Clocks run slower near massive bodies by factor:
```
Δt = Δt₀√(1 - 2GM/rc²)
```

## Visualization in ParaView

### Metric Perturbation
Shows how Sun and planets "dent" spacetime:
- Dark blue = strong curvature (time dilation)
- Red = weak curvature
- Sun creates deepest "well"
- Jupiter creates secondary well

### Ricci Curvature
Proportional to matter density via Einstein equations:
```
R_μν - ½Rg_μν = (8πG/c⁴)T_μν
```

### Field Lines
Stream tracers show geodesic paths that objects follow in curved spacetime.

## Mathematical Details

### Kepler's Equation
Orbital position requires solving:
```
M = E - e sin(E)
```
where M is mean anomaly, E is eccentric anomaly, e is eccentricity.

### Metric Components
The simulation computes:
- Temporal component: g_00 = -1 + 2Φ/c²
- Spatial components: g_ij = δ_ij(1 - 2Φ/c²)

### Einstein Constraint
The Hamiltonian constraint relates curvature to matter:
```
∇²h_00 = (8πG/c⁴)ρ
```

## Limitations

- Post-Newtonian approximation (valid for v << c, weak fields)
- No gravitational waves
- Static mesh (not adaptive)
- Simplified matter treatment
- Not full general relativity

For more rigorous implementations, see the BSSN formulation simulations.

## Technical Specifications

- **Formulation**: Post-Newtonian (PN) approximation
- **Spatial Domain**: ±15 AU cube
- **Temporal Domain**: 0-10 years
- **FEM Elements**: Lagrange P1 (linear)
- **Solvers**: CG with Hypre preconditioner
- **Output Format**: ADIOS2 BP4 (ParaView compatible)

## References

1. Poisson & Will (2014). "Gravity: Newtonian, Post-Newtonian, Relativistic"
2. Will, C.M. (2014). "The Confrontation between General Relativity and Experiment"
3. Einstein, A. (1915). "The Field Equations of Gravitation"

## Directory Structure

```
outputs/
├── data/
│   └── vtx/              # Time series VTX data
├── plots/                # Static PNG plots
└── [auto-generated ParaView scripts]
```

