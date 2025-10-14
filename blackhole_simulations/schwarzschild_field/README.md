# Schwarzschild Black Hole Gravitational Field Simulation

A FEniCS-based simulation of the time-evolving gravitational field around a static Schwarzschild black hole using finite element methods.

## Overview

This simulation models the gravitational potential field in the vicinity of a black hole with Schwarzschild geometry. It solves the time-dependent Poisson equation in 3D space to compute how the gravitational field evolves over time with periodic perturbations.

## Features

- **3D Spatial Mesh**: Hexahedral finite element discretization
- **Time Evolution**: Solves time-dependent field equations with perturbations
- **Schwarzschild Geometry**: Uses Schwarzschild radius as the fundamental length scale
- **Multiple Field Outputs**: 
  - Gravitational potential
  - Field magnitude  
  - Energy density distribution
  - Curvature-like quantities
  - 3D vector field (gravitational force direction)
- **ParaView Compatible**: Exports VTX and XDMF formats for visualization
- **Test Particles**: Simulates particle trajectories in the gravitational field

## Physical Parameters

- **Schwarzschild Radius**: rs = 2GM/c² = 2.0 (geometric units)
- **Domain Size**: ±20 rs
- **Mesh Resolution**: 30³ cells (configurable)
- **Time Steps**: 50 steps
- **Time Step Size**: 0.1 time units

## Output Files

All outputs are saved in the `outputs/` subdirectory:

### Data Files
- `outputs/data/vtx/gravitational_field_potential.bp` - Potential field time series
- `outputs/data/vtx/gravitational_field_scalar_fields.bp` - Magnitude, energy, curvature
- `outputs/data/vtx/gravitational_field_vector_field.bp` - Force vector field
- `outputs/data/xdmf/gravitational_field.xdmf` - All fields in XDMF format

### Visualizations
- `outputs/visualizations/plots/energy_evolution.png` - Field energy over time
- `outputs/visualizations/plots/particle_trajectories.png` - Test particle orbits

### ParaView Scripts
- `outputs/visualizations/paraview/paraview_script.py` - Auto-setup script
- `outputs/visualizations/paraview/PARAVIEW_GUIDE.md` - Visualization guide

## Running the Simulation

### Prerequisites
```bash
# Requires DOLFINx (FEniCSx) installation
# See: https://github.com/FEniCS/dolfinx
```

### Execute
```bash
python3 blackhole_simulation.py
```

### Configuration
Edit parameters in the `main()` function:
```python
schwarzschild_radius = 2.0      # rs in geometric units
domain_size = 20.0              # Domain extent (×rs)
mesh_resolution = 30            # Cells per dimension
time_steps = 50                 # Number of timesteps
dt = 0.1                        # Time step size
```

## Visualization in ParaView

### Quick Start
1. Open ParaView
2. Load: `outputs/data/vtx/gravitational_field_scalar_fields.bp`
3. Color by: `field_magnitude`
4. Representation: `Volume` or `Surface`
5. Use time slider to animate

### Recommended Views

**Volume Rendering**: Shows 3D field structure
- Color by `energy_density`
- Adjust opacity for transparency

**Slice View**: Cross-sectional analysis  
- Apply Slice filter
- Set origin at [0, 0, 0]
- Color by `field_magnitude`

**Field Lines**: Visualize gravitational force
- Load `gravitational_field_vector_field.bp`
- Apply Stream Tracer filter
- Color by field magnitude

**Contours**: Equipotential surfaces
- Load `gravitational_field_potential.bp`
- Apply Contour filter
- Add 5-10 isovalues

## Mathematical Background

### Time-Dependent Poisson Equation

The simulation solves:

```
α(∂u/∂t) + ∇²u = f(x,t)
```

where:
- u is the gravitational potential
- α = 1/Δt (inverse time step)
- f(x,t) is a time-dependent source term with perturbations

### Schwarzschild Potential

Initial conditions use the Schwarzschild potential:

```
Φ(r) = -rs/(2r)
```

with safe radius cutoff to avoid singularities.

### Derived Quantities

- **Field Magnitude**: |∇Φ|
- **Energy Density**: ½|∇Φ|²  
- **Curvature**: √((∇²Φ)² + ε)
- **Field Strength**: -∇Φ (force per unit mass)

## Test Particle Trajectories

The simulation includes particle trajectory integration using RK4 (Runge-Kutta 4th order):

- 8 particles initialized at 10rs radius
- Tangential velocities for orbital motion
- Integration stops at Schwarzschild radius
- Produces beautiful spiral trajectories

## Technical Details

- **FEM Discretization**: Lagrange P1 elements
- **Linear Solver**: Conjugate Gradient with Hypre preconditioner
- **Time Integration**: Implicit Euler scheme
- **Output Engine**: ADIOS2 BP4 format
- **Parallel**: MPI-enabled (via DOLFINx/PETSc)

## Limitations

- Simplified model (not full general relativity)
- Static background geometry (Schwarzschild)
- Weak-field approximation
- No self-consistent geometry evolution
- Periodic perturbations are artificial

For full general relativistic simulations, see the proper BSSN implementation in the solar system simulations.

## References

1. Schwarzschild, K. (1916). "On the gravitational field of a mass point"
2. DOLFINx documentation: https://docs.fenicsproject.org
3. ParaView User Guide: https://www.paraview.org/documentation

## Output Structure

```
outputs/
├── data/
│   ├── vtx/          # Time series data (BP format)
│   └── xdmf/         # Static snapshots (XDMF format)
└── visualizations/
    ├── plots/        # PNG visualizations
    └── paraview/     # ParaView helper scripts
```

