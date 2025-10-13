# Spacetime Simulations with Finite Element Methods

A collection of spacetime simulations using the Finite Element Method (FEM) to model gravitational phenomena from flat 3D Euclidean space to curved 4D spacetime. This repository explores relativistic physics through computational methods, progressing from simple Newtonian gravitational fields to full general relativistic spacetime curvature.

## Project Vision

This project aims to create a comprehensive suite of simulations that explore:

- **3D Flat Space**: Newtonian gravitational potentials in Euclidean geometry
- **3D Curved Space**: Spatial curvature around massive objects
- **4D Minkowski Spacetime**: Special relativistic effects in flat spacetime
- **4D Curved Spacetime**: General relativistic spacetime curvature (Schwarzschild, Kerr metrics)
- **Dynamic Spacetime**: Time-evolving gravitational fields and gravitational waves

## Current Simulations

### Black Hole Gravitational Potential (3D)
Computes the gravitational potential field around supermassive black holes using real astronomical data from Astropy. Currently configured for Sgr A* (Sagittarius A*), the supermassive black hole at the center of our galaxy.

## Features

- **Finite Element Method**: Sophisticated numerical technique for solving partial differential equations on complex geometries
- **Astronomical Integration**: Real astronomical data from Astropy for physically accurate parameters
- **Progressive Complexity**: From simple 3D simulations to full 4D spacetime curvature
- **Interactive Visualization**: ParaView-compatible output for exploring 3D/4D fields
- **Parallel Computing**: MPI-based parallel computation via PETSc for large-scale problems
- **Modular Design**: Extensible framework for adding new spacetime configurations

## What is the Finite Element Method?

The Finite Element Method (FEM) is a powerful numerical technique for finding approximate solutions to partial differential equations (PDEs) that govern physical phenomena. Unlike finite difference methods that work on regular grids, FEM discretizes space into a mesh of elements with flexible shapes and sizes.

### Why FEM for Spacetime Simulations?

**Geometric Flexibility**: FEM handles complex geometries naturally, essential for curved spacetime regions near black holes, neutron stars, and other exotic objects.

**Weak Formulation**: FEM converts differential equations into integral form, allowing solutions even when derivatives don't exist classically (important for singularities).

**Adaptive Refinement**: Mesh resolution can be concentrated where needed (near event horizons, strong curvature regions) while keeping coarse elements elsewhere.

**Mathematical Rigor**: FEM provides error estimates and convergence guarantees, ensuring simulation accuracy.

### How FEM Works

1. **Domain Discretization**: Divide the spacetime region into finite elements (tetrahedra, hexahedra in 3D)
2. **Basis Functions**: Represent the unknown field as a combination of polynomial basis functions on each element
3. **Weak Form**: Convert PDEs to integral equations through variational principles
4. **Assembly**: Build a large sparse linear system Ax = b representing the entire problem
5. **Solve**: Use iterative or direct solvers to find the solution vector
6. **Visualization**: Reconstruct the continuous field from discrete solution

### FEM in General Relativity

For spacetime simulations, FEM discretizes the Einstein field equations or their approximations:

**Einstein Field Equations**: Gμν = 8πG/c⁴ Tμν

These nonlinear PDEs describe how matter-energy curves spacetime. FEM allows us to:
- Solve for metric tensor components in curved spacetime
- Compute geodesics and light paths
- Model gravitational wave propagation
- Simulate black hole mergers and cosmological evolution

## Requirements

### System Dependencies
- Python 3.8+
- FEniCSx (DOLFINx)
- MPI implementation (OpenMPI or MPICH)
- ParaView (for visualization)

### Python Dependencies
```
numpy
matplotlib
scipy
astropy>=7.1.1
mpi4py
petsc4py
```

## Installation

### Option 1: Using DOLFINx Docker Container
```bash
docker pull dolfinx/dolfinx:stable
docker run -ti -v $(pwd):/workspace dolfinx/dolfinx:stable
```

### Option 2: Local Installation
```bash
pip install numpy scipy matplotlib astropy mpi4py petsc4py
```

Install FEniCSx following the official documentation at https://fenicsproject.org/download/

## Usage

### Running Simulations

**Black Hole Gravitational Potential (Current)**
```bash
python3 blackhole.py
```

For parallel execution with MPI:
```bash
mpirun -n 4 python3 blackhole.py
```

**Future Simulations** (Planned)
- `flat_space_gravity.py` - Newtonian gravity in 3D Euclidean space
- `minkowski_spacetime.py` - Special relativistic spacetime
- `schwarzschild_metric.py` - Static black hole spacetime geometry
- `kerr_metric.py` - Rotating black hole spacetime
- `gravitational_waves.py` - Dynamic spacetime ripples

### Expected Output

```
FEniCSx loaded successfully
Astropy loaded successfully
Creating geometry for Sgr A*
Mass: 4100000.0 solMass
Schwarzschild radius: 12108325312.011024 m
Domain size: 1.21e+12 m
Created mesh with 8000 cells
Schwarzschild radius: 1.21e+10 m
Solving gravitational potential field...
Black hole simulation complete!
Potential field degrees of freedom: 9261
Potential field L2 norm: 1.58e+16
Results saved to blackhole_simulation.bp for ParaView visualization
```

## Visualization with ParaView

### Opening the Results

1. Launch ParaView
2. Go to **File → Open**
3. Navigate to the simulation directory
4. Select `blackhole_simulation.bp`
5. Click **Apply** in the Properties panel

### Visualization Options

**Basic Visualization:**
- In the dropdown menu, select `gravitational_potential`
- Choose representation: Surface, Volume, or Points
- Apply a color map to visualize field intensity

**Advanced Techniques:**
- **Slice Filter**: Cut through the 3D field to see cross-sections
- **Contour Filter**: Create isosurfaces of constant potential
- **Clip Filter**: Remove portions of the domain for interior views
- **Glyph Filter**: Show field vectors or gradients
- **Volume Rendering**: Enable opacity mapping for volumetric visualization

### Recommended Workflow

1. Apply a **Slice** filter along the Z-axis to see the equatorial plane
2. Use **Contour** filter to visualize gravitational equipotential surfaces
3. Adjust color scale to highlight regions near the event horizon
4. Enable **Show Color Legend** for quantitative reference

## Technical Details

### Current Implementation: Black Hole Gravitational Potential

**Simulation Parameters**

| Parameter | Value | Description |
|-----------|-------|-------------|
| Black Hole | Sgr A* | Supermassive black hole at Galactic center |
| Mass | 4.1 × 10⁶ M☉ | Solar masses |
| Schwarzschild Radius | ~1.21 × 10¹⁰ m | Event horizon radius |
| Domain Size | 100 × rs | Computational domain extends 100 Schwarzschild radii |
| Mesh Resolution | 20 × 20 × 20 | Hexahedral elements |
| Total Cells | 8000 | Finite elements |
| Degrees of Freedom | 9261 | Solution vector size |

**Mathematical Model**

Solves Laplace's equation for the gravitational potential:

∇²φ = 0

With Dirichlet boundary conditions based on the Schwarzschild potential:

φ(r) = -rs / (2r)

where rs is the Schwarzschild radius and r is the distance from the black hole center.

**Numerical Method**

- **Discretization**: Finite Element Method (FEM)
- **Element Type**: Hexahedral cells
- **Function Space**: Lagrange polynomials of degree 1
- **Solver**: Direct LU factorization via PETSc
- **Parallel**: MPI-based domain decomposition

### Future Implementations

**Schwarzschild Metric (4D Spacetime)**

Metric tensor components:
```
ds² = -(1 - rs/r)dt² + (1 - rs/r)⁻¹dr² + r²(dθ² + sin²θ dφ²)
```

**Kerr Metric (Rotating Black Hole)**

Includes angular momentum parameter a:
```
ds² = -(1 - rs·r/Σ)dt² + (Σ/Δ)dr² + Σdθ² + ... (frame dragging terms)
```

**Gravitational Waves**

Time-dependent metric perturbations hμν:
```
gμν = ημν + hμν(t, x, y, z)
∂²hμν/∂t² - ∇²hμν = source terms
```

## Project Structure

```
.
├── blackhole.py                    # Black hole gravitational potential (3D)
├── flat_space_gravity.py           # Newtonian gravity (3D) [planned]
├── minkowski_spacetime.py          # Special relativity (4D) [planned]
├── schwarzschild_metric.py         # Static black hole metric (4D) [planned]
├── kerr_metric.py                  # Rotating black hole metric (4D) [planned]
├── gravitational_waves.py          # Dynamic spacetime (4D) [planned]
├── requirements.txt                # Python dependencies
├── README.md                       # This file
└── output/                         # Simulation results
    ├── blackhole_simulation.bp/    # ParaView format
    └── ...
```

## Developing New Simulations

### Simulation Development Roadmap

**Phase 1: 3D Flat Space** (Newtonian Limit)
- Solve Poisson equation: ∇²φ = 4πGρ
- Multiple mass configurations
- Tidal forces and equipotential surfaces

**Phase 2: 3D Curved Space** (Spatial Geometry)
- Embed curved surfaces in higher dimensions
- Geodesics on curved manifolds
- Intrinsic vs extrinsic curvature

**Phase 3: 4D Minkowski Spacetime** (Special Relativity)
- Light cones and causality
- Lorentz transformations
- Time dilation and length contraction fields

**Phase 4: 4D Schwarzschild Spacetime** (Static Black Holes)
- Solve for metric components gμν
- Proper time along geodesics
- Photon orbits and gravitational lensing

**Phase 5: 4D Kerr Spacetime** (Rotating Black Holes)
- Frame dragging visualization
- Ergosphere geometry
- Penrose process energy extraction

**Phase 6: Dynamic Spacetime** (Gravitational Waves)
- Time-dependent metric evolution
- Wave propagation and interference
- Binary merger simulations

### Customization Guide

**Modifying Existing Simulations**

For `blackhole.py`:
- **Mesh resolution**: `[20, 20, 20]` in `mesh.create_box()`
- **Domain size**: multiplier `100 * rs.value`
- **Black hole mass**: Change in `create_astronomical_blackhole_geometry()`
- **Solver options**: `petsc_options` dictionary parameters

**Creating New Astronomical Objects**

```python
elif blackhole_name == "M87*":
    mass = 6.5e9 * u.M_sun
    distance = 16.8 * u.Mpc
    coordinates = SkyCoord('12h30m49.4233s', '+12d23m28.044s', frame='icrs')
```

**Implementing New Spacetime Geometries**

1. Define the metric tensor gμν or potential φ
2. Formulate governing PDEs (Einstein equations, geodesic equations, etc.)
3. Convert to weak form for FEM
4. Set up mesh and function spaces in DOLFINx
5. Implement boundary conditions
6. Solve and export for visualization

## Performance Notes

- Computation time scales with mesh resolution (O(n³) for 3D, O(n⁴) for 4D spacetime)
- Memory usage depends on degrees of freedom and solver type
- Parallel execution recommended for high-resolution meshes (100+ elements per dimension)
- Direct solvers (LU) are robust but memory-intensive for large problems
- Iterative solvers (GMRES, CG) scale better for massive simulations
- 4D spacetime simulations require significantly more resources than 3D spatial problems

## Contributing

This is an active research and educational project. Contributions are welcome in several areas:

### Simulation Development
- Implement planned simulations (flat space, Minkowski, Kerr, etc.)
- Add new spacetime metrics or gravitational configurations
- Optimize existing solvers for performance

### Visualization
- Create ParaView state files for common visualization workflows
- Develop custom filters for GR-specific quantities (Riemann tensor, Christoffel symbols)
- Build interactive dashboards for parameter exploration

### Documentation
- Add tutorials for specific physics concepts
- Create Jupyter notebooks with step-by-step explanations
- Document FEM theory for relativistic PDEs

### Validation
- Compare results against analytical solutions
- Benchmark performance across different mesh resolutions
- Verify physical accuracy of computed fields

**How to Contribute**
1. Fork the repository
2. Create a feature branch
3. Implement your simulation/improvement
4. Add tests and documentation
5. Submit a pull request

## Roadmap

**Short Term**
-  Black hole gravitational potential (3D)
-  Newtonian gravity with multiple masses
-  Geodesic path computation and visualization
-  Gravitational lensing ray tracing

**Medium Term**
-  Schwarzschild metric full 4D implementation
-  Proper time and coordinate time fields
-  Kerr metric for rotating black holes
-  Tidal tensor visualization

**Long Term**
-  Time-evolving gravitational wave simulation
-  Binary black hole merger
-  Cosmological spacetime expansion
-  Numerical relativity integration (3+1 formalism)

## Acknowledgments

This project builds upon powerful scientific computing tools:

- **FEniCSx**: High-performance finite element library for solving PDEs
- **Astropy**: Comprehensive astronomical data and coordinate transformations
- **PETSc**: Portable, Extensible Toolkit for Scientific Computation
- **ParaView**: Open-source scientific visualization platform
- **MPI**: Message Passing Interface for parallel computing

## References

### General Relativity & Numerical Methods
- Misner, Thorne & Wheeler (1973). "Gravitation" - The classic GR reference
- Schwarzschild, K. (1916). "On the Gravitational Field of a Mass Point"
- Alcubierre, M. (2008). "Introduction to 3+1 Numerical Relativity"
- Baumgarte & Shapiro (2010). "Numerical Relativity"

### Finite Element Methods
- Zienkiewicz & Taylor (2000). "The Finite Element Method"
- Hughes, T.J.R. (2000). "The Finite Element Method: Linear Static and Dynamic FEA"
- Logg, Mardal & Wells (2012). "Automated Solution of Differential Equations by FEM"

### Software Documentation
- FEniCSx Documentation: https://fenicsproject.org/
- Astropy Documentation: https://www.astropy.org/
- PETSc Documentation: https://petsc.org/
- ParaView User Guide: https://www.paraview.org/

## License

MIT License - Feel free to use, modify, and extend for research, education, or personal projects.

## Contact & Citation

If you use this code in your research or educational materials, please cite this repository and acknowledge the underlying software packages (FEniCSx, Astropy, PETSc).

