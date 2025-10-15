# FEniCSx Spacetime Simulations

A collection of General Relativity and gravitational physics simulations using the Finite Element Method (FEM) through FEniCSx. This repository explores gravitational phenomena from Newtonian fields to full BSSN numerical relativity.

## Overview

This project provides a suite of progressively complex gravitational simulations:

1. **Black Hole Simulations**: Time-evolving gravitational fields around Schwarzschild black holes
2. **Solar System Simulations**: Curved spacetime geometry with real planetary data
   - Post-Newtonian approximations
   - Advanced BSSN formulation concepts
   - Proper BSSN evolution equations

## Repository Structure

```
.
 blackhole_simulations/
    schwarzschild_field/       # Time-evolving Schwarzschild gravitational field
        blackhole_simulation.py
        paraview_setup.py
        README.md
        outputs/
 solar_system_simulations/
    post_newtonian/            # Post-Newtonian solar system dynamics
       solar_system_spacetime.py
       README.md
       outputs/
    advanced_bssn/             # BSSN concepts demonstration
       solar_system_advanced.py
       README.md
       outputs/
    proper_bssn/               # Proper BSSN time evolution
        solar_system_proper_bssn.py
        solar_system_proper_bssn_2.py
        README.md
        outputs/
 .gitattributes                 # Git LFS configuration
 .gitignore
 README.md                      # This file
```

## Simulations

### 1. Schwarzschild Black Hole Gravitational Field

**Location**: `blackhole_simulations/schwarzschild_field/`

A FEniCS-based simulation of time-evolving gravitational fields around a static Schwarzschild black hole.

**Features**:
- 3D spatial mesh with time evolution
- Schwarzschild geometry as initial condition
- Multiple derived fields (potential, magnitude, energy density, curvature)
- Test particle trajectory integration
- ParaView-compatible VTX and XDMF output formats

**Quick Start**:
```bash
cd blackhole_simulations/schwarzschild_field/
python3 blackhole_simulation.py
```

**Output**: All outputs saved in `outputs/` subdirectory (data files, plots, ParaView scripts)

**Documentation**: See `blackhole_simulations/schwarzschild_field/README.md`

### 2. Solar System Post-Newtonian Spacetime

**Location**: `solar_system_simulations/post_newtonian/`

Models the solar system as a curved spacetime using Post-Newtonian corrections to Newtonian gravity.

**Features**:
- Real astronomical data (Sun + 7 planets)
- Accurate Keplerian orbits with eccentricities
- Relativistic v²/c² corrections
- Metric perturbations h_μν
- Ricci curvature computation
- Observable effects like Mercury's perihelion precession

**Quick Start**:
```bash
cd solar_system_simulations/post_newtonian/
python3 solar_system_spacetime.py
```

**Documentation**: See `solar_system_simulations/post_newtonian/README.md`

### 3. Advanced BSSN Solar System

**Location**: `solar_system_simulations/advanced_bssn/`

An educational implementation demonstrating BSSN formulation concepts applied to the solar system.

**Features**:
- Full 10-component metric tensor
- BSSN variables (φ, γ̃_ij, K̃_ij, K, α, β^i)
- 3+1 ADM decomposition framework
- Conformal decomposition
- Weyl curvature tensor
- AMR concepts

**Note**: This is a **pedagogical implementation** showing BSSN structure, not production-grade numerical relativity.

**Quick Start**:
```bash
cd solar_system_simulations/advanced_bssn/
python3 solar_system_advanced.py
```

**Documentation**: See `solar_system_simulations/advanced_bssn/README.md`

### 4. Production-Ready BSSN Solar System

**Location**: `solar_system_simulations/proper_bssn/`

A production-ready BSSN implementation with advanced features bridging educational and research-grade numerical relativity.

**Core Features**:
-  Full BSSN evolution: φ, K, α, β^i, B^i (optionally: γ̃ᵢ, Ãᵢ)
-  Gamma-driver shift condition with full evolution
-  Z4c-style constraint damping (κ₁=0.02, κ₂=0.1)
-  4th-order Runge-Kutta (RK4) time integration
-  High-order finite elements (P1-P4, CG/DG)

**Advanced Features**:
-  Adaptive Mesh Refinement (AMR) with moving box grids
-  Real ephemeris data (JPL DE440/Astropy)
-  Post-Newtonian corrections (1PN+2PN+3PN)
-  HPC support with MPI parallelization
-  Production diagnostics (GW extraction, horizon tracking)
-  Einstein Toolkit interface framework

**Quick Start**:
```bash
cd solar_system_simulations/proper_bssn/
python3 solar_system_proper_bssn.py

# With MPI (HPC)
mpirun -n 8 python3 solar_system_proper_bssn.py
```

**Configuration Flags**:
- `IS_HPC = True/False` - Enable HPC mode
- `USE_REAL_EPHEMERIS = True/False` - Use real planetary data
- `USE_POST_NEWTONIAN = True/False` - Apply PN corrections

**Documentation**: See `solar_system_simulations/proper_bssn/README.md`

## Requirements

### Core Dependencies
- **Python 3.8+**
- **FEniCSx (DOLFINx)** - Finite element library
- **MPI** - Parallel computing (OpenMPI or MPICH)
- **ParaView** - Visualization (optional but recommended)

### Python Packages
```bash
numpy
matplotlib
scipy
mpi4py
petsc4py
```

## Installation

### Option 1: Docker (Recommended)
```bash
docker pull dolfinx/dolfinx:stable
docker run -ti -v $(pwd):/workspace dolfinx/dolfinx:stable
```

### Option 2: Conda Environment
```bash
conda create -n fenicsx-env
conda activate fenicsx-env
conda install -c conda-forge fenics-dolfinx mpich pyvista
pip install matplotlib scipy
```

### Option 3: System Installation
Follow the official FEniCSx installation guide:
https://github.com/FEniCS/dolfinx#installation

## Visualization with ParaView

All simulations export data in ParaView-compatible formats:

1. **VTX/BP Format**: Time-series data (ADIOS2 BP4)
2. **XDMF Format**: Universal compatibility

### Basic ParaView Workflow

1. Open ParaView
2. File → Open → Navigate to simulation `outputs/data/vtx/` or `outputs/data/xdmf/`
3. Select `.bp` or `.xdmf` file
4. Click **Apply**
5. Color by desired field
6. Use filters: Slice, Contour, Stream Tracer, Clip

### Automated Setup

Most simulations include auto-generated ParaView scripts:
- `paraview_script.py` - Basic visualization setup
- `PARAVIEW_GUIDE.md` - Detailed instructions

## Understanding the Finite Element Method

### What is FEM?

The Finite Element Method numerically solves partial differential equations by:
1. Dividing space into finite elements (mesh)
2. Approximating solutions with polynomial basis functions
3. Converting PDEs to integral form (weak formulation)
4. Solving large sparse linear systems

### Why FEM for General Relativity?

- **Geometric Flexibility**: Handles curved spacetime naturally
- **Adaptive Refinement**: Concentrate resolution near strong curvature
- **Weak Formulation**: Works even with singularities
- **Mathematical Rigor**: Error estimates and convergence guarantees

### FEM in These Simulations

All simulations use:
- **Element Type**: Hexahedral (3D) or tetrahedral cells
- **Function Space**: Lagrange polynomials (P1 or P2)
- **Linear Solvers**: CG, GMRES with preconditioners (Hypre, Jacobi)
- **Parallel**: MPI domain decomposition via PETSc
- **Output**: ADIOS2 BP4 format for scalability

## Mathematical Background

### Black Hole Simulation

Solves time-dependent Poisson equation:
```
α(∂u/∂t) + ∇²u = f(x,t)
```
with Schwarzschild initial conditions.

### Post-Newtonian Solar System

Metric perturbation:
```
g_μν = η_μν + h_μν
h_00 ≈ 2Φ/c²
```
with corrections:
```
Φ_PN = Φ_Newton + (v²/2c²)Φ_Newton + (GM/rc²)Φ_Newton
```

### BSSN Formulation

3+1 decomposition of Einstein equations:
```
ds² = -α²dt² + γ_ij(dx^i + β^idt)(dx^j + β^jdt)
```

Key evolution equations:
- Conformal factor: ∂_t φ = -(1/6) α K + β^i ∂_i φ
- Trace K: ∂_t K = -D_i D^i α + α(A_ij A^ij + K²/3) + source
- Lapse (1+log): ∂_t α = -2 α K + β^i ∂_i α

## Performance Considerations

### Computational Complexity
- 3D simulations: O(n³) scaling with resolution
- Time evolution: O(N_t × n³) for N_t timesteps
- Memory: Depends on DOF and solver type

### Recommended Resources
- **Black Hole**: 4 CPU cores, 8 GB RAM (30³ mesh)
- **Post-Newtonian**: 4-8 cores, 16 GB RAM (25³ mesh)
- **BSSN**: 8+ cores, 32 GB RAM (20³ mesh)

### Parallelization
All simulations support MPI:
```bash
mpirun -n 4 python3 simulation_script.py
```

## Git LFS Configuration

Large output files (>99MB) are managed with Git LFS:

- Tracked file types: `*.bp`, `*.h5`, `*.xdmf`
- Automatic tracking for `outputs/data/` directories
- Pre-commit hook for automatic LFS tracking

**Setup**:
```bash
git lfs install
```

See `.gitattributes` for tracking rules.

## Learning Path

### Beginner
1. Start with **Schwarzschild Black Hole** simulation
2. Understand basic field visualization in ParaView
3. Explore test particle trajectories

### Intermediate
1. Run **Post-Newtonian Solar System**
2. Compare Newtonian vs relativistic effects
3. Visualize metric perturbations and curvature

### Advanced
1. Study **Advanced BSSN** for formulation concepts
2. Run **Proper BSSN** to see evolution equations
3. Monitor constraint violations
4. Explore Einstein Toolkit for production simulations

## Production Numerical Relativity

For research-grade simulations, use:

### Einstein Toolkit
- Full BSSN/CCZ4 evolution
- Carpet AMR
- Gravitational waveform extraction
- https://einsteintoolkit.org

### GRChombo
- Modern C++ implementation
- Chombo AMR framework
- Designed for scalar field cosmology
- https://github.com/GRChombo/GRChombo

### SpEC (Caltech)
- Spectral methods
- Binary black hole simulations
- SXS waveform catalog
- https://www.black-holes.org/code/SpEC.html

## References

### General Relativity
1. Misner, Thorne & Wheeler (1973). "Gravitation"
2. Carroll, S. (2004). "Spacetime and Geometry"
3. Wald, R. (1984). "General Relativity"

### Numerical Relativity
4. Baumgarte & Shapiro (2010). "Numerical Relativity"
5. Alcubierre, M. (2008). "Introduction to 3+1 Numerical Relativity"
6. Gourgoulhon, E. (2012). "3+1 Formalism in General Relativity"

### BSSN Formulation
7. Baumgarte & Shapiro (1999). Phys. Rev. D 59, 024007
8. Shibata & Nakamura (1995). Phys. Rev. D 52, 5428

### Finite Element Methods
9. Logg, Mardal & Wells (2012). "Automated Solution of Differential Equations by FEM"
10. Zienkiewicz & Taylor (2000). "The Finite Element Method"

### Software Documentation
- FEniCSx: https://docs.fenicsproject.org
- ParaView: https://www.paraview.org/documentation
- PETSc: https://petsc.org/release/docs/
- ADIOS2: https://adios2.readthedocs.io

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Contributing

Contributions are welcome! Areas of interest:

- New simulation implementations
- Performance optimizations
- Enhanced visualization scripts
- Documentation improvements
- Validation against analytical solutions

By contributing, you agree that your contributions will be licensed under the same MIT License.

## Acknowledgments

This project uses:
- **FEniCSx/DOLFINx** - High-performance FEM library
- **PETSc** - Scalable scientific computation
- **MPI** - Parallel computing
- **ADIOS2** - I/O framework for large-scale simulations
- **ParaView** - Scientific visualization platform

## Contact

For questions, issues, or contributions, please use the GitHub issue tracker.

---


