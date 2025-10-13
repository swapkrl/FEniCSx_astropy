# Black Hole Gravitational Potential Simulation

A finite element simulation of the gravitational potential field around supermassive black holes using FEniCSx and astronomical data from Astropy. This project computes and visualizes the Schwarzschild potential in 3D space around real astronomical objects like Sagittarius A*.

## Features

- **Astronomical Integration**: Uses real astronomical data from Astropy for accurate black hole parameters
- **FEM Simulation**: Solves the gravitational potential field using finite element methods via DOLFINx
- **3D Visualization**: Exports results to ParaView-compatible format for interactive 3D visualization
- **Parallel Computing**: Supports MPI-based parallel computation via PETSc
- **Real Black Holes**: Preconfigured for Sgr A* (Sagittarius A*), the supermassive black hole at our galaxy's center

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

### Running the Simulation

```bash
python3 blackhole.py
```

For parallel execution with MPI:
```bash
mpirun -n 4 python3 blackhole.py
```

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

### Simulation Parameters

| Parameter | Value | Description |
|-----------|-------|-------------|
| Black Hole | Sgr A* | Supermassive black hole at Galactic center |
| Mass | 4.1 × 10⁶ M☉ | Solar masses |
| Schwarzschild Radius | ~1.21 × 10¹⁰ m | Event horizon radius |
| Domain Size | 100 × rs | Computational domain extends 100 Schwarzschild radii |
| Mesh Resolution | 20 × 20 × 20 | Hexahedral elements |
| Total Cells | 8000 | Finite elements |
| Degrees of Freedom | 9261 | Solution vector size |

### Mathematical Model

The simulation solves Laplace's equation for the gravitational potential:

∇²φ = 0

With Dirichlet boundary conditions based on the Schwarzschild potential:

φ(r) = -rs / (2r)

where rs is the Schwarzschild radius and r is the distance from the black hole center.

### Numerical Method

- **Discretization**: Finite Element Method (FEM)
- **Element Type**: Hexahedral cells
- **Function Space**: Lagrange polynomials of degree 1
- **Solver**: Direct LU factorization via PETSc
- **Parallel**: MPI-based domain decomposition

## Project Structure

```
.
├── blackhole.py              # Main simulation script
├── requirements.txt          # Python dependencies
├── README.md                 # This file
└── blackhole_simulation.bp/  # ParaView output (generated)
    ├── data.0
    ├── md.0
    ├── md.idx
    └── profiling.json
```

## Customization

### Modifying Simulation Parameters

Edit `blackhole.py` to adjust:

- **Mesh resolution**: Change `[20, 20, 20]` array in `mesh.create_box()`
- **Domain size**: Modify multiplier in `100 * rs.value`
- **Black hole**: Add custom astronomical objects in `create_astronomical_blackhole_geometry()`
- **Boundary conditions**: Adjust the `boundary()` function threshold

### Adding New Black Holes

Extend the `create_astronomical_blackhole_geometry()` function:

```python
elif blackhole_name == "M87*":
    mass = 6.5e9 * u.M_sun
    distance = 16.8 * u.Mpc
    coordinates = SkyCoord('12h30m49.4233s', '+12d23m28.044s', frame='icrs')
```

## Performance Notes

- Computation time scales with mesh resolution (O(n³) for 3D problems)
- Memory usage depends on degrees of freedom and solver type
- Parallel execution recommended for high-resolution meshes
- Direct solver (LU) is robust but memory-intensive for large problems

## Acknowledgments

This project uses:
- **FEniCSx**: High-performance finite element library
- **Astropy**: Astronomical data and calculations
- **PETSc**: Scalable linear algebra solvers
- **ParaView**: Scientific visualization platform

## References

- Schwarzschild, K. (1916). "On the Gravitational Field of a Mass Point"
- FEniCSx Documentation: https://fenicsproject.org/
- Astropy Documentation: https://www.astropy.org/
- ParaView User Guide: https://www.paraview.org/

## License

MIT License - Feel free to use and modify for your research or educational purposes.

