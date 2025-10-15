# Installation Scripts

This directory contains optional installation scripts for advanced features of the Production-Ready BSSN simulation.

## Real Ephemeris Data

### Option 1: Astropy (Built-in) Yes **RECOMMENDED**

**Status**: Already installed and working!

- No additional setup required
- Uses Astropy's built-in ephemeris data
- Meter-level accuracy (sufficient for most applications)
- Automatically enabled when `USE_REAL_EPHEMERIS = True`

### Option 2: SPICE Toolkit (JPL DE440)

**Accuracy**: Millimeter-level precision  
**Download Size**: ~114 MB  
**Setup Time**: 5-15 minutes (depending on internet speed)

```bash
# Run the installation script
./scripts/install_spice_kernels.sh
```

The script will download:
- `de440.bsp` - JPL DE440 planetary ephemeris (114 MB)
- `naif0012.tls` - Leap seconds kernel (5 KB)
- `pck00010.tpc` - Planetary constants (124 KB)

After installation, update `solar_system_proper_bssn.py`:
```python
# In RealEphemerisData.__init__():
self.use_spice = use_spice and SPICE_AVAILABLE
# Set use_spice=True when creating instance
```

## Einstein Toolkit Integration

**Status**: Framework ready, full installation optional  
**Complexity**: Advanced

### Option 1: Use the Existing Framework  (Recommended)

The `EinsteinToolkitInterface` class is already implemented and ready!
No installation needed.

```python
# Already available in solar_system_proper_bssn.py
USE_EINSTEIN_TOOLKIT = False  # Framework is ready when needed
```

### Option 2: Docker Image  (Best for Full ET)

```bash
# Pull the official Einstein Toolkit image
docker pull einsteintoolkit/toolkit:latest

# Run with your workspace mounted
docker run -it -v $(pwd):/work einsteintoolkit/toolkit:latest
```

### Option 3: Minimal Test Environment 

```bash
# Create basic directory structure for testing
./scripts/install_einstein_toolkit_simple.sh
```

Creates a minimal framework for testing the interface without compilation.

### Option 4: Full Compilation  (Advanced Users)

**Note**: The full compilation is complex due to changing ET repositories.

**Requirements**:
- Build Time: 2-4 hours  
- Disk Space: ~3 GB  
- Dependencies: gcc, gfortran, MPI, HDF5, etc.

For production ET work, the authors recommend:
1. Using a pre-configured cluster with ET installed
2. The official Docker image
3. Contacting the Einstein Toolkit community for the latest build instructions

### What the Framework Provides

The existing `EinsteinToolkitInterface` class in your simulation provides:
-  Cactus thorn configuration (ADMBase, HydroBase, TmunuBase, Carpet)
-  Data exchange structure (FEniCS ↔ Cactus)
-  Field conversion methods
-  Ready for coupling when ET is available

## Quick Start Guide

### For Most Users (Astropy ephemeris):
```bash
# Nothing to install! Just run:
cd solar_system_simulations/proper_bssn
python3 solar_system_proper_bssn.py
```

### For High-Precision Ephemeris (SPICE):
```bash
# Install SPICE kernels
./scripts/install_spice_kernels.sh

# Update code to use SPICE (see script output)
# Then run simulation
```

### For Research-Grade Simulations (Einstein Toolkit):
```bash
# Install Einstein Toolkit (takes 2-4 hours)
./scripts/install_einstein_toolkit.sh

# Update code configuration
# Run with ET integration
```

## Verification

After installation, verify your setup:

```bash
# Check Python dependencies
python3 -c "import astropy; print(f'Astropy {astropy.__version__} Yes')"
python3 -c "import spiceypy; print(f'SPICE {spiceypy.tkvrsn(\"TOOLKIT\")} Yes')"

# Check SPICE kernels
ls -lh /workspace/data/spice_kernels/

# Check Einstein Toolkit
ls -la /workspace/external/einstein_toolkit/Cactus/exe/
```

## Troubleshooting

### SPICE Installation Issues
- **Slow download**: The DE440 kernel is 114 MB. Use a stable internet connection.
- **Incomplete download**: Re-run the script; it will resume from where it stopped.
- **Kernel not found**: Ensure kernels are loaded in `get_states_from_spice()` using `spice.furnsh()`.

### Einstein Toolkit Build Errors
- **Compilation fails**: Check `build.log` for specific error messages.
- **Missing dependencies**: Re-run with `sudo apt-get install` for missing packages.
- **Out of memory**: ET compilation needs ~4 GB RAM minimum.

### Runtime Issues
- **ImportError**: Ensure virtual environment has all dependencies.
- **SPICE errors**: Verify kernel files exist in `/workspace/data/spice_kernels/`.
- **Performance**: Start with low resolution (`mesh_resolution=10`) for testing.

## See Also

- Main README: `/workspace/README.md`
- BSSN Documentation: `/workspace/solar_system_simulations/proper_bssn/README.md`
- License: `/workspace/LICENSE`

