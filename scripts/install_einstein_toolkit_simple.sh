#!/bin/bash

echo "================================================"
echo "Einstein Toolkit - Simplified Installation"
echo "================================================"
echo ""
echo "NOTE: Full Einstein Toolkit installation is complex and"
echo "      requires 2-4 hours. This creates a minimal setup"
echo "      for testing the framework integration."
echo ""

read -p "Continue? (y/N): " -n 1 -r
echo
if [[ ! $REPLY =~ ^[Yy]$ ]]; then
    echo "Installation canceled."
    exit 0
fi

ET_DIR="/workspace/external/einstein_toolkit_minimal"
mkdir -p "$ET_DIR"
cd "$ET_DIR"

echo ""
echo "Creating minimal Einstein Toolkit test environment..."
echo ""

# Create a minimal Cactus-like directory structure
mkdir -p Cactus/configs
mkdir -p Cactus/arrangements
mkdir -p Cactus/exe
mkdir -p Cactus/lib

# Create a basic parameter file for testing
cat > Cactus/test.par << 'EOF'
# Minimal test parameter file for Einstein Toolkit interface testing
ActiveThorns = "CarpetLib Carpet CarpetReduce CartGrid3D CoordBase SymBase Boundary Time MoL"

Cactus::cctk_run_title = "BSSN Test"
Cactus::cctk_full_warnings         = yes

# Grid setup
CartGrid3D::type         = "coordbase"
CartGrid3D::domain       = "full"
CartGrid3D::avoid_origin = "no"

CoordBase::domainsize = "minmax"

CoordBase::xmin = -10.0
CoordBase::ymin = -10.0
CoordBase::zmin = -10.0
CoordBase::xmax = +10.0
CoordBase::ymax = +10.0
CoordBase::zmax = +10.0

CoordBase::dx = 0.5
CoordBase::dy = 0.5
CoordBase::dz = 0.5

# Time evolution
Cactus::cctk_initial_time = 0.0
Cactus::cctk_final_time   = 1.0
Time::dtfac = 0.25

# Output
IO::out_dir = "$parfile"
EOF

# Create a README
cat > Cactus/README.md << 'EOF'
# Minimal Einstein Toolkit Test Environment

This is a minimal setup for testing Einstein Toolkit integration.

## Full Installation

For a complete Einstein Toolkit installation, you need to:

1. Download the full toolkit (2-4 hours):
   ```bash
   git clone https://github.com/gridaphobe/CRL.git einsteintoolkit
   cd einsteintoolkit
   curl -kLO https://raw.githubusercontent.com/gridaphobe/CRL/ET_2023_05/GetComponents
   chmod a+x GetComponents
   ./GetComponents --parallel https://bitbucket.org/einsteintoolkit/manifest/raw/ET_2023_05/einsteintoolkit.th
   ```

2. Configure and build (requires make, compilers, MPI, HDF5, etc.)

## Alternative: Use Pre-Built Docker Image

```bash
docker pull einsteintoolkit/toolkit:latest
```

## Current Setup

This minimal environment allows testing the FEniCS-Einstein Toolkit
interface without the full compilation overhead.
EOF

echo ""
echo "================================================"
echo "Minimal Einstein Toolkit Environment Created!"
echo "================================================"
echo ""
echo "Location: $ET_DIR"
echo ""
echo "This provides a framework structure for testing the"
echo "EinsteinToolkitInterface class without full compilation."
echo ""
echo "For production use, consider:"
echo "  1. Using Einstein Toolkit Docker image"
echo "  2. Full compilation (2-4 hours)"
echo "  3. Accessing a cluster with ET pre-installed"
echo ""
echo "Your BSSN simulation already works with Astropy!"
echo "Run: cd /workspace/solar_system_simulations/proper_bssn && python3 solar_system_proper_bssn.py"

