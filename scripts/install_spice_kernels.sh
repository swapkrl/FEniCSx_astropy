#!/bin/bash

echo "================================================"
echo "SPICE Kernel Installation Script"
echo "================================================"
echo ""
echo "This script downloads JPL SPICE kernels for high-precision"
echo "planetary ephemeris data (DE440 - 114 MB download)"
echo ""

KERNEL_DIR="/workspace/data/spice_kernels"
mkdir -p "$KERNEL_DIR"
cd "$KERNEL_DIR"

echo "Downloading kernels to: $KERNEL_DIR"
echo ""

echo "[1/3] Downloading leap seconds kernel (naif0012.tls)..."
wget -q --show-progress https://naif.jpl.nasa.gov/pub/naif/generic_kernels/lsk/naif0012.tls || {
    echo "Error: Failed to download naif0012.tls"
    exit 1
}
echo "✓ Downloaded naif0012.tls"

echo ""
echo "[2/3] Downloading planetary constants kernel (pck00010.tpc)..."
wget -q --show-progress https://naif.jpl.nasa.gov/pub/naif/generic_kernels/pck/pck00010.tpc || {
    echo "Error: Failed to download pck00010.tpc"
    exit 1
}
echo "✓ Downloaded pck00010.tpc"

echo ""
echo "[3/3] Downloading DE440 planetary ephemeris (de440.bsp - 114 MB)..."
echo "This may take 5-15 minutes depending on connection speed..."
wget --show-progress https://naif.jpl.nasa.gov/pub/naif/generic_kernels/spk/planets/de440.bsp || {
    echo "Error: Failed to download de440.bsp"
    exit 1
}
echo "✓ Downloaded de440.bsp"

echo ""
echo "================================================"
echo "SPICE kernels installed successfully!"
echo "================================================"
echo ""
echo "Kernel files:"
ls -lh "$KERNEL_DIR"

echo ""
echo "To use SPICE ephemeris, update solar_system_proper_bssn.py:"
echo "  1. Set USE_REAL_EPHEMERIS = True"
echo "  2. In RealEphemerisData.__init__(), set use_spice=True"
echo "  3. Load kernels in get_states_from_spice() using:"
echo "     spice.furnsh('$KERNEL_DIR/de440.bsp')"
echo "     spice.furnsh('$KERNEL_DIR/naif0012.tls')"
echo "     spice.furnsh('$KERNEL_DIR/pck00010.tpc')"

