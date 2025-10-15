#!/bin/bash

echo "================================================"
echo "Einstein Toolkit Installation Script"
echo "================================================"
echo ""
echo "WARNING: Einstein Toolkit is a large framework that takes"
echo "         2-4 hours to compile on most systems."
echo ""
echo "This script will guide you through the installation process."
echo ""

read -p "Continue with installation? (y/N): " -n 1 -r
echo
if [[ ! $REPLY =~ ^[Yy]$ ]]; then
    echo "Installation canceled."
    exit 0
fi

ET_DIR="/workspace/external/einstein_toolkit"
mkdir -p "$ET_DIR"
cd "$ET_DIR"

echo ""
echo "Step 1: Installing dependencies..."
apt-get update
apt-get install -y \
    git \
    subversion \
    gcc \
    g++ \
    gfortran \
    make \
    patch \
    pkg-config \
    libhdf5-dev \
    libopenmpi-dev \
    libfftw3-dev \
    libgsl-dev \
    libjpeg-dev \
    libssl-dev \
    libtool \
    zlib1g-dev

echo ""
echo "Step 2: Downloading Einstein Toolkit manifest..."
git clone https://bitbucket.org/einsteintoolkit/manifest.git ET_manifest || {
    echo "Error: Failed to clone Einstein Toolkit manifest"
    exit 1
}

cd ET_manifest

echo ""
echo "Step 3: Downloading Einstein Toolkit components..."
echo "This will download ~2 GB of source code..."
./GetComponents --parallel https://bitbucket.org/einsteintoolkit/manifest/raw/ET_2023_05/einsteintoolkit.th || {
    echo "Error: Failed to download components"
    exit 1
}

cd Cactus

echo ""
echo "Step 4: Configuring build..."
cp repos/simfactory2/mdb/optionlists/ubuntu.cfg config-optionlist.cfg || {
    echo "Creating custom configuration..."
    cat > config-optionlist.cfg << 'EOF'
# Einstein Toolkit configuration for Docker/Ubuntu
CPP = cpp
CC = gcc
CXX = g++
FPP = cpp
FC = gfortran
F90 = gfortran

CFLAGS = -g -std=gnu11
CXXFLAGS = -g -std=gnu++11
F90FLAGS = -g -fcray-pointer -ffixed-line-length-none
FPPFLAGS = -traditional

LDFLAGS = -rdynamic

C_LINE_DIRECTIVES = yes
F_LINE_DIRECTIVES = yes

VECTORISE = yes
VECTORISE_INLINE = no

DEBUG = no
CPP_DEBUG_FLAGS = -DCARPET_DEBUG
C_DEBUG_FLAGS = -fbounds-check -fsanitize=undefined -fstack-protector-all -ftrapv
CXX_DEBUG_FLAGS = -fbounds-check -fsanitize=undefined -fstack-protector-all -ftrapv
FPP_DEBUG_FLAGS = -DCARPET_DEBUG
F90_DEBUG_FLAGS = -fcheck=bounds,do,mem,pointer,recursion -finit-character=65 -finit-integer=42424242 -finit-real=nan -fsanitize=undefined -fstack-protector-all -ftrapv

OPTIMISE = yes
C_OPTIMISE_FLAGS = -O3 -march=native
CXX_OPTIMISE_FLAGS = -O3 -march=native
F90_OPTIMISE_FLAGS = -O3 -march=native

OPENMP = yes
CPP_OPENMP_FLAGS = -fopenmp
FPP_OPENMP_FLAGS = -D_OPENMP

WARN = yes
C_WARN_FLAGS = -Wall
CXX_WARN_FLAGS = -Wall
F90_WARN_FLAGS = -Wall

HDF5_DIR = /usr
MPI_DIR = /usr
GSL_DIR = /usr
FFTW3_DIR = /usr
EOF
}

echo ""
echo "Step 5: Building Einstein Toolkit..."
echo "This will take 2-4 hours. Output will be logged to build.log"
echo ""
./simfactory/bin/sim build --optionlist=config-optionlist.cfg --thornlist=thornlists/einsteintoolkit.th 2>&1 | tee build.log || {
    echo "Error: Build failed. Check build.log for details."
    exit 1
}

echo ""
echo "================================================"
echo "Einstein Toolkit Installation Complete!"
echo "================================================"
echo ""
echo "Installation directory: $ET_DIR/Cactus"
echo "Executable: $ET_DIR/Cactus/exe/cactus_sim"
echo ""
echo "To use with your simulation:"
echo "  1. Set USE_EINSTEIN_TOOLKIT = True in solar_system_proper_bssn.py"
echo "  2. The EinsteinToolkitInterface class will handle integration"
echo ""
echo "To run a test simulation:"
echo "  cd $ET_DIR/Cactus"
echo "  ./simfactory/bin/sim create-run test --parfile arrangements/CactusExamples/WaveToy/par/WaveToyFreeF90.par"

