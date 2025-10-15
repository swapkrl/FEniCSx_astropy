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
