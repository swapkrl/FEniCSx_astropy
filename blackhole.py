import os
os.environ['FI_PROVIDER'] = 'tcp'
os.environ['MPICH_ASYNC_PROGRESS'] = '1'

from dolfinx import mesh
from mpi4py import MPI
import numpy as np

comm = MPI.COMM_SELF
print(f"Running in serial mode")

# Test basic FEniCS functionality
domain = mesh.create_unit_square(comm, 10, 10)
num_cells = domain.topology.index_map(domain.topology.dim).size_local
print(f"Mesh has {num_cells} cells")
