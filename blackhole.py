import os
os.environ['FI_PROVIDER'] = 'tcp'
os.environ['MPICH_ASYNC_PROGRESS'] = '1'

import numpy as np

DOLFINX_AVAILABLE = False
comm = None

try:
    from dolfinx import mesh
    import dolfinx.cpp as cpp
    comm = cpp.MPI.COMM_SELF
    DOLFINX_AVAILABLE = True
    print("FEniCSx loaded successfully in serial mode")
except Exception as e:
    print(f"Warning: FEniCSx not available: {e}")
    print("Continuing without FEM features...")

try:
    from astropy.coordinates import SkyCoord
    from astropy import units as u
    from astropy import constants as const
    ASTROPY_AVAILABLE = True
    print("Astropy loaded successfully")
except ImportError:
    ASTROPY_AVAILABLE = False
    print("Astropy not available - using default parameters")

def create_astronomical_blackhole_geometry(blackhole_name="Sgr A*"):
    if not ASTROPY_AVAILABLE:
        return {
            'mass': 1.0,
            'schwarzschild_radius': 2.0,
            'domain_size': 20.0
        }
    
    if blackhole_name == "Sgr A*":
        mass = 4.1e6 * u.M_sun
        distance = 8.2 * u.kpc
        coordinates = SkyCoord('17h45m40.0409s', '-29d00m28.118s', frame='icrs')
        
        rs = 2 * const.G * mass / (const.c**2)
        domain_size = 100 * rs.to(u.m).value
        
        print(f"Creating geometry for {blackhole_name}")
        print(f"Mass: {mass}")
        print(f"Schwarzschild radius: {rs}")
        print(f"Domain size: {domain_size:.2e} m")
        
        return {
            'mass': mass,
            'distance': distance,
            'coordinates': coordinates,
            'schwarzschild_radius': rs.to(u.m).value,
            'domain_size': domain_size
        }

astronomical_data = create_astronomical_blackhole_geometry()
rs = astronomical_data['schwarzschild_radius']
domain_size = astronomical_data['domain_size']

if DOLFINX_AVAILABLE and comm is not None:
    domain = mesh.create_box(
        comm,
        [[-domain_size, -domain_size, -domain_size], 
         [domain_size, domain_size, domain_size]],
        [10, 10, 10],
        cell_type=mesh.CellType.hexahedron
    )
    
    num_cells = domain.topology.index_map(domain.topology.dim).size_local
    print(f"Astronomical mesh has {num_cells} cells")
    print(f"Schwarzschild radius: {rs:.2e} m")
else:
    print("Cannot create mesh - FEniCSx not available")
    print(f"Schwarzschild radius: {rs:.2e} m")
