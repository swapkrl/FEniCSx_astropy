import numpy as np

DOLFINX_AVAILABLE = False
comm = None

try:
    from dolfinx import mesh, fem, default_scalar_type
    from dolfinx.fem.petsc import LinearProblem
    from dolfinx.io import VTXWriter
    from mpi4py import MPI
    import ufl
    from petsc4py import PETSc
    comm = MPI.COMM_WORLD
    DOLFINX_AVAILABLE = True
    print("FEniCSx loaded successfully")
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
        
        mass_kg = mass.to(u.kg)
        rs = (2 * const.G * mass_kg / (const.c**2)).to(u.m)
        domain_size = 100 * rs.value
        
        print(f"Creating geometry for {blackhole_name}")
        print(f"Mass: {mass}")
        print(f"Schwarzschild radius: {rs}")
        print(f"Domain size: {domain_size:.2e} m")
        
        return {
            'mass': mass,
            'distance': distance,
            'coordinates': coordinates,
            'schwarzschild_radius': rs.value,
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
        [20, 20, 20],
        cell_type=mesh.CellType.hexahedron
    )
    
    num_cells = domain.topology.index_map(domain.topology.dim).size_local
    print(f"Created mesh with {num_cells} cells")
    print(f"Schwarzschild radius: {rs:.2e} m")
    
    V = fem.functionspace(domain, ("Lagrange", 1))
    
    def schwarzschild_potential(x):
        r = np.sqrt(x[0]**2 + x[1]**2 + x[2]**2)
        r = np.maximum(r, rs * 0.1)
        return -rs / (2 * r)
    
    def boundary(x):
        boundary_size = domain_size * 0.99
        return np.logical_or(
            np.logical_or(
                np.abs(x[0]) > boundary_size,
                np.abs(x[1]) > boundary_size
            ),
            np.abs(x[2]) > boundary_size
        )
    
    boundary_dofs = fem.locate_dofs_geometrical(V, boundary)
    bc_func = fem.Function(V)
    bc_func.interpolate(schwarzschild_potential)
    bc = fem.dirichletbc(bc_func, boundary_dofs)
    
    u = ufl.TrialFunction(V)
    v = ufl.TestFunction(V)
    
    f = fem.Function(V)
    x = ufl.SpatialCoordinate(domain)
    r_expr = ufl.sqrt(x[0]**2 + x[1]**2 + x[2]**2)
    
    a = ufl.dot(ufl.grad(u), ufl.grad(v)) * ufl.dx
    L = f * v * ufl.dx
    
    problem = LinearProblem(a, L, bcs=[bc], 
                           petsc_options={"ksp_type": "preonly", "pc_type": "lu"})
    
    print("Solving gravitational potential field...")
    uh = problem.solve()
    
    print(f"Black hole simulation complete!")
    print(f"Potential field degrees of freedom: {V.dofmap.index_map.size_global}")
    
    potential_norm = np.sqrt(comm.allreduce(
        fem.assemble_scalar(fem.form(ufl.inner(uh, uh) * ufl.dx)),
        op=MPI.SUM
    ))
    print(f"Potential field L2 norm: {potential_norm:.2e}")
    
    output_filename = "blackhole_simulation.bp"
    uh.name = "gravitational_potential"
    with VTXWriter(comm, output_filename, [uh], engine="BP4") as vtx:
        vtx.write(0.0)
    print(f"Results saved to {output_filename} for ParaView visualization")
    
else:
    print("Cannot create mesh - FEniCSx not available")
    print(f"Schwarzschild radius: {rs:.2e} m")
