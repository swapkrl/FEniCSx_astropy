import os
os.environ['FI_PROVIDER'] = 'tcp'
os.environ['MPICH_ASYNC_PROGRESS'] = '1'

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

try:
    from astropy.coordinates import SkyCoord
    from astropy import units as u
    from astropy import constants as const
    ASTROPY_AVAILABLE = True
except ImportError:
    ASTROPY_AVAILABLE = False

try:
    import dolfinx
    from dolfinx import mesh, fem
    import dolfinx.cpp as cpp
    import ufl
    DOLFINX_AVAILABLE = True
    comm = cpp.MPI.COMM_SELF
except ImportError:
    DOLFINX_AVAILABLE = False
    comm = None

class AstronomicalBlackHoleFEM:
    def __init__(self, blackhole_name="Sgr A*", mesh_resolution=30):
        self.blackhole_name = blackhole_name
        self.mesh_resolution = mesh_resolution
        self.astronomical_data = self._get_astronomical_data()
        self.domain = None
        self.function_space = None
        
    def _get_astronomical_data(self):
        if not ASTROPY_AVAILABLE:
            return {
                'mass': 1.0,
                'schwarzschild_radius': 2.0,
                'domain_size': 20.0,
                'coordinates': None
            }
        
        if self.blackhole_name == "Sgr A*":
            mass = 4.1e6 * u.M_sun
            distance = 8.2 * u.kpc
            coordinates = SkyCoord('17h45m40.0409s', '-29d00m28.118s', frame='icrs')
            
            rs = 2 * const.G * mass / (const.c**2)
            domain_size = 50 * rs.to(u.m).value
            
            print(f"Astronomical Data for {self.blackhole_name}:")
            print(f"  Mass: {mass}")
            print(f"  Distance: {distance}")
            print(f"  Schwarzschild radius: {rs}")
            print(f"  Domain size: {domain_size:.2e} m")
            
            return {
                'mass': mass,
                'distance': distance,
                'coordinates': coordinates,
                'schwarzschild_radius': rs.to(u.m).value,
                'domain_size': domain_size
            }
    
    def create_mesh(self):
        if not DOLFINX_AVAILABLE:
            print("FEniCSx not available - cannot create mesh")
            return None
            
        rs = self.astronomical_data['schwarzschild_radius']
        domain_size = self.astronomical_data['domain_size']
        
        self.domain = mesh.create_box(
            comm,
            [[-domain_size, -domain_size, -domain_size], 
             [domain_size, domain_size, domain_size]],
            [self.mesh_resolution, self.mesh_resolution, self.mesh_resolution],
            cell_type=mesh.CellType.hexahedron
        )
        
        self.function_space = fem.functionspace(self.domain, ("Lagrange", 1))
        
        num_cells = self.domain.topology.index_map(self.domain.topology.dim).size_local
        num_vertices = self.domain.topology.index_map(0).size_local
        
        print(f"Mesh created:")
        print(f"  Cells: {num_cells}")
        print(f"  Vertices: {num_vertices}")
        print(f"  DOFs: {self.function_space.dofmap.index_map.size_local}")
        
        return self.domain
    
    def compute_schwarzschild_metric(self):
        if not DOLFINX_AVAILABLE or self.domain is None:
            print("Cannot compute metric - FEniCSx not available or mesh not created")
            return None
            
        rs = self.astronomical_data['schwarzschild_radius']
        
        g_tt = fem.Function(self.function_space)
        g_rr = fem.Function(self.function_space)
        
        def g_tt_expr(x):
            r = np.sqrt(x[0]**2 + x[1]**2 + x[2]**2)
            r = np.maximum(r, rs * 1.01)
            return -(1 - rs / r)
        
        def g_rr_expr(x):
            r = np.sqrt(x[0]**2 + x[1]**2 + x[2]**2)
            r = np.maximum(r, rs * 1.01)
            return 1 / (1 - rs / r)
        
        g_tt.interpolate(g_tt_expr)
        g_rr.interpolate(g_rr_expr)
        
        print("Schwarzschild metric components computed")
        return g_tt, g_rr
    
    def compute_effective_potential(self):
        if not DOLFINX_AVAILABLE or self.domain is None:
            print("Cannot compute potential - FEniCSx not available or mesh not created")
            return None
            
        rs = self.astronomical_data['schwarzschild_radius']
        domain_size = self.astronomical_data['domain_size']
        
        potential = fem.Function(self.function_space)
        
        def potential_expr(x):
            center = np.array([0.0, 0.0, 0.0])
            r = np.sqrt((x[0] - center[0])**2 + (x[1] - center[1])**2 + (x[2] - center[2])**2)
            r = np.maximum(r, rs * 1.1)
            return -rs / (2 * r)
        
        potential.interpolate(potential_expr)
        
        print("Effective potential computed")
        return potential
    
    def solve_geodesic_equations(self, initial_position, initial_velocity, t_max=100):
        if not DOLFINX_AVAILABLE:
            print("Cannot solve geodesics - FEniCSx not available")
            return None
            
        rs = self.astronomical_data['schwarzschild_radius']
        
        def geodesic_equations(t, y):
            r, theta, phi, pr, ptheta, pphi = y
            
            if r <= rs * 1.01:
                return np.zeros(6)
            
            f = 1 - rs / r
            
            dr_dt = f * pr
            dtheta_dt = ptheta / (r ** 2)
            dphi_dt = pphi / (r ** 2 * np.sin(theta) ** 2)
            
            dpr_dt = (rs / (2 * r ** 2)) * pr ** 2 / f - f * (ptheta ** 2 + pphi ** 2 / np.sin(theta) ** 2) / (r ** 3)
            dptheta_dt = pphi ** 2 * np.cos(theta) / (r ** 2 * np.sin(theta) ** 3)
            dpphi_dt = 0
            
            return np.array([dr_dt, dtheta_dt, dphi_dt, dpr_dt, dptheta_dt, dpphi_dt])
        
        from scipy.integrate import solve_ivp
        
        r0, theta0, phi0 = initial_position
        vr0, vtheta0, vphi0 = initial_velocity
        
        y0 = np.array([r0, theta0, phi0, vr0, vtheta0, vphi0])
        t_span = (0, t_max)
        t_eval = np.linspace(0, t_max, 1000)
        
        solution = solve_ivp(geodesic_equations, t_span, y0, t_eval=t_eval, 
                            method='RK45', rtol=1e-8, atol=1e-10)
        
        print(f"Geodesic solved with {len(solution.t)} time points")
        return solution
    
    def visualize_results(self, geodesic_solution=None):
        fig = plt.figure(figsize=(15, 10))
        
        if geodesic_solution is not None:
            ax1 = fig.add_subplot(221, projection='3d')
            ax1.set_title('Particle Geodesic in Curved Spacetime')
            
            r, theta, phi = geodesic_solution.y[0], geodesic_solution.y[1], geodesic_solution.y[2]
            x = r * np.sin(theta) * np.cos(phi)
            y = r * np.sin(theta) * np.sin(phi)
            z = r * np.cos(theta)
            
            ax1.plot(x, y, z, 'b-', linewidth=2, label='Geodesic')
            
            rs = self.astronomical_data['schwarzschild_radius']
            u_sphere = np.linspace(0, 2 * np.pi, 20)
            v_sphere = np.linspace(0, np.pi, 20)
            x_sphere = rs * np.outer(np.cos(u_sphere), np.sin(v_sphere))
            y_sphere = rs * np.outer(np.sin(u_sphere), np.sin(v_sphere))
            z_sphere = rs * np.outer(np.ones(np.size(u_sphere)), np.cos(v_sphere))
            ax1.plot_surface(x_sphere, y_sphere, z_sphere, color='black', alpha=0.7)
            
            ax1.set_xlabel('X')
            ax1.set_ylabel('Y')
            ax1.set_zlabel('Z')
            ax1.legend()
        
        ax2 = fig.add_subplot(222)
        ax2.set_title('Radial Distance vs Time')
        
        if geodesic_solution is not None:
            r_normalized = geodesic_solution.y[0] / self.astronomical_data['schwarzschild_radius']
            ax2.plot(geodesic_solution.t, r_normalized, 'r-', linewidth=2)
            ax2.axhline(y=1.0, color='black', linestyle='--', label='Event Horizon')
            ax2.set_xlabel('Time')
            ax2.set_ylabel('r / rs')
            ax2.legend()
            ax2.grid(True)
        
        ax3 = fig.add_subplot(223)
        ax3.set_title('Spacetime Curvature (g_tt)')
        
        r_range = np.linspace(self.astronomical_data['schwarzschild_radius'] * 1.01, 
                             20 * self.astronomical_data['schwarzschild_radius'], 1000)
        g_tt = -(1 - self.astronomical_data['schwarzschild_radius'] / r_range)
        
        ax3.plot(r_range / self.astronomical_data['schwarzschild_radius'], g_tt, 'b-', linewidth=2)
        ax3.axvline(x=1.0, color='red', linestyle='--', label='Event Horizon')
        ax3.set_xlabel('r / rs')
        ax3.set_ylabel('g_tt')
        ax3.legend()
        ax3.grid(True)
        
        ax4 = fig.add_subplot(224)
        ax4.set_title('Spacetime Curvature (g_rr)')
        
        g_rr = 1 / (1 - self.astronomical_data['schwarzschild_radius'] / r_range)
        ax4.plot(r_range / self.astronomical_data['schwarzschild_radius'], g_rr, 'r-', linewidth=2)
        ax4.axvline(x=1.0, color='red', linestyle='--', label='Event Horizon')
        ax4.set_xlabel('r / rs')
        ax4.set_ylabel('g_rr')
        ax4.set_ylim([0, 20])
        ax4.legend()
        ax4.grid(True)
        
        plt.tight_layout()
        plt.savefig('astronomical_fem_results.png', dpi=150, bbox_inches='tight')
        print("Results saved as 'astronomical_fem_results.png'")
        plt.show()

def main():
    print("=" * 80)
    print("ASTRONOMICAL BLACK HOLE SIMULATION WITH FEniCSx")
    print("=" * 80)
    
    print(f"Astropy available: {ASTROPY_AVAILABLE}")
    print(f"FEniCSx available: {DOLFINX_AVAILABLE}")
    
    if not DOLFINX_AVAILABLE:
        print("Warning: FEniCSx not available. Install with: pip install fenicsx")
        return
    
    bh_fem = AstronomicalBlackHoleFEM(blackhole_name="Sgr A*", mesh_resolution=20)
    
    print("\n" + "-" * 80)
    print("Creating astronomical mesh...")
    domain = bh_fem.create_mesh()
    
    if domain is not None:
        print("\n" + "-" * 80)
        print("Computing Schwarzschild metric...")
        metric_components = bh_fem.compute_schwarzschild_metric()
        
        print("\n" + "-" * 80)
        print("Computing effective potential...")
        potential = bh_fem.compute_effective_potential()
        
        print("\n" + "-" * 80)
        print("Solving geodesic equations...")
        rs = bh_fem.astronomical_data['schwarzschild_radius']
        initial_pos = (10 * rs, np.pi/2, 0)
        initial_vel = (-0.05, 0, 0.1)
        
        geodesic_solution = bh_fem.solve_geodesic_equations(initial_pos, initial_vel, t_max=200)
        
        print("\n" + "-" * 80)
        print("Generating visualizations...")
        bh_fem.visualize_results(geodesic_solution)
    
    print("\n" + "=" * 80)
    print("SIMULATION COMPLETE")
    print("=" * 80)

if __name__ == "__main__":
    main()
