import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import os

DOLFINX_AVAILABLE = False
comm = None

def ensure_dir(path):
    os.makedirs(path, exist_ok=True)
    return path

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
OUTPUT_DIR = ensure_dir(os.path.join(SCRIPT_DIR, "outputs"))
SOLAR_DATA_DIR = ensure_dir(os.path.join(OUTPUT_DIR, "data"))
SOLAR_VTX_DIR = ensure_dir(os.path.join(SOLAR_DATA_DIR, "vtx"))
SOLAR_PLOTS_DIR = ensure_dir(os.path.join(OUTPUT_DIR, "plots"))

try:
    from dolfinx import mesh, fem, default_scalar_type
    from dolfinx.fem.petsc import LinearProblem
    from dolfinx.io import VTXWriter, XDMFFile
    from mpi4py import MPI
    import ufl
    from petsc4py import PETSc
    comm = MPI.COMM_WORLD
    DOLFINX_AVAILABLE = True
    print("FEniCSx loaded successfully")
except Exception as e:
    print(f"Warning: FEniCSx not available: {e}")
    print("Continuing without FEM features...")

class SolarSystemData:
    def __init__(self):
        self.G = 6.67430e-11
        self.c = 299792458.0
        self.AU = 1.496e11
        
        self.bodies = {
            'Sun': {
                'mass': 1.989e30,
                'radius': 6.96e8,
                'position': np.array([0.0, 0.0, 0.0]),
                'velocity': np.array([0.0, 0.0, 0.0]),
                'color': 'yellow'
            },
            'Mercury': {
                'mass': 3.301e23,
                'radius': 2.439e6,
                'semi_major_axis': 0.387 * self.AU,
                'eccentricity': 0.2056,
                'orbital_period': 87.969 * 86400,
                'inclination': 7.005 * np.pi/180,
                'color': 'gray'
            },
            'Venus': {
                'mass': 4.867e24,
                'radius': 6.051e6,
                'semi_major_axis': 0.723 * self.AU,
                'eccentricity': 0.0068,
                'orbital_period': 224.701 * 86400,
                'inclination': 3.395 * np.pi/180,
                'color': 'orange'
            },
            'Earth': {
                'mass': 5.972e24,
                'radius': 6.371e6,
                'semi_major_axis': 1.0 * self.AU,
                'eccentricity': 0.0167,
                'orbital_period': 365.256 * 86400,
                'inclination': 0.0 * np.pi/180,
                'color': 'blue'
            },
            'Mars': {
                'mass': 6.417e23,
                'radius': 3.389e6,
                'semi_major_axis': 1.524 * self.AU,
                'eccentricity': 0.0934,
                'orbital_period': 686.980 * 86400,
                'inclination': 1.850 * np.pi/180,
                'color': 'red'
            },
            'Jupiter': {
                'mass': 1.898e27,
                'radius': 6.991e7,
                'semi_major_axis': 5.203 * self.AU,
                'eccentricity': 0.0484,
                'orbital_period': 4332.589 * 86400,
                'inclination': 1.303 * np.pi/180,
                'color': 'brown'
            },
            'Saturn': {
                'mass': 5.683e26,
                'radius': 5.823e7,
                'semi_major_axis': 9.537 * self.AU,
                'eccentricity': 0.0539,
                'orbital_period': 10759.22 * 86400,
                'inclination': 2.485 * np.pi/180,
                'color': 'goldenrod'
            }
        }
        
    def initialize_orbits(self, t=0):
        for name, body in self.bodies.items():
            if name == 'Sun':
                continue
                
            a = body['semi_major_axis']
            e = body['eccentricity']
            T = body['orbital_period']
            inc = body['inclination']
            
            mean_anomaly = 2 * np.pi * t / T
            eccentric_anomaly = self.solve_kepler(mean_anomaly, e)
            
            r = a * (1 - e * np.cos(eccentric_anomaly))
            theta = 2 * np.arctan2(np.sqrt(1 + e) * np.sin(eccentric_anomaly/2),
                                   np.sqrt(1 - e) * np.cos(eccentric_anomaly/2))
            
            x = r * np.cos(theta)
            y = r * np.sin(theta) * np.cos(inc)
            z = r * np.sin(theta) * np.sin(inc)
            
            body['position'] = np.array([x, y, z])
            
            n = 2 * np.pi / T
            v_mag = n * a / np.sqrt(1 - e**2) * np.sqrt(1 + 2*e*np.cos(theta) + e**2)
            vx = -v_mag * np.sin(theta)
            vy = v_mag * np.cos(theta) * np.cos(inc)
            vz = v_mag * np.cos(theta) * np.sin(inc)
            
            body['velocity'] = np.array([vx, vy, vz])
    
    def solve_kepler(self, M, e, tol=1e-10):
        E = M
        for _ in range(100):
            E_new = M + e * np.sin(E)
            if abs(E_new - E) < tol:
                return E_new
            E = E_new
        return E

class SpacetimeEvolution:
    def __init__(self,
                 domain_size=15.0,
                 mesh_resolution=30,
                 simulation_years=10.0,
                 time_steps=100,
                 use_post_newtonian=True):
        
        self.solar_data = SolarSystemData()
        self.domain_size = domain_size * self.solar_data.AU
        self.mesh_resolution = mesh_resolution
        self.simulation_time = simulation_years * 365.25 * 86400
        self.time_steps = time_steps
        self.dt = self.simulation_time / time_steps
        self.use_post_newtonian = use_post_newtonian
        
        self.G = self.solar_data.G
        self.c = self.solar_data.c
        
    def create_mesh(self):
        L = self.domain_size
        n = self.mesh_resolution
        
        domain = mesh.create_box(
            comm,
            [[-L, -L, -L], [L, L, L]],
            [n, n, n],
            cell_type=mesh.CellType.hexahedron
        )
        
        num_cells = domain.topology.index_map(domain.topology.dim).size_local
        print(f"Created spacetime mesh with {num_cells} cells")
        print(f"Domain size: ±{L/self.solar_data.AU:.1f} AU")
        
        return domain
    
    def setup_function_space(self, domain):
        V_scalar = fem.functionspace(domain, ("Lagrange", 1))
        V_vector = fem.functionspace(domain, ("Lagrange", 1, (3,)))
        
        print(f"Scalar space DOFs: {V_scalar.dofmap.index_map.size_global}")
        print(f"Vector space DOFs: {V_vector.dofmap.index_map.size_global}")
        
        return V_scalar, V_vector
    
    def compute_newtonian_potential(self, x, t):
        self.solar_data.initialize_orbits(t)
        
        potential = np.zeros(x.shape[1])
        
        for name, body in self.solar_data.bodies.items():
            pos = body['position']
            mass = body['mass']
            
            dx = x[0] - pos[0]
            dy = x[1] - pos[1]
            dz = x[2] - pos[2]
            
            r = np.sqrt(dx**2 + dy**2 + dz**2)
            r_safe = np.maximum(r, body['radius'])
            
            potential += -self.G * mass / r_safe
        
        return potential
    
    def compute_post_newtonian_correction(self, x, t):
        self.solar_data.initialize_orbits(t)
        
        pn_correction = np.zeros(x.shape[1])
        
        for name, body in self.solar_data.bodies.items():
            pos = body['position']
            vel = body['velocity']
            mass = body['mass']
            
            dx = x[0] - pos[0]
            dy = x[1] - pos[1]
            dz = x[2] - pos[2]
            
            r = np.sqrt(dx**2 + dy**2 + dz**2)
            r_safe = np.maximum(r, body['radius'])
            
            v_squared = np.sum(vel**2)
            
            pn_term = (self.G * mass / r_safe) * (
                (v_squared / (2 * self.c**2)) + 
                (self.G * mass / (r_safe * self.c**2))
            )
            
            pn_correction += pn_term
        
        return pn_correction
    
    def compute_metric_perturbation(self, domain, V_scalar, t):
        u = fem.Function(V_scalar)
        u.name = "metric_perturbation"
        
        u.interpolate(lambda x: self.compute_newtonian_potential(x, t) / self.c**2)
        
        if self.use_post_newtonian:
            u_pn = fem.Function(V_scalar)
            u_pn.interpolate(lambda x: self.compute_post_newtonian_correction(x, t))
            u.x.array[:] += u_pn.x.array[:]
        
        return u
    
    def compute_curvature_scalar(self, domain, V_scalar, metric_pert):
        curvature = fem.Function(V_scalar)
        curvature.name = "ricci_scalar"
        
        grad_u = ufl.grad(metric_pert)
        laplacian = ufl.div(grad_u)
        
        curv_expr = fem.Expression(
            8 * np.pi * self.G / self.c**4 * ufl.sqrt(laplacian**2 + 1e-20),
            V_scalar.element.interpolation_points()
        )
        curvature.interpolate(curv_expr)
        
        return curvature
    
    def compute_gravitational_field(self, domain, V_vector, metric_pert):
        field = fem.Function(V_vector)
        field.name = "gravitational_field"
        
        grad_u = ufl.grad(metric_pert)
        field_expr = fem.Expression(-grad_u * self.c**2, V_vector.element.interpolation_points())
        field.interpolate(field_expr)
        
        return field
    
    def solve_einstein_constraints(self, domain, V_scalar, u_n, t):
        u = ufl.TrialFunction(V_scalar)
        v = ufl.TestFunction(V_scalar)
        
        f = fem.Function(V_scalar)
        f.interpolate(lambda x: 8 * np.pi * self.G / self.c**4 * 
                      self.compute_matter_density(x, t))
        
        a = ufl.dot(ufl.grad(u), ufl.grad(v)) * ufl.dx
        L = f * v * ufl.dx
        
        problem = LinearProblem(
            a, L,
            petsc_options={"ksp_type": "cg", "pc_type": "hypre", "ksp_rtol": 1e-8}
        )
        
        u_new = problem.solve()
        
        return u_new
    
    def compute_matter_density(self, x, t):
        self.solar_data.initialize_orbits(t)
        
        density = np.zeros(x.shape[1])
        
        for name, body in self.solar_data.bodies.items():
            pos = body['position']
            mass = body['mass']
            radius = body['radius']
            
            dx = x[0] - pos[0]
            dy = x[1] - pos[1]
            dz = x[2] - pos[2]
            
            r = np.sqrt(dx**2 + dy**2 + dz**2)
            
            gaussian_width = radius * 10
            density += (mass / (gaussian_width**3 * (2*np.pi)**1.5)) * \
                      np.exp(-r**2 / (2 * gaussian_width**2))
        
        return density
    
    def simulate(self):
        print("=" * 70)
        print("SOLAR SYSTEM SPACETIME EVOLUTION SIMULATION")
        print("Post-Newtonian Approximation with Real Orbital Data")
        print("=" * 70)
        
        print(f"\nSimulation parameters:")
        print(f"  Domain size: ±{self.domain_size/self.solar_data.AU:.1f} AU")
        print(f"  Mesh resolution: {self.mesh_resolution}³")
        print(f"  Simulation time: {self.simulation_time/(365.25*86400):.1f} years")
        print(f"  Time steps: {self.time_steps}")
        print(f"  Time step size: {self.dt/86400:.1f} days")
        print(f"  Post-Newtonian corrections: {self.use_post_newtonian}")
        
        domain = self.create_mesh()
        V_scalar, V_vector = self.setup_function_space(domain)
        
        print("\nInitializing solar system orbits...")
        self.solar_data.initialize_orbits(0)
        
        print("\nStarting spacetime evolution...")
        
        vtx_metric_path = os.path.join(SOLAR_VTX_DIR, "metric_perturbation.bp")
        vtx_curvature_path = os.path.join(SOLAR_VTX_DIR, "ricci_curvature.bp")
        vtx_field_path = os.path.join(SOLAR_VTX_DIR, "gravitational_field.bp")
        
        metric_pert = self.compute_metric_perturbation(domain, V_scalar, 0)
        curvature = self.compute_curvature_scalar(domain, V_scalar, metric_pert)
        grav_field = self.compute_gravitational_field(domain, V_vector, metric_pert)
        
        vtx_metric = VTXWriter(comm, vtx_metric_path, [metric_pert], engine="BP4")
        vtx_curvature = VTXWriter(comm, vtx_curvature_path, [curvature], engine="BP4")
        vtx_field = VTXWriter(comm, vtx_field_path, [grav_field], engine="BP4")
        
        vtx_metric.write(0.0)
        vtx_curvature.write(0.0)
        vtx_field.write(0.0)
        
        planetary_trajectories = {name: [] for name in self.solar_data.bodies.keys()}
        time_array = []
        
        for step in range(self.time_steps):
            current_time = (step + 1) * self.dt
            time_years = current_time / (365.25 * 86400)
            
            metric_pert = self.compute_metric_perturbation(domain, V_scalar, current_time)
            curvature = self.compute_curvature_scalar(domain, V_scalar, metric_pert)
            grav_field = self.compute_gravitational_field(domain, V_vector, metric_pert)
            
            vtx_metric.write(time_years)
            vtx_curvature.write(time_years)
            vtx_field.write(time_years)
            
            for name, body in self.solar_data.bodies.items():
                planetary_trajectories[name].append(body['position'].copy())
            time_array.append(time_years)
            
            if (step + 1) % max(1, self.time_steps // 10) == 0:
                print(f"  Step {step+1}/{self.time_steps}, t={time_years:.2f} years")
        
        vtx_metric.close()
        vtx_curvature.close()
        vtx_field.close()
        
        print(f"\nSpacetime evolution complete!")
        print(f"Results saved to {SOLAR_VTX_DIR}")
        
        return planetary_trajectories, time_array
    
    def plot_solar_system_evolution(self, trajectories, times):
        fig = plt.figure(figsize=(16, 12))
        ax = fig.add_subplot(111, projection='3d')
        
        for name, positions in trajectories.items():
            if len(positions) == 0:
                continue
            
            positions = np.array(positions)
            body_data = self.solar_data.bodies[name]
            
            if name == 'Sun':
                ax.scatter(0, 0, 0, c=body_data['color'], s=200, 
                          marker='*', label='Sun', edgecolors='orange', linewidths=2)
            else:
                ax.plot(positions[:, 0]/self.solar_data.AU,
                       positions[:, 1]/self.solar_data.AU,
                       positions[:, 2]/self.solar_data.AU,
                       c=body_data['color'], linewidth=2, label=name, alpha=0.7)
                
                ax.scatter(positions[-1, 0]/self.solar_data.AU,
                          positions[-1, 1]/self.solar_data.AU,
                          positions[-1, 2]/self.solar_data.AU,
                          c=body_data['color'], s=50, marker='o')
        
        max_range = self.domain_size / self.solar_data.AU
        ax.set_xlim([-max_range, max_range])
        ax.set_ylim([-max_range, max_range])
        ax.set_zlim([-max_range/10, max_range/10])
        
        ax.set_xlabel('X (AU)', fontsize=12)
        ax.set_ylabel('Y (AU)', fontsize=12)
        ax.set_zlabel('Z (AU)', fontsize=12)
        ax.set_title(f'Solar System Evolution over {times[-1]:.1f} Years\n(Curved Spacetime Simulation)', 
                    fontsize=14, fontweight='bold')
        ax.legend(loc='upper right', fontsize=10)
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plot_path = os.path.join(SOLAR_PLOTS_DIR, 'solar_system_trajectories.png')
        plt.savefig(plot_path, dpi=150, bbox_inches='tight')
        print(f"Solar system trajectories saved to '{plot_path}'")
        plt.close()
        
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        
        for idx, (name, positions) in enumerate(list(trajectories.items())[1:5]):
            row, col = idx // 2, idx % 2
            ax = axes[row, col]
            
            positions = np.array(positions)
            ax.plot(positions[:, 0]/self.solar_data.AU,
                   positions[:, 1]/self.solar_data.AU,
                   c=self.solar_data.bodies[name]['color'], linewidth=2)
            
            ax.scatter(0, 0, c='yellow', s=100, marker='*', 
                      edgecolors='orange', linewidths=1)
            ax.scatter(positions[-1, 0]/self.solar_data.AU,
                      positions[-1, 1]/self.solar_data.AU,
                      c=self.solar_data.bodies[name]['color'], s=50, marker='o')
            
            ax.set_xlabel('X (AU)', fontsize=10)
            ax.set_ylabel('Y (AU)', fontsize=10)
            ax.set_title(f'{name} Orbit', fontsize=12, fontweight='bold')
            ax.grid(True, alpha=0.3)
            ax.set_aspect('equal')
        
        plt.tight_layout()
        plot_path = os.path.join(SOLAR_PLOTS_DIR, 'inner_planets_orbits.png')
        plt.savefig(plot_path, dpi=150, bbox_inches='tight')
        print(f"Inner planets orbits saved to '{plot_path}'")
        plt.close()

def main():
    if not DOLFINX_AVAILABLE:
        print("ERROR: FEniCSx is required for this simulation")
        print("Please install DOLFINx or run in a DOLFINx container")
        return
    
    simulation_years = 10.0
    domain_size_au = 15.0
    mesh_resolution = 25
    time_steps = 100
    
    spacetime_sim = SpacetimeEvolution(
        domain_size=domain_size_au,
        mesh_resolution=mesh_resolution,
        simulation_years=simulation_years,
        time_steps=time_steps,
        use_post_newtonian=True
    )
    
    trajectories, times = spacetime_sim.simulate()
    
    print("\n" + "=" * 70)
    print("GENERATING VISUALIZATIONS")
    print("=" * 70)
    
    spacetime_sim.plot_solar_system_evolution(trajectories, times)
    
    print("\n" + "=" * 70)
    print("SIMULATION COMPLETE")
    print("=" * 70)
    print("\nGenerated files:")
    print(f"  Data files:")
    print(f"    - {os.path.join(SOLAR_VTX_DIR, 'metric_perturbation.bp')}")
    print(f"    - {os.path.join(SOLAR_VTX_DIR, 'ricci_curvature.bp')}")
    print(f"    - {os.path.join(SOLAR_VTX_DIR, 'gravitational_field.bp')}")
    print(f"  Visualization files:")
    print(f"    - {os.path.join(SOLAR_PLOTS_DIR, 'solar_system_trajectories.png')}")
    print(f"    - {os.path.join(SOLAR_PLOTS_DIR, 'inner_planets_orbits.png')}")
    
    print("\n" + "=" * 70)
    print("PARAVIEW VISUALIZATION INSTRUCTIONS")
    print("=" * 70)
    print("\n1. Open ParaView and load:")
    print(f"   - {os.path.join(SOLAR_VTX_DIR, 'metric_perturbation.bp')}")
    print(f"   - {os.path.join(SOLAR_VTX_DIR, 'ricci_curvature.bp')}")
    print(f"   - {os.path.join(SOLAR_VTX_DIR, 'gravitational_field.bp')}")
    
    print("\n2. Visualization suggestions:")
    print("   A. Metric Perturbation (Spacetime Curvature):")
    print("      - Show as volume rendering or contours")
    print("      - Color by metric_perturbation value")
    print("      - Animate over time to see how geometry changes")
    
    print("\n   B. Ricci Curvature (Matter Distribution):")
    print("      - Show where spacetime is most curved")
    print("      - Use isosurfaces to highlight regions")
    
    print("\n   C. Gravitational Field:")
    print("      - Add glyphs to show field direction")
    print("      - Use streamlines to visualize field lines")
    
    print("\n" + "=" * 70)

if __name__ == "__main__":
    main()

