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
DATA_DIR = ensure_dir(os.path.join(OUTPUT_DIR, "data"))
VTX_DIR = ensure_dir(os.path.join(DATA_DIR, "vtx"))
XDMF_DIR = ensure_dir(os.path.join(DATA_DIR, "xdmf"))
PLOT_DIR = ensure_dir(os.path.join(OUTPUT_DIR, "visualizations", "plots"))
PARAVIEW_DIR = ensure_dir(os.path.join(OUTPUT_DIR, "visualizations", "paraview"))

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

class TimeEvolvingGravitationalField:
    def __init__(self, 
                 domain_size=20.0, 
                 mesh_resolution=30,
                 schwarzschild_radius=2.0,
                 time_steps=50,
                 dt=0.1,
                 output_basename="gravitational_field",
                 output_format="both",
                 write_interval=1,
                 vtx_engine="BP4"):
        
        self.domain_size = domain_size
        self.mesh_resolution = mesh_resolution
        self.rs = schwarzschild_radius
        self.time_steps = time_steps
        self.dt = dt
        self.current_time = 0.0
        self.output_basename = output_basename
        self.output_format = output_format
        self.write_interval = max(1, int(write_interval))
        self.vtx_engine = vtx_engine
        
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
        print(f"Created mesh with {num_cells} cells")
        
        return domain
    
    def setup_function_space(self, domain):
        V = fem.functionspace(domain, ("Lagrange", 1))
        print(f"Function space DOFs: {V.dofmap.index_map.size_global}")
        return V
    
    def initial_potential(self, x):
        r = np.sqrt(x[0]**2 + x[1]**2 + x[2]**2)
        r = np.maximum(r, self.rs * 0.1)
        return -self.rs / (2 * r)
    
    def perturbed_source(self, x, t, omega=0.5, amplitude=0.1):
        r = np.sqrt(x[0]**2 + x[1]**2 + x[2]**2)
        r_safe = np.maximum(r, self.rs * 0.5)
        
        static_source = self.rs / (r_safe**2)
        
        perturbation = amplitude * np.sin(omega * t) * np.exp(-r / (5 * self.rs))
        
        return static_source + perturbation
    
    def boundary_condition(self, x):
        boundary_size = self.domain_size * 0.99
        return np.logical_or(
            np.logical_or(
                np.abs(x[0]) > boundary_size,
                np.abs(x[1]) > boundary_size
            ),
            np.abs(x[2]) > boundary_size
        )
    
    def solve_timestep(self, domain, V, u_n, t):
        u = ufl.TrialFunction(V)
        v = ufl.TestFunction(V)
        
        f = fem.Function(V)
        f.interpolate(lambda x: self.perturbed_source(x, t))
        
        bc_func = fem.Function(V)
        bc_func.interpolate(self.initial_potential)
        boundary_dofs = fem.locate_dofs_geometrical(V, self.boundary_condition)
        bc = fem.dirichletbc(bc_func, boundary_dofs)
        
        alpha = fem.Constant(domain, default_scalar_type(1.0 / self.dt))
        
        a = (alpha * u * v + ufl.dot(ufl.grad(u), ufl.grad(v))) * ufl.dx
        L = (alpha * u_n * v + f * v) * ufl.dx
        
        problem = LinearProblem(
            a, L, bcs=[bc],
            petsc_options={"ksp_type": "cg", "pc_type": "hypre", "ksp_rtol": 1e-8}
        )
        
        u_new = problem.solve()
        
        return u_new
    
    def compute_field_energy(self, domain, u):
        energy_form = ufl.inner(u, u) * ufl.dx
        energy = comm.allreduce(
            fem.assemble_scalar(fem.form(energy_form)),
            op=MPI.SUM
        )
        return np.sqrt(energy)
    
    def compute_derived_fields(self, domain, V, u):
        grad_u = ufl.grad(u)
        
        V_vec = fem.functionspace(domain, ("DG", 0, (3,)))
        field_strength = fem.Function(V_vec)
        field_strength.name = "field_strength"
        
        field_expr = fem.Expression(-grad_u, V_vec.element.interpolation_points())
        field_strength.interpolate(field_expr)
        
        V_scalar = fem.functionspace(domain, ("DG", 0))
        magnitude = fem.Function(V_scalar)
        magnitude.name = "field_magnitude"
        
        mag_expr = fem.Expression(ufl.sqrt(ufl.dot(grad_u, grad_u)), V_scalar.element.interpolation_points())
        magnitude.interpolate(mag_expr)
        
        energy_density = fem.Function(V_scalar)
        energy_density.name = "energy_density"
        energy_expr = fem.Expression(0.5 * ufl.dot(grad_u, grad_u), V_scalar.element.interpolation_points())
        energy_density.interpolate(energy_expr)
        
        curvature = fem.Function(V_scalar)
        curvature.name = "curvature_scalar"
        laplacian = ufl.div(grad_u)
        curv_expr = fem.Expression(ufl.sqrt(laplacian**2 + 1e-10), V_scalar.element.interpolation_points())
        curvature.interpolate(curv_expr)
        
        return field_strength, magnitude, energy_density, curvature
    
    def simulate(self):
        print("=" * 70)
        print("TIME-EVOLVING GRAVITATIONAL FIELD SIMULATION")
        print("Solving time-dependent Poisson equation in flat 3D spacetime")
        print("=" * 70)
        
        print(f"\nSimulation parameters:")
        print(f"  Domain size: ±{self.domain_size}")
        print(f"  Schwarzschild radius: {self.rs}")
        print(f"  Mesh resolution: {self.mesh_resolution}³")
        print(f"  Time steps: {self.time_steps}")
        print(f"  Time step size: {self.dt}")
        print(f"  Total time: {self.time_steps * self.dt}")
        
        domain = self.create_mesh()
        V = self.setup_function_space(domain)
        
        u_n = fem.Function(V)
        u_n.interpolate(self.initial_potential)
        u_n.name = "gravitational_potential"
        
        field_strength, magnitude, energy_density, curvature = self.compute_derived_fields(domain, V, u_n)
        
        print("\nStarting time evolution...")
        
        write_vtx = self.output_format in ["vtx", "both"]
        write_xdmf = self.output_format in ["xdmf", "both"]
        
        base_path = self.output_basename
        vtx_potential_path = os.path.join(VTX_DIR, f"{base_path}_potential.bp")
        vtx_scalar_path = os.path.join(VTX_DIR, f"{base_path}_scalar_fields.bp")
        vtx_vector_path = os.path.join(VTX_DIR, f"{base_path}_vector_field.bp")
        xdmf_path = os.path.join(XDMF_DIR, f"{base_path}.xdmf")
        
        energies = []
        times = []
        
        vtx_potential = None
        vtx_scalar_fields = None
        vtx_vector_field = None
        
        xdmf = None
        if write_xdmf:
            xdmf = XDMFFile(comm, xdmf_path, "w")
            xdmf.write_mesh(domain)
            xdmf.write_function(u_n, 0.0)
            xdmf.write_function(magnitude, 0.0)
            xdmf.write_function(energy_density, 0.0)
            xdmf.write_function(curvature, 0.0)
        
        if write_vtx:
            vtx_potential = VTXWriter(comm, vtx_potential_path, [u_n], engine=self.vtx_engine)
            vtx_scalar_fields = VTXWriter(comm, vtx_scalar_path, [magnitude, energy_density, curvature], engine=self.vtx_engine)
            vtx_vector_field = VTXWriter(comm, vtx_vector_path, [field_strength], engine=self.vtx_engine)
            
            vtx_potential.write(0.0)
            vtx_scalar_fields.write(0.0)
            vtx_vector_field.write(0.0)
        
        for step in range(self.time_steps):
            self.current_time = (step + 1) * self.dt
            u_new = self.solve_timestep(domain, V, u_n, self.current_time)
            energy = self.compute_field_energy(domain, u_new)
            energies.append(energy)
            times.append(self.current_time)
            u_n.x.array[:] = u_new.x.array
            field_strength, magnitude, energy_density, curvature = self.compute_derived_fields(domain, V, u_n)
            
            if (step + 1) % self.write_interval == 0:
                if write_vtx:
                    vtx_potential.write(self.current_time)
                    vtx_scalar_fields.write(self.current_time)
                    vtx_vector_field.write(self.current_time)
                
                if write_xdmf and xdmf is not None:
                    xdmf.write_function(u_n, self.current_time)
                    xdmf.write_function(magnitude, self.current_time)
                    xdmf.write_function(energy_density, self.current_time)
                    xdmf.write_function(curvature, self.current_time)
            
            if (step + 1) % 10 == 0:
                print(f"  Step {step+1}/{self.time_steps}, t={self.current_time:.2f}, Energy={energy:.2e}")
        
        if xdmf is not None:
            xdmf.close()
            
        if write_vtx:
            vtx_potential.close()
            vtx_scalar_fields.close()
            vtx_vector_field.close()
        
        print(f"\nTime evolution complete!")
        if write_vtx:
            print(f"VTX results saved to:")
            print(f"  - {vtx_potential_path}     (gravitational potential)")
            print(f"  - {vtx_scalar_path} (field magnitude, energy density, curvature)")
            print(f"  - {vtx_vector_path}  (field strength vector)")
        if write_xdmf:
            print(f"XDMF results saved to {xdmf_path}")
        
        return times, energies, u_n
    
    def plot_energy_evolution(self, times, energies):
        plt.figure(figsize=(10, 6))
        plt.plot(times, energies, 'b-', linewidth=2, label='Field Energy')
        plt.xlabel('Time', fontsize=12)
        plt.ylabel('L² Norm of Potential', fontsize=12)
        plt.title('Gravitational Field Energy Evolution', fontsize=14)
        plt.grid(True, alpha=0.3)
        plt.legend(fontsize=11)
        plt.tight_layout()
        
        plot_path = os.path.join(PLOT_DIR, 'energy_evolution.png')
        plt.savefig(plot_path, dpi=150, bbox_inches='tight')
        print(f"Energy evolution plot saved to '{plot_path}'")
        plt.close()

class ParticleTrajectories:
    def __init__(self, schwarzschild_radius=2.0):
        self.rs = schwarzschild_radius
        
    def gravitational_acceleration(self, position):
        r = np.linalg.norm(position)
        r_safe = max(r, self.rs * 0.5)
        
        acc_magnitude = self.rs / (2 * r_safe**2)
        
        acceleration = -acc_magnitude * position / r_safe
        
        return acceleration
    
    def rk4_step(self, position, velocity, dt):
        k1_v = self.gravitational_acceleration(position)
        k1_x = velocity
        
        k2_v = self.gravitational_acceleration(position + 0.5 * dt * k1_x)
        k2_x = velocity + 0.5 * dt * k1_v
        
        k3_v = self.gravitational_acceleration(position + 0.5 * dt * k2_x)
        k3_x = velocity + 0.5 * dt * k2_v
        
        k4_v = self.gravitational_acceleration(position + dt * k3_x)
        k4_x = velocity + dt * k3_v
        
        new_velocity = velocity + (dt / 6.0) * (k1_v + 2*k2_v + 2*k3_v + k4_v)
        new_position = position + (dt / 6.0) * (k1_x + 2*k2_x + 2*k3_x + k4_x)
        
        return new_position, new_velocity
    
    def simulate_particle(self, initial_position, initial_velocity, time_steps, dt):
        positions = [initial_position.copy()]
        velocities = [initial_velocity.copy()]
        
        pos = initial_position.copy()
        vel = initial_velocity.copy()
        
        for _ in range(time_steps):
            if np.linalg.norm(pos) < self.rs * 1.01:
                break
                
            pos, vel = self.rk4_step(pos, vel, dt)
            positions.append(pos.copy())
            velocities.append(vel.copy())
        
        return np.array(positions), np.array(velocities)
    
    def plot_trajectories(self, n_particles=8):
        fig = plt.figure(figsize=(12, 10))
        ax = fig.add_subplot(111, projection='3d')
    
        dt = 0.1
        time_steps = 500
        
        colors = plt.cm.viridis(np.linspace(0, 1, n_particles))
        
        for i in range(n_particles):
            angle = 2 * np.pi * i / n_particles
            r0 = 10 * self.rs
            
            initial_pos = np.array([
                r0 * np.cos(angle),
                r0 * np.sin(angle),
                0.5 * self.rs * (-1)**i
            ])
            
            speed = 0.3
            tangent_angle = angle + np.pi / 2
            initial_vel = np.array([
                speed * np.cos(tangent_angle) - 0.05 * np.cos(angle),
                speed * np.sin(tangent_angle) - 0.05 * np.sin(angle),
                0.0
            ])
            
            positions, velocities = self.simulate_particle(
                initial_pos, initial_vel, time_steps, dt
            )
            
            ax.plot(positions[:, 0], positions[:, 1], positions[:, 2],
                   color=colors[i], linewidth=1.5, alpha=0.8,
                   label=f'Particle {i+1}')
        
        u = np.linspace(0, 2 * np.pi, 50)
        v = np.linspace(0, np.pi, 50)
        x_sphere = self.rs * np.outer(np.cos(u), np.sin(v))
        y_sphere = self.rs * np.outer(np.sin(u), np.sin(v))
        z_sphere = self.rs * np.outer(np.ones(np.size(u)), np.cos(v))
        ax.plot_surface(x_sphere, y_sphere, z_sphere, color='black', alpha=0.9)
        
        max_range = 12 * self.rs
        ax.set_xlim([-max_range, max_range])
        ax.set_ylim([-max_range, max_range])
        ax.set_zlim([-max_range, max_range])
    
        ax.set_xlabel('X', fontsize=11)
        ax.set_ylabel('Y', fontsize=11)
        ax.set_zlabel('Z', fontsize=11)
        ax.set_title('Particle Trajectories in Time-Evolving Gravitational Field', fontsize=13)
        
        plt.tight_layout()
        
        plot_path = os.path.join(PLOT_DIR, 'particle_trajectories.png')
        plt.savefig(plot_path, dpi=150, bbox_inches='tight')
        print(f"Particle trajectories saved to '{plot_path}'")
        plt.close()

def main():
    if not DOLFINX_AVAILABLE:
        print("ERROR: FEniCSx is required for this simulation")
        print("Please install DOLFINx or run in a DOLFINx container")
        return
    
    schwarzschild_radius = 2.0
    domain_size = 20.0
    mesh_resolution = 30
    time_steps = 50
    dt = 0.1
    
    field_sim = TimeEvolvingGravitationalField(
        domain_size=domain_size,
        mesh_resolution=mesh_resolution,
        schwarzschild_radius=schwarzschild_radius,
        time_steps=time_steps,
        dt=dt,
        output_basename="gravitational_field",
        output_format="both",
        write_interval=1,
        vtx_engine="BP4"
    )
    
    times, energies, final_field = field_sim.simulate()
    
    print("\n" + "=" * 70)
    print("GENERATING VISUALIZATIONS")
    print("=" * 70)
    
    field_sim.plot_energy_evolution(times, energies)
    
    particle_sim = ParticleTrajectories(schwarzschild_radius=schwarzschild_radius)
    particle_sim.plot_trajectories(n_particles=8)
    
    print("\n" + "=" * 70)
    print("CREATING PARAVIEW SETUP FILES")
    print("=" * 70)
    
    try:
        import paraview_setup
        paraview_setup.OUTPUT_DIR = PARAVIEW_DIR
        paraview_setup.create_paraview_python_script()
        paraview_setup.create_quick_start_guide()
        print("ParaView helper files created successfully!")
    except Exception as e:
        print(f"Note: Could not create ParaView setup files: {e}")
    
    print("\n" + "=" * 70)
    print("SIMULATION COMPLETE")
    print("=" * 70)
    print("\nGenerated files:")
    print("  Data files:")
    print(f"    - {os.path.join(VTX_DIR, 'gravitational_field_potential.bp')}     (VTX format - potential field)")
    print(f"    - {os.path.join(VTX_DIR, 'gravitational_field_scalar_fields.bp')} (VTX format - magnitude, energy, curvature)")
    print(f"    - {os.path.join(VTX_DIR, 'gravitational_field_vector_field.bp')}  (VTX format - vector field)")
    print(f"    - {os.path.join(XDMF_DIR, 'gravitational_field.xdmf')}            (XDMF format - all fields)")
    print("  Visualization files:")
    print(f"    - {os.path.join(PLOT_DIR, 'energy_evolution.png')}                (Field energy over time)")
    print(f"    - {os.path.join(PLOT_DIR, 'particle_trajectories.png')}           (Test particle orbits)")
    print("  ParaView helper files:")
    print(f"    - {os.path.join(PARAVIEW_DIR, 'paraview_script.py')}              (Auto-setup script for ParaView)")
    print(f"    - {os.path.join(PARAVIEW_DIR, 'PARAVIEW_GUIDE.md')}               (Detailed visualization guide)")
    
    print("\n" + "=" * 70)
    print("PARAVIEW VISUALIZATION GUIDE")
    print("=" * 70)
    print("\n1. QUICK START (EASIEST):")
    print("   - Open ParaView")
    print("   - Tools → Python Shell → Run Script")
    print("   - Select 'paraview_script.py'")
    print("   - Visualization auto-configured!")
    
    print("\n2. MANUAL SETUP:")
    print("   OPTION A (Recommended - time series):")
    print("      - File → Open → Select one of the BP files:")
    print("        * gravitational_field_potential.bp    (for potential field)")
    print("        * gravitational_field_scalar_fields.bp (for magnitude, energy density)")
    print("        * gravitational_field_vector_field.bp  (for vector field)")
    print("      - Click 'Apply' in Properties panel")
    print("   OPTION B (Alternative - all fields in one file):")
    print("      - File → Open → Select 'gravitational_field.xdmf'")
    print("      - Click 'Apply' in Properties panel")
    
    print("\n3. AVAILABLE FIELDS TO VISUALIZE:")
    print("   - gravitational_potential : Gravitational potential field")
    print("   - field_magnitude        : Magnitude of gravitational force")
    print("   - energy_density         : Energy density distribution")
    print("   - curvature_scalar       : Curvature-like quantity")
    print("   - field_strength         : 3D vector field (gravitational force)")
    
    print("\n4. RECOMMENDED VISUALIZATION STEPS:")
    print("\n   A. Volume Rendering (Best for initial view):")
    print("      - Select 'field_magnitude' from the coloring dropdown")
    print("      - Change representation to 'Volume'")
    print("      - Adjust opacity in Color Map Editor")
    print("      - Use 'Rescale to Custom Range' for better contrast")
    
    print("\n   B. Slice View (For cross-sections):")
    print("      - Filters → Slice")
    print("      - Set Origin to [0, 0, 0]")
    print("      - Color by 'energy_density' or 'field_magnitude'")
    print("      - Enable 'Show Plane' for orientation")
    
    print("\n   C. Contour/Isosurface (For equipotential surfaces):")
    print("      - Filters → Contour")
    print("      - Select 'gravitational_potential' as Contour By")
    print("      - Add 5-10 isovalues")
    print("      - Color by same field or different field")
    
    print("\n   D. Vector Field (Glyph visualization):")
    print("      - Filters → Glyph")
    print("      - Set 'Glyph Type' to Arrow")
    print("      - Set 'Vectors' to 'field_strength'")
    print("      - Scale by 'field_magnitude'")
    print("      - Adjust 'Scale Factor' for visibility")
    
    print("\n   E. Streamlines (Field lines):")
    print("      - Filters → Stream Tracer")
    print("      - Set 'Vectors' to 'field_strength'")
    print("      - Choose seed type (Point Cloud or Line)")
    print("      - Adjust integration parameters")
    
    print("\n5. ANIMATION CONTROLS:")
    print("   - Use time slider at top to scrub through timesteps")
    print("   - Click play button to animate evolution")
    print("   - Adjust animation speed in Settings")
    
    print("\n6. TIPS FOR BETTER VISUALIZATION:")
    print("   - Use logarithmic color scale for high dynamic range")
    print("   - Combine multiple filters (Clip + Slice)")
    print("   - Adjust camera position for better perspective")
    print("   - Save screenshots or animations via File → Save Screenshot/Animation")
    print("   - Use 'Color Map Editor' to customize color schemes")
    
    print("\n" + "=" * 70)

if __name__ == "__main__":
    main()

