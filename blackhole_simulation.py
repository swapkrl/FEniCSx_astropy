import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

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

class TimeEvolvingGravitationalField:
    def __init__(self, 
                 domain_size=20.0, 
                 mesh_resolution=30,
                 schwarzschild_radius=2.0,
                 time_steps=50,
                 dt=0.1):
        
        self.domain_size = domain_size
        self.mesh_resolution = mesh_resolution
        self.rs = schwarzschild_radius
        self.time_steps = time_steps
        self.dt = dt
        self.current_time = 0.0
        
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
        
        print("\nStarting time evolution...")
        
        output_file = "time_evolution.bp"
        energies = []
        times = []
        
        with VTXWriter(comm, output_file, [u_n], engine="BP4") as vtx:
            vtx.write(0.0)
            
            for step in range(self.time_steps):
                self.current_time = (step + 1) * self.dt
                
                u_new = self.solve_timestep(domain, V, u_n, self.current_time)
                
                energy = self.compute_field_energy(domain, u_new)
                energies.append(energy)
                times.append(self.current_time)
                
                u_n.x.array[:] = u_new.x.array
                
                vtx.write(self.current_time)
                
                if (step + 1) % 10 == 0:
                    print(f"  Step {step+1}/{self.time_steps}, t={self.current_time:.2f}, Energy={energy:.2e}")
        
        print(f"\nTime evolution complete!")
        print(f"Results saved to {output_file}")
        
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
        plt.savefig('energy_evolution.png', dpi=150, bbox_inches='tight')
        print("Energy evolution plot saved to 'energy_evolution.png'")
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
        plt.savefig('particle_trajectories.png', dpi=150, bbox_inches='tight')
        print("Particle trajectories saved to 'particle_trajectories.png'")
        plt.close()

def main():
    if not DOLFINX_AVAILABLE:
        print("ERROR: FEniCSx is required for this simulation")
        print("Please install DOLFINx or run in a DOLFINx container")
        return
    
    schwarzschild_radius = 2.0
    domain_size = 20.0
    mesh_resolution = 20
    time_steps = 50
    dt = 0.1
    
    field_sim = TimeEvolvingGravitationalField(
        domain_size=domain_size,
        mesh_resolution=mesh_resolution,
        schwarzschild_radius=schwarzschild_radius,
        time_steps=time_steps,
        dt=dt
    )
    
    times, energies, final_field = field_sim.simulate()
    
    print("\n" + "=" * 70)
    print("GENERATING VISUALIZATIONS")
    print("=" * 70)
    
    field_sim.plot_energy_evolution(times, energies)
    
    particle_sim = ParticleTrajectories(schwarzschild_radius=schwarzschild_radius)
    particle_sim.plot_trajectories(n_particles=8)
    
    print("\n" + "=" * 70)
    print("SIMULATION COMPLETE")
    print("=" * 70)
    print("\nGenerated files:")
    print("  - time_evolution.bp/        (ParaView time series)")
    print("  - energy_evolution.png      (Field energy over time)")
    print("  - particle_trajectories.png (Test particle orbits)")
    print("\nTo visualize in ParaView:")
    print("  1. Open 'time_evolution.bp'")
    print("  2. Use the time controls to animate the evolution")
    print("  3. Apply filters (Slice, Contour) to explore the field")
    print("=" * 70)

if __name__ == "__main__":
    main()

