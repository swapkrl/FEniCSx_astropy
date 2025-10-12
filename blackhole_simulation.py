import os
import sys

os.environ['FI_PROVIDER'] = 'tcp'
os.environ['MPICH_ASYNC_PROGRESS'] = '1'
os.environ['I_MPI_FABRICS'] = 'shm'
os.environ['MPICH_INTERFACE_HOSTNAME'] = 'localhost'

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from scipy.integrate import solve_ivp

DOLFINX_AVAILABLE = False
MPI_COMM = None

try:
    import dolfinx
    from dolfinx import mesh, fem, plot
    import dolfinx.cpp as cpp
    import ufl
    
    MPI_COMM = cpp.MPI.COMM_SELF
    DOLFINX_AVAILABLE = True
    print("FEniCSx loaded successfully in serial mode")
except Exception as e:
    print(f"Warning: FEniCSx not available: {e}")
    print("Continuing without FEM features...")

class SchwarzschildBlackHole:
    def __init__(self, mass=1.0, speed_of_light=1.0, gravitational_constant=1.0):
        self.M = mass
        self.c = speed_of_light
        self.G = gravitational_constant
        self.rs = 2 * self.G * self.M / (self.c ** 2)
        
    def geodesic_equations(self, t, y):
        r, theta, phi, pr, ptheta, pphi = y
        
        if r <= self.rs * 1.01:
            return np.zeros(6)
        
        f = 1 - self.rs / r
        
        dr_dt = f * pr
        dtheta_dt = ptheta / (r ** 2)
        dphi_dt = pphi / (r ** 2 * np.sin(theta) ** 2)
        
        dpr_dt = (self.rs / (2 * r ** 2)) * pr ** 2 / f - f * (ptheta ** 2 + pphi ** 2 / np.sin(theta) ** 2) / (r ** 3)
        dptheta_dt = pphi ** 2 * np.cos(theta) / (r ** 2 * np.sin(theta) ** 3)
        dpphi_dt = 0
        
        return np.array([dr_dt, dtheta_dt, dphi_dt, dpr_dt, dptheta_dt, dpphi_dt])
    
    def compute_geodesic(self, initial_position, initial_velocity, t_max=100, n_points=10000):
        r0, theta0, phi0 = initial_position
        vr0, vtheta0, vphi0 = initial_velocity
        
        y0 = np.array([r0, theta0, phi0, vr0, vtheta0, vphi0])
        t_span = (0, t_max)
        t_eval = np.linspace(0, t_max, n_points)
        
        solution = solve_ivp(self.geodesic_equations, t_span, y0, t_eval=t_eval, 
                            method='RK45', rtol=1e-8, atol=1e-10)
        
        return solution
    
    def spherical_to_cartesian(self, r, theta, phi):
        x = r * np.sin(theta) * np.cos(phi)
        y = r * np.sin(theta) * np.sin(phi)
        z = r * np.cos(theta)
        return x, y, z

class BlackHolePotentialFEM:
    def __init__(self, schwarzschild_radius=2.0, domain_size=20.0, mesh_resolution=50):
        self.rs = schwarzschild_radius
        self.domain_size = domain_size
        self.mesh_resolution = mesh_resolution
        
    def create_mesh_and_function_space(self):
        domain = mesh.create_box(
            MPI_COMM,
            [[0.0, 0.0, 0.0], [self.domain_size, self.domain_size, self.domain_size]],
            [self.mesh_resolution, self.mesh_resolution, self.mesh_resolution],
            cell_type=mesh.CellType.hexahedron
        )
        
        V = fem.functionspace(domain, ("Lagrange", 1))
        return domain, V
    
    def compute_effective_potential(self, domain, V):
        u = fem.Function(V)
        
        def potential_expr(x):
            center = np.array([self.domain_size/2, self.domain_size/2, self.domain_size/2])
            r = np.sqrt((x[0] - center[0])**2 + (x[1] - center[1])**2 + (x[2] - center[2])**2)
            r = np.maximum(r, self.rs * 1.1)
            return -self.rs / (2 * r)
        
        u.interpolate(potential_expr)
        return u

def simulate_multiple_geodesics(black_hole, n_trajectories=5):
    trajectories = []
    
    for i in range(n_trajectories):
        angle = 2 * np.pi * i / n_trajectories
        
        r0 = 10 * black_hole.rs
        theta0 = np.pi / 2
        phi0 = angle
        
        vr0 = -0.05
        vtheta0 = 0.0
        vphi0 = 0.15 / r0
        
        initial_pos = (r0, theta0, phi0)
        initial_vel = (vr0, vtheta0, vphi0)
        
        sol = black_hole.compute_geodesic(initial_pos, initial_vel, t_max=500, n_points=5000)
        trajectories.append(sol)
    
    return trajectories

def plot_geodesics(black_hole, trajectories):
    fig = plt.figure(figsize=(15, 5))
    
    ax1 = fig.add_subplot(131, projection='3d')
    ax1.set_title('Particle Geodesics around Schwarzschild Black Hole', fontsize=10, pad=10)
    ax1.set_xlabel('X', fontsize=9)
    ax1.set_ylabel('Y', fontsize=9)
    ax1.set_zlabel('Z', fontsize=9)
    
    for sol in trajectories:
        r, theta, phi = sol.y[0], sol.y[1], sol.y[2]
        x, y, z = black_hole.spherical_to_cartesian(r, theta, phi)
        ax1.plot(x, y, z, linewidth=0.8, alpha=0.7)
    
    u = np.linspace(0, 2 * np.pi, 50)
    v = np.linspace(0, np.pi, 50)
    rs = black_hole.rs
    x_sphere = rs * np.outer(np.cos(u), np.sin(v))
    y_sphere = rs * np.outer(np.sin(u), np.sin(v))
    z_sphere = rs * np.outer(np.ones(np.size(u)), np.cos(v))
    ax1.plot_surface(x_sphere, y_sphere, z_sphere, color='black', alpha=0.9)
    
    max_range = 15 * black_hole.rs
    ax1.set_xlim([-max_range, max_range])
    ax1.set_ylim([-max_range, max_range])
    ax1.set_zlim([-max_range, max_range])
    
    ax2 = fig.add_subplot(132)
    ax2.set_title('Radial Distance vs Time', fontsize=10, pad=10)
    ax2.set_xlabel('Time', fontsize=9)
    ax2.set_ylabel('r / rs', fontsize=9)
    
    for sol in trajectories:
        r_normalized = sol.y[0] / black_hole.rs
        ax2.plot(sol.t, r_normalized, linewidth=1.0, alpha=0.7)
    
    ax2.axhline(y=1.0, color='red', linestyle='--', linewidth=1.5, label='Event Horizon')
    ax2.legend(fontsize=8)
    ax2.grid(True, alpha=0.3)
    
    ax3 = fig.add_subplot(133, projection='3d')
    ax3.set_title('Photon Orbit (Critical)', fontsize=10, pad=10)
    ax3.set_xlabel('X', fontsize=9)
    ax3.set_ylabel('Y', fontsize=9)
    ax3.set_zlabel('Z', fontsize=9)
    
    r_photon = 1.5 * black_hole.rs
    theta_photon = np.pi / 2
    phi_photon = 0.0
    vr_photon = 0.0
    vtheta_photon = 0.0
    vphi_photon = 1.0 / (r_photon * np.sqrt(3/2))
    
    photon_sol = black_hole.compute_geodesic(
        (r_photon, theta_photon, phi_photon),
        (vr_photon, vtheta_photon, vphi_photon),
        t_max=200,
        n_points=3000
    )
    
    r_ph, theta_ph, phi_ph = photon_sol.y[0], photon_sol.y[1], photon_sol.y[2]
    x_ph, y_ph, z_ph = black_hole.spherical_to_cartesian(r_ph, theta_ph, phi_ph)
    ax3.plot(x_ph, y_ph, z_ph, 'yellow', linewidth=1.2, label='Photon Orbit')
    ax3.plot_surface(x_sphere, y_sphere, z_sphere, color='black', alpha=0.9)
    
    ax3.set_xlim([-5*rs, 5*rs])
    ax3.set_ylim([-5*rs, 5*rs])
    ax3.set_zlim([-5*rs, 5*rs])
    ax3.legend(fontsize=8)
    
    plt.tight_layout()
    plt.savefig('blackhole_geodesics.png', dpi=150, bbox_inches='tight')
    print("Geodesics plot saved as 'blackhole_geodesics.png'")
    plt.show()

def plot_spacetime_curvature_2d(black_hole):
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    r_range = np.linspace(black_hole.rs * 1.01, 20 * black_hole.rs, 1000)
    
    g_tt = -(1 - black_hole.rs / r_range)
    g_rr = 1 / (1 - black_hole.rs / r_range)
    
    axes[0].plot(r_range / black_hole.rs, g_tt, 'b-', linewidth=2)
    axes[0].set_xlabel('r / rs', fontsize=11)
    axes[0].set_ylabel('g_tt', fontsize=11)
    axes[0].set_title('Temporal Metric Component', fontsize=12)
    axes[0].axvline(x=1.0, color='red', linestyle='--', linewidth=1.5, label='Event Horizon')
    axes[0].axhline(y=0, color='gray', linestyle=':', linewidth=1)
    axes[0].grid(True, alpha=0.3)
    axes[0].legend(fontsize=9)
    
    axes[1].plot(r_range / black_hole.rs, g_rr, 'r-', linewidth=2)
    axes[1].set_xlabel('r / rs', fontsize=11)
    axes[1].set_ylabel('g_rr', fontsize=11)
    axes[1].set_title('Radial Metric Component', fontsize=12)
    axes[1].axvline(x=1.0, color='red', linestyle='--', linewidth=1.5, label='Event Horizon')
    axes[1].grid(True, alpha=0.3)
    axes[1].legend(fontsize=9)
    axes[1].set_ylim([0, 20])
    
    plt.tight_layout()
    plt.savefig('spacetime_curvature.png', dpi=150, bbox_inches='tight')
    print("Spacetime curvature plot saved as 'spacetime_curvature.png'")
    plt.show()

def plot_embedding_diagram(black_hole):
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')
    
    r = np.linspace(black_hole.rs * 1.01, 10 * black_hole.rs, 100)
    theta = np.linspace(0, 2 * np.pi, 100)
    R, Theta = np.meshgrid(r, theta)
    
    Z = -2 * np.sqrt(black_hole.rs * (R - black_hole.rs))
    
    X = R * np.cos(Theta)
    Y = R * np.sin(Theta)
    
    surf = ax.plot_surface(X, Y, Z, cmap='viridis', alpha=0.8, edgecolor='none')
    
    ax.set_xlabel('X', fontsize=11)
    ax.set_ylabel('Y', fontsize=11)
    ax.set_zlabel('Z (Embedding Height)', fontsize=11)
    ax.set_title('Schwarzschild Spacetime Embedding Diagram', fontsize=13, pad=15)
    
    fig.colorbar(surf, ax=ax, shrink=0.5, aspect=5)
    
    plt.savefig('embedding_diagram.png', dpi=150, bbox_inches='tight')
    print("Embedding diagram saved as 'embedding_diagram.png'")
    plt.show()

def main():
    print("=" * 70)
    print("BLACK HOLE SIMULATION USING FEniCSx")
    print("Schwarzschild Metric in Geometric Units (G=c=1)")
    print("=" * 70)
    
    black_hole_mass = 1.0
    bh = SchwarzschildBlackHole(mass=black_hole_mass)
    
    print(f"\nBlack Hole Parameters:")
    print(f"  Mass (M): {bh.M}")
    print(f"  Schwarzschild Radius (rs): {bh.rs}")
    print(f"  Photon Sphere Radius: {1.5 * bh.rs}")
    print(f"  ISCO Radius: {3 * bh.rs}")
    
    print("\n" + "-" * 70)
    print("Computing geodesic trajectories...")
    trajectories = simulate_multiple_geodesics(bh, n_trajectories=8)
    print(f"Computed {len(trajectories)} geodesic trajectories")
    
    print("\n" + "-" * 70)
    print("Generating visualizations...")
    plot_geodesics(bh, trajectories)
    plot_spacetime_curvature_2d(bh)
    plot_embedding_diagram(bh)
    
    if DOLFINX_AVAILABLE:
        print("\n" + "-" * 70)
        print("Computing effective potential using FEM...")
        try:
            fem_solver = BlackHolePotentialFEM(schwarzschild_radius=bh.rs, domain_size=20.0, mesh_resolution=30)
            domain, V = fem_solver.create_mesh_and_function_space()
            potential = fem_solver.compute_effective_potential(domain, V)
            print(f"FEM mesh created with {domain.topology.index_map(0).size_global} vertices")
            print(f"Effective potential computed on {V.dofmap.index_map.size_global} degrees of freedom")
        except Exception as e:
            print(f"FEM computation failed: {str(e)}")
    else:
        print("\n" + "-" * 70)
        print("Skipping FEM computation (FEniCSx not available)")
    
    print("\n" + "=" * 70)
    print("SIMULATION COMPLETE")
    print("=" * 70)
    print("\nGenerated files:")
    print("  - blackhole_geodesics.png")
    print("  - spacetime_curvature.png")
    print("  - embedding_diagram.png")
    print("\nPhysical insights:")
    print("  - Geodesics show particle trajectories in curved spacetime")
    print("  - Metric components diverge at event horizon (r = rs)")
    print("  - Embedding diagram visualizes spatial curvature")
    print("=" * 70)

if __name__ == "__main__":
    main()

