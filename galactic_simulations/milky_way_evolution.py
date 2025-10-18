import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.animation import FuncAnimation, PillowWriter
from matplotlib import cm
import os

def ensure_dir(path):
    os.makedirs(path, exist_ok=True)
    return path

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
OUTPUT_DIR = ensure_dir(os.path.join(SCRIPT_DIR, "outputs"))
GALAXY_DATA_DIR = ensure_dir(os.path.join(OUTPUT_DIR, "data"))
GALAXY_PLOTS_DIR = ensure_dir(os.path.join(OUTPUT_DIR, "plots"))

class MilkyWayGalaxy:
    def __init__(self,
                 num_stars=50000,
                 num_dark_matter=20000,
                 simulation_time_myr=500,
                 time_steps=200,
                 include_dark_matter=True,
                 include_spiral_arms=True):
        
        self.num_stars = num_stars
        self.num_dark_matter = num_dark_matter
        self.simulation_time = simulation_time_myr * 1e6 * 365.25 * 86400
        self.time_steps = time_steps
        self.dt = self.simulation_time / time_steps
        
        self.include_dark_matter = include_dark_matter
        self.include_spiral_arms = include_spiral_arms
        
        self.kpc = 3.086e19
        self.Msun = 1.989e30
        self.G = 6.67430e-11
        
        self.bulge_mass = 1.5e10 * self.Msun
        self.disk_mass = 6.0e10 * self.Msun
        self.halo_mass = 1.0e12 * self.Msun
        
        self.disk_scale_length = 3.5 * self.kpc
        self.disk_scale_height = 0.3 * self.kpc
        self.bulge_scale = 1.0 * self.kpc
        self.halo_scale = 20.0 * self.kpc
        
        self.spiral_arms = 2
        self.spiral_pitch_angle = 12.0 * np.pi / 180
        
        self.history = {
            'times': [],
            'star_positions': [],
            'dark_matter_positions': [],
            'spiral_phase': []
        }
        
        print("=" * 70)
        print("MILKY WAY GALAXY EVOLUTION SIMULATION")
        print("=" * 70)
        print(f"\nGalaxy Parameters:")
        print(f"  Total stellar mass: {self.disk_mass/self.Msun:.2e} M☉")
        print(f"  Bulge mass: {self.bulge_mass/self.Msun:.2e} M☉")
        print(f"  Dark matter halo: {self.halo_mass/self.Msun:.2e} M☉")
        print(f"  Disk scale length: {self.disk_scale_length/self.kpc:.1f} kpc")
        print(f"  Number of star particles: {num_stars:,}")
        if include_dark_matter:
            print(f"  Number of DM particles: {num_dark_matter:,}")
        print(f"\nSimulation Settings:")
        print(f"  Time span: {simulation_time_myr} Myr")
        print(f"  Time steps: {time_steps}")
        print(f"  Time step: {simulation_time_myr/time_steps:.2f} Myr")
        print("=" * 70)
    
    def initialize_galaxy(self):
        print("\nInitializing galaxy structure...")
        
        stars_pos = np.zeros((self.num_stars, 3))
        stars_vel = np.zeros((self.num_stars, 3))
        stars_type = np.zeros(self.num_stars, dtype=int)
        
        num_bulge = int(0.15 * self.num_stars)
        num_disk = self.num_stars - num_bulge
        
        for i in range(num_bulge):
            r = np.random.exponential(self.bulge_scale)
            theta = np.random.uniform(0, 2 * np.pi)
            phi = np.random.uniform(0, np.pi)
            
            stars_pos[i] = [
                r * np.sin(phi) * np.cos(theta),
                r * np.sin(phi) * np.sin(theta),
                r * np.cos(phi) * 0.8
            ]
            
            v_circ = self.circular_velocity(np.linalg.norm(stars_pos[i]))
            v_circ *= np.random.uniform(0.7, 1.0)
            
            stars_vel[i] = [
                -v_circ * np.sin(theta),
                v_circ * np.cos(theta),
                np.random.normal(0, 30000)
            ]
            
            stars_type[i] = 0
        
        for i in range(num_bulge, self.num_stars):
            R = np.random.exponential(self.disk_scale_length)
            R = min(R, 25 * self.kpc)
            
            theta = np.random.uniform(0, 2 * np.pi)
            
            if self.include_spiral_arms:
                spiral_prob = self.spiral_density(R, theta)
                if np.random.random() > spiral_prob:
                    theta = np.random.uniform(0, 2 * np.pi)
            
            z = np.random.normal(0, self.disk_scale_height)
            
            stars_pos[i] = [
                R * np.cos(theta),
                R * np.sin(theta),
                z
            ]
            
            v_circ = self.circular_velocity(R)
            v_r = np.random.normal(0, 15000)
            v_z = np.random.normal(0, 10000)
            
            stars_vel[i] = [
                v_r * np.cos(theta) - v_circ * np.sin(theta),
                v_r * np.sin(theta) + v_circ * np.cos(theta),
                v_z
            ]
            
            stars_type[i] = 1
        
        self.stars_pos = stars_pos
        self.stars_vel = stars_vel
        self.stars_type = stars_type
        
        if self.include_dark_matter:
            dm_pos = np.zeros((self.num_dark_matter, 3))
            dm_vel = np.zeros((self.num_dark_matter, 3))
            
            for i in range(self.num_dark_matter):
                r = np.random.exponential(self.halo_scale)
                r = min(r, 50 * self.kpc)
                
                theta = np.random.uniform(0, 2 * np.pi)
                phi = np.random.uniform(0, np.pi)
                
                dm_pos[i] = [
                    r * np.sin(phi) * np.cos(theta),
                    r * np.sin(phi) * np.sin(theta),
                    r * np.cos(phi)
                ]
                
                v_circ = self.circular_velocity(r) * 0.5
                
                dm_vel[i] = [
                    -v_circ * np.sin(theta) + np.random.normal(0, 50000),
                    v_circ * np.cos(theta) + np.random.normal(0, 50000),
                    np.random.normal(0, 50000)
                ]
            
            self.dm_pos = dm_pos
            self.dm_vel = dm_vel
        
        print(f"  ✓ Initialized {num_bulge:,} bulge stars")
        print(f"  ✓ Initialized {num_disk:,} disk stars")
        if self.include_dark_matter:
            print(f"  ✓ Initialized {self.num_dark_matter:,} dark matter particles")
    
    def spiral_density(self, R, theta):
        density = 0.0
        
        for i in range(self.spiral_arms):
            arm_angle = 2 * np.pi * i / self.spiral_arms
            
            spiral_theta = arm_angle + np.log(R / self.kpc) / np.tan(self.spiral_pitch_angle)
            
            delta_theta = theta - spiral_theta
            delta_theta = np.arctan2(np.sin(delta_theta), np.cos(delta_theta))
            
            arm_width = 0.5
            density += np.exp(-delta_theta**2 / (2 * arm_width**2))
        
        return min(density / self.spiral_arms * 2, 1.0)
    
    def circular_velocity(self, r):
        if r < 1e-10:
            return 0.0
        
        M_bulge = self.bulge_mass * (1 - np.exp(-r / self.bulge_scale))
        
        M_disk = self.disk_mass * (r / self.disk_scale_length)**2 / (1 + r / self.disk_scale_length)**2
        
        M_halo = self.halo_mass * (r / self.halo_scale) / (1 + r / self.halo_scale)**2
        
        M_total = M_bulge + M_disk + M_halo
        
        v_circ = np.sqrt(self.G * M_total / r)
        
        return v_circ
    
    def compute_acceleration(self, pos, use_smoothing=True):
        a_bulge = self.bulge_acceleration(pos)
        a_disk = self.disk_acceleration(pos)
        a_halo = self.halo_acceleration(pos)
        
        return a_bulge + a_disk + a_halo
    
    def bulge_acceleration(self, pos):
        r = np.linalg.norm(pos, axis=1, keepdims=True)
        r = np.maximum(r, 0.1 * self.kpc)
        
        M_enc = self.bulge_mass * (1 - np.exp(-r / self.bulge_scale))
        
        a = -self.G * M_enc * pos / r**3
        
        return a
    
    def disk_acceleration(self, pos):
        R = np.sqrt(pos[:, 0:1]**2 + pos[:, 1:2]**2)
        R = np.maximum(R, 0.1 * self.kpc)
        
        M_enc = self.disk_mass * (R / self.disk_scale_length)**2 / (1 + R / self.disk_scale_length)**2
        
        a_R = -self.G * M_enc / R**2
        
        a = np.zeros_like(pos)
        a[:, 0:1] = a_R * pos[:, 0:1] / R
        a[:, 1:2] = a_R * pos[:, 1:2] / R
        
        z_force = -self.G * self.disk_mass / (2 * self.disk_scale_length**2) * pos[:, 2:3]
        a[:, 2:3] = z_force / (1 + np.abs(pos[:, 2:3]) / self.disk_scale_height)
        
        return a
    
    def halo_acceleration(self, pos):
        r = np.linalg.norm(pos, axis=1, keepdims=True)
        r = np.maximum(r, 0.1 * self.kpc)
        
        M_enc = self.halo_mass * (r / self.halo_scale) / (1 + r / self.halo_scale)**2
        
        a = -self.G * M_enc * pos / r**3
        
        return a
    
    def leapfrog_step(self, dt):
        a_stars = self.compute_acceleration(self.stars_pos)
        
        self.stars_vel += 0.5 * dt * a_stars
        
        self.stars_pos += dt * self.stars_vel
        
        a_stars = self.compute_acceleration(self.stars_pos)
        
        self.stars_vel += 0.5 * dt * a_stars
        
        if self.include_dark_matter:
            a_dm = self.compute_acceleration(self.dm_pos)
            self.dm_vel += 0.5 * dt * a_dm
            self.dm_pos += dt * self.dm_vel
            a_dm = self.compute_acceleration(self.dm_pos)
            self.dm_vel += 0.5 * dt * a_dm
    
    def evolve_galaxy(self):
        print("\nStarting galaxy evolution...")
        
        self.initialize_galaxy()
        
        self.history['times'].append(0.0)
        self.history['star_positions'].append(self.stars_pos.copy())
        if self.include_dark_matter:
            self.history['dark_matter_positions'].append(self.dm_pos.copy())
        self.history['spiral_phase'].append(0.0)
        
        for step in range(self.time_steps):
            current_time = (step + 1) * self.dt
            time_myr = current_time / (1e6 * 365.25 * 86400)
            
            self.leapfrog_step(self.dt)
            
            self.history['times'].append(time_myr)
            self.history['star_positions'].append(self.stars_pos.copy())
            if self.include_dark_matter:
                self.history['dark_matter_positions'].append(self.dm_pos.copy())
            
            spiral_phase = 2 * np.pi * time_myr / 200
            self.history['spiral_phase'].append(spiral_phase)
            
            if (step + 1) % max(1, self.time_steps // 10) == 0:
                print(f"  Step {step+1}/{self.time_steps}, t={time_myr:.1f} Myr")
                
                R_stars = np.sqrt(self.stars_pos[:, 0]**2 + self.stars_pos[:, 1]**2)
                print(f"    Mean stellar radius: {np.mean(R_stars)/self.kpc:.2f} kpc")
                print(f"    Max stellar radius: {np.max(R_stars)/self.kpc:.2f} kpc")
        
        print(f"\n✓ Galaxy evolution complete!")
        print(f"  Final time: {self.history['times'][-1]:.1f} Myr")
        
        return self.history
    
    def plot_galaxy_face_on(self, time_index=-1):
        fig, ax = plt.subplots(figsize=(14, 14))
        
        ax.set_facecolor('#000814')
        fig.patch.set_facecolor('#001021')
        
        pos = self.history['star_positions'][time_index]
        time_myr = self.history['times'][time_index]
        
        x = pos[:, 0] / self.kpc
        y = pos[:, 1] / self.kpc
        
        bulge_mask = self.stars_type == 0
        disk_mask = self.stars_type == 1
        
        R = np.sqrt(x**2 + y**2)
        colors_disk = cm.plasma(np.clip((R[disk_mask] - 2) / 15, 0, 1))
        
        ax.scatter(x[disk_mask], y[disk_mask], s=0.5, c=colors_disk, alpha=0.6, edgecolors='none')
        
        ax.scatter(x[bulge_mask], y[bulge_mask], s=2, c='#FFD700', alpha=0.8, edgecolors='none')
        
        if self.include_dark_matter and time_index < len(self.history['dark_matter_positions']):
            dm_pos = self.history['dark_matter_positions'][time_index]
            dm_x = dm_pos[:, 0] / self.kpc
            dm_y = dm_pos[:, 1] / self.kpc
            ax.scatter(dm_x, dm_y, s=0.2, c='#8B00FF', alpha=0.1, edgecolors='none')
        
        ax.set_xlabel('X (kpc)', fontsize=14, fontweight='bold', color='white')
        ax.set_ylabel('Y (kpc)', fontsize=14, fontweight='bold', color='white')
        ax.set_title(f'Milky Way Galaxy - Face-On View\nTime = {time_myr:.1f} Myr',
                    fontsize=16, fontweight='bold', color='white', pad=20)
        
        ax.set_xlim(-25, 25)
        ax.set_ylim(-25, 25)
        ax.set_aspect('equal')
        ax.grid(True, alpha=0.15, color='#4a4a6a', linestyle='--')
        ax.tick_params(colors='white', labelsize=11)
        
        plt.tight_layout()
        plot_path = os.path.join(GALAXY_PLOTS_DIR, 'milky_way_face_on.png')
        plt.savefig(plot_path, dpi=200, bbox_inches='tight', facecolor='#001021')
        print(f"Face-on view saved to '{plot_path}'")
        plt.close()
    
    def plot_galaxy_edge_on(self, time_index=-1):
        fig, ax = plt.subplots(figsize=(16, 8))
        
        ax.set_facecolor('#000814')
        fig.patch.set_facecolor('#001021')
        
        pos = self.history['star_positions'][time_index]
        time_myr = self.history['times'][time_index]
        
        R = np.sqrt(pos[:, 0]**2 + pos[:, 1]**2) / self.kpc
        z = pos[:, 2] / self.kpc
        
        bulge_mask = self.stars_type == 0
        disk_mask = self.stars_type == 1
        
        colors_disk = cm.plasma(np.clip((R[disk_mask] - 2) / 15, 0, 1))
        
        ax.scatter(R[disk_mask], z[disk_mask], s=0.5, c=colors_disk, alpha=0.6, edgecolors='none')
        ax.scatter(R[bulge_mask], z[bulge_mask], s=2, c='#FFD700', alpha=0.8, edgecolors='none')
        
        ax.set_xlabel('Radius (kpc)', fontsize=14, fontweight='bold', color='white')
        ax.set_ylabel('Height (kpc)', fontsize=14, fontweight='bold', color='white')
        ax.set_title(f'Milky Way Galaxy - Edge-On View\nTime = {time_myr:.1f} Myr',
                    fontsize=16, fontweight='bold', color='white', pad=20)
        
        ax.set_xlim(0, 25)
        ax.set_ylim(-5, 5)
        ax.grid(True, alpha=0.15, color='#4a4a6a', linestyle='--')
        ax.tick_params(colors='white', labelsize=11)
        
        plt.tight_layout()
        plot_path = os.path.join(GALAXY_PLOTS_DIR, 'milky_way_edge_on.png')
        plt.savefig(plot_path, dpi=200, bbox_inches='tight', facecolor='#001021')
        print(f"Edge-on view saved to '{plot_path}'")
        plt.close()
    
    def plot_rotation_curve(self):
        fig, ax = plt.subplots(figsize=(12, 8))
        
        radii = np.linspace(0.1, 30, 100) * self.kpc
        v_circ = np.array([self.circular_velocity(r) for r in radii])
        
        ax.plot(radii / self.kpc, v_circ / 1000, 'c-', linewidth=3, label='Total')
        
        ax.set_xlabel('Radius (kpc)', fontsize=13, fontweight='bold')
        ax.set_ylabel('Circular Velocity (km/s)', fontsize=13, fontweight='bold')
        ax.set_title('Milky Way Rotation Curve', fontsize=15, fontweight='bold')
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=11)
        
        plt.tight_layout()
        plot_path = os.path.join(GALAXY_PLOTS_DIR, 'rotation_curve.png')
        plt.savefig(plot_path, dpi=150, bbox_inches='tight')
        print(f"Rotation curve saved to '{plot_path}'")
        plt.close()
    
    def create_galaxy_animation(self):
        print("\nCreating galaxy evolution animation...")
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 10))
        
        fig.patch.set_facecolor('#001021')
        ax1.set_facecolor('#000814')
        ax2.set_facecolor('#000814')
        
        num_frames = min(len(self.history['times']), 150)
        frame_indices = np.linspace(0, len(self.history['times']) - 1, num_frames, dtype=int)
        
        def update(frame_idx):
            idx = frame_indices[frame_idx]
            ax1.clear()
            ax2.clear()
            
            ax1.set_facecolor('#000814')
            ax2.set_facecolor('#000814')
            
            pos = self.history['star_positions'][idx]
            time_myr = self.history['times'][idx]
            
            x = pos[:, 0] / self.kpc
            y = pos[:, 1] / self.kpc
            R = np.sqrt(pos[:, 0]**2 + pos[:, 1]**2) / self.kpc
            z = pos[:, 2] / self.kpc
            
            bulge_mask = self.stars_type == 0
            disk_mask = self.stars_type == 1
            
            colors_disk = cm.plasma(np.clip((R[disk_mask] - 2) / 15, 0, 1))
            
            ax1.scatter(x[disk_mask], y[disk_mask], s=0.3, c=colors_disk, alpha=0.5, edgecolors='none')
            ax1.scatter(x[bulge_mask], y[bulge_mask], s=1.5, c='#FFD700', alpha=0.7, edgecolors='none')
            
            ax1.set_xlabel('X (kpc)', fontsize=12, fontweight='bold', color='white')
            ax1.set_ylabel('Y (kpc)', fontsize=12, fontweight='bold', color='white')
            ax1.set_title('Face-On View', fontsize=13, fontweight='bold', color='white')
            ax1.set_xlim(-25, 25)
            ax1.set_ylim(-25, 25)
            ax1.set_aspect('equal')
            ax1.grid(True, alpha=0.1, color='#4a4a6a', linestyle='--')
            ax1.tick_params(colors='white', labelsize=10)
            
            colors_edge = cm.plasma(np.clip((R[disk_mask] - 2) / 15, 0, 1))
            ax2.scatter(R[disk_mask], z[disk_mask], s=0.3, c=colors_edge, alpha=0.5, edgecolors='none')
            ax2.scatter(R[bulge_mask], z[bulge_mask], s=1.5, c='#FFD700', alpha=0.7, edgecolors='none')
            
            ax2.set_xlabel('Radius (kpc)', fontsize=12, fontweight='bold', color='white')
            ax2.set_ylabel('Height (kpc)', fontsize=12, fontweight='bold', color='white')
            ax2.set_title('Edge-On View', fontsize=13, fontweight='bold', color='white')
            ax2.set_xlim(0, 25)
            ax2.set_ylim(-5, 5)
            ax2.grid(True, alpha=0.1, color='#4a4a6a', linestyle='--')
            ax2.tick_params(colors='white', labelsize=10)
            
            fig.suptitle(f'Milky Way Galaxy Evolution - Time = {time_myr:.1f} Myr',
                        fontsize=16, fontweight='bold', color='white', y=0.98)
        
        anim = FuncAnimation(fig, update, frames=num_frames,
                           interval=100, blit=False, repeat=True)
        
        animation_path = os.path.join(GALAXY_PLOTS_DIR, 'milky_way_evolution.gif')
        writer = PillowWriter(fps=15)
        anim.save(animation_path, writer=writer, dpi=120)
        print(f"Galaxy evolution animation saved to '{animation_path}'")
        plt.close()

def main():
    galaxy = MilkyWayGalaxy(
        num_stars=50000,
        num_dark_matter=20000,
        simulation_time_myr=500,
        time_steps=200,
        include_dark_matter=True,
        include_spiral_arms=True
    )
    
    history = galaxy.evolve_galaxy()
    
    print("\n" + "=" * 70)
    print("GENERATING VISUALIZATIONS")
    print("=" * 70)
    
    print("\n1. Face-on view...")
    galaxy.plot_galaxy_face_on()
    
    print("\n2. Edge-on view...")
    galaxy.plot_galaxy_edge_on()
    
    print("\n3. Rotation curve...")
    galaxy.plot_rotation_curve()
    
    print("\n4. Galaxy evolution animation...")
    galaxy.create_galaxy_animation()
    
    print("\n" + "=" * 70)
    print("MILKY WAY SIMULATION COMPLETE")
    print("=" * 70)
    print("\nGenerated files:")
    print(f"  - {os.path.join(GALAXY_PLOTS_DIR, 'milky_way_face_on.png')}")
    print(f"  - {os.path.join(GALAXY_PLOTS_DIR, 'milky_way_edge_on.png')}")
    print(f"  - {os.path.join(GALAXY_PLOTS_DIR, 'rotation_curve.png')}")
    print(f"  - {os.path.join(GALAXY_PLOTS_DIR, 'milky_way_evolution.gif')} (Animation)")
    print("\n" + "=" * 70)

if __name__ == "__main__":
    main()

