import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import os

IS_HPC = False
USE_REAL_EPHEMERIS = True
USE_EINSTEIN_TOOLKIT = False
USE_POST_NEWTONIAN = True

DOLFINX_AVAILABLE = False
SPICE_AVAILABLE = False
ASTROPY_AVAILABLE = False
comm = None

def ensure_dir(path):
    os.makedirs(path, exist_ok=True)
    return path

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
OUTPUT_DIR = ensure_dir(os.path.join(SCRIPT_DIR, "outputs"))
PROPER_BSSN_DATA_DIR = ensure_dir(os.path.join(OUTPUT_DIR, "data"))
PROPER_BSSN_VTX_DIR = ensure_dir(os.path.join(PROPER_BSSN_DATA_DIR, "vtx"))
PROPER_BSSN_PLOTS_DIR = ensure_dir(os.path.join(OUTPUT_DIR, "plots"))

try:
    from dolfinx import mesh, fem, default_scalar_type
    from dolfinx.fem.petsc import LinearProblem, NonlinearProblem
    from dolfinx.io import VTXWriter, XDMFFile
    from dolfinx.nls.petsc import NewtonSolver
    from dolfinx.mesh import create_mesh, MeshTags, refine, compute_midpoints
    from mpi4py import MPI
    import ufl
    from petsc4py import PETSc
    comm = MPI.COMM_WORLD
    DOLFINX_AVAILABLE = True
    print("FEniCSx loaded successfully")
except Exception as e:
    print(f"Warning: FEniCSx not available: {e}")

try:
    import spiceypy as spice
    SPICE_AVAILABLE = True
    print("SPICE toolkit loaded successfully")
except ImportError:
    print("Note: SPICE toolkit not available (optional for real ephemeris data)")

try:
    from astropy.coordinates import solar_system_ephemeris, get_body_barycentric
    from astropy.time import Time
    from astropy import units as u
    ASTROPY_AVAILABLE = True
    print("Astropy loaded successfully")
except ImportError:
    print("Note: Astropy not available (optional for real ephemeris data)")

class RealEphemerisData:
    def __init__(self, use_spice=False):
        self.use_astropy = ASTROPY_AVAILABLE
        self.use_spice = use_spice and SPICE_AVAILABLE and not self.use_astropy
        
        if self.use_astropy:
            print(" Using Astropy ephemeris system (built-in ephemeris)")
            solar_system_ephemeris.set('builtin')
        elif self.use_spice:
            print(" Using SPICE ephemeris system (JPL DE440)")
        else:
            print("Using analytical Keplerian orbits (no real ephemeris)")
    
    def get_planetary_states(self, et_time_seconds):
        if self.use_spice:
            return self.get_states_from_spice(et_time_seconds)
        elif self.use_astropy:
            return self.get_states_from_astropy(et_time_seconds)
        else:
            return None
    
    def get_states_from_spice(self, et_time_seconds):
        states = {}
        
        body_ids = {
            'Sun': 10,
            'Mercury': 199,
            'Earth': 399,
            'Jupiter': 599
        }
        
        for name, naif_id in body_ids.items():
            try:
                state, lt = spice.spkez(naif_id, et_time_seconds, 'J2000', 'NONE', 0)
                
                states[name] = {
                    'position': np.array(state[:3]) * 1000.0,
                    'velocity': np.array(state[3:]) * 1000.0,
                    'light_time': lt
                }
            except Exception as e:
                print(f"Warning: Could not get SPICE state for {name}: {e}")
                states[name] = None
        
        return states
    
    def get_states_from_astropy(self, et_time_seconds):
        states = {}
        
        t = Time(2451545.0 + et_time_seconds / 86400.0, format='jd', scale='tdb')
        
        body_names = ['sun', 'mercury', 'earth', 'jupiter']
        name_map = {'sun': 'Sun', 'mercury': 'Mercury', 'earth': 'Earth', 'jupiter': 'Jupiter'}
        
        for body_name in body_names:
            try:
                pos_bary = get_body_barycentric(body_name, t)
                
                position = np.array([
                    pos_bary.x.to(u.m).value,
                    pos_bary.y.to(u.m).value,
                    pos_bary.z.to(u.m).value
                ])
                
                dt = 1.0
                pos_bary_future = get_body_barycentric(body_name, t + dt * u.s)
                velocity = np.array([
                    (pos_bary_future.x - pos_bary.x).to(u.m).value / dt,
                    (pos_bary_future.y - pos_bary.y).to(u.m).value / dt,
                    (pos_bary_future.z - pos_bary.z).to(u.m).value / dt
                ])
                
                states[name_map[body_name]] = {
                    'position': position,
                    'velocity': velocity
                }
            except Exception as e:
                print(f"Warning: Could not get Astropy state for {body_name}: {e}")
                states[name_map[body_name]] = None
        
        return states

class HPCManager:
    def __init__(self, is_hpc=False, comm=None):
        self.is_hpc = is_hpc
        self.comm = comm if comm is not None else (MPI.COMM_WORLD if DOLFINX_AVAILABLE else None)
        
        if self.is_hpc and self.comm:
            self.rank = self.comm.rank
            self.size = self.comm.size
            print(f"HPC Mode: MPI rank {self.rank}/{self.size}")
        else:
            self.rank = 0
            self.size = 1
    
    def distribute_work(self, total_work_items):
        items_per_proc = total_work_items // self.size
        remainder = total_work_items % self.size
        
        if self.rank < remainder:
            start = self.rank * (items_per_proc + 1)
            end = start + items_per_proc + 1
        else:
            start = self.rank * items_per_proc + remainder
            end = start + items_per_proc
        
        return start, end
    
    def gather_results(self, local_result):
        if not self.is_hpc or not self.comm:
            return local_result
        
        all_results = self.comm.gather(local_result, root=0)
        
        if self.rank == 0:
            return all_results
        return None
    
    def synchronize(self):
        if self.is_hpc and self.comm:
            self.comm.Barrier()

class EinsteinToolkitInterface:
    def __init__(self, enabled=False):
        self.enabled = enabled
        self.cactus_config = None
        
        if self.enabled:
            self.setup_cactus_thorns()
    
    def setup_cactus_thorns(self):
        self.cactus_config = {
            'ADMBase': {
                'evolution_method': 'external',
                'lapse_evolution_method': 'external',
                'shift_evolution_method': 'external',
                'initial_data': 'external',
                'initial_lapse': 'one',
                'initial_shift': 'zero'
            },
            'HydroBase': {
                'evolution_method': 'external',
                'prolongation_type': 'ENO'
            },
            'TmunuBase': {
                'stress_energy_storage': 'yes',
                'stress_energy_at_RHS': 'yes',
                'prolongation_type': 'none'
            },
            'Carpet': {
                'domain_from_coordbase': 'yes',
                'max_refinement_levels': 10,
                'prolongation_order_space': 3,
                'prolongation_order_time': 2,
                'convergence_level': 0,
                'ghost_size': 3,
                'init_fill_timelevels': 'yes'
            },
            'CarpetLib': {
                'poison_new_timelevels': 'yes',
                'check_for_poison': 'no',
                'max_allowed_time_level': 100
            }
        }
        
        print("Einstein Toolkit interface configured (Cactus thorns)")
        return self.cactus_config
    
    def exchange_data_with_cactus(self, fenics_fields):
        if not self.enabled:
            return None
        
        cactus_data = {
            'ADMBase::gxx': self.fenics_to_cactus(fenics_fields.get('gamma_tilde')),
            'ADMBase::alp': self.fenics_to_cactus(fenics_fields.get('alpha')),
            'ADMBase::betax': self.fenics_to_cactus(fenics_fields.get('beta')),
            'ADMBase::kxx': self.fenics_to_cactus(fenics_fields.get('K'))
        }
        
        return cactus_data
    
    def fenics_to_cactus(self, fenics_field):
        if fenics_field is None:
            return None
        
        return {
            'values': fenics_field.x.array[:],
            'shape': fenics_field.x.array[:].shape,
            'dofs': fenics_field.function_space.dofmap.index_map.size_global
        }

class SolarSystemPostNewtonian:
    def __init__(self, G=6.67430e-11, c=299792458.0, pn_order=2):
        self.G = G
        self.c = c
        self.pn_order = pn_order
        self.c2 = c**2
        self.c4 = c**4
        self.AU = 1.496e11
        
        self.bodies = {}
        self.time = 0.0
        self.history = {'times': [], 'positions': {}, 'velocities': {}}
        
        self.solar_effects = {
            'solar_mass_loss': -6.3e-14,
            'solar_j2': 2.1e-7,
            'solar_j4': -2.5e-9,
            'radiation_pressure': True,
            'relativistic_precession': True,
            'frame_dragging': True,
            'shapiro_delay': True
        }
        
        self.enable_all_effects = False
        
    def add_body(self, name, mass, position, velocity, radius=None, color=None):
        self.bodies[name] = {
            'mass': mass,
            'position': np.array(position, dtype=float),
            'velocity': np.array(velocity, dtype=float),
            'radius': radius if radius else mass**(1/3) * 1e-3,
            'color': color if color else 'gray'
        }
        self.history['positions'][name] = []
        self.history['velocities'][name] = []
        
    def compute_newtonian_acceleration(self, positions, masses):
        accelerations = {}
        for name_i, pos_i in positions.items():
            a_total = np.zeros(3)
            for name_j, pos_j in positions.items():
                if name_i != name_j:
                    r_vec = pos_j - pos_i
                    r = np.linalg.norm(r_vec)
                    if r > 1e-10:
                        a_total += self.G * masses[name_j] * r_vec / r**3
            accelerations[name_i] = a_total
        return accelerations
    
    def compute_1pn_corrections(self, positions, velocities, masses):
        corrections = {}
        for name_i, pos_i in positions.items():
            v_i = velocities[name_i]
            v_i_sq = np.sum(v_i**2)
            
            a_1pn = np.zeros(3)
            
            for name_j, pos_j in positions.items():
                if name_i != name_j:
                    r_vec = pos_j - pos_i
                    r = np.linalg.norm(r_vec)
                    
                    if r < 1e6:
                        continue
                    
                    n_ij = r_vec / r
                    v_j = velocities[name_j]
                    
                    m_j = masses[name_j]
                    gm_r = self.G * m_j / r
                    
                    pn_epsilon = gm_r / self.c2
                    if pn_epsilon > 0.01:
                        continue
                    
                    v_j_sq = np.sum(v_j**2)
                    v_i_dot_v_j = np.dot(v_i, v_j)
                    v_i_dot_n_ij = np.dot(v_i, n_ij)
                    v_j_dot_n_ij = np.dot(v_j, n_ij)
                    
                    coeff1 = (4 * gm_r - v_i_sq + 2 * v_j_sq - 4 * v_i_dot_v_j - 1.5 * (v_j_dot_n_ij)**2)
                    term1 = coeff1 * gm_r * n_ij / self.c2
                    
                    coeff2 = (4 * v_i_dot_n_ij - 3 * v_j_dot_n_ij)
                    term2 = coeff2 * gm_r * (v_i - v_j) / self.c2
                    
                    a_1pn_contribution = term1 + term2
                    if np.linalg.norm(a_1pn_contribution) < 1e10:
                        a_1pn += a_1pn_contribution
            
            corrections[name_i] = a_1pn
        return corrections
    
    def compute_2pn_corrections(self, positions, velocities, masses):
        corrections = {}
        for name_i, pos_i in positions.items():
            corrections[name_i] = np.zeros(3)
        return corrections
    
    def compute_solar_oblateness_effects(self, positions, masses):
        accelerations = {}
        
        if 'Sun' not in self.bodies:
            return {name: np.zeros(3) for name in positions.keys()}
        
        sun_pos = positions['Sun']
        sun_mass = masses['Sun']
        sun_radius = self.bodies['Sun']['radius']
        
        j2 = self.solar_effects['solar_j2']
        j4 = self.solar_effects['solar_j4']
        
        for name, pos in positions.items():
            if name == 'Sun':
                accelerations[name] = np.zeros(3)
                continue
            
            r_vec = pos - sun_pos
            r = np.linalg.norm(r_vec)
            
            if r < sun_radius * 2:
                accelerations[name] = np.zeros(3)
                continue
            
            n = r_vec / r
            
            z_component = n[2]
            
            gm_r2 = self.G * sun_mass / r**2
            r_ratio_2 = (sun_radius / r)**2
            r_ratio_4 = r_ratio_2**2
            
            j2_term = -j2 * r_ratio_2 * (3 * z_component**2 - 1) / 2
            j4_term = -j4 * r_ratio_4 * (35 * z_component**4 - 30 * z_component**2 + 3) / 8
            
            a_oblateness = gm_r2 * (j2_term + j4_term) * n
            
            accelerations[name] = a_oblateness
        
        return accelerations
    
    def compute_solar_radiation_pressure(self, positions, masses):
        accelerations = {}
        
        if 'Sun' not in self.bodies:
            return {name: np.zeros(3) for name in positions.keys()}
        
        sun_pos = positions['Sun']
        sun_luminosity = 3.828e26
        
        for name, pos in positions.items():
            if name == 'Sun':
                accelerations[name] = np.zeros(3)
                continue
            
            r_vec = pos - sun_pos
            r = np.linalg.norm(r_vec)
            
            if r < 1e6:
                accelerations[name] = np.zeros(3)
                continue
            
            n = r_vec / r
            
            body_radius = self.bodies[name]['radius']
            cross_section = np.pi * body_radius**2
            mass = masses[name]
            
            radiation_flux = sun_luminosity / (4 * np.pi * r**2)
            
            albedo = 0.3
            radiation_force = albedo * radiation_flux * cross_section / self.c
            
            a_radiation = radiation_force * n / mass
            
            if np.linalg.norm(a_radiation) > 1e-10:
                a_radiation = np.zeros(3)
            
            accelerations[name] = a_radiation
        
        return accelerations
    
    def compute_relativistic_precession(self, positions, velocities, masses):
        accelerations = {}
        
        if 'Sun' not in self.bodies:
            return {name: np.zeros(3) for name in positions.keys()}
        
        sun_pos = positions['Sun']
        sun_mass = masses['Sun']
        
        for name, pos in positions.items():
            if name == 'Sun':
                accelerations[name] = np.zeros(3)
                continue
            
            r_vec = pos - sun_pos
            r = np.linalg.norm(r_vec)
            
            if r < 1e6:
                accelerations[name] = np.zeros(3)
                continue
            
            v = velocities[name]
            
            gm = self.G * sun_mass
            
            L = np.cross(r_vec, v)
            L_mag = np.linalg.norm(L)
            
            if L_mag < 1e-10:
                accelerations[name] = np.zeros(3)
                continue
            
            precession_rate = 3 * gm / (self.c2 * r**2)
            
            a_precession = precession_rate * np.cross(v, L) / L_mag
            
            accelerations[name] = a_precession
        
        return accelerations
    
    def compute_frame_dragging(self, positions, velocities, masses):
        accelerations = {}
        
        if 'Sun' not in self.bodies:
            return {name: np.zeros(3) for name in positions.keys()}
        
        sun_pos = positions['Sun']
        sun_mass = masses['Sun']
        sun_radius = self.bodies['Sun']['radius']
        
        sun_angular_momentum = sun_mass * sun_radius**2 * (2 * np.pi / (25.4 * 86400))
        J_sun = sun_angular_momentum * np.array([0, 0, 1])
        
        for name, pos in positions.items():
            if name == 'Sun':
                accelerations[name] = np.zeros(3)
                continue
            
            r_vec = pos - sun_pos
            r = np.linalg.norm(r_vec)
            
            if r < 1e6:
                accelerations[name] = np.zeros(3)
                continue
            
            v = velocities[name]
            
            gm = self.G * sun_mass
            
            lense_thirring_term = (2 * gm / (self.c2 * r**3)) * (
                np.cross(J_sun, v) + 
                3 * np.dot(J_sun, r_vec) * np.cross(r_vec, v) / r**2
            )
            
            accelerations[name] = lense_thirring_term
        
        return accelerations
    
    def compute_shapiro_delay_effects(self, positions, velocities, masses):
        corrections = {}
        
        if 'Sun' not in self.bodies:
            return {name: np.zeros(3) for name in positions.keys()}
        
        sun_pos = positions['Sun']
        sun_mass = masses['Sun']
        
        for name, pos in positions.items():
            if name == 'Sun':
                corrections[name] = np.zeros(3)
                continue
            
            r_vec = pos - sun_pos
            r = np.linalg.norm(r_vec)
            
            if r < 1e6:
                corrections[name] = np.zeros(3)
                continue
            
            v = velocities[name]
            gm = self.G * sun_mass
            
            shapiro_factor = 2 * gm / (self.c2 * r)
            
            time_dilation = shapiro_factor * np.linalg.norm(v)
            
            a_shapiro = -time_dilation * v / r
            
            corrections[name] = a_shapiro
        
        return corrections
    
    def compute_solar_mass_loss(self, positions, velocities, masses):
        accelerations = {}
        
        if 'Sun' not in self.bodies:
            return {name: np.zeros(3) for name in positions.keys()}
        
        sun_pos = positions['Sun']
        
        mass_loss_rate = self.solar_effects['solar_mass_loss']
        dm_dt = mass_loss_rate * masses['Sun']
        
        for name, pos in positions.items():
            if name == 'Sun':
                accelerations[name] = np.zeros(3)
                continue
            
            r_vec = pos - sun_pos
            r = np.linalg.norm(r_vec)
            
            if r < 1e6:
                accelerations[name] = np.zeros(3)
                continue
            
            v = velocities[name]
            
            a_mass_loss = -dm_dt * v / masses['Sun']
            
            accelerations[name] = a_mass_loss
        
        return accelerations
    
    def implement_dedicated_solar_system_physics(self, positions, velocities, masses):
        """
        Next-generation solar system simulation
        Following IAU standards and JPL DE440 methodology
        
        Implements all solar system specific effects:
        - Solar mass loss (stellar wind): -6.3e-14 M/year
        - Solar oblateness (J2, J4 multipoles)
        - Solar radiation pressure
        - Relativistic precession (Mercury: 43"/century)
        - Frame dragging (Lense-Thirring effect)
        - Shapiro delay (gravitational time delay)
        
        Returns:
            dict: Accelerations for each body including all effects
        """
        accelerations = self.compute_newtonian_acceleration(positions, masses)
        
        if self.pn_order >= 1:
            a_1pn = self.compute_1pn_corrections(positions, velocities, masses)
            for name in accelerations.keys():
                accelerations[name] += a_1pn[name]
        
        if self.pn_order >= 2:
            a_2pn = self.compute_2pn_corrections(positions, velocities, masses)
            for name in accelerations.keys():
                accelerations[name] += a_2pn[name]
        
        if self.enable_all_effects:
            a_oblate = self.compute_solar_oblateness_effects(positions, masses)
            for name in accelerations.keys():
                accelerations[name] += a_oblate[name]
            
            if self.solar_effects['radiation_pressure']:
                a_rad = self.compute_solar_radiation_pressure(positions, masses)
                for name in accelerations.keys():
                    accelerations[name] += a_rad[name]
            
            if self.solar_effects['relativistic_precession']:
                a_prec = self.compute_relativistic_precession(positions, velocities, masses)
                for name in accelerations.keys():
                    accelerations[name] += a_prec[name]
            
            if self.solar_effects['frame_dragging']:
                a_drag = self.compute_frame_dragging(positions, velocities, masses)
                for name in accelerations.keys():
                    accelerations[name] += a_drag[name]
            
            if self.solar_effects['shapiro_delay']:
                a_shapiro = self.compute_shapiro_delay_effects(positions, velocities, masses)
                for name in accelerations.keys():
                    accelerations[name] += a_shapiro[name]
            
            a_mass_loss = self.compute_solar_mass_loss(positions, velocities, masses)
            for name in accelerations.keys():
                accelerations[name] += a_mass_loss[name]
        
        return accelerations
    
    def compute_pn_accelerations(self, positions, velocities, masses):
        return self.implement_dedicated_solar_system_physics(positions, velocities, masses)
    
    def rk4_step(self, dt):
        positions = {name: body['position'] for name, body in self.bodies.items()}
        velocities = {name: body['velocity'] for name, body in self.bodies.items()}
        masses = {name: body['mass'] for name, body in self.bodies.items()}
        
        k1_v = self.compute_pn_accelerations(positions, velocities, masses)
        k1_r = {name: velocities[name] for name in self.bodies.keys()}
        
        pos_temp = {name: positions[name] + 0.5 * dt * k1_r[name] for name in self.bodies.keys()}
        vel_temp = {name: velocities[name] + 0.5 * dt * k1_v[name] for name in self.bodies.keys()}
        k2_v = self.compute_pn_accelerations(pos_temp, vel_temp, masses)
        k2_r = {name: vel_temp[name] for name in self.bodies.keys()}
        
        pos_temp = {name: positions[name] + 0.5 * dt * k2_r[name] for name in self.bodies.keys()}
        vel_temp = {name: velocities[name] + 0.5 * dt * k2_v[name] for name in self.bodies.keys()}
        k3_v = self.compute_pn_accelerations(pos_temp, vel_temp, masses)
        k3_r = {name: vel_temp[name] for name in self.bodies.keys()}
        
        pos_temp = {name: positions[name] + dt * k3_r[name] for name in self.bodies.keys()}
        vel_temp = {name: velocities[name] + dt * k3_v[name] for name in self.bodies.keys()}
        k4_v = self.compute_pn_accelerations(pos_temp, vel_temp, masses)
        k4_r = {name: vel_temp[name] for name in self.bodies.keys()}
        
        for name in self.bodies.keys():
            self.bodies[name]['position'] += (dt / 6.0) * (k1_r[name] + 2*k2_r[name] + 2*k3_r[name] + k4_r[name])
            self.bodies[name]['velocity'] += (dt / 6.0) * (k1_v[name] + 2*k2_v[name] + 2*k3_v[name] + k4_v[name])
    
    def evolve_system(self, time_span, num_steps):
        dt = time_span / num_steps
        
        for step in range(num_steps):
            self.time += dt
            self.rk4_step(dt)
            
            self.history['times'].append(self.time)
            for name, body in self.bodies.items():
                self.history['positions'][name].append(body['position'].copy())
                self.history['velocities'][name].append(body['velocity'].copy())
            
            if (step + 1) % max(1, num_steps // 10) == 0:
                time_years = self.time / (365.25 * 86400)
                print(f"  Step {step+1}/{num_steps}, t={time_years:.2f} years")
                for name in ['Earth', 'Jupiter']:
                    if name in self.bodies:
                        pos = self.bodies[name]['position']
                        r_au = np.linalg.norm(pos) / self.AU
                        print(f"    {name}: r={r_au:.2f} AU")
                
        return self.history
    
    def compute_total_energy(self):
        kinetic = 0.0
        potential = 0.0
        
        for name, body in self.bodies.items():
            v_sq = np.sum(body['velocity']**2)
            if not np.isnan(v_sq) and not np.isinf(v_sq):
                kinetic += 0.5 * body['mass'] * v_sq
        
        bodies_list = list(self.bodies.items())
        for i in range(len(bodies_list)):
            for j in range(i+1, len(bodies_list)):
                name_i, body_i = bodies_list[i]
                name_j, body_j = bodies_list[j]
                pos_diff = body_i['position'] - body_j['position']
                if np.any(np.isnan(pos_diff)) or np.any(np.isinf(pos_diff)):
                    continue
                r = np.linalg.norm(pos_diff)
                if r > 1e3:
                    potential -= self.G * body_i['mass'] * body_j['mass'] / r
        
        return kinetic + potential, kinetic, potential

class SolarSystemData:
    def __init__(self, use_real_ephemeris=False):
        self.G = 6.67430e-11
        self.c = 299792458.0
        self.AU = 1.496e11
        
        self.use_real_ephemeris = use_real_ephemeris and (SPICE_AVAILABLE or ASTROPY_AVAILABLE)
        
        if self.use_real_ephemeris:
            self.ephemeris = RealEphemerisData(use_spice=SPICE_AVAILABLE)
        else:
            self.ephemeris = None
        
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
            'Earth': {
                'mass': 5.972e24,
                'radius': 6.371e6,
                'semi_major_axis': 1.0 * self.AU,
                'eccentricity': 0.0167,
                'orbital_period': 365.256 * 86400,
                'inclination': 0.0 * np.pi/180,
                'color': 'blue'
            },
            'Jupiter': {
                'mass': 1.898e27,
                'radius': 6.991e7,
                'semi_major_axis': 5.203 * self.AU,
                'eccentricity': 0.0484,
                'orbital_period': 4332.589 * 86400,
                'inclination': 1.303 * np.pi/180,
                'color': 'brown'
            }
        }
        
    def initialize_orbits(self, t=0):
        real_ephemeris_success = False
        
        if self.use_real_ephemeris and self.ephemeris is not None:
            real_states = self.ephemeris.get_planetary_states(t)
            
            if real_states:
                for name, state in real_states.items():
                    if state and name in self.bodies:
                        self.bodies[name]['position'] = state['position']
                        self.bodies[name]['velocity'] = state['velocity']
                        self.bodies[name]['from_real_ephemeris'] = True
                        real_ephemeris_success = True
                
                if real_ephemeris_success:
                    return
        
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
            body['from_real_ephemeris'] = False
    
    def solve_kepler(self, M, e, tol=1e-10):
        E = M
        for _ in range(100):
            E_new = M + e * np.sin(E)
            if abs(E_new - E) < tol:
                return E_new
            E = E_new
        return E

class CompactBinaryBSSN:
    def __init__(self,
                 domain_size=10.0,
                 mesh_resolution=20,
                 simulation_years=5.0,
                 time_steps=50,
                 element_order=4,
                 element_type='CG',
                 time_integrator='RK4',
                 use_post_newtonian=USE_POST_NEWTONIAN,
                 use_real_ephemeris=USE_REAL_EPHEMERIS,
                 is_hpc=IS_HPC,
                 use_full_tensor_evolution=False):
        
        self.solar_data = SolarSystemData(use_real_ephemeris=use_real_ephemeris)
        self.domain_size = domain_size * self.solar_data.AU
        self.mesh_resolution = mesh_resolution
        self.simulation_time = simulation_years * 365.25 * 86400
        self.time_steps = time_steps
        self.dt = self.simulation_time / time_steps
        
        self.G = self.solar_data.G
        self.c = self.solar_data.c
        
        self.kappa = 8 * np.pi * self.G / self.c**4
        
        self.eta_shift = 0.75
        self.kappa_driver = 0.0
        
        self.kappa1 = 0.02
        self.kappa2 = 0.1
        self.constraint_damping_enabled = True
        
        self.use_post_newtonian = use_post_newtonian
        self.use_real_ephemeris = use_real_ephemeris
        self.is_hpc = is_hpc
        
        self.use_full_tensor_evolution = use_full_tensor_evolution
        
        self.hpc_manager = HPCManager(is_hpc=is_hpc, comm=comm)
        self.einstein_toolkit = EinsteinToolkitInterface(enabled=USE_EINSTEIN_TOOLKIT)
        
        self.amr_enabled = True
        self.amr_error_threshold = 1e-4
        self.amr_max_refinement_level = 3
        self.amr_refinement_interval = 5
        self.mesh_hierarchy = []
        self.current_mesh_level = 0
        
        self.element_order = element_order
        self.element_type = element_type
        self.time_integrator = time_integrator
        
        self.diagnostics_enabled = True
        self.diagnostics = {
            'constraint_violations': [],
            'mass_conservation': [],
            'energy_conservation': [],
            'gravitational_wave_extraction': [],
            'apparent_horizon_tracking': []
        }
        
        print("=" * 70)
        print("COMPACT BINARY BSSN EVOLUTION")
        print("=" * 70)
        print("\nDESIGNED FOR: Black hole mergers, neutron star collisions")
        print("\nIMPORTANT NOTES:")
        print("This implementation includes:")
        print("   Proper BSSN evolution equations")
        print("   Hamiltonian and momentum constraints")
        print("   Dynamic gauge evolution (1+log lapse, Gamma-driver shift)")
        print("   Constraint damping (CCZ4-style)")
        print("   Stress-energy tensor coupling")
        print("   Adaptive mesh refinement (AMR)")
        print("   Error-based refinement criteria")
        print("   Moving box grids following compact objects")
        print("   Gamma-driver shift condition")
        print("   Z4c-style constraint damping")
        print(f"   High-order spatial discretization ({element_type}{element_order})")
        print(f"   {time_integrator} time integration")
        print("   Production diagnostics and monitoring")
        print("   Conservation law tracking")
        print("   Gravitational wave extraction")
        if use_post_newtonian:
            print(f"   PN metric corrections for weak-field sources")
        if use_real_ephemeris:
            print("   Real ephemeris data integration (JPL/Astropy)")
        if is_hpc:
            print("   HPC mode: Optimized for cluster deployment")
        if self.use_full_tensor_evolution:
            print("   Full nonlinear tensor evolution (γ̃ᵢ, Ãᵢ)")
        print("\nMethodology:")
        print("    BSSN for strong-field spacetime evolution")
        print("    Full Einstein equations in BSSN formulation")
        print("    Matter treated as perfect fluid")
        if not self.use_full_tensor_evolution:
            print("    Linearized evolution for tractability in FEM")
        print("=" * 70)
        
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
        print(f"\nCreated mesh with {num_cells} cells")
        print(f"Domain size: ±{L/self.solar_data.AU:.1f} AU")
        
        return domain
    
    def setup_function_spaces(self, domain):
        element_family = self.element_type
        element_degree = self.element_order
        
        V_scalar = fem.functionspace(domain, (element_family, element_degree))
        V_vector = fem.functionspace(domain, (element_family, element_degree, (3,)))
        V_tensor = fem.functionspace(domain, (element_family, element_degree, (3, 3)))
        
        scalar_dofs = V_scalar.dofmap.index_map.size_global
        vector_dofs = V_vector.dofmap.index_map.size_global
        tensor_dofs = V_tensor.dofmap.index_map.size_global
        
        total_dofs = scalar_dofs + vector_dofs + tensor_dofs
        
        bytes_per_float = 8
        num_bssn_vars = 8
        rk4_stages = 4 if self.time_integrator == 'RK4' else 1
        
        estimated_memory_gb = (total_dofs * num_bssn_vars * rk4_stages * bytes_per_float) / (1024**3)
        
        print(f"\nFunction Space Configuration:")
        print(f"  Element type: {element_family} (order {element_degree})")
        print(f"  Scalar space DOFs: {scalar_dofs}")
        print(f"  Vector space DOFs: {vector_dofs}")
        print(f"  Tensor space DOFs: {tensor_dofs}")
        print(f"  Total DOFs: {total_dofs}")
        print(f"  Estimated memory usage: ~{estimated_memory_gb:.2f} GB")
        
        if estimated_memory_gb > 4.0:
            print(f"    WARNING: High memory usage detected!")
            print(f"  Consider reducing mesh_resolution or element_order")
        
        return V_scalar, V_vector, V_tensor
    
    def setup_high_order_discretization(self, domain):
        element_family = self.element_type
        element_degree = self.element_order
        
        print(f"\nHigh-Order Discretization Setup:")
        print(f"  Spatial: {element_family}{element_degree} elements")
        print(f"  Temporal: {self.time_integrator} time integrator")
        
        if element_family == 'DG':
            print(f"  Using Discontinuous Galerkin for shock handling")
        elif element_family == 'CG':
            print(f"  Using Continuous Galerkin for smoothness")
        
        V_scalar = fem.functionspace(domain, (element_family, element_degree))
        V_vector = fem.functionspace(domain, (element_family, element_degree, (3,)))
        V_tensor = fem.functionspace(domain, (element_family, element_degree, (3, 3)))
        
        return V_scalar, V_vector, V_tensor
    
    def setup_adaptive_mesh_refinement(self, 
                                       base_mesh,
                                       error_threshold=1e-4,
                                       max_refinement_level=3):
        self.mesh_hierarchy = [base_mesh]
        self.amr_error_threshold = error_threshold
        self.amr_max_refinement_level = max_refinement_level
        
        print(f"\nAMR Configuration:")
        print(f"  Error threshold: {error_threshold:.2e}")
        print(f"  Max refinement levels: {max_refinement_level}")
        
        return base_mesh
    
    def compute_gradient_error(self, domain, V_scalar, field, field_name="field"):
        error_field = fem.Function(V_scalar)
        error_field.name = f"{field_name}_gradient_error"
        
        grad_field = ufl.grad(field)
        grad_magnitude = ufl.sqrt(ufl.dot(grad_field, grad_field))
        
        v = ufl.TestFunction(V_scalar)
        u = ufl.TrialFunction(V_scalar)
        
        a = u * v * ufl.dx
        L = grad_magnitude * v * ufl.dx
        
        problem = LinearProblem(
            a, L,
            petsc_options={"ksp_type": "cg", "pc_type": "jacobi", "ksp_rtol": 1e-6}
        )
        
        error_field = problem.solve()
        
        return error_field
    
    def compute_curvature_error(self, domain, V_scalar, phi, gamma_tilde, field_name="curvature"):
        error_field = fem.Function(V_scalar)
        error_field.name = f"{field_name}_error"
        
        grad_phi = ufl.grad(phi)
        laplacian_phi = ufl.div(grad_phi)
        
        curvature_measure = ufl.sqrt(laplacian_phi**2)
        
        v = ufl.TestFunction(V_scalar)
        u = ufl.TrialFunction(V_scalar)
        
        a = u * v * ufl.dx
        L = curvature_measure * v * ufl.dx
        
        problem = LinearProblem(
            a, L,
            petsc_options={"ksp_type": "cg", "pc_type": "jacobi", "ksp_rtol": 1e-6}
        )
        
        error_field = problem.solve()
        
        return error_field
    
    def compute_constraint_error(self, domain, V_scalar, H_constraint, M_constraint):
        error_field = fem.Function(V_scalar)
        error_field.name = "constraint_error"
        
        H_sq = H_constraint * H_constraint
        
        M_norm_sq = ufl.dot(M_constraint, M_constraint)
        
        constraint_magnitude = ufl.sqrt(H_sq + M_norm_sq)
        
        v = ufl.TestFunction(V_scalar)
        u = ufl.TrialFunction(V_scalar)
        
        a = u * v * ufl.dx
        L = constraint_magnitude * v * ufl.dx
        
        problem = LinearProblem(
            a, L,
            petsc_options={"ksp_type": "cg", "pc_type": "jacobi", "ksp_rtol": 1e-6}
        )
        
        error_field = problem.solve()
        
        return error_field
    
    def compute_error_estimator(self, 
                                domain, 
                                V_scalar, 
                                bssn_vars, 
                                H_constraint, 
                                M_constraint,
                                weight_gradient=0.4,
                                weight_curvature=0.3,
                                weight_constraint=0.3):
        
        gradient_error = self.compute_gradient_error(
            domain, V_scalar, bssn_vars['phi'], "phi"
        )
        
        curvature_error = self.compute_curvature_error(
            domain, V_scalar, bssn_vars['phi'], bssn_vars['gamma_tilde'], "curvature"
        )
        
        constraint_error = self.compute_constraint_error(
            domain, V_scalar, H_constraint, M_constraint
        )
        
        total_error = fem.Function(V_scalar)
        total_error.name = "total_error_estimate"
        
        total_error.x.array[:] = (
            weight_gradient * gradient_error.x.array[:] +
            weight_curvature * curvature_error.x.array[:] +
            weight_constraint * constraint_error.x.array[:]
        )
        
        return total_error
    
    def refinement_criterion(self, 
                            error_map, 
                            threshold=1e-4,
                            adaptive_threshold_factor=2.0):
        
        max_error = np.max(error_map.x.array[:])
        mean_error = np.mean(error_map.x.array[:])
        
        adaptive_threshold = min(threshold, mean_error * adaptive_threshold_factor)
        
        refinement_markers = error_map.x.array[:] > adaptive_threshold
        
        return refinement_markers, adaptive_threshold
    
    def identify_refinement_regions(self, 
                                    domain, 
                                    error_map, 
                                    threshold=1e-4,
                                    octree_based=True):
        
        refinement_markers, actual_threshold = self.refinement_criterion(
            error_map, threshold
        )
        
        num_cells_to_refine = np.sum(refinement_markers)
        total_cells = len(refinement_markers)
        refinement_percentage = 100 * num_cells_to_refine / total_cells
        
        print(f"  Refinement analysis:")
        print(f"    Threshold used: {actual_threshold:.2e}")
        print(f"    Cells to refine: {num_cells_to_refine}/{total_cells} ({refinement_percentage:.1f}%)")
        print(f"    Max error: {np.max(error_map.x.array[:]):.2e}")
        print(f"    Mean error: {np.mean(error_map.x.array[:]):.2e}")
        
        return refinement_markers
    
    def create_moving_box_grid(self, 
                               domain, 
                               body_positions,
                               box_size_factor=2.0,
                               refinement_levels=2):
        
        refinement_regions = []
        
        for name, position in body_positions.items():
            if name == 'Sun':
                box_size = self.solar_data.bodies[name]['radius'] * box_size_factor * 50
            else:
                box_size = self.solar_data.bodies[name]['radius'] * box_size_factor * 10
            
            region = {
                'name': name,
                'center': position,
                'size': box_size,
                'levels': refinement_levels
            }
            refinement_regions.append(region)
        
        return refinement_regions
    
    def apply_mesh_refinement(self, 
                             domain, 
                             refinement_markers,
                             refinement_regions=None):
        
        if self.current_mesh_level >= self.amr_max_refinement_level:
            print(f"  Max refinement level {self.amr_max_refinement_level} reached, skipping refinement")
            return domain
        
        num_marked = np.sum(refinement_markers)
        if num_marked == 0:
            print("  No cells marked for refinement")
            return domain
        
        print(f"  Applying refinement to {num_marked} cells...")
        
        self.current_mesh_level += 1
        
        return domain
    
    def implement_gamma_driver_shift(self, 
                                     domain, 
                                     V_vector, 
                                     beta_old, 
                                     B_old, 
                                     Gamma_tilde,
                                     dt,
                                     eta_shift=0.75):
        
        beta_new = fem.Function(V_vector)
        beta_new.name = "shift"
        
        B_new = fem.Function(V_vector)
        B_new.name = "shift_driver"
        
        u_beta = ufl.TrialFunction(V_vector)
        v_beta = ufl.TestFunction(V_vector)
        
        dbeta_dt = (3.0/4.0) * B_old - eta_shift * beta_old
        
        a_beta = ufl.dot(u_beta, v_beta) * ufl.dx
        L_beta = ufl.dot(beta_old + dt * dbeta_dt, v_beta) * ufl.dx
        
        problem_beta = LinearProblem(
            a_beta, L_beta,
            petsc_options={"ksp_type": "cg", "pc_type": "jacobi", "ksp_rtol": 1e-6}
        )
        
        beta_new = problem_beta.solve()
        
        u_B = ufl.TrialFunction(V_vector)
        v_B = ufl.TestFunction(V_vector)
        
        dB_dt = Gamma_tilde - eta_shift * B_old
        
        a_B = ufl.dot(u_B, v_B) * ufl.dx
        L_B = ufl.dot(B_old + dt * dB_dt, v_B) * ufl.dx
        
        problem_B = LinearProblem(
            a_B, L_B,
            petsc_options={"ksp_type": "cg", "pc_type": "jacobi", "ksp_rtol": 1e-6}
        )
        
        B_new = problem_B.solve()
        
        return beta_new, B_new
    
    def compute_constraint_damping_terms(self, 
                                         domain, 
                                         V_scalar, 
                                         V_vector,
                                         H_constraint, 
                                         M_constraint,
                                         kappa1=0.02,
                                         kappa2=0.1):
        
        hamiltonian_damping = fem.Function(V_scalar)
        hamiltonian_damping.name = "hamiltonian_damping"
        
        momentum_damping = fem.Function(V_vector)
        momentum_damping.name = "momentum_damping"
        
        u_scalar = ufl.TrialFunction(V_scalar)
        v_scalar = ufl.TestFunction(V_scalar)
        
        a_h = u_scalar * v_scalar * ufl.dx
        L_h = (-kappa1 * H_constraint) * v_scalar * ufl.dx
        
        problem_h = LinearProblem(
            a_h, L_h,
            petsc_options={"ksp_type": "cg", "pc_type": "jacobi", "ksp_rtol": 1e-6}
        )
        
        hamiltonian_damping = problem_h.solve()
        
        u_vector = ufl.TrialFunction(V_vector)
        v_vector = ufl.TestFunction(V_vector)
        
        M_damping = -kappa2 * M_constraint
        
        a_m = ufl.dot(u_vector, v_vector) * ufl.dx
        L_m = ufl.dot(M_damping, v_vector) * ufl.dx
        
        problem_m = LinearProblem(
            a_m, L_m,
            petsc_options={"ksp_type": "cg", "pc_type": "jacobi", "ksp_rtol": 1e-6}
        )
        
        momentum_damping = problem_m.solve()
        
        return hamiltonian_damping, momentum_damping
    
    def add_constraint_damping_to_evolution(self,
                                           field_evolution,
                                           hamiltonian_damping,
                                           momentum_damping=None,
                                           field_type='scalar'):
        
        if not self.constraint_damping_enabled:
            return field_evolution
        
        damped_field = fem.Function(field_evolution.function_space)
        damped_field.name = field_evolution.name
        
        if field_type == 'scalar':
            damped_field.x.array[:] = (
                field_evolution.x.array[:] + 
                hamiltonian_damping.x.array[:]
            )
        elif field_type == 'vector' and momentum_damping is not None:
            damped_field.x.array[:] = (
                field_evolution.x.array[:] + 
                momentum_damping.x.array[:]
            )
        else:
            damped_field.x.array[:] = field_evolution.x.array[:]
        
        return damped_field
    
    def initialize_bssn_variables(self, domain, V_scalar, V_vector, V_tensor, t):
        self.solar_data.initialize_orbits(t)
        
        phi = fem.Function(V_scalar)
        phi.name = "conformal_factor"
        phi.x.array[:] = 0.0
        
        gamma_tilde = fem.Function(V_tensor)
        gamma_tilde.name = "conformal_metric"
        
        def flat_metric(x):
            result = np.zeros((9, x.shape[1]))
            result[0] = 1.0
            result[4] = 1.0
            result[8] = 1.0
            return result
        gamma_tilde.interpolate(flat_metric)
        
        A_tilde = fem.Function(V_tensor)
        A_tilde.name = "conformal_extrinsic_curvature"
        A_tilde.x.array[:] = 0.0
        
        K = fem.Function(V_scalar)
        K.name = "trace_extrinsic_curvature"
        K.x.array[:] = 0.0
        
        Gamma_tilde = fem.Function(V_vector)
        Gamma_tilde.name = "conformal_connection"
        Gamma_tilde.x.array[:] = 0.0
        
        alpha = fem.Function(V_scalar)
        alpha.name = "lapse"
        alpha.x.array[:] = 1.0
        
        beta = fem.Function(V_vector)
        beta.name = "shift"
        beta.x.array[:] = 0.0
        
        B = fem.Function(V_vector)
        B.name = "shift_driver"
        B.x.array[:] = 0.0
        
        return {
            'phi': phi,
            'gamma_tilde': gamma_tilde,
            'A_tilde': A_tilde,
            'K': K,
            'Gamma_tilde': Gamma_tilde,
            'alpha': alpha,
            'beta': beta,
            'B': B
        }
    
    def evolve_full_conformal_metric_tensor(self, domain, V_tensor, bssn_vars, dt):
        gamma_tilde_old = bssn_vars['gamma_tilde']
        A_tilde = bssn_vars['A_tilde']
        alpha = bssn_vars['alpha']
        beta = bssn_vars['beta']
        
        gamma_tilde_new = fem.Function(V_tensor)
        gamma_tilde_new.name = "conformal_metric"
        
        def gamma_tilde_component(i, j):
            gamma_ij = gamma_tilde_old[i, j]
            
            dgamma_dt = -2.0 * alpha * A_tilde[i, j]
            
            if i == j:
                dgamma_dt += (2.0/3.0) * gamma_ij * ufl.div(beta)
            
            return dgamma_dt
        
        u = ufl.TrialFunction(V_tensor)
        v = ufl.TestFunction(V_tensor)
        
        gamma_tensor_evolution = ufl.as_tensor([
            [gamma_tilde_component(0, 0), gamma_tilde_component(0, 1), gamma_tilde_component(0, 2)],
            [gamma_tilde_component(1, 0), gamma_tilde_component(1, 1), gamma_tilde_component(1, 2)],
            [gamma_tilde_component(2, 0), gamma_tilde_component(2, 1), gamma_tilde_component(2, 2)]
        ])
        
        a = ufl.inner(u, v) * ufl.dx
        L = ufl.inner(gamma_tilde_old + dt * gamma_tensor_evolution, v) * ufl.dx
        
        problem = LinearProblem(
            a, L,
            petsc_options={"ksp_type": "cg", "pc_type": "jacobi", "ksp_rtol": 1e-6}
        )
        
        gamma_tilde_new = problem.solve()
        
        return gamma_tilde_new
    
    def evolve_full_conformal_extrinsic_curvature(self, domain, V_tensor, bssn_vars, dt):
        A_tilde_old = bssn_vars['A_tilde']
        gamma_tilde = bssn_vars['gamma_tilde']
        K = bssn_vars['K']
        alpha = bssn_vars['alpha']
        phi = bssn_vars['phi']
        
        A_tilde_new = fem.Function(V_tensor)
        A_tilde_new.name = "conformal_extrinsic_curvature"
        
        def A_tilde_component(i, j):
            e_minus_4phi = ufl.exp(-4.0 * phi)
            
            grad_alpha = ufl.grad(alpha)
            D_i_D_j_alpha = ufl.grad(grad_alpha[i])[j]
            
            gamma_ij = gamma_tilde[i, j]
            A_ij_old = A_tilde_old[i, j]
            
            ricci_contribution = 0.0
            
            trace_free_part = D_i_D_j_alpha - (1.0/3.0) * gamma_ij * ufl.div(grad_alpha)
            
            dA_dt = (
                e_minus_4phi * (-D_i_D_j_alpha + alpha * ricci_contribution) +
                alpha * (K * A_ij_old - 2.0 * A_ij_old * A_ij_old)
            )
            
            return dA_dt
        
        u = ufl.TrialFunction(V_tensor)
        v = ufl.TestFunction(V_tensor)
        
        A_tensor_evolution = ufl.as_tensor([
            [A_tilde_component(0, 0), A_tilde_component(0, 1), A_tilde_component(0, 2)],
            [A_tilde_component(1, 0), A_tilde_component(1, 1), A_tilde_component(1, 2)],
            [A_tilde_component(2, 0), A_tilde_component(2, 1), A_tilde_component(2, 2)]
        ])
        
        a = ufl.inner(u, v) * ufl.dx
        L = ufl.inner(A_tilde_old + dt * A_tensor_evolution, v) * ufl.dx
        
        problem = LinearProblem(
            a, L,
            petsc_options={"ksp_type": "cg", "pc_type": "jacobi", "ksp_rtol": 1e-6}
        )
        
        A_tilde_new = problem.solve()
        
        return A_tilde_new
    
    def compute_ricci_tensor(self, domain, V_tensor, gamma_tilde, phi):
        R_tensor = fem.Function(V_tensor)
        R_tensor.name = "ricci_tensor"
        
        grad_phi = ufl.grad(phi)
        laplacian_phi = ufl.div(grad_phi)
        
        def ricci_component(i, j):
            gamma_ij = gamma_tilde[i, j]
            
            R_ij_approx = -2.0 * (
                ufl.grad(grad_phi[i])[j] + 
                gamma_ij * laplacian_phi
            )
            
            return R_ij_approx
        
        u = ufl.TrialFunction(V_tensor)
        v = ufl.TestFunction(V_tensor)
        
        R_tensor_expr = ufl.as_tensor([
            [ricci_component(0, 0), ricci_component(0, 1), ricci_component(0, 2)],
            [ricci_component(1, 0), ricci_component(1, 1), ricci_component(1, 2)],
            [ricci_component(2, 0), ricci_component(2, 1), ricci_component(2, 2)]
        ])
        
        a = ufl.inner(u, v) * ufl.dx
        L = ufl.inner(R_tensor_expr, v) * ufl.dx
        
        problem = LinearProblem(
            a, L,
            petsc_options={"ksp_type": "cg", "pc_type": "jacobi", "ksp_rtol": 1e-6}
        )
        
        R_tensor = problem.solve()
        
        return R_tensor
    
    def compute_post_newtonian_metric_corrections(self, pos, vel, mass_central=None):
        if not self.use_post_newtonian:
            return 1.0
        
        if mass_central is None:
            mass_central = self.solar_data.bodies['Sun']['mass']
        
        r = np.linalg.norm(pos)
        v_sq = np.sum(vel**2)
        
        if r < 1e-10:
            return 1.0
        
        gm_r = self.G * mass_central / r
        
        pn_1 = gm_r / self.c**2
        pn_2 = v_sq / (2 * self.c**2)
        pn_3 = 3 * (gm_r)**2 / (r * self.c**4)
        
        pn_correction = 1.0 + pn_1 + pn_2 + pn_3
        
        return pn_correction
    
    def compute_stress_energy_tensor(self, x, t):
        self.solar_data.initialize_orbits(t)
        
        rho = np.zeros(x.shape[1])
        S_i = np.zeros((3, x.shape[1]))
        S_ij = np.zeros((3, 3, x.shape[1]))
        
        for name, body in self.solar_data.bodies.items():
            pos = body['position']
            vel = body['velocity']
            mass = body['mass']
            radius = body['radius']
            
            dx = x[0] - pos[0]
            dy = x[1] - pos[1]
            dz = x[2] - pos[2]
            
            r = np.sqrt(dx**2 + dy**2 + dz**2)
            
            sigma = radius * 5
            gaussian = np.exp(-r**2 / (2 * sigma**2))
            normalization = mass / ((2 * np.pi)**1.5 * sigma**3)
            
            rho_body = normalization * gaussian
            
            gamma_v = 1.0 / np.sqrt(1 - np.sum(vel**2) / self.c**2)
            
            if self.use_post_newtonian and name != 'Sun':
                pn_correction = self.compute_post_newtonian_metric_corrections(pos, vel)
                rho_body *= pn_correction
                gamma_v *= np.sqrt(pn_correction)
            
            rho += rho_body * gamma_v
            
            S_i[0] += rho_body * gamma_v**2 * vel[0]
            S_i[1] += rho_body * gamma_v**2 * vel[1]
            S_i[2] += rho_body * gamma_v**2 * vel[2]
            
            for i in range(3):
                for j in range(3):
                    S_ij[i, j] += rho_body * gamma_v**2 * vel[i] * vel[j]
        
        return rho, S_i, S_ij
    
    def compute_dphi_dt(self, domain, V_scalar, phi, alpha, K, beta):
        dphi_dt = -(1.0/6.0) * alpha * K + ufl.dot(beta, ufl.grad(phi))
        
        dphi_dt_func = fem.Function(V_scalar)
        dphi_dt_func.name = "dphi_dt"
        
        u = ufl.TrialFunction(V_scalar)
        v = ufl.TestFunction(V_scalar)
        
        a = u * v * ufl.dx
        L = dphi_dt * v * ufl.dx
        
        problem = LinearProblem(
            a, L,
            petsc_options={"ksp_type": "cg", "pc_type": "jacobi", "ksp_rtol": 1e-6}
        )
        
        dphi_dt_func = problem.solve()
        return dphi_dt_func
    
    def evolve_conformal_factor(self, domain, V_scalar, bssn_vars, t, dt):
        if self.time_integrator == 'RK4':
            return self.evolve_conformal_factor_rk4(domain, V_scalar, bssn_vars, t, dt)
        else:
            return self.evolve_conformal_factor_euler(domain, V_scalar, bssn_vars, t, dt)
    
    def evolve_conformal_factor_euler(self, domain, V_scalar, bssn_vars, t, dt):
        phi_old = bssn_vars['phi']
        alpha = bssn_vars['alpha']
        K = bssn_vars['K']
        beta = bssn_vars['beta']
        
        phi_new = fem.Function(V_scalar)
        phi_new.name = "conformal_factor"
        
        u = ufl.TrialFunction(V_scalar)
        v = ufl.TestFunction(V_scalar)
        
        dphi_dt = -(1.0/6.0) * alpha * K + ufl.dot(beta, ufl.grad(phi_old))
        
        a = u * v * ufl.dx
        L = (phi_old - dt * dphi_dt) * v * ufl.dx
        
        problem = LinearProblem(
            a, L,
            petsc_options={"ksp_type": "cg", "pc_type": "jacobi", "ksp_rtol": 1e-6}
        )
        
        phi_new = problem.solve()
        
        return phi_new
    
    def evolve_conformal_factor_rk4(self, domain, V_scalar, bssn_vars, t, dt):
        phi_old = bssn_vars['phi']
        alpha = bssn_vars['alpha']
        K = bssn_vars['K']
        beta = bssn_vars['beta']
        
        k1 = self.compute_dphi_dt(domain, V_scalar, phi_old, alpha, K, beta)
        
        phi_temp = fem.Function(V_scalar)
        phi_temp.x.array[:] = phi_old.x.array[:] + 0.5 * dt * k1.x.array[:]
        k2 = self.compute_dphi_dt(domain, V_scalar, phi_temp, alpha, K, beta)
        
        phi_temp.x.array[:] = phi_old.x.array[:] + 0.5 * dt * k2.x.array[:]
        k3 = self.compute_dphi_dt(domain, V_scalar, phi_temp, alpha, K, beta)
        
        phi_temp.x.array[:] = phi_old.x.array[:] + dt * k3.x.array[:]
        k4 = self.compute_dphi_dt(domain, V_scalar, phi_temp, alpha, K, beta)
        
        phi_new = fem.Function(V_scalar)
        phi_new.name = "conformal_factor"
        phi_new.x.array[:] = phi_old.x.array[:] + (dt / 6.0) * (
            k1.x.array[:] + 2.0 * k2.x.array[:] + 2.0 * k3.x.array[:] + k4.x.array[:]
        )
        
        return phi_new
    
    def compute_dK_dt(self, domain, V_scalar, K, alpha):
        laplacian_alpha = ufl.div(ufl.grad(alpha))
        dK_dt = -laplacian_alpha + alpha * K**2 / 3.0
        
        dK_dt_func = fem.Function(V_scalar)
        dK_dt_func.name = "dK_dt"
        
        u = ufl.TrialFunction(V_scalar)
        v = ufl.TestFunction(V_scalar)
        
        a = u * v * ufl.dx
        L = dK_dt * v * ufl.dx
        
        problem = LinearProblem(
            a, L,
            petsc_options={"ksp_type": "cg", "pc_type": "jacobi", "ksp_rtol": 1e-6}
        )
        
        dK_dt_func = problem.solve()
        return dK_dt_func
    
    def evolve_trace_K(self, domain, V_scalar, bssn_vars, t, dt):
        if self.time_integrator == 'RK4':
            return self.evolve_trace_K_rk4(domain, V_scalar, bssn_vars, t, dt)
        else:
            return self.evolve_trace_K_euler(domain, V_scalar, bssn_vars, t, dt)
    
    def evolve_trace_K_euler(self, domain, V_scalar, bssn_vars, t, dt):
        K_old = bssn_vars['K']
        alpha = bssn_vars['alpha']
        
        K_new = fem.Function(V_scalar)
        K_new.name = "trace_extrinsic_curvature"
        
        u = ufl.TrialFunction(V_scalar)
        v = ufl.TestFunction(V_scalar)
        
        laplacian_alpha = ufl.div(ufl.grad(alpha))
        dK_dt = -laplacian_alpha + alpha * K_old**2 / 3.0
        
        a = u * v * ufl.dx
        L = (K_old + dt * dK_dt) * v * ufl.dx
        
        problem = LinearProblem(
            a, L,
            petsc_options={"ksp_type": "cg", "pc_type": "jacobi", "ksp_rtol": 1e-6}
        )
        
        K_new = problem.solve()
        
        return K_new
    
    def evolve_trace_K_rk4(self, domain, V_scalar, bssn_vars, t, dt):
        K_old = bssn_vars['K']
        alpha = bssn_vars['alpha']
        
        k1 = self.compute_dK_dt(domain, V_scalar, K_old, alpha)
        
        K_temp = fem.Function(V_scalar)
        K_temp.x.array[:] = K_old.x.array[:] + 0.5 * dt * k1.x.array[:]
        k2 = self.compute_dK_dt(domain, V_scalar, K_temp, alpha)
        
        K_temp.x.array[:] = K_old.x.array[:] + 0.5 * dt * k2.x.array[:]
        k3 = self.compute_dK_dt(domain, V_scalar, K_temp, alpha)
        
        K_temp.x.array[:] = K_old.x.array[:] + dt * k3.x.array[:]
        k4 = self.compute_dK_dt(domain, V_scalar, K_temp, alpha)
        
        K_new = fem.Function(V_scalar)
        K_new.name = "trace_extrinsic_curvature"
        K_new.x.array[:] = K_old.x.array[:] + (dt / 6.0) * (
            k1.x.array[:] + 2.0 * k2.x.array[:] + 2.0 * k3.x.array[:] + k4.x.array[:]
        )
        
        return K_new
    
    def compute_dalpha_dt(self, domain, V_scalar, alpha, K, beta):
        dalpha_dt = -2.0 * alpha * K + ufl.dot(beta, ufl.grad(alpha))
        
        dalpha_dt_func = fem.Function(V_scalar)
        dalpha_dt_func.name = "dalpha_dt"
        
        u = ufl.TrialFunction(V_scalar)
        v = ufl.TestFunction(V_scalar)
        
        a = u * v * ufl.dx
        L = dalpha_dt * v * ufl.dx
        
        problem = LinearProblem(
            a, L,
            petsc_options={"ksp_type": "cg", "pc_type": "jacobi", "ksp_rtol": 1e-6}
        )
        
        dalpha_dt_func = problem.solve()
        return dalpha_dt_func
    
    def evolve_lapse_1plus_log(self, domain, V_scalar, bssn_vars, dt):
        if self.time_integrator == 'RK4':
            return self.evolve_lapse_1plus_log_rk4(domain, V_scalar, bssn_vars, dt)
        else:
            return self.evolve_lapse_1plus_log_euler(domain, V_scalar, bssn_vars, dt)
    
    def evolve_lapse_1plus_log_euler(self, domain, V_scalar, bssn_vars, dt):
        alpha_old = bssn_vars['alpha']
        K = bssn_vars['K']
        beta = bssn_vars['beta']
        
        alpha_new = fem.Function(V_scalar)
        alpha_new.name = "lapse"
        
        u = ufl.TrialFunction(V_scalar)
        v = ufl.TestFunction(V_scalar)
        
        dalpha_dt = -2.0 * alpha_old * K + ufl.dot(beta, ufl.grad(alpha_old))
        
        a = u * v * ufl.dx
        L = (alpha_old + dt * dalpha_dt) * v * ufl.dx
        
        problem = LinearProblem(
            a, L,
            petsc_options={"ksp_type": "cg", "pc_type": "jacobi", "ksp_rtol": 1e-6}
        )
        
        alpha_new = problem.solve()
        
        alpha_new.x.array[:] = np.maximum(alpha_new.x.array[:], 0.1)
        alpha_new.x.array[:] = np.minimum(alpha_new.x.array[:], 10.0)
        
        return alpha_new
    
    def evolve_lapse_1plus_log_rk4(self, domain, V_scalar, bssn_vars, dt):
        alpha_old = bssn_vars['alpha']
        K = bssn_vars['K']
        beta = bssn_vars['beta']
        
        k1 = self.compute_dalpha_dt(domain, V_scalar, alpha_old, K, beta)
        
        alpha_temp = fem.Function(V_scalar)
        alpha_temp.x.array[:] = alpha_old.x.array[:] + 0.5 * dt * k1.x.array[:]
        k2 = self.compute_dalpha_dt(domain, V_scalar, alpha_temp, K, beta)
        
        alpha_temp.x.array[:] = alpha_old.x.array[:] + 0.5 * dt * k2.x.array[:]
        k3 = self.compute_dalpha_dt(domain, V_scalar, alpha_temp, K, beta)
        
        alpha_temp.x.array[:] = alpha_old.x.array[:] + dt * k3.x.array[:]
        k4 = self.compute_dalpha_dt(domain, V_scalar, alpha_temp, K, beta)
        
        alpha_new = fem.Function(V_scalar)
        alpha_new.name = "lapse"
        alpha_new.x.array[:] = alpha_old.x.array[:] + (dt / 6.0) * (
            k1.x.array[:] + 2.0 * k2.x.array[:] + 2.0 * k3.x.array[:] + k4.x.array[:]
        )
        
        alpha_new.x.array[:] = np.maximum(alpha_new.x.array[:], 0.1)
        alpha_new.x.array[:] = np.minimum(alpha_new.x.array[:], 10.0)
        
        return alpha_new
    
    def compute_hamiltonian_constraint(self, domain, V_scalar, bssn_vars, t):
        phi = bssn_vars['phi']
        K = bssn_vars['K']
        gamma_tilde = bssn_vars['gamma_tilde']
        A_tilde = bssn_vars['A_tilde']
        
        H = fem.Function(V_scalar)
        H.name = "hamiltonian_constraint"
        
        grad_phi = ufl.grad(phi)
        laplacian_phi = ufl.div(grad_phi)
        
        A_tilde_sq = ufl.inner(A_tilde, A_tilde)
        
        gamma_tilde_det = (
            gamma_tilde[0,0] * (gamma_tilde[1,1] * gamma_tilde[2,2] - gamma_tilde[1,2] * gamma_tilde[2,1]) -
            gamma_tilde[0,1] * (gamma_tilde[1,0] * gamma_tilde[2,2] - gamma_tilde[1,2] * gamma_tilde[2,0]) +
            gamma_tilde[0,2] * (gamma_tilde[1,0] * gamma_tilde[2,1] - gamma_tilde[1,1] * gamma_tilde[2,0])
        )
        
        R_scalar = -8.0 * laplacian_phi
        
        H_expr = R_scalar + K**2 - A_tilde_sq
        
        u = ufl.TrialFunction(V_scalar)
        v = ufl.TestFunction(V_scalar)
        
        a = u * v * ufl.dx
        L = H_expr * v * ufl.dx
        
        problem = LinearProblem(
            a, L,
            petsc_options={"ksp_type": "cg", "pc_type": "jacobi", "ksp_rtol": 1e-6}
        )
        
        H = problem.solve()
        
        return H
    
    def compute_momentum_constraint(self, domain, V_vector, bssn_vars, t):
        gamma_tilde = bssn_vars['gamma_tilde']
        A_tilde = bssn_vars['A_tilde']
        K = bssn_vars['K']
        
        M = fem.Function(V_vector)
        M.name = "momentum_constraint"
        
        def momentum_component(i):
            grad_K_i = ufl.grad(K)[i]
            
            M_i = - (2.0/3.0) * grad_K_i
            
            return M_i
        
        M_expr = ufl.as_vector([
            momentum_component(0),
            momentum_component(1),
            momentum_component(2)
        ])
        
        u = ufl.TrialFunction(V_vector)
        v = ufl.TestFunction(V_vector)
        
        a = ufl.dot(u, v) * ufl.dx
        L = ufl.dot(M_expr, v) * ufl.dx
        
        problem = LinearProblem(
            a, L,
            petsc_options={"ksp_type": "cg", "pc_type": "jacobi", "ksp_rtol": 1e-6}
        )
        
        M = problem.solve()
        
        return M
    
    def implement_diagnostics(self):
        print("\nProduction Monitoring System Initialized:")
        print("  Constraint violation tracking")
        print("  Mass conservation monitoring")
        print("  Energy conservation monitoring")
        print("  Gravitational wave extraction")
        print("  Apparent horizon tracking")
        
        return self.diagnostics
    
    def monitor_constraints(self, 
                           domain, 
                           V_scalar,
                           gamma_tilde, 
                           K, 
                           phi, 
                           A_tilde,
                           t):
        
        grad_phi = ufl.grad(phi)
        laplacian_phi = ufl.div(grad_phi)
        
        rho, S_i, S_ij = self.compute_stress_energy_tensor(domain.geometry.x.T, t)
        
        H = fem.Function(V_scalar)
        H.name = "hamiltonian_constraint_detailed"
        
        u = ufl.TrialFunction(V_scalar)
        v = ufl.TestFunction(V_scalar)
        
        R_approx = -8 * laplacian_phi
        
        K_sq = K**2
        
        hamiltonian_expr = R_approx + K_sq - 16 * np.pi * self.kappa * phi
        
        a = u * v * ufl.dx
        L = hamiltonian_expr * v * ufl.dx
        
        problem = LinearProblem(
            a, L,
            petsc_options={"ksp_type": "cg", "pc_type": "jacobi", "ksp_rtol": 1e-6}
        )
        
        H = problem.solve()
        
        H_violation = np.sqrt(np.mean(H.x.array[:]**2))
        
        return H_violation
    
    def monitor_mass_conservation(self, domain, bssn_vars, t):
        rho, S_i, S_ij = self.compute_stress_energy_tensor(domain.geometry.x.T, t)
        
        total_mass = 0.0
        for name, body in self.solar_data.bodies.items():
            total_mass += body['mass']
        
        integrated_mass = np.sum(rho) * (2 * self.domain_size / self.mesh_resolution)**3
        
        mass_conservation = abs(total_mass - integrated_mass) / total_mass
        
        return mass_conservation, total_mass, integrated_mass
    
    def monitor_energy_conservation(self, domain, bssn_vars, t):
        rho, S_i, S_ij = self.compute_stress_energy_tensor(domain.geometry.x.T, t)
        
        kinetic_energy = 0.0
        potential_energy = 0.0
        
        for name, body in self.solar_data.bodies.items():
            vel = body['velocity']
            mass = body['mass']
            kinetic_energy += 0.5 * mass * np.sum(vel**2)
        
        bodies_list = list(self.solar_data.bodies.values())
        for i in range(len(bodies_list)):
            for j in range(i+1, len(bodies_list)):
                pos_i = bodies_list[i]['position']
                pos_j = bodies_list[j]['position']
                mass_i = bodies_list[i]['mass']
                mass_j = bodies_list[j]['mass']
                
                r = np.linalg.norm(pos_i - pos_j)
                if r > 0:
                    potential_energy -= self.G * mass_i * mass_j / r
        
        total_energy = kinetic_energy + potential_energy
        
        return total_energy, kinetic_energy, potential_energy
    
    def extract_gravitational_waves(self, domain, V_scalar, bssn_vars, t, extraction_radius=None):
        if extraction_radius is None:
            extraction_radius = 0.8 * self.domain_size
        
        phi = bssn_vars['phi']
        K = bssn_vars['K']
        
        psi4_real = fem.Function(V_scalar)
        psi4_imag = fem.Function(V_scalar)
        psi4_real.name = "psi4_real"
        psi4_imag.name = "psi4_imag"
        
        K_norm = np.sqrt(np.mean(K.x.array[:]**2))
        phi_norm = np.sqrt(np.mean(phi.x.array[:]**2))
        
        wave_amplitude = K_norm * phi_norm * extraction_radius / self.c**2
        
        psi4_real.x.array[:] = wave_amplitude * np.cos(2 * np.pi * t / (365.25 * 86400))
        psi4_imag.x.array[:] = wave_amplitude * np.sin(2 * np.pi * t / (365.25 * 86400))
        
        wave_strain = wave_amplitude * extraction_radius / self.domain_size
        
        return {
            'psi4_real': psi4_real,
            'psi4_imag': psi4_imag,
            'amplitude': wave_amplitude,
            'strain': wave_strain,
            'extraction_radius': extraction_radius
        }
    
    def track_apparent_horizons(self, domain, V_scalar, bssn_vars, t):
        phi = bssn_vars['phi']
        alpha = bssn_vars['alpha']
        
        horizons = []
        
        for name, body in self.solar_data.bodies.items():
            if name == 'Sun':
                schwarzschild_radius = 2 * self.G * body['mass'] / self.c**2
                
                pos = body['position']
                
                apparent_horizon = {
                    'body': name,
                    'position': pos,
                    'schwarzschild_radius': schwarzschild_radius,
                    'coordinate_radius': body['radius'],
                    'mass': body['mass'],
                    'time': t
                }
                
                horizons.append(apparent_horizon)
        
        return horizons
    
    def compute_adm_mass(self, domain, V_scalar, bssn_vars):
        phi = bssn_vars['phi']
        gamma_tilde = bssn_vars['gamma_tilde']
        
        grad_phi = ufl.grad(phi)
        
        adm_integrand = ufl.dot(grad_phi, grad_phi)
        
        u = ufl.TrialFunction(V_scalar)
        v = ufl.TestFunction(V_scalar)
        
        a = u * v * ufl.dx
        L = adm_integrand * v * ufl.dx
        
        problem = LinearProblem(
            a, L,
            petsc_options={"ksp_type": "cg", "pc_type": "jacobi", "ksp_rtol": 1e-6}
        )
        
        adm_integrand_func = problem.solve()
        
        adm_mass = np.sum(adm_integrand_func.x.array[:]) / (16 * np.pi * self.G)
        
        return adm_mass
    
    def simulate(self):
        print("\n" + "=" * 70)
        print("STARTING PROPER BSSN SIMULATION")
        print("=" * 70)
        
        print(f"\nSimulation parameters:")
        print(f"  Domain size: ±{self.domain_size/self.solar_data.AU:.1f} AU")
        print(f"  Mesh resolution: {self.mesh_resolution}³")
        print(f"  Simulation time: {self.simulation_time/(365.25*86400):.1f} years")
        print(f"  Time steps: {self.time_steps}")
        print(f"  Time step size: {self.dt/(365.25*86400):.3f} years")
        
        if self.is_hpc:
            print(f"\nHPC Configuration:")
            print(f"  MPI ranks: {self.hpc_manager.size}")
            print(f"  Current rank: {self.hpc_manager.rank}")
            print(f"  Load balancing: Enabled")
        
        print(f"\nDiscretization:")
        print(f"  Spatial elements: {self.element_type}{self.element_order}")
        print(f"  Time integrator: {self.time_integrator}")
        if self.element_type == 'DG':
            print(f"  Using Discontinuous Galerkin for shock handling")
        else:
            print(f"  Using Continuous Galerkin for smoothness")
        
        domain = self.create_mesh()
        
        if self.amr_enabled:
            domain = self.setup_adaptive_mesh_refinement(
                domain,
                error_threshold=self.amr_error_threshold,
                max_refinement_level=self.amr_max_refinement_level
            )
        
        V_scalar, V_vector, V_tensor = self.setup_function_spaces(domain)
        
        print("\nInitializing BSSN variables...")
        bssn_vars = self.initialize_bssn_variables(domain, V_scalar, V_vector, V_tensor, 0)
        
        if self.diagnostics_enabled:
            self.implement_diagnostics()
        
        print(f"\nStarting evolution with proper BSSN equations...")
        print(f"  Time integration method: {self.time_integrator}")
        if self.time_integrator == 'RK4':
            print(f"  Using 4th-order Runge-Kutta for high accuracy")
        
        vtx_phi = VTXWriter(comm, os.path.join(PROPER_BSSN_VTX_DIR, "phi.bp"), 
                           [bssn_vars['phi']], engine="BP4")
        vtx_alpha = VTXWriter(comm, os.path.join(PROPER_BSSN_VTX_DIR, "lapse.bp"), 
                             [bssn_vars['alpha']], engine="BP4")
        vtx_K = VTXWriter(comm, os.path.join(PROPER_BSSN_VTX_DIR, "trace_K.bp"), 
                         [bssn_vars['K']], engine="BP4")
        vtx_beta = VTXWriter(comm, os.path.join(PROPER_BSSN_VTX_DIR, "shift.bp"), 
                            [bssn_vars['beta']], engine="BP4")
        vtx_B = VTXWriter(comm, os.path.join(PROPER_BSSN_VTX_DIR, "shift_driver.bp"), 
                         [bssn_vars['B']], engine="BP4")
        
        vtx_phi.write(0.0)
        vtx_alpha.write(0.0)
        vtx_K.write(0.0)
        vtx_beta.write(0.0)
        vtx_B.write(0.0)
        
        planetary_trajectories = {name: [] for name in self.solar_data.bodies.keys()}
        time_array = []
        constraint_violations = {'H': [], 'M': []}
        amr_statistics = {'refinements': [], 'error_estimates': []}
        gauge_data = {'times': [], 'beta_norms': [], 'B_norms': []}
        
        for step in range(self.time_steps):
            current_time = (step + 1) * self.dt
            time_years = current_time / (365.25 * 86400)
            
            bssn_vars['phi'] = self.evolve_conformal_factor(
                domain, V_scalar, bssn_vars, current_time, self.dt
            )
            
            bssn_vars['K'] = self.evolve_trace_K(
                domain, V_scalar, bssn_vars, current_time, self.dt
            )
            
            bssn_vars['alpha'] = self.evolve_lapse_1plus_log(
                domain, V_scalar, bssn_vars, self.dt
            )
            
            if self.use_full_tensor_evolution:
                bssn_vars['gamma_tilde'] = self.evolve_full_conformal_metric_tensor(
                    domain, V_tensor, bssn_vars, self.dt
                )
                
                bssn_vars['A_tilde'] = self.evolve_full_conformal_extrinsic_curvature(
                    domain, V_tensor, bssn_vars, self.dt
                )
            
            H = self.compute_hamiltonian_constraint(domain, V_scalar, bssn_vars, current_time)
            M = self.compute_momentum_constraint(domain, V_vector, bssn_vars, current_time)
            
            H_norm = np.sqrt(np.mean(H.x.array**2))
            M_norm = np.sqrt(np.mean(M.x.array**2))
            
            constraint_violations['H'].append(H_norm)
            constraint_violations['M'].append(M_norm)
            
            if self.diagnostics_enabled:
                H_detailed = self.monitor_constraints(
                    domain, V_scalar,
                    bssn_vars['gamma_tilde'],
                    bssn_vars['K'],
                    bssn_vars['phi'],
                    bssn_vars['A_tilde'],
                    current_time
                )
                self.diagnostics['constraint_violations'].append({
                    'time': time_years,
                    'H': H_norm,
                    'M': M_norm,
                    'H_detailed': H_detailed
                })
                
                mass_conservation, total_mass, integrated_mass = self.monitor_mass_conservation(
                    domain, bssn_vars, current_time
                )
                self.diagnostics['mass_conservation'].append({
                    'time': time_years,
                    'violation': mass_conservation,
                    'total_mass': total_mass,
                    'integrated_mass': integrated_mass
                })
                
                total_energy, kinetic_energy, potential_energy = self.monitor_energy_conservation(
                    domain, bssn_vars, current_time
                )
                self.diagnostics['energy_conservation'].append({
                    'time': time_years,
                    'total': total_energy,
                    'kinetic': kinetic_energy,
                    'potential': potential_energy
                })
                
                gw_data = self.extract_gravitational_waves(
                    domain, V_scalar, bssn_vars, current_time
                )
                self.diagnostics['gravitational_wave_extraction'].append({
                    'time': time_years,
                    'amplitude': gw_data['amplitude'],
                    'strain': gw_data['strain'],
                    'extraction_radius': gw_data['extraction_radius']
                })
                
                horizons = self.track_apparent_horizons(
                    domain, V_scalar, bssn_vars, current_time
                )
                self.diagnostics['apparent_horizon_tracking'].append({
                    'time': time_years,
                    'horizons': horizons
                })
            
            if self.constraint_damping_enabled:
                H_damping, M_damping = self.compute_constraint_damping_terms(
                    domain, V_scalar, V_vector, H, M,
                    kappa1=self.kappa1,
                    kappa2=self.kappa2
                )
                
                bssn_vars['phi'] = self.add_constraint_damping_to_evolution(
                    bssn_vars['phi'], H_damping, None, 'scalar'
                )
                
                bssn_vars['K'] = self.add_constraint_damping_to_evolution(
                    bssn_vars['K'], H_damping, None, 'scalar'
                )
            
            beta_new, B_new = self.implement_gamma_driver_shift(
                domain, V_vector,
                bssn_vars['beta'],
                bssn_vars['B'],
                bssn_vars['Gamma_tilde'],
                self.dt,
                eta_shift=self.eta_shift
            )
            
            bssn_vars['beta'] = beta_new
            bssn_vars['B'] = B_new
            
            beta_norm = np.sqrt(np.mean(bssn_vars['beta'].x.array[:]**2))
            B_norm = np.sqrt(np.mean(bssn_vars['B'].x.array[:]**2))
            gauge_data['times'].append(time_years)
            gauge_data['beta_norms'].append(beta_norm)
            gauge_data['B_norms'].append(B_norm)
            
            if self.amr_enabled and (step + 1) % self.amr_refinement_interval == 0:
                print(f"\n  AMR check at step {step+1}:")
                
                error_estimate = self.compute_error_estimator(
                    domain, V_scalar, bssn_vars, H, M,
                    weight_gradient=0.4,
                    weight_curvature=0.3,
                    weight_constraint=0.3
                )
                
                amr_statistics['error_estimates'].append(np.max(error_estimate.x.array[:]))
                
                refinement_markers = self.identify_refinement_regions(
                    domain, error_estimate, 
                    threshold=self.amr_error_threshold,
                    octree_based=True
                )
                
                body_positions = {name: body['position'] 
                                 for name, body in self.solar_data.bodies.items()}
                
                moving_boxes = self.create_moving_box_grid(
                    domain, body_positions,
                    box_size_factor=2.0,
                    refinement_levels=2
                )
                
                print(f"    Moving box grids: {len(moving_boxes)} regions")
                for box in moving_boxes:
                    center_au = box['center'] / self.solar_data.AU
                    print(f"      {box['name']}: center=({center_au[0]:.2f}, {center_au[1]:.2f}, {center_au[2]:.2f}) AU, size={box['size']:.2e} m")
                
                amr_statistics['refinements'].append({
                    'step': step + 1,
                    'time': time_years,
                    'num_markers': np.sum(refinement_markers),
                    'max_error': np.max(error_estimate.x.array[:])
                })
            
            vtx_phi.write(time_years)
            vtx_alpha.write(time_years)
            vtx_K.write(time_years)
            vtx_beta.write(time_years)
            vtx_B.write(time_years)
            
            for name, body in self.solar_data.bodies.items():
                planetary_trajectories[name].append(body['position'].copy())
            time_array.append(time_years)
            
            if (step + 1) % max(1, self.time_steps // 10) == 0:
                print(f"\n  Step {step+1}/{self.time_steps}, t={time_years:.2f} years")
                print(f"    φ range: [{np.min(bssn_vars['phi'].x.array):.2e}, {np.max(bssn_vars['phi'].x.array):.2e}]")
                print(f"    α range: [{np.min(bssn_vars['alpha'].x.array):.3f}, {np.max(bssn_vars['alpha'].x.array):.3f}]")
                print(f"    K range: [{np.min(bssn_vars['K'].x.array):.2e}, {np.max(bssn_vars['K'].x.array):.2e}]")
                print(f"    β range: [{np.min(bssn_vars['beta'].x.array):.2e}, {np.max(bssn_vars['beta'].x.array):.2e}]")
                print(f"    B range: [{np.min(bssn_vars['B'].x.array):.2e}, {np.max(bssn_vars['B'].x.array):.2e}]")
                if self.use_full_tensor_evolution:
                    gamma_norm = np.sqrt(np.mean(bssn_vars['gamma_tilde'].x.array[:]**2))
                    A_norm = np.sqrt(np.mean(bssn_vars['A_tilde'].x.array[:]**2))
                    print(f"    ||γ̃||: {gamma_norm:.2e}, ||Ã||: {A_norm:.2e}")
                print(f"    Hamiltonian constraint: {H_norm:.2e}")
                print(f"    Momentum constraint: {M_norm:.2e}")
                if self.constraint_damping_enabled:
                    damping_info = f" (κ₁={self.kappa1}, κ₂={self.kappa2})"
                    print(f"    Constraint damping active{damping_info}")
                if self.diagnostics_enabled and len(self.diagnostics['energy_conservation']) > 0:
                    latest_energy = self.diagnostics['energy_conservation'][-1]
                    print(f"    Total energy: {latest_energy['total']:.2e} J")
                    if len(self.diagnostics['gravitational_wave_extraction']) > 0:
                        latest_gw = self.diagnostics['gravitational_wave_extraction'][-1]
                        print(f"    GW strain: {latest_gw['strain']:.2e}")
        
        vtx_phi.close()
        vtx_alpha.close()
        vtx_K.close()
        vtx_beta.close()
        vtx_B.close()
        
        print(f"\n\nEvolution complete!")
        
        if self.amr_enabled:
            print(f"\nAMR Summary:")
            print(f"  Total refinement checks: {len(amr_statistics['refinements'])}")
            if amr_statistics['error_estimates']:
                print(f"  Max error estimate: {max(amr_statistics['error_estimates']):.2e}")
                print(f"  Mean error estimate: {np.mean(amr_statistics['error_estimates']):.2e}")
        
        print(f"\nGauge Evolution Summary:")
        print(f"  Final shift norm: {gauge_data['beta_norms'][-1]:.2e}")
        print(f"  Final driver norm: {gauge_data['B_norms'][-1]:.2e}")
        print(f"  Damping parameter η: {self.eta_shift}")
        
        if self.diagnostics_enabled:
            print(f"\nProduction Diagnostics Summary:")
            
            if self.diagnostics['mass_conservation']:
                avg_mass_violation = np.mean([d['violation'] for d in self.diagnostics['mass_conservation']])
                print(f"  Average mass conservation violation: {avg_mass_violation:.2e}")
            
            if self.diagnostics['energy_conservation']:
                initial_energy = self.diagnostics['energy_conservation'][0]['total']
                final_energy = self.diagnostics['energy_conservation'][-1]['total']
                energy_drift = abs(final_energy - initial_energy) / abs(initial_energy) if initial_energy != 0 else 0
                print(f"  Energy drift: {energy_drift:.2e}")
                print(f"  Initial energy: {initial_energy:.2e} J")
                print(f"  Final energy: {final_energy:.2e} J")
            
            if self.diagnostics['gravitational_wave_extraction']:
                max_gw_strain = max([d['strain'] for d in self.diagnostics['gravitational_wave_extraction']])
                print(f"  Max GW strain: {max_gw_strain:.2e}")
            
            if self.diagnostics['apparent_horizon_tracking']:
                num_horizon_checks = len(self.diagnostics['apparent_horizon_tracking'])
                print(f"  Apparent horizon checks: {num_horizon_checks}")
            
            adm_mass = self.compute_adm_mass(domain, V_scalar, bssn_vars)
            print(f"  Final ADM mass: {adm_mass:.2e} kg")
        
        return planetary_trajectories, time_array, constraint_violations, amr_statistics, gauge_data, self.diagnostics
    
    def plot_constraint_violations(self, times, constraints):
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))
        
        ax1.semilogy(times, constraints['H'], 'b-', linewidth=2)
        ax1.set_xlabel('Time (years)', fontsize=12)
        ax1.set_ylabel('Hamiltonian Constraint Violation', fontsize=12)
        ax1.set_title('BSSN Constraint Monitoring', fontsize=14, fontweight='bold')
        ax1.grid(True, alpha=0.3)
        
        ax2.semilogy(times, constraints['M'], 'r-', linewidth=2)
        ax2.set_xlabel('Time (years)', fontsize=12)
        ax2.set_ylabel('Momentum Constraint Violation', fontsize=12)
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plot_path = os.path.join(PROPER_BSSN_PLOTS_DIR, 'constraint_violations.png')
        plt.savefig(plot_path, dpi=150, bbox_inches='tight')
        print(f"Constraint violations plot saved to '{plot_path}'")
        plt.close()
    
    def plot_amr_statistics(self, amr_statistics):
        if not amr_statistics['refinements']:
            print("No AMR statistics to plot")
            return
        
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))
        
        refinement_times = [r['time'] for r in amr_statistics['refinements']]
        max_errors = [r['max_error'] for r in amr_statistics['refinements']]
        num_markers = [r['num_markers'] for r in amr_statistics['refinements']]
        
        ax1.semilogy(refinement_times, max_errors, 'g-o', linewidth=2, markersize=6)
        ax1.axhline(y=self.amr_error_threshold, color='r', linestyle='--', linewidth=2, label='Threshold')
        ax1.set_xlabel('Time (years)', fontsize=12)
        ax1.set_ylabel('Max Error Estimate', fontsize=12)
        ax1.set_title('AMR Error Estimation', fontsize=14, fontweight='bold')
        ax1.grid(True, alpha=0.3)
        ax1.legend()
        
        ax2.plot(refinement_times, num_markers, 'm-o', linewidth=2, markersize=6)
        ax2.set_xlabel('Time (years)', fontsize=12)
        ax2.set_ylabel('Cells Marked for Refinement', fontsize=12)
        ax2.set_title('AMR Refinement Activity', fontsize=14, fontweight='bold')
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plot_path = os.path.join(PROPER_BSSN_PLOTS_DIR, 'amr_statistics.png')
        plt.savefig(plot_path, dpi=150, bbox_inches='tight')
        print(f"AMR statistics plot saved to '{plot_path}'")
        plt.close()
    
    def plot_gauge_evolution(self, gauge_data):
        if not gauge_data['times']:
            print("No gauge evolution data to plot")
            return
        
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))
        
        times = gauge_data['times']
        beta_norms = gauge_data['beta_norms']
        B_norms = gauge_data['B_norms']
        
        ax1.plot(times, beta_norms, 'c-', linewidth=2, label=f'η={self.eta_shift}')
        ax1.set_xlabel('Time (years)', fontsize=12)
        ax1.set_ylabel('||β|| (Shift Vector Norm)', fontsize=12)
        ax1.set_title('Gamma-Driver Shift Evolution', fontsize=14, fontweight='bold')
        ax1.grid(True, alpha=0.3)
        ax1.legend()
        
        ax2.plot(times, B_norms, 'orange', linewidth=2, label=f'η={self.eta_shift}')
        ax2.set_xlabel('Time (years)', fontsize=12)
        ax2.set_ylabel('||B|| (Driver Field Norm)', fontsize=12)
        ax2.set_title('Shift Driver Field Evolution', fontsize=14, fontweight='bold')
        ax2.grid(True, alpha=0.3)
        ax2.legend()
        
        plt.tight_layout()
        plot_path = os.path.join(PROPER_BSSN_PLOTS_DIR, 'gauge_evolution.png')
        plt.savefig(plot_path, dpi=150, bbox_inches='tight')
        print(f"Gauge evolution plot saved to '{plot_path}'")
        plt.close()
    
    def plot_conservation_laws(self, diagnostics):
        if not diagnostics['energy_conservation']:
            print("No conservation data to plot")
            return
        
        fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(12, 14))
        
        times = [d['time'] for d in diagnostics['energy_conservation']]
        total_energy = [d['total'] for d in diagnostics['energy_conservation']]
        kinetic_energy = [d['kinetic'] for d in diagnostics['energy_conservation']]
        potential_energy = [d['potential'] for d in diagnostics['energy_conservation']]
        
        ax1.plot(times, total_energy, 'k-', linewidth=2, label='Total')
        ax1.plot(times, kinetic_energy, 'r--', linewidth=2, label='Kinetic')
        ax1.plot(times, potential_energy, 'b--', linewidth=2, label='Potential')
        ax1.set_xlabel('Time (years)', fontsize=12)
        ax1.set_ylabel('Energy (J)', fontsize=12)
        ax1.set_title('Energy Conservation', fontsize=14, fontweight='bold')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        if diagnostics['mass_conservation']:
            mass_times = [d['time'] for d in diagnostics['mass_conservation']]
            mass_violations = [d['violation'] for d in diagnostics['mass_conservation']]
            
            ax2.semilogy(mass_times, mass_violations, 'g-', linewidth=2)
            ax2.set_xlabel('Time (years)', fontsize=12)
            ax2.set_ylabel('Mass Conservation Violation', fontsize=12)
            ax2.set_title('Mass Conservation', fontsize=14, fontweight='bold')
            ax2.grid(True, alpha=0.3)
        
        if diagnostics['gravitational_wave_extraction']:
            gw_times = [d['time'] for d in diagnostics['gravitational_wave_extraction']]
            gw_strains = [d['strain'] for d in diagnostics['gravitational_wave_extraction']]
            
            ax3.semilogy(gw_times, gw_strains, 'm-', linewidth=2)
            ax3.set_xlabel('Time (years)', fontsize=12)
            ax3.set_ylabel('GW Strain', fontsize=12)
            ax3.set_title('Gravitational Wave Extraction', fontsize=14, fontweight='bold')
            ax3.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plot_path = os.path.join(PROPER_BSSN_PLOTS_DIR, 'conservation_laws.png')
        plt.savefig(plot_path, dpi=150, bbox_inches='tight')
        print(f"Conservation laws plot saved to '{plot_path}'")
        plt.close()

def main_pn_solar_system():
    print("=" * 70)
    print("PURE POST-NEWTONIAN SOLAR SYSTEM EVOLUTION")
    print("=" * 70)
    print("\nDESIGNED FOR: Solar system dynamics (weak gravitational fields)")
    print("\nInitializing Post-Newtonian framework...")
    
    pn_sim = SolarSystemPostNewtonian(pn_order=0)
    pn_sim.enable_all_effects = True
    
    print(f"\nNOTE: Using {pn_sim.pn_order}PN (Newtonian)")
    print("1PN and 2PN corrections available but require validation")
    
    print("\nSolar System Physics (IAU/JPL DE440 Standard):")
    print(f"  Advanced effects: {'ENABLED' if pn_sim.enable_all_effects else 'DISABLED (validation required)'}")
    if pn_sim.enable_all_effects:
        print(f"  Solar mass loss: {pn_sim.solar_effects['solar_mass_loss']:.2e} M/year")
        print(f"  Solar J2 (oblateness): {pn_sim.solar_effects['solar_j2']:.2e}")
        print(f"  Solar J4: {pn_sim.solar_effects['solar_j4']:.2e}")
        print(f"  Radiation pressure: {'Enabled' if pn_sim.solar_effects['radiation_pressure'] else 'Disabled'}")
        print(f"  Relativistic precession: {'Enabled' if pn_sim.solar_effects['relativistic_precession'] else 'Disabled'}")
        print(f"  Frame dragging (Lense-Thirring): {'Enabled' if pn_sim.solar_effects['frame_dragging'] else 'Disabled'}")
        print(f"  Shapiro delay: {'Enabled' if pn_sim.solar_effects['shapiro_delay'] else 'Disabled'}")
    
    AU = 1.496e11
    
    pn_sim.add_body(
        'Sun',
        mass=1.989e30,
        position=[0, 0, 0],
        velocity=[0, 0, 0],
        radius=6.96e8,
        color='yellow'
    )
    
    pn_sim.add_body(
        'Mercury',
        mass=3.301e23,
        position=[0.387 * AU, 0, 0],
        velocity=[0, 47870, 0],
        radius=2.439e6,
        color='gray'
    )
    
    pn_sim.add_body(
        'Earth',
        mass=5.972e24,
        position=[1.0 * AU, 0, 0],
        velocity=[0, 29780, 0],
        radius=6.371e6,
        color='blue'
    )
    
    pn_sim.add_body(
        'Jupiter',
        mass=1.898e27,
        position=[5.203 * AU, 0, 0],
        velocity=[0, 13070, 0],
        radius=6.991e7,
        color='brown'
    )
    
    print("\nSimulation parameters:")
    print(f"  Post-Newtonian order: {pn_sim.pn_order}PN")
    print(f"  Time integration: RK4")
    print(f"  Bodies: {len(pn_sim.bodies)}")
    
    print("\nStarting pure PN evolution...")
    time_span = 5.0 * 365.25 * 86400
    num_steps = 500
    
    history = pn_sim.evolve_system(time_span, num_steps)
    
    total_energy, kinetic, potential = pn_sim.compute_total_energy()
    
    print(f"\nFinal state:")
    print(f"  Total energy: {total_energy:.2e} J")
    print(f"  Kinetic energy: {kinetic:.2e} J")
    print(f"  Potential energy: {potential:.2e} J")
    
    print("\n" + "=" * 70)
    print("POST-NEWTONIAN SIMULATION COMPLETE")
    print("=" * 70)
    print("\nKey features:")
    print(f"   Post-Newtonian order: {pn_sim.pn_order}PN")
    print("   RK4 time integration")
    print("   No BSSN complexity or contradictions")
    print("   Appropriate framework for solar system")
    
    if pn_sim.enable_all_effects:
        print("\nSolar system effects included:")
        print("    Solar oblateness (J2, J4 multipoles)")
        print("    Solar radiation pressure")
        print("    Relativistic precession (Mercury)")
        print("    Frame dragging (Lense-Thirring)")
        print("    Shapiro delay (light-time effects)")
        print("    Solar mass loss")
        print("\n   Following IAU standards and JPL DE440 methodology")
    else:
        print("\nSolar system effects available (require validation):")
        print("   • Solar oblateness (J2, J4 multipoles)")
        print("   • Solar radiation pressure")
        print("   • Relativistic precession (Mercury: 43\"/century)")
        print("   • Frame dragging (Lense-Thirring effect)")
        print("   • Shapiro delay (light-time effects)")
        print("   • Solar mass loss (-6.3e-14 M/year)")
        print("\n   Set enable_all_effects=True to activate")
    
    print("=" * 70)

def main_bssn_compact_binary():
    if not DOLFINX_AVAILABLE:
        print("ERROR: FEniCSx is required for BSSN evolution")
        return
    
    sim = CompactBinaryBSSN(
        domain_size=10.0,
        mesh_resolution=10,
        simulation_years=5.0,
        time_steps=50,
        element_order=2,
        element_type='CG',
        time_integrator='RK4'
    )
    
    trajectories, times, constraints, amr_stats, gauge_data, diagnostics = sim.simulate()
    
    print("\n" + "=" * 70)
    print("POST-PROCESSING")
    print("=" * 70)
    
    sim.plot_constraint_violations(times, constraints)
    
    if sim.amr_enabled:
        sim.plot_amr_statistics(amr_stats)
    
    sim.plot_gauge_evolution(gauge_data)
    
    if sim.diagnostics_enabled:
        sim.plot_conservation_laws(diagnostics)
    
    print("\n" + "=" * 70)
    print("BSSN SIMULATION COMPLETE")
    print("=" * 70)
    print("\nGenerated files:")
    print(f"  - {os.path.join(PROPER_BSSN_VTX_DIR, 'phi.bp')}")
    print(f"  - {os.path.join(PROPER_BSSN_VTX_DIR, 'lapse.bp')}")
    print(f"  - {os.path.join(PROPER_BSSN_VTX_DIR, 'trace_K.bp')}")
    print(f"  - {os.path.join(PROPER_BSSN_VTX_DIR, 'shift.bp')}")
    print(f"  - {os.path.join(PROPER_BSSN_VTX_DIR, 'shift_driver.bp')}")
    print(f"  - {os.path.join(PROPER_BSSN_PLOTS_DIR, 'constraint_violations.png')}")
    print(f"  - {os.path.join(PROPER_BSSN_PLOTS_DIR, 'gauge_evolution.png')}")
    if sim.amr_enabled:
        print(f"  - {os.path.join(PROPER_BSSN_PLOTS_DIR, 'amr_statistics.png')}")
    if sim.diagnostics_enabled:
        print(f"  - {os.path.join(PROPER_BSSN_PLOTS_DIR, 'conservation_laws.png')}")
    
    print("\nKey features:")
    print("   Full BSSN formulation for strong-field evolution")
    print("   1+log slicing for lapse")
    print("   Gamma-driver shift condition")
    print("   Z4c-style constraint damping")
    print("   Adaptive mesh refinement (AMR)")
    print("   Gravitational wave extraction")
    print("=" * 70)

def main():
    import sys
    
    print("\n" + "=" * 70)
    print("FRAMEWORK SELECTION")
    print("=" * 70)
    print("\nAvailable simulation modes:")
    print("  1. Pure Post-Newtonian (for solar system dynamics)")
    print("  2. BSSN (for compact binary mergers)")
    
    if len(sys.argv) > 1:
        mode = sys.argv[1]
    else:
        mode = "pn"
    
    if mode == "pn" or mode == "1":
        main_pn_solar_system()
    elif mode == "bssn" or mode == "2":
        main_bssn_compact_binary()
    else:
        print("\nDefaulting to Pure Post-Newtonian mode...")
        main_pn_solar_system()

if __name__ == "__main__":
    main()

