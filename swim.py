import numpy as np

from scipy.spatial import KDTree

class LJ:
    
    def __init__(self):
        """
        param['epsilon'] = epsilon: height of barrier;
        param['sigma'] = sigma: minima of double well.
        """
        self.param={}
        
    def act(self, pos1, pos2):
        """
        defining an act function taking simulation variable as input argument.
        """
        dist = np.linalg.norm(pos1 - pos2, axis=0)
        R = self.param['sigma'] / dist
        return 4 * self.param['epsilon'] * (R**12 - R**6)
    
    def deriv(self, pos1, pos2):
        """derivative."""
        dist = np.linalg.norm(pos1 - pos2, axis=0)
        sigma = self.param['sigma']
        R = self.param['sigma'] / dist
        return -4 * self.param['epsilon'] * (12 * R**12 - 6 * R**6) * (pos1 - pos2)
    
class simple_md3d:
    
    def __init__(self, seed=0):
        self._rng = np.random.default_rng(seed)
        self.pe = None
        self.nlist = None
        
    def set_param(self, dt, kT=1.0, damping=1.0):
        self.dt = dt
        self.kT = kT
        self.damping = damping
        
    def set_init_config(self, init_x3d, init_v3d, num_particles, box):
        """
        set initial configuration.
        """
        self.init_x3d = init_x3d
        self.init_v3d = init_v3d
        self.num_particles = num_particles
        self.box = box
        self.traj = [[0, self.init_x3d, self.init_v3d]]
    
    def dvdt_f(self, x, v):

        dim=3
        grad = np.zeros((self.num_particles, dim))
        f = np.array([self.pe.deriv(x[self.nlist][idx][0], 
                                    x[self.nlist][idx][1]) for idx in range(len(self.nlist))])
        self.neighbor_sum(grad, self.nlist[:,0], f)

        return - grad - self.damping * v
    
    def xi(self, kT, damping):
        """
        Wiener noise.
        """
        sigma = np.sqrt( 6 * kT * damping )
        return np.random.normal(loc=0.0, 
                                scale=sigma, 
                                size=(self.num_particles,3))

    def nlist_search(self, x, rcut):
        """
        Neighbor list within rcut.
        """
        kdtree = KDTree(x)
        nlist = kdtree.query_pairs(r=rcut, output_type='ndarray')
        
        return nlist
    
    def neighbor_sum(self, target, idx, vals):
        """
        target[idx] += vals
        """
        np.add.at(target, idx.ravel(), vals)
        
    def pbc(self, coord_in, box):
        """
        apply periodic boundary condition to the coordinates.
        """
        cell = np.array(box)
        coord_out = coord_in - np.floor(coord_in/cell + 0.5) * cell
    
        return coord_out
    
    def v_rescale(self, v, kT_target):
        """
        apply velocity rescaling method for constant kT.
        """
        ke = 0.5 * np.sum(v**2)
        kT_inst = 2 * ke / 3 / self.num_particles
        v_rescaling = np.sqrt(kT_target / kT_inst)
        v *= v_rescaling

        return v
    
    def compute_pe(self, x, nlist):
        """
        compute potential energy on-the-fly.
        """
        pe_val = np.sum([self.pe.act(x[nlist][idx][0], x[nlist][idx][1]) for idx in range(len(nlist))])
        return pe_val

    def first_run(self):
        
        self.nlist = self.nlist_search(self.init_x3d, rcut=self.pe.param['r_cut'])
        
        first_pe = self.compute_pe(self.init_x3d, self.nlist)
        self.traj_pe = [first_pe]
    
    def run(self, nsteps):
        """
        nsteps: number of steps.
        """
        
        self.first_run()
        
        for i in range(int(nsteps)):
            
            xi0 = self.xi(self.kT, self.damping)
            
            _, x, v = self.traj[-1]
            self.nlist = self.nlist_search(x, rcut=self.pe.param['r_cut'])
            
            predict_x = x + self.dt * v
            predict_v = v + self.dt * self.dvdt_f(x,v) \
                          + xi0 * np.sqrt(self.dt)
            
            # apply periodic boundary condition
            predict_x = self.pbc(predict_x, self.box)
            # perform velocity rescaling for constant kT
            predict_v = self.v_rescale(predict_v, self.kT)
            
            xf = x + 0.5 * self.dt * (predict_v + v)
            vf = v + 0.5 * self.dt * (self.dvdt_f(predict_x,predict_v) + self.dvdt_f(x,v)) \
                   + xi0 * np.sqrt(self.dt)
            
            # apply periodic boundary condition
            xf = self.pbc(xf, self.box)
            # perform velocity rescaling for constant kT
            vf = self.v_rescale(vf, self.kT)
    
            self.traj.append([i, xf, vf])
            
            # save total potential energy
            pe_val = self.compute_pe(x, self.nlist)
            self.traj_pe.append( pe_val )
    