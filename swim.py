import numpy as np

from scipy.spatial import cKDTree

class LJ:
    
    def __init__(self):
        """
        param['epsilon'] = epsilon: height of barrier;
        param['sigma'] = sigma: minima of double well.
        """
        self.param={}

    def act(self, pos1, pos2):
        """
        defining an act function performing calculation on positions.
        """
        dpos = pos1 - pos2
        dist = np.linalg.norm(dpos, axis=1)
        R = self.param['sigma'] / dist
        
        return 4 * self.param['epsilon'] * (R**12 - R**6)
    
    def deriv(self, pos1, pos2):
        """
        derivatives.
        """
        dpos = pos1 - pos2
        dist = np.linalg.norm(dpos, axis=1)
        dist_sqr = dist*dist
        R = self.param['sigma'] / dist
        force_divr2 = -4 * self.param['epsilon'] * (12 * R**12 - 6 * R**6) / dist_sqr
        
        return force_divr2[:, np.newaxis] * dpos
        
class simple_md3d:
    
    def __init__(self, seed=0, calc_pe=False):
        self._rng = np.random.default_rng(seed)
        self.calc_pe = calc_pe
        self.pe = None
        self.nlist = None
        self.ndim = 3
        
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
    
    def dvdt_f(self, x, v, nlist):

        grad = np.zeros((self.num_particles, self.ndim))
        pos1 = x[nlist][:,0,:]
        pos2 = x[nlist][:,1,:]
        f = self.pe.deriv(pos1, pos2)
        self.neighbor_sum(grad, nlist[:,0], f)

        return - grad - self.damping * v
    
    def xi(self, kT, damping):
        """
        Wiener noise.
        """
        sigma = np.sqrt( 2 * self.ndim * kT * damping )
        return self._rng.normal(loc=0.0, 
                                scale=sigma, 
                                size=(self.num_particles, self.ndim))

    def nlist_search(self, x, rcut, full=False):
        """
        Neighbor list within rcut.
        """
        half_box = np.array(self.box)
        ckdtree = cKDTree(x+half_box/2, boxsize=self.box)
        # only count [i,j]
        nlist = ckdtree.query_pairs(r=rcut, output_type='ndarray')
        if full==True:
            # [i,j] and [j,i] both in nlist.
            nlist_swap = np.roll(nlist, shift=1, axis=1)
            nlist = np.vstack((nlist, nlist_swap))
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
        compute potential energy.
        """
        pos1 = x[nlist][:,0,:]
        pos2 = x[nlist][:,1,:]
        pe_val = np.sum(self.pe.act(pos1, pos2))
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
            # use full nlist for force calculation.
            nlist = self.nlist_search(x, rcut=self.pe.param['r_cut'], full=True)
            self.nlist = nlist
            
            predict_x = x + self.dt * v
            predict_v = v + self.dt * self.dvdt_f(x,v,nlist) + xi0 * np.sqrt(self.dt)
            
            # apply periodic boundary condition
            predict_x = self.pbc(predict_x, self.box)
            # use full nlist for force calculation.
            nlist_pred = self.nlist_search(predict_x, rcut=self.pe.param['r_cut'], full=True)
            # perform velocity rescaling for constant kT
            predict_v = self.v_rescale(predict_v, self.kT)
            
            xf = x + 0.5 * self.dt * (predict_v + v)
            f_temp = self.dvdt_f(predict_x,predict_v,nlist_pred) + self.dvdt_f(x,v,nlist)
            vf = v + 0.5 * self.dt * f_temp + xi0 * np.sqrt(self.dt)
            
            # apply periodic boundary condition
            xf = self.pbc(xf, self.box)
            # perform velocity rescaling for constant kT
            if i%1==0:
                vf = self.v_rescale(vf, self.kT)
    
            self.traj.append([i, xf, vf])

            if self.calc_pe==True:
                nlist_pe = self.nlist_search(xf, rcut=self.pe.param['r_cut'], full=False)
                self.pe = self.compute_pe(x, nlist_pe)
    