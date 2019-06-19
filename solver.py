#Flow past circular cylinder using RVM
import numpy as np

import matplotlib.pyplot as plt
from numba import jit

import pdb

sin = np.sin
cos = np.cos
pi = np.pi
sqrt = np.sqrt
log = np.log


class panel():
    def __init__(self, z1, z2, phi):
        self.z1 = z1    # Start point of panel
        self.z2 = z2    # End point of panel
        self.z_cen = (z1+z2)/2.0    # Control point of panel
        self.phi = phi  # Angle of inclination of panel from horizontal
        self.beta = self.phi+(pi/2.0)   # Angle of inclination of normal of panel from horizontal
        self.normal = cos(self.beta) + 1j*sin(self.beta)
        self.gamma1 = 0
        self.gamma2 = 0
        self.strength = 0
        self.V_w = 0
        self.tangent = (z1-z2) / np.absolute(z1-z2)

    def local_integral(self, panel_j):
        # Velocity per unit strength induced by panel j on panel i based on coordinate of panel j      
        l = (panel_j.z2-panel_j.z1)   
        m = log((self.z_cen-panel_j.z2)/(self.z_cen-panel_j.z1))
        n = ((self.z_cen-panel_j.z1)*m)/l
        return ((1j/(2*pi))*np.asarray([m-n-1,n+1])).conjugate()
    
    def global_integral(self, panel_j):
        # Velocity per unit strength induced by panel j based on global co-ordinate  
        vel_global = (self.local_integral(panel_j)) * (cos(panel_j.phi)+1j*sin(panel_j.phi))
        a_Vn = dotprod(vel_global, self.normal) 
        return a_Vn

    def vel_induce(self, z):
        # Velocity induced by panel i at z based on frame of reference of panel i
        l = (self.z2-self.z1)   
        k = (self.gamma2-self.gamma1)/l
        m = log((z-self.z2)/(z-self.z1))
        vel = (1j/(2*pi))*(k*(l+(z-self.z1)*m)+(self.gamma1*m))
        # Converts panel frame of reference to global frame of reference
        return vel.conjugate()*(cos(self.phi)+1j*sin(self.phi))


def panel_create(D, n):
    # Panels stores list of panel classes
    panels = []
    panel_endpoints = []

    # Appends end points of each panel to panel_endpoints list
    for i in np.arange(pi+(pi/n), -pi+(pi/n), -(2*pi)/n):
        panel_endpoints.append((D/2.0)*cos(i) + 1j*(D/2.0)*sin(i))

    # Generates inclination of panels
    theta = pi/2.0
    phi = []
    for j in range(n):
        if theta < 0:
            theta = 2*pi-abs(theta)
        phi.append(theta)
        theta = theta-((2*pi)/n)
    
    # Creates panel classes for different end points and inclination of panels
    for k in range(len(panel_endpoints)):
        if k != len(panel_endpoints)-1:
            panels.append(panel(panel_endpoints[k], panel_endpoints[k+1], phi[k]))   
        else:
            panels.append(panel(panel_endpoints[k], panel_endpoints[0], phi[k]))   
         
    return panels[:]


def dotprod(z1,z2):
    return (z1.real*z2.real)+(z1.imag*z2.imag) 


def vel_body():
    vel = 0.0 + 0.0*1j
    return vel


def vel_freestream(vel=1.0+0.0*1j):
    return vel


def vel_doublet(z, D):
    R = 0.5*D
    vel = (-1*vel_freestream()*(R**2))/(z**2)
    return vel.conjugate()


def net_velocity(z, D):
    return vel_freestream()+vel_doublet(z, D)    



def gamma_panels(panel_class_list, D=2.0, n=8, gamma=0, delta=0, vor_z=np.asarray([complex(0,0)]), vel_infi=1.0+0*1j):
    panels = panel_class_list[:] 

    V_fs = vel_infi
    V_body = vel_body()

    # Stores LHS matrix and RHS matrix for equation a*gamma = b
    a_Vn = []
    b_Vn = [0]*n
    for i in range(n):
        a_Vn.append([0]*n)

    for i,panel_i in enumerate(panels):
        for j,panel_j in enumerate(panels):
               if j!=len(panels)-1:  
                   a_Vn[i][j] += panel_i.global_integral(panel_j)[0] 
                   a_Vn[i][j+1] += panel_i.global_integral(panel_j)[1]
               else:
                   a_Vn[i][j] += panel_i.global_integral(panel_j)[0]
                   a_Vn[i][0] += panel_i.global_integral(panel_j)[1]
        V_w = blob_velocity(panel_i.z_cen, vor_z, gamma, delta)
        panel_i.V_w = V_w
        b_Vn[i] = -1*(dotprod(V_w, panel_i.normal))+dotprod(V_body, panel_i.normal)-dotprod(V_fs, panel_i.normal)
    # Imposed condition of sum of vortex strength of panels is zero
    a_Vn.append([1.0]*n)
    b_Vn.append(0.0)
    
    # Solves matrix using least squares to get strength of each vortex panel 
    gamma_i = np.linalg.lstsq(np.array(a_Vn),np.array(b_Vn))[0]
    panel_length = np.absolute(panels[0].z1-panels[0].z2)
    for k in range(len(gamma_i)):
        if k != len(gamma_i)-1:
            panels[k].gamma1,panels[k].gamma2 = gamma_i[k],gamma_i[k+1]
            panels[k].strength = 0.5*panel_length*(panels[k].gamma1+panels[k].gamma2) 
        else:
            panels[k].gamma1,panels[k].gamma2 = gamma_i[k],gamma_i[0]
            panels[k].strength = 0.5*panel_length*(panels[k].gamma1+panels[k].gamma2) 
    return panels[:]


@jit
def vel_by_panels(gamma_panel, z, D, n):
    # To compute flow field induced by vortex panels
    panels = gamma_panel
    vel = np.zeros(len(z), dtype=complex)
    for i in panels:
        vel += i.vel_induce(z)
    return vel

def vel_by_panels_on_blob(panels, vor_z, D, n, gamma, delta, vel_infi):
    vel = np.zeros(len(vor_z),dtype=complex)
    for i in panels:
        vel += i.vel_induce(vor_z)
    return vel.copy()

def blob_velocity(z, z_vor, gamma, delta):
    r = z-z_vor
    # For r < delta
    def vel1(r, gamma): 
        return (-1j*gamma*np.exp(-1j*np.arctan2(r.imag,r.real))/(2*pi*delta)).conjugate()
    # For r > delta
    def vel2(r, gamma): 
        return ((-1j*gamma)/(2*pi*r)).conjugate()
    
    if isinstance(z_vor, np.ndarray) == True and isinstance(gamma, np.ndarray) == True:     
        vel = list(map(lambda r, gamma: vel1(r, gamma) if np.absolute(r)<delta else vel2(r, gamma), r, gamma))
        return sum(vel) 
    elif isinstance(z_vor, np.ndarray) == False and isinstance(gamma, np.ndarray) == False:
        if np.absolute(r) < delta:
            return vel1(r, gamma)
        else:
            return vel2(r, gamma)
    else:
        gamma= 0.0
        return gamma*z_vor 

def runge_kutta_integrate(panel_class_list, g_panels, pos, D, n, gamma, delta, dt, tf, vel_infi):
    # Integrates path of vortex due to velocity induced by panels
    t = dt
    while t < tf:

         vel = vel_by_panels_on_blob(g_panels, pos.copy(), D, n, gamma, delta, vel_infi) + blob_array_velocity(pos.copy(),gamma.copy(), delta)+ vel_freestream()
         pos_mid = pos + (vel*dt)/2.0

         panels = gamma_panels(panel_class_list, D, n, gamma, delta, pos_mid.copy(), vel_infi)

         vel_mid = vel_by_panels_on_blob(panels, pos_mid.copy(), D, n, gamma, delta, vel_infi) + blob_array_velocity(pos_mid.copy(),gamma, delta) + vel_freestream()
         pos += dt*vel_mid

         t += dt

    return pos.copy(), gamma.copy()


@jit
def blob_array_velocity(pos, gamma, delta):
    vel = np.zeros_like(pos)
    for i, z_i in enumerate(pos):
        for j, z_j in enumerate(pos):
            if z_i!=z_j :
                vel[i] += blob_velocity(z_i, z_j, gamma[j], delta)
    return vel.copy()


def blob_from_panels(n, gamma_panel, delta, V_fs):
    l = np.absolute(gamma_panel[0].z1 - gamma_panel[0].z2)
    pos = []
    gamma = []
    for i in range(n):
        V_p = 0
        for j in range(n):
            if i != j:
                V_p += gamma_panel[j].vel_induce(gamma_panel[i].z_cen)
        pos.append(gamma_panel[i].z_cen + delta*(gamma_panel[i].normal/np.absolute(gamma_panel[i].normal)))
        if isinstance(gamma_panel[i].V_w , np.ndarray) == True:
            dp = dotprod(gamma_panel[i].V_w[0] + V_fs + V_p, -gamma_panel[i].tangent)
        else:
            dp = dotprod(gamma_panel[i].V_w + V_fs + V_p, -gamma_panel[i].tangent)
        gamma.append(-(dp * l))
    return pos[:], gamma[:]

def blob_from_panels_split(pos, gamma, gamma_max):
    # Splits blobs such that each smaller blob has gamma <= gamma_max
    pos1 = []
    gamma1 = []
    for i,g in zip(pos,gamma):
        if abs(g) > gamma_max:
            if abs(g)%gamma_max != 0:
                num_small_blob = int(abs(g)//gamma_max + 1)
                num_small_blob_gamma_max = num_small_blob-1
                if g<0:
                    gamma1.extend([-1*gamma_max]*(num_small_blob_gamma_max)+[-1*(abs(g)%gamma_max)])
                else:
                    gamma1.extend([gamma_max]*(num_small_blob_gamma_max)+[abs(g)%gamma_max])
            else:
                num_small_blob = int(abs(g)//gamma_max)
                num_small_blob_gamma_max = num_small_blob
                if g<0:
                    gamma1.extend([-1*gamma_max]*(num_small_blob_gamma_max))
                else:
                    gamma1.extend([gamma_max]*(num_small_blob_gamma_max))
            pos1.extend([i]*num_small_blob)
        else:
            pos1.append(i)
            gamma1.append(g)
    return np.array(pos1).copy(),np.array(gamma1).copy()


def random_walk(num_particles, kinematic_visco, t):
    sigma = sqrt(2*kinematic_visco*t)
    mean = 0.0
    gaussian_pos = np.random.normal(mean, sigma, num_particles)
    return gaussian_pos.copy()

@jit
def blob_diffusion(pos, kinematic_visco, dt, tf):
    t = dt
    while t<tf:
        pos += random_walk(len(pos), kinematic_visco, t)+1j*random_walk(len(pos), kinematic_visco, t)
        t += dt
    return pos.copy()


@jit
def check_blob_inside_cylinder(pos_after_diffuse, pos_b4_diffuse, r):
    # Check if blob within cylinder of radius r and deflects blob outside if it is within cylinder
    return np.asarray([z2 if np.absolute(z2)>r else (1.0+2.0*((1.0/np.absolute(z1))-1.0))*z1 
                       if np.absolute(z1)<1 else (2.0-(r/np.absolute(z1)))*z1 
                       for z1,z2 in zip(pos_b4_diffuse, pos_after_diffuse)]).copy()  


def timesteps(t = 0.1):
    D = 2.0
    n = 75

    panel_class_list = panel_create(D, n)
    panel_length = np.absolute(panel_class_list[0].z1-panel_class_list[0].z2)

    delta = panel_length/pi

    t_step = 0.1
    tf = t
    Re = 1000.0

    vel_infi = vel_freestream()
    visco = (vel_infi.real*D)/Re

    gamma_max = 0.1

    gamma_panel = gamma_panels(panel_class_list, D, n)

    pos_all = []
    gam_all = []
    t_list = []

    pos_blob_from_panel, gamma_blob_from_panel = blob_from_panels(n, gamma_panel, delta, vel_infi)
    pos_all.append(pos_blob_from_panel[:])
    gam_all.append(gamma_blob_from_panel[:])
    t_list.append(0)

    pos_blob_from_panel, gamma_blob_from_panel = blob_from_panels_split(pos_blob_from_panel[:], gamma_blob_from_panel[:], gamma_max)
    pos_all.append(pos_blob_from_panel[:])
    gam_all.append(gamma_blob_from_panel[:])
    t_list.append(0)

    pos_b4_diff = np.array(pos_blob_from_panel[:])
    gamma = np.array(gamma_blob_from_panel[:])

    pos = blob_diffusion(pos_b4_diff.copy(), visco, t_step, 2*t_step)
    pos = check_blob_inside_cylinder(pos.copy(), pos_b4_diff.copy(), D/2.0)
    pos_all.append(pos.copy())
    gam_all.append(gamma.copy())
    t_list.append(0)

    for t in np.arange(t_step, tf, t_step):
        # Updated panel class list with strength of each panel
        gamma_panel = gamma_panels(panel_class_list, D, n, gamma.copy(), delta, pos.copy())

        # Position and strength of blobs due to vorticity of respective panel, blob_from_panels returns lists
        pos_blob_from_panel, gamma_blob_from_panel = blob_from_panels(n, gamma_panel, delta, vel_infi)
        pos_all.append(pos_blob_from_panel[:])
        gam_all.append(gamma_blob_from_panel[:])
        t_list.append(t)

        # Postion and strength of blobs after splitting into smaller blobs of
        # strength <= gamma_max, return arrays
        pos_blob_from_panel, gamma_blob_from_panel = blob_from_panels_split(pos_blob_from_panel[:], gamma_blob_from_panel[:], gamma_max)
        pos_all.append(pos_blob_from_panel[:])
        gam_all.append(gamma_blob_from_panel[:])
        t_list.append(t)


        # Position after advection of blobs, influence from panels as well as other n-1 blobs
        pos_and_gamma = runge_kutta_integrate(panel_class_list, gamma_panel[:], pos.copy(), D, n, gamma.copy(), delta, t_step, 2*t_step, vel_infi)

        pos_b4_diff = np.concatenate((pos_and_gamma[0], pos_blob_from_panel), axis=0)
        gamma = np.concatenate((pos_and_gamma[1], gamma_blob_from_panel), axis=0)
        pos_all.append(pos.copy())
        gam_all.append(gamma.copy())
        t_list.append(t)

        pos = blob_diffusion(pos_b4_diff.copy(), visco, t_step, 2*t_step)
        pos = check_blob_inside_cylinder(pos.copy(), pos_b4_diff.copy(), D/2.0)
        pos_all.append(pos.copy())
        gam_all.append(gamma.copy())
        t_list.append(t)

    return t_list, pos_all, gam_all


#if __name__=="__main__":
#    timesteps()

