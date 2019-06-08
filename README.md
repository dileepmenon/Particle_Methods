# Particle Methods
# Simulates the 2D flow around a circular cylinder using the Random Vortex Method


The flow past a circular cylinder has been simulated using vortex
methods. Each iteration of the simulation is split into two parts
(operator splitting). That is, each time step is split into separate
convection and diffusion steps. The convection step is carried out by
imposing the boundary conditions using the vortex panel method and then
integrating using the Range Kutta second order integrator. The diffusion
step is carried out using the Random Vortex Method (RVM). The simulation
tries to satisfy the no-slip conditions by placing vortex blobs of the
appropriate strength over each panel


The following algorithm has been used for each iteration in the
simulation:
1) Solve for the panel strengths to satisfy the normal velocity
constraints.
2) Calculate the tangential velocity at the control points.
If the tangential velocity is V along the tangent, place vortex blobs
of strength V.L at a distance delta over the panel (where L
is the length of the panel and delta is the core radius of the
vortex blob). The core radius here is taken as L/pi. The
vortex blobs used are chorin blobs and have a kernel function
K(r) = r/delta. The vortex blobs have additionally been split
into multiple vortices, each with a maximum vortex strength. 
3) Convect the vortex blobs (excluding the ones placed in step 2).
4) Diffuse the vortex blobs using random walks.
To ensure the no penetration condition is satisfied, particles are
reflected back if they are found to intersect any of the panels.

The following system was simulated using parameters
maximum vortex strength = 0.1, timestep = 0.1, Re = 1000 and with 75 panels
each having linear vorticity distributions.

![Alt Text](https://im.ezgif.com/tmp/ezgif-1-8e7bc7cd6069.gif)
