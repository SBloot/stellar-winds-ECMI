# Compute free-free flux density for given stellar parameters and freq list
# Author: H. K. Vedantham
# ASTRON, Oct 2023
#
import numpy as np
import matplotlib.pyplot as plt

# Relevant constants in Gaussian units
C_c = 2.99792458e10
C_k = 1.380658e-16
C_h = 6.62606885e-27
C_G = 6.67428e-8
C_Msun = 1.99e33
C_Rsun = 6.96e10
C_au = 1.496e13
C_pc = 3.086e18
C_me = 9.1093897e-28
C_mp = 1.6726231e-24
year = 3.15567e7
A=0.09
from scipy.interpolate import RegularGridInterpolator as interp2
from ffa_utils_parker import *



def ffe_int(r,n,T,nu,A=0.09): 
   # Calculate the emergent optical depth and specific intensity of free-free emission
   #  r = radius vector values in cm (code assumes a regular grid)
   #  n = Density vector in cm^-3
   #  T = Temp. vector (or scalar) in K
   #  nu = Frequency (scalar) in Hz
   
   n_proton=n/(1+2*A+A+1)
   dr = np.absolute(r[1]-r[0])      # distance increment
   alpha, nu_p = alpha_ffa(n_proton,T,nu) 

   tau_tot = np.sum(alpha*dr)    # Total optical depth 
   source_func = 2*C_k*T/C_c**2*nu**2  # Source function in vacuum (Rayleigh-Jeans) 

   
   if tau_tot>=5: # Optically thick regime, save computation time by assuming blackbody
      I = source_func

   elif tau_tot<1e-2: # Optically thin regime, save time by assuming exp(-tau) = 1-tau
      opt_depth = np.array([dr*np.sum(alpha[i:]) for i in range(len(alpha))]) 
      I = source_func*dr*np.sum(alpha*(1-opt_depth)) 

   else:          # Intermediate case, need full calculation
      opt_depth = np.array([dr*np.sum(alpha[i:]) for i in range(len(alpha))]) 
      I = source_func*np.sum(alpha*np.exp(-opt_depth))*dr 
   
   return I



def scale_height(M,R,T, A=0.09): # Density scale height
   # M = mass of the star in gm
   # R = radius of the star in cm
   # T = plasma temperature
   # A = helium fraction
   return 2*C_k*T*(R)**2/(C_G*M*C_mp*mean_molecular_weight(A))



def ffe_flux_density(mstar, rstar, T, nu, Mdot, rmax_rstar, dist, A):
   # 2D ray trace to compute the free-free flux density of a Parker wind
   # mstar = Stellar mass in grams
   # rstar = stellar radius in cm
   # T = coronal temp in Kelvin
   # nu = freq in Hz
   # n0 = base density in cm^-3
   # Returns the flux density in Jy

   hp = scale_height(mstar,rstar,T) # Scale height
   dr = min(rstar/5,hp/5)     # numerical grid resolution
   rmax = rstar*rmax_rstar    # numerical grid extent
   r = np.arange(rstar,rmax,dr) 

   iso_rgrid, iso_vgrid=velocity_profile_isothermal(mstar, rstar, rmax_rstar, (rmax/dr), T, A=0.09)

   rho_base = Mdot/(4.0 * np.pi * R_grid[0]**2.0)/(iso_vgrid[0] / C_Msun * year) 
   n_base=rho_base/(mean_molecular_weight(A)*C_mp)
   n_proton, Mdot_out=density_profile(iso_rgrid, iso_vgrid, n_base, A=A)

   # Compute the Zone 1 radial size
   n_e=n_proton*(1+2*A)
   nup_list = plasma_frequency(n_e)
   if nu > nup_list[0]:
      xmin_plasma = rstar
   else:
      xmin_plasma = r[np.where(nu>nup_list)[0][0]]

   # Compute dh and h_grid: the grid for lateral (plane of sky) distance from stellar centre
   dh = iso_rgrid[1]-iso_rgrid[0]
   h_grid = np.arange(0.0,rmax,dh)

   # Initialize flux density and optical depth vectors (functions of lateral distance)
   flux_density = []
   tau = []
   
   norm = dist**2/1e23  # To convert to Jy from c.g.s with 1/D^2 scaling

   for h in h_grid:
      if h<xmin_plasma: # If the ray ends on the surface of the Zone 1 sphere
         xmin = (xmin_plasma**2-h**2)**0.5
         xmax = (rmax**2-h**2)**0.5
         xvec=np.arange(xmin,xmax,dh)
         rvec = (h**2+xvec**2)**0.5
         nvec = np.interp(rvec,r,n)

         flux_density.append(ff_int(xvec,nvec,T,nu) *2*np.pi*dh*h/norm)
      else:    # If ray misses the surface of the Zone 1 sphere
         xmax = (rmax**2-h**2)**0.5 
         xvec=np.arange(-xmax,xmax,dh)
         rvec = (h**2+xvec**2)**0.5
         nvec = np.interp(rvec,r,n)
         flux_density.append(ff_int(xvec,nvec,T,nu) *2*np.pi*dh*h/norm)

   return np.nansum(np.array(flux_density))



# Properties of the star
Lx = 3.36*1e28 # X-ray luminosity in erg/
Rstar = 0.419 # Radius in solar radii
Mstar = 0.43 # Mass in solar masses
Fx = np.array(Lx)/(4*np.pi*(np.array(Rstar)*C_Rsun)**2) # Surface X-ray flux in erg/s/cm^2
grid_temp = 0.11*np.array(Fx)**0.26/1.36 # Wind temperature in MK

Mdot_msun = 5000 # Mass-loss rate in units of the solar mass-loss rate
dist_pc = 18.47 # Distance in parsec


nu = 45e9 # Observing frequency in Hz
Mdot = Mdot_msun*2e-14*C_Msun/(365*24*3600) # Mass loss rate in g/s
rstar = Rstar*C_Rsun       # Stellar radius in cm
mstar = Mstar*C_Msun       # Stellar mass in g
T = grid_temp * 1e6          # Coronal temp in K


flux_density=ffe_flux_density(mstar=mstar,rstar=rstar,T=T,nu=nu,Mdot=Mdot,rmax_rstar=200.0,dist=C_pc*dist_pc)



