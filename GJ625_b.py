import numpy as np
import math
import matplotlib.pyplot as plt

#  Script to calculate the dynamic pressure and the magnetic pressure on GJ 625 b,
#  given a detection of ECMI at 120 MHz.
#  Authors: S. Bloot, R. D. Kavanagh, H. K. Vedantham
#  ASTRON, Nov 2024
#

from ffa_utils_parker import *

#Relevant constants in Gaussian units
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

A=0.09 # Helium fraction

name='GJ_625'
Mstar=0.317
Rstar=0.332
Lx=0.04*1e28
Fx=Lx/(4*np.pi*(Rstar*C_Rsun)**2)

a_planet=0.078*C_au
v_planet=5.83e6

grid_rho=np.logspace(6, 12, 5000)
grid_temp=coronal_temp(Fx)/1.36

frequency=120e6

B0s=np.logspace(np.log10(10),np.log10(4e3), 100)

Mdots_iso=np.zeros(len(grid_rho))
taus_iso=np.zeros([len(grid_rho), len(B0s)])

# Calculate the velocity profile of the wind
iso_rgrid, iso_vgrid=velocity_profile_isothermal(Mstar*C_Msun, Rstar*C_Rsun, 100*C_Rsun, 50000, grid_temp*1e6, A=A)
second_harmonic=True

# Calculate the density structure and optical depths for a range of base densities
for i in range(len(grid_rho)):
   iso_density, Mdots_iso[i]=density_profile(iso_rgrid, iso_vgrid, grid_rho[i], A=A)
   for k in range(len(B0s)):
      taus_iso[i,k]=optical_depth(iso_density, iso_rgrid, iso_vgrid, grid_temp*1e6,Rstar*C_Rsun, frequency, A=A, second_harmonic=second_harmonic, B0= B0s[k])


all_maxes_iso=[]
Pdyn=[]
Bsa=[]
indexes=np.zeros(len(B0s))

# For each value of the magnetic field, calculate the mass-loss rate limit
# and the corresponding dynamic and magnetic pressure at the location of the planet.
for k in range(len(B0s)):
   indexes[k]=np.argwhere(Mdots_iso[(taus_iso[:,k]<1)])[-1]
   Bgrid=B0s[k]*(iso_rgrid/(Rstar*C_Rsun))**-3.0
   B_loc=Bgrid[iso_rgrid>a_planet][0]
   iso_density, temp=density_profile(iso_rgrid, iso_vgrid, grid_rho[int(indexes[k])], A=0.09)
   rho=iso_density*(mean_molecular_weight(A)*C_mp)*(1+2*A+A+1)

   # Find the wind density and velocity at the location of the planet
   rho_loc=rho[iso_rgrid>a_planet][0]
   v_loc=iso_vgrid[iso_rgrid>a_planet][0]
   pdyn=rho_loc*((v_loc)**2+(v_planet)**2)
   Pdyn.append(pdyn)
   Bsa.append(B_loc**2/(8*np.pi))

   all_maxes_iso.append(np.nanmax(Mdots_iso[(taus_iso[:,k]<1)]))


np.save('GJ625_b_Mdot', np.array([all_maxes_iso]))
np.save('GJ625_b_Pdyn', np.array([Pdyn]))
np.save('GJ625_b_BSA', np.array([Bsa]))

