import numpy as np
import math
import matplotlib.pyplot as plt

#  Script to calculate mass-loss rate upper limits for a sample of M dwarfs
#  Authors: S. Bloot, R. D. Kavanagh, H. K. Vedantham
#  ASTRON, Nov 2024


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


# Properties of the sample
names=['DO_Cep','WX_UMa', 'AD_Leo', 'GJ_625','GJ1151', 'GJ450','LP_169-22', 'CW_UMa', 'HAT_182-00605', 'LP_212-62', 'DG_CVn', 'GJ_3861', 'CR_Dra', 'GJ_3729', 'G_240-45', '2MASS_J09481615+5114518', 'LP_259-39', '2MASS_J10534129+5253040', '2MASS_J14333139+3417472']
# Radius in solar radii
Rstar=[0.332,0.121,0.431, 0.332, 0.190, 0.474, 0.138, 0.322, 0.422, 0.183, 0.567, 0.430, 0.827,0.481,0.162, 0.161, 0.202, 0.379, 0.136]
# Mass in solar masses
Mstar=[0.316,0.095, 0.420, 0.317, 0.167, 0.460, 0.111, 0.306, 0.442, 0.161, 0.560, 0.419, 0.823, 0.472, 0.125, 0.122, 0.173, 0.408, 0.101]
# X-ray luminosity in erg/s
Lx=np.array([0.23, 0.36, 3.20, 0.04, 0.02, 0.66, 0.03, 5.37, 3.40, 0.38, 10.72, 3.36, 36.65, 7.54, 0.02, 0.28, 18.70, 28.01, 0.83])*1e28
# Surface X-ray flux in erg/s/cm^2
Fx=np.array(Lx)/(4*np.pi*(np.array(Rstar)*C_Rsun)**2)
# Wind temperature estimated from the surface X-ray flux, in MK
grid_temp=coronal_temp(Fx)/1.36

# Estimated magnetic field strengths of the sample
B0_low=magnetic_field_average(Mstar)
B0_high=magnetic_field_conservative(Mstar)

# Measured magnetic fields
B0_low[1]=4.3e3
B0_low[2]=1.0e3
B0_low[4]=150

B0_high[1]=4.3e3
B0_high[2]=1.0e3
B0_high[4]=150

# Define the range of base densities to test
grid_rho=np.logspace(6, 18, 10000)


frequency=120e6
all_maxes_iso_low=[]
all_maxes_iso_high=[]

for l in range(len(names)):
     print(names[l])
     Mdots_iso=np.zeros(len(grid_rho))
     taus_iso_low=np.zeros(len(grid_rho))
     taus_iso_high=np.zeros(len(grid_rho))

     # Calculate the velocity profile of the wind
     iso_rgrid, iso_vgrid=velocity_profile_isothermal(Mstar[l]*C_Msun, Rstar[l]*C_Rsun, 100*C_Rsun, 50000, grid_temp[l]*1e6, A=0.09)
     if names[l]=='WX_UMa':
          second_harmonic=False
     else:
          second_harmonic=True

     # For each base density, calculate the density profile of the wind
     # and the optical depth along the path of the ECMI emission
     for i in range(len(grid_rho)):
          iso_density, Mdots_iso[i]=density_profile(iso_rgrid, iso_vgrid, grid_rho[i], A=A)
          taus_iso_low[i]=optical_depth(iso_density, iso_rgrid, iso_vgrid, grid_temp[l]*1e6,Rstar[l]*C_Rsun, frequency, A=A, second_harmonic=second_harmonic, B0= B0_low[l])
          taus_iso_high[i]=optical_depth(iso_density, iso_rgrid, iso_vgrid, grid_temp[l]*1e6,Rstar[l]*C_Rsun, frequency, A=A, second_harmonic=second_harmonic, B0= B0_high[l])
     # Find the highest mass loss rate for which the optical depth is less than 1
     Mdot_max_low=np.max(Mdots_iso[(taus_iso_low<1)])
     Mdot_max_high=np.max(Mdots_iso[(taus_iso_high<1)])
     all_maxes_iso_low.append(Mdot_max_low)
     all_maxes_iso_high.append(Mdot_max_high)
     


np.save('Mdot_lotss_average_B_iso', np.array([all_maxes_iso_low]))
np.save('Mdot_lotss_conservative_B_iso', np.array([all_maxes_iso_high]))

