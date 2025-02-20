# stellar-winds-ECMI
This repository contains the code used to obtain the mass-loss rate constraints in Bloot et al. (2025).

ffa_utils_parker.py: Defines the functions to localise the emission region, calculate the density profile of a Parker wind, and determine the optical depth.
test-star.py: Calculates the mass-loss rate upper limits for a star of a given mass and radius as a function of the wind temperature and the magnetic field strength, given a detection of ECMI.
sample-lotss.py: Calculates the mass-loss rate upper limits for a sample of M dwarfs detected with LOFAR (Callingham et al. 2020).
GJ625_b.py: Calculates mass-loss rate upper limits for GJ 625 as a function of stellar magnetic field strength and determines the corresponding dynamic and magnetic pressure on GJ625 b.
ffe_GJ3861.py: Calculates the free-free emission from a stellar wind for a given mass-loss rate. GJ 3861 is used as an example.
