# soilFluxes3D library
SoilFluxes3D is a numerical solution for flow equations of water and heat in the soil, in a three-dimensional domain.
Surface water flow is described by the two-dimensional parabolic approximation of the St. Venant equation, 
using Manning’s equation of motion; subsurface water flow is described by the three-dimensional Richards’ 
equation for the unsaturated zone and by three-dimensional Darcy’s law for the saturated zone, using an integrated finite difference formulation.

Water fluxes equations may be coupled with the heat flux equations, which include diffusive, latent and advective terms. 
Atmospheric data (net irradiance, air temperature and relative humidity, wind speed) could be used as top boundary conditions. 

See [CRITERIA-3D paper](http://www.sciencedirect.com/science/article/pii/S0309170809001754) for more information.
