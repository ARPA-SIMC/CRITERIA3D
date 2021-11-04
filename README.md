# CRITERIA-3D
CRITERIA-3D is a three-dimensional water balance for small catchments.

It includes a numerical solution for three-dimensional water and heat flow in the soil, coupled surface and subsurface flow, meteorological data interpolation, radiation budget, crop development and crop water uptake. It needs hourly meteo data as input (air temperature, precipitation, solar irradiance, air relative humidity, wind speed). 
See [CRITERIA3D paper](https://github.com/ARPA-SIMC/CRITERIA3D/blob/master/DOC/CRITERIA3D.pdf) for more information. 

![](https://github.com/ARPA-SIMC/CRITERIA3D/blob/master/DOC/img/CRITERIA3D.png)

## How to compile CRITERIA-3D
Dependencies:
- [Qt libraries](https://www.qt.io/download-qt-installer): Qt 5.x or following is needed (download also QtCharts).

Build:
- Build first the project *MapGraphics/MapGraphics.pro* With Qt Creator
- then the project *bin/Makeall_CRITERIA3D/Makeall_CRITERIA3D.pro*

Warning: deselect the flag 'Shadow build' in 'Build settings' of the Qt Creator, for both the projects *MapGraphics.pro* and *Makeall_CRITERIA3D.pro*

## soilFluxes3D library 
agrolib/soilFluxed3D is a numerical solution of flow equations of water and heat in the soil, in a three-dimensional domain.

Surface water flow is described by the two-dimensional parabolic approximation of the St. Venant equation, using Manning’s equation of motion. Subsurface water flow is described by the three-dimensional Richards’ equation for the unsaturated zone and by three-dimensional Darcy’s law for the saturated zone, using an integrated finite difference formulation. The available boundary conditions are surface runoff, culvert runoff, free drainage (lateral or deep) and prescribed total potential.

Water fluxes equations may be coupled with the heat flux equations, which include diffusive, latent and advective terms. Atmospheric data (net irradiance, air temperature and relative humidity, wind speed) could be used as top boundary conditions.

![](https://github.com/ARPA-SIMC/CRITERIA3D/blob/master/DOC/img/ravone.png)

## Authors
- Fausto Tomei      
- Gabriele Antolini 
- Antonio Volta
- Laura Costantini
- Alberto Pistocchi  
- Marco Bittelli  

## Contacts
- ftomei@arpae.it   (CRITERIA3D)
- gantolini@arpae.it  (VINE3D)
- avolta@arpae.it  (Grapevine library)

## License
CRITERIA-3D has been developed under contract issued by [ARPAE Hydro-Meteo-Climate Service](https://github.com/ARPA-SIMC), Emilia-Romagna, Italy.
The executables (*CRITERIA3D, VINE3D*) are released under the terms of the GNU GPL license, libreries (*agrolib*) are released under the the terms of GNU LGPL license.
