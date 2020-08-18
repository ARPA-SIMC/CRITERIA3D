### updates
CRITERIA-1D distribution has been moved to https://github.com/ARPA-SIMC/CRITERIA1D

WG1D and WG2D have been moved to https://github.com/ARPA-SIMC/WeatherGenerator

# CRITERIA-3D
CRITERIA-3D is a three-dimensional water balance for small catchments.

It includes a numerical solution for three-dimensional water and heat flow in the soil, coupled surface and subsurface flow, meteorological data interpolation, radiation budget, crop development and crop water uptake. It needs hourly meteo data as input. 
See [CRITERIA3D paper](https://github.com/ARPA-SIMC/CRITERIA3D/blob/master/DOC/CRITERIA3D.pdf) for more information. 

![](https://github.com/ARPA-SIMC/CRITERIA3D/blob/master/DOC/img/CRITERIA3D.png)

## How to compile CRITERIA-3D
Dependencies:
- [Qt libraries](https://www.qt.io/download-qt-installer): Qt 5.x or following is needed (download also QtCharts).
- [MapGraphics library](https://github.com/raptorswing/MapGraphics) (modified version): build with Qt Creator the project *mapGraphics/MapGraphics.pro*

Build:
- build with Qt Creator the project *bin/Makeall_CRITERIA3D/Makeall_CRITERIA3D.pro*

Qt Creator warning: deselect 'Shadow build' in 'Build settings' for the projects *MapGraphics.pro* and *Makeall_CRITERIA3D.pro*

## soilFluxes3D library 
agrolib/soilFluxed3D is a numerical solution for flow equations of water and heat in the soil, in a three-dimensional domain.
Surface water flow is described by the two-dimensional parabolic approximation of the St. Venant equation, using Manning’s equation of motion; subsurface water flow is described by the three-dimensional Richards’ equation for the unsaturated zone and by three-dimensional Darcy’s law for the saturated zone, using an integrated finite difference formulation.

Water fluxes equations may be coupled with the heat flux equations, which include diffusive, latent and advective terms. Atmospheric data (net irradiance, air temperature and relative humidity, wind speed) could be used as top boundary conditions.


## License
CRITERIA-3D has been developed under contract issued by [ARPAE Hydro-Meteo-Climate Service](https://github.com/ARPA-SIMC), Emilia-Romagna, Italy.

The executables (*CRITERIA3D, VINE3D*) are released under the GNU GPL license, libreries (*agrolib*) are released under the GNU LGPL license.
