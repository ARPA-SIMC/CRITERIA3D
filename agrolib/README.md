# agrolib
Agrolib are a set of libraries for agrometeorological modeling and climate analysis. 
They include a numerical solution for three-dimensional water and heat flow in soil, 
water balance 1D, meteorological data interpolation, daily data weather generator (1D and 2D), radiation budget, 
snow accumulation and melt, phenology, plant development, root development, rainfall interception and plant water uptake.

## soilFluxes3D library
SoilFluxes3D is a numerical solution for flow equations of water and heat in the soil, in a three-dimensional domain.
Surface water flow is described by the two-dimensional parabolic approximation of the St. Venant equation, using Manning’s equation of motion; subsurface water flow is described by the three-dimensional Richards’ equation for the unsaturated zone and by three-dimensional Darcy’s law for the saturated zone, using an integrated finite difference formulation.

Water fluxes equations may be coupled with the heat flux equations, which include diffusive, latent and advective terms. Atmospheric data (net irradiance, air temperature and relative humidity, wind speed) could be used as top boundary conditions. See [CRITERIA3D](http://www.sciencedirect.com/science/article/pii/S0309170809001754) for more information.

## criteriaModel library
Algorithms for 1D water balance ([CRITERIA1D](https://github.com/ARPA-SIMC/CRITERIA1D)): soil water infiltration, drainage and capillary rise, crop water demand, evaporation and crop transpiration. 

## crop library
Algorithms for crop development, leaf area index, root growth and distribution, based on daily temperature.

## soil library
Modified Van Genuchten-Mualem model for soil water retention curve and water conductivity, USDA soil texture classification.

## solarRadiation library
Algorithms for potential/actual solar radiation computation.

## interpolation library
Algorithms for the spatialization of meteorological data ([PRAGA](https://github.com/ARPA-SIMC/PRAGA)).

## License
agrolib has been developed under contract issued by [ARPAE Hydro-Meteo-Climate Service](https://github.com/ARPA-SIMC), Emilia-Romagna, Italy.  
agrolib is released under the GNU LGPL license.

## Authors
- Fausto Tomei <ftomei@arpae.it>
- Gabriele Antolini	 <gantolini@arpae.it>
- Antonio Volta		<avolta@arpae.it>
- Caterina Topscano <ctoscano@arpae.it>
- Laura Costantini  <laura.costantini0@gmail.com>

### Contributions
- Vittorio Marletto <vmarletto@arpae.it>
- Marco Bittelli <marco.bittelli@unibo.it>
- Alberto Pistocchi	 <alberto.pistocchi@jrc.ec.europa.eu>
- Tomaso Tonelli <ttonelli@arpae.it>
- Margot Van Soetendaal <margot@farnet.eu>
- Franco Zinoni <fzinoni@arpae.it>
- Fabrizio Nerozzi <fnerozzi@arpae.it>
