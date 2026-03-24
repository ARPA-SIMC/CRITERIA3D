[![Build Status](https://simc.arpae.it/moncic-ci/CRITERIA3D/rocky8.png)](https://simc.arpae.it/moncic-ci/CRITERIA3D/)
[![Build Status](https://simc.arpae.it/moncic-ci/CRITERIA3D/rocky9.png)](https://simc.arpae.it/moncic-ci/CRITERIA3D/)
[![Build Status](https://simc.arpae.it/moncic-ci/CRITERIA3D/fedora40.png)](https://simc.arpae.it/moncic-ci/CRITERIA3D/)
[![Build Status](https://simc.arpae.it/moncic-ci/CRITERIA3D/fedora42.png)](https://simc.arpae.it/moncic-ci/CRITERIA3D/)

# CRITERIA-3D
CRITERIA-3D is a three-dimensional water balance for small catchments.  
It includes a numerical solution for three-dimensional water and heat flow in the soil, coupled surface and subsurface flow, meteorological data interpolation, radiation budget, crop development and crop water uptake. It needs hourly meteo data as input (air temperature, precipitation, solar irradiance, air relative humidity, wind speed).  

See [CRITERIA3D](https://github.com/ARPA-SIMC/CRITERIA3D/blob/master/DOC/CRITERIA3D.pdf) for more technical information, [user guide](https://github.com/ARPA-SIMC/CRITERIA1D/blob/master/DOC/CRITERIA3D_user_manual.pdfl) for user documentation and [last release](https://github.com/ARPA-SIMC/CRITERIA3D/releases) to download precompiled binaries.

CRITERIA is operational at [Arpae Emilia-Romagna](https://www.arpae.it/it/temi-ambientali/meteo/scopri-di-piu/strumenti-di-modellistica/criteria/criteria-modello-di-bilancio-idrico). It has been used in several international projects (Vintage, Highlander, Arcadia) and it is reported in the [International Soil Modeling Consortium](https://soil-modeling.org/resources-links/model-portal/criteria).

![](https://github.com/ARPA-SIMC/CRITERIA3D/blob/master/DOC/img/CRITERIA3D.png)

## Step-by-Step Compilation Guide
#### 1️⃣ Install required software

Make sure you have:  
- **Qt 5.x or later**  
- **QtCharts module** (install with Qt)  
- **Qt5 Compatibility module** (Only for Qt 6.x)

Also, install **Qt Creator** (the IDE for building Qt projects).

#### 2️⃣ Build projects in Qt Creator  

Open and build the **MapGraphics** (GUI) project:    
`\MapGraphics\MapGraphics.pro`  
Go to Build → Build Project

Open and build the **main** project:  
`\bin\Makeall_CRITERIA3D\Makeall_CRITERIA3D.pro`  
Build it the same way

#### ⚠️ Important Tips
- Always build MapGraphics.pro first, then the main project.
- For both projects, go to **Projects → Build Settings** in Qt Creator and **uncheck “Shadow build”**. This prevents common compilation errors.

#### 3️⃣ Run CRITERIA-3D  

After successful compilation:  
Locate the executable in the build folder  
Run it directly from Qt Creator or your system file explorer.

## How to compile VINE3D
Follow the same steps of CRITERIA-3D, using the project  *bin/Makeall_CRITERIA3D/Makeall_VINE3D.pro*

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
- Giada Sannino
- Alberto Pistocchi  
- Marco Bittelli
- Caterina Toscano  

## Contacts
- ftomei@arpae.it   (CRITERIA3D)
- gantolini@arpae.it  (VINE3D)
- gsannino@arpae.it (slope stability)
- avolta@arpae.it  (grapevine)
- ctoscano@arpae.it (hydrall model)

## License
CRITERIA-3D has been developed under contract issued by [ARPAE Hydro-Meteo-Climate Service](https://github.com/ARPA-SIMC), Emilia-Romagna, Italy.  
The executables (*CRITERIA3D, VINE3D*) are released under the terms of the GNU GPL license, libreries (*agrolib*) are released under the the terms of GNU LGPL license.
