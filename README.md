[![rocky9](https://simc.arpae.it/moncic-ci/CRITERIA3D/rocky9.png)](https://simc.arpae.it/moncic-ci/CRITERIA3D/)
[![rocky10](https://simc.arpae.it/moncic-ci/CRITERIA3D/rocky10.png)](https://simc.arpae.it/moncic-ci/CRITERIA3D/)
[![fedora42](https://simc.arpae.it/moncic-ci/CRITERIA3D/fedora42.png)](https://simc.arpae.it/moncic-ci/CRITERIA3D/)
[![fedora44](https://simc.arpae.it/moncic-ci/CRITERIA3D/fedora44.png)](https://simc.arpae.it/moncic-ci/CRITERIA3D/)

# CRITERIA-3D
CRITERIA-3D is an open-source agro-hydrological model for simulating water flow, crop water use, and slope stability in small to medium-sized catchments.

Its fully three-dimensional numerical solution of soil water flow is parallelized using OpenMP to efficiently exploit multi-core processors. The model also includes meteorological interpolation, radiation modelling, crop development, root water uptake, snow processes, and slope stability analysis.

The model requires hourly meteorological data as input, including:
- air temperature
- precipitation
- solar irradiance
- relative humidity
- wind speed

See [latest release](https://github.com/ARPA-SIMC/CRITERIA3D/releases) to download precompiled binaries.

![](https://github.com/ARPA-SIMC/CRITERIA3D/blob/master/DOC/img/ravone.png)
_Case study of the Ravone creek catchment (Bologna, Italy)_

## Key Features
- Three-dimensional water flow
- Parallel numerical solver based on OpenMP
- Fully coupled surface–subsurface flow simulation
- Hourly meteorological data interpolation
- Surface radiation budget
- Crop canopy development and root water uptake
- Snow accumulation and melt
- Slope stability analysis

## Applications
- Watershed hydrology and water balance studies
- Agricultural water management
- Landslide susceptibility and slope stability assessment
- Climate change impact studies
- Research on coupled surface–subsurface hydrological processes

![](https://github.com/ARPA-SIMC/CRITERIA3D/blob/master/DOC/img/CRITERIA3D.png)
_Screenshot of the CRITERIA-3D interface_

## Repository structure

CRITERIA3D/  
├─ agrolib/        Core numerical libraries  
├─ DATA/           Templates and sample projects  
├─ DOC/            Documentation  
├─ MapGraphics/    GUI library  
├─ bin/            Applications  
├─ ..

## Requirements

| Component | Version |
|-----------|---------|
| C++ | C++17 |
| Qt | 5.15 or later |
| Build system | qmake |
| Platforms | Windows, Linux |

# Step-by-Step Compilation Guide

### 1️⃣ Install the Required Software

The project uses qmake and can be built with Qt Creator on Windows and Linux.  
Make sure the following software is installed:

- A C++ compiler
- Qt 5.x or later
- Qt Charts module (included with Qt)
- Qt5 Compatibility module (required for Qt 6.x or later)
- Qt Creator (recommended IDE for building Qt projects)

### 2️⃣ Build the Projects in Qt Creator

1. Open the **MapGraphics** project:
   ```
   MapGraphics/MapGraphics.pro
   ```
2. Build the project using **Build → Build Project**.

3. Open the main project:
   ```
   bin/Makeall_CRITERIA3D/Makeall_CRITERIA3D.pro
   ```
4. Build it in the same way.

> **Important**
>
> - Always build **MapGraphics** before building the main project.
> - For both projects, open **Projects → Build Settings** in Qt Creator and disable **Shadow build**. This prevents common compilation errors.


### 3️⃣ Install and Run CRITERIA-3D

After successfully compiling the project:

1. Create a directory named `CRITERIA3D` containing the following subdirectories:
   ```
   CRITERIA3D/
   ├── bin/
   ├── DATA/
   └── DOC/
   ```

2. Copy the compiled executable from the build directory to `CRITERIA3D/bin`.

3. **Windows only:** open the `bin` directory and run:
   ```bash
   windeployqt CRITERIA3D.exe
   ```

4. Copy the `DATA/TEMPLATE` and `DATA/SETTINGS` directories from the repository into `CRITERIA3D/DATA`.

5. Create the directory `CRITERIA3D/DATA/PROJECT` and copy one of the sample projects (for example, `DATA/PROJECT/Montue`) from the repository.

6. Copy the entire contents of the repository's `DOC` directory into `CRITERIA3D/DOC`.

7. **Linux only:** define the environment variable `CRITERIA3D_HOME` and set it to the path of `CRITERIA3D/DATA`.

8. Run `CRITERIA3D` from the `bin` directory.

### Command-Line Compilation (Fedora)

For command-line compilation and RPM package creation on Fedora, see the `fedora/SPECS/CRITERIA3D.spec` file.

# soilFluxes3D library 
The `agrolib/soilFluxes3D` library implements a fully coupled numerical solution for three-dimensional water and heat transport in soil.

Features include:
- 3D Richards equation for unsaturated flow
- 3D Darcy equation for saturated flow
- 2D Saint-Venant surface runoff
- Integrated finite difference discretization
- Coupled heat transport
- Multiple hydraulic boundary conditions
 
Surface water flow is described by the two-dimensional parabolic approximation of the St. Venant equation, using Manning’s equation of motion. Subsurface water flow is described by the three-dimensional Richards’ equation for the unsaturated zone and by three-dimensional Darcy’s law for the saturated zone, using an integrated finite difference formulation. The available boundary conditions are surface runoff, culvert runoff, free drainage (lateral or deep) and prescribed total potential.  
The water flow equations may be coupled with the heat transport equations, which include diffusive, latent and advective terms. Atmospheric variables (net irradiance, air temperature, relative humidity and wind speed) can be used as upper boundary conditions.

# Documentation & References
- Model description [(PDF)](https://github.com/ARPA-SIMC/CRITERIA3D/blob/master/DOC/CRITERIA3D.pdf)  

- The user manual is available in the `DOC` directory.

CRITERIA is operational at [Arpae](https://www.arpae.it/it/temi-ambientali/meteo/scopri-di-piu/strumenti-di-modellistica/criteria/criteria-modello-di-bilancio-idrico) Emilia-Romagna. It has been used in several international projects (Vintage, Highlander, Arcadia) and it is reported in the [International Soil Modeling Consortium](https://soil-modeling.org/resources-links/model-portal/criteria).

### How to cite
1. Bittelli, M., Tomei, F., Pistocchi, A., Flury, M., Boll, J., Brooks, E. S., & Antolini, G. (2010). Development and testing of a physically based, three-dimensional model of surface and subsurface hydrology. Advances in Water Resources, 33(1), 106-122.
2. Bittelli, M., Pistocchi, A., Tomei, F., Roggero, P. P., Orsini, R., Toderi, M., ... & Flury, M. (2011). CRITERIA-3D: a mechanistic model for surface and subsurface hydrology for small catchments. In Soil hydrology, land use and agriculture: measurement and modelling (pp. 253-265). Wallingford UK: CAB International.
3. Sannino, G., Tomei, F., Bittelli, M., Meisina, C., Bordoni, M., & Valentino, R. (2025). A three-dimensional agro-hydrological model for predictive analysis of shallow landslides: CRITERIA-3D. Engineering Geology, 352, 108073.

# Authors
- Fausto Tomei      
- Gabriele Antolini
- Laura Costantini
- Antonio Volta
- Caterina Toscano
  
## Contributors
- Alberto Pistocchi  
- Marco Bittelli
- Giada Sannino

## Contacts
- ftomei@arpae.it   (CRITERIA3D)
- gantolini@arpae.it  (VINE3D)
- avolta@arpae.it  (grapevine model)
- ctoscano@arpae.it (hydrall model)
- gsannino@arpae.it (slope stability model)

# License
CRITERIA-3D is developed by ARPAE Hydro-Meteo-Climate Service (Emilia-Romagna, Italy).  
The applications (`CRITERIA3D`, `VINE3D`) are distributed under the GNU GPL license.
The `agrolib` libraries are distributed under the GNU LGPL license.  
See the `LICENSE` and `COPYING.LESSER` files for the complete license terms.
