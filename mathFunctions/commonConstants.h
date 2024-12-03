#ifndef COMMONCONSTANTS_H
#define COMMONCONSTANTS_H

    #ifndef MINVALUE
        #define MINVALUE(a, b) (((a) < (b))? (a) : (b))
    #endif

    #ifndef MAXVALUE
        #define MAXVALUE(a, b) (((a) > (b))? (a) : (b))
    #endif

    #ifndef sgnVariable
        #define sgnVariable(a) (((a) < 0 )? -1 : 1)
    #endif

    #ifndef NODATA
        #define NODATA -9999
    #endif

    #ifndef NODATA_UNSIGNED_SHORT
        #define NODATA_UNSIGNED_SHORT 65535
    #endif

    #ifndef HOUR_SECONDS
        #define HOUR_SECONDS 3600.
    #endif

    #ifndef DAY_SECONDS
        #define DAY_SECONDS 86400.
    #endif

    // --------------- modalities ------------------
    #define MODE_GUI 0
    #define MODE_BATCH 1
    #define MODE_CONSOLE 2

    // --------------- DB types --------------------
    #define DB_SQLITE 0
    #define DB_MYSQL 1
    #define DB_POSTGRES 2

    // --------------- path ------------------------
    #define PATH_DEM "DEM/"
    #define PATH_METEOPOINT "METEOPOINT/"
    #define PATH_METEOGRID "METEOGRID/"
    #define PATH_GEO "GEO/"
    #define PATH_SOIL "SOIL/"
    #define PATH_PROJECT "PROJECT/"
    #define PATH_TEMPLATE "TEMPLATE/"
    #define PATH_SETTINGS "SETTINGS/"
    #define PATH_LOG "LOG/"
    #define PATH_OUTPUT "OUTPUT/"
    #define PATH_TD "TD/"
    #define PATH_GLOCAL "GLOCAL/"
    #define PATH_STATES "STATES/"
    #define PATH_NETCDF "NETCDF/"

    // --------------- PRAGA constants  ----------------
    #define PRAGA_OK 0
    #define PRAGA_ERROR 100
    #define PRAGA_INVALID_COMMAND 101
    #define PRAGA_MISSING_FILE 102
    #define PRAGA_ENV_ERROR 103
    #define NO_ACTIVE -8888
    #define MAXDAYS_DOWNLOAD_DAILY 180
    #define MAXDAYS_DOWNLOAD_HOURLY 10

    // --------------- soilFluxes3D ----------------
    #define NOLINK -1

    #define PROCESS_WATER 0
    #define PROCESS_HEAT 1
    #define PROCESS_SOLUTES 2

    #define UP 1
    #define DOWN 2
    #define LATERAL 3

    #define INDEX_ERROR -1111
    #define MEMORY_ERROR -2222
    #define TOPOGRAPHY_ERROR -3333
    #define BOUNDARY_ERROR -4444
    #define MISSING_DATA_ERROR -9999
    #define PARAMETER_ERROR -7777

    #define CRIT1D_OK 0
    #define CRIT3D_OK 1

    #define VANGENUCHTEN 0
    #define MODIFIEDVANGENUCHTEN 1
    #define CAMPBELL 2

    #define MEAN_GEOMETRIC 0
    #define MEAN_LOGARITHMIC 1

    // maximum soil depth for evaporation computation [m]
    #define MAX_EVAPORATION_DEPTH 0.25

    //#define BOUNDARY_SURFACE 1
    #define BOUNDARY_RUNOFF 2
    #define BOUNDARY_FREEDRAINAGE 3
    #define BOUNDARY_FREELATERALDRAINAGE 4
    #define BOUNDARY_PRESCRIBEDTOTALPOTENTIAL 5
    #define BOUNDARY_URBAN 10
    #define BOUNDARY_ROAD 11
    #define BOUNDARY_CULVERT 12

    #define BOUNDARY_HEAT_SURFACE 10
    #define BOUNDARY_SOLUTEFLUX 30
    #define BOUNDARY_NONE 99

    #define RELAXATION 1

    // --------------- heat model -----------------
    #define SAVE_HEATFLUXES_NONE 0
    #define SAVE_HEATFLUXES_TOTAL 1
    #define SAVE_HEATFLUXES_ALL 2
    #define HEATFLUX_TOTAL 0
    #define HEATFLUX_DIFFUSIVE 1
    #define HEATFLUX_LATENT_ISOTHERMAL 2
    #define HEATFLUX_LATENT_THERMAL 3
    #define HEATFLUX_ADVECTIVE 4    
    #define WATERFLUX_LIQUID_ISOTHERMAL 5
    #define WATERFLUX_LIQUID_THERMAL 6
    #define WATERFLUX_VAPOR_ISOTHERMAL 7
    #define WATERFLUX_VAPOR_THERMAL 8

    // maximum number of solutes
    #define MAX_SOLUTES   6

    #define MAX_NUMBER_APPROXIMATIONS 50
    #define MAX_NUMBER_ITERATIONS 1000


    // -------------------ASTRONOMY--------------------
    // [J s-1 m-2] solar constant
    #define SOLAR_CONSTANT  1367.
    // [m s-2] GRAVITY acceleration
    #define	GRAVITY	9.80665
    // [m s-1]
    #define LIGHT_SPEED 299792458.

    // -------------------CHEMISTRY--------------------
    // [mol-1] Avogadro number
    #define AVOGADRO 6.022E23
    // [J s]
    #define PLANCK 6.626E-34

    // [kg m-3]
    #define WATER_DENSITY 1000.

    // [kg m-3] air density, temperature 0 °C
    #define  AIR_DENSITY 1.29

    // ---------------------PHYSICS-------------------
    // [kg mol-1] molecular mass of water
    #define	MH2O	0.018
    // [kg mol-1] mass of molecular oxygen (O2)
    #define	MO2		0.032
    // [kg mol-1] mass of molecular nitrogen (N2)
    #define	MN2		0.028
    // [K] zero Celsius
    #define	ZEROCELSIUS	273.15
    // [] ratio molecular weight of water vapour/dry air
    #define RATIO_WATER_VD 0.622
    // [J K-1 mol-1] universal gas constant
    #define R_GAS 8.31447215
    // [J kg-1 K-1] specific gas constant for dry air
    #define R_DRY_AIR 287.058

    // [K m-1] constant lapse rate of moist air
    #define LAPSE_RATE_MOIST_AIR 0.0065
    // [Pa] standard atmospheric pressure at sea level
    #define P0 101300.
    // [K] temperature at reference pressure level (P0)
    #define TP0 293.16
    // [g s-2] surface tension at 25 degC
    #define GAMMA0 71.89

    // [W m-1 K-1] thermal conductivity of water
    #define KH_H2O 0.57
    // [W m-1 K-1] average thermal conductivity of soil minerals (no quartz)
    #define KH_mineral 2.5

    // [W m-2 K-4] Stefan-Boltzmann constant
    #define STEFAN_BOLTZMANN 5.670373E-8
    // [-] Von Kármán constant
    #define VON_KARMAN_CONST 0.41
    // [J kg-1 K-1] specific heat at constant pressure
    #define CP 1013.

    // [g cm3-1]
    #define QUARTZ_DENSITY 2.648

    // [J m-3 K-1] volumetric specific heat
    #define HEAT_CAPACITY_WATER 4182000.
    #define HEAT_CAPACITY_AIR  1290.
    #define HEAT_CAPACITY_SNOW 2100000.
    #define HEAT_CAPACITY_MINERAL 231000

    // [J kg-1 K-1] specific heat
    #define HEAT_CAPACITY_WATER_VAPOR 1996.

    // [J mol-1 K-1] molar specific heat of air at constant pressure
    #define HEAT_CAPACITY_AIR_MOLAR 29.31

    // [m2 s-1] vapor diffusivity at standard conditions
    #define	 VAPOR_DIFFUSIVITY0 0.0000212

    // [Pa] default atmospheric pressure at sea level
    #define SEA_LEVEL_PRESSURE 101325.

    #define ALBEDO_WATER 0.05
    #define ALBEDO_SOIL 0.15
    #define ALBEDO_CROP 0.25
    #define ALBEDO_CROP_REFERENCE 0.23

    #define TRANSMISSIVITY_SAMANI_COEFF_DEFAULT  0.17
    #define WINKLERTHRESHOLD 10

    // --------------------MATH---------------------
    #ifndef PI
        #define PI 3.141592653589793238462643383
    #endif
    #ifndef EPSILON
        #define EPSILON 0.0000001
    #endif
    #define EULER 2.718281828459
    #define DEG_TO_RAD 0.0174532925
    #define RAD_TO_DEG 57.2957795
    #define SQRT_2 1.41421356237
    #define GOLDEN_SECTION 1.6180339887498948482

    #define MINIMUM_PERCENTILE_DATA 3


#endif // COMMONCONSTANTS_H
