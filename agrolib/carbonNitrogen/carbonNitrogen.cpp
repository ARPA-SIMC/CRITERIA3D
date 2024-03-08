#include "carbonNitrogen.h"
#include "commonConstants.h"


Crit3DCarbonNitrogenLayer::Crit3DCarbonNitrogenLayer()
{
    // TODO

    // NITROGEN
    // contents
    N_NO3 = NODATA;
    // ...

    // CARBON
    // contents
    // ...

    // PARAMETERS
    // temperatureCorrectionFactor
    // ...
}


Crit3DFertilizerProperties::Crit3DFertilizerProperties()
{
    // TODO set default values for a common fertilizer
    // TODO read from db for a specific fertilizer
}


Crit3DCarbonNitrogenSettings::Crit3DCarbonNitrogenSettings()
{
    // default values
    rate_C_humusMin = 0.000005;
    rate_C_litterMin = 0.01;
    rate_N_NH4_volatilization = 0.4;
    rate_N_denitrification = 0.001;
    max_afp_denitr = 0.1;
    constant_sat_denitr = 10;
    rate_urea_hydr = 0.43;
    rate_N_nitrification = 0.0018;
    limRatio_nitr = 8;
    FE = 0.5;
    FH = 0.2;
    Q10 = 2.3;
    baseTemperature = 20;
    Kd_NH4 = 4;
    CN_RATIO_NOTHARVESTED = 30;
    LITTERINI_C_DEFAULT = 1200;         // [kg ha-1] initial litter carbon
    LITTERINI_N_DEFAULT = 40;           // [kg ha-1] initial litter nitrogen
    LITTERINI_DEPTH_DEFAULT = 30;       // [cm] initial litter depth
    ratioBiomassCN = 7;
    ratioHumusCN = 7;
    ratioLitterCN = 7;
}


double convertToGramsPerM3(double myQuantity, soil::Crit1DLayer &soilLayer)
{
    // convert [g m-2] -> [g m-3] = [mg dm-3]
    return myQuantity / soilLayer.thickness;
}

double convertToGramsPerLiter(double myQuantity, soil::Crit1DLayer &soilLayer)
{
    // convert [g m-2] -> [g m-3] -> [g l-1]
    return (convertToGramsPerM3(myQuantity, soilLayer) / 1000);
}

double convertToGramsPerKg(double myQuantity, soil::Crit1DLayer &soilLayer)
{
    // convert [g m-2] -> [g m-3] -> [g kg-1]
    return (convertToGramsPerM3(myQuantity, soilLayer) / 1000) / soilLayer.horizonPtr->bulkDensity;
}


double CNRatio(double C, double N, int flagOrganicMatter)
{
    // 2004.02.20.VM
    // computes the C/N ratio
    if (flagOrganicMatter != 1)
        return 20.;
    if (N > 0.000001)
        return MAXVALUE(0.001, C/N);
    else
        return 100.;
}



