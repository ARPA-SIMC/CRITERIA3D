/*######################################################################################################################
#
#  RothC C++ version
#
#  This C++ version was translated from the Python code by Caterina Toscano, Antonio Volta and Enrico Balugani 03/03/2025
#
#  The Rothamsted Carbon Model: RothC
#  Developed by David Jenkinson and Kevin Coleman
#
#  INPUTS:
#
#  clay:  clay content of the soil (units: %)
#  depth: depth of topsoil (units: cm)
#  IOM: inert organic matter (t C /ha)
#  nsteps: number of timesteps
#
#  year:    year
#  month:   month (1-12)
#  modern:   %modern
#  TMP:      Air temperature (C)
#  Rain:     Rainfall (mm)
#  Evap:     open pan evaporation (mm)
#  C_inp:    carbon input to the soil each month (units: t C /ha)
#  FYM:      Farmyard manure input to the soil each month (units: t C /ha)
#  PC:       Plant cover (0 = no cover, 1 = covered by a crop)
#  DPM/RPM:  Ratio of DPM to RPM for carbon additions to the soil (units: none)
#
#  OUTPUTS:
#
#  All pools are carbon and not organic matter
#
#  DPM:   Decomposable Plant Material (units: t C /ha)
#  RPM:   Resistant Plant Material    (units: t C /ha)
#  Bio:   Microbial Biomass           (units: t C /ha)
#  Hum:   Humified Organic Matter     (units: t C /ha)
#  IOM:   Inert Organic Matter        (units: t C /ha)
#  SOC:   Soil Organic Matter / Total organic Matter (units: t C / ha)
#
#  DPM_Rage:   radiocarbon age of DPM
#  RPM_Rage:   radiocarbon age of RPM
#  Bio_Rage:   radiocarbon age of Bio
#  HUM_Rage:   radiocarbon age of Hum
#  Total_Rage: radiocarbon age of SOC (/ TOC)
#
#  SWC:       soil moisture deficit (mm per soil depth)
#  RM_TMP:    rate modifying fator for temperature (0.0 - ~5.0)
#  RM_Moist:  rate modifying fator for moisture (0.0 - 1.0)
#  RM_PC:     rate modifying fator for plant retainment (0.6 or 1.0)

######################################################################################################################*/

#ifndef ROTHCPLUSPLUS_H
#define ROTHCPLUSPLUS_H

#ifndef GIS_H
    #include "gis.h"
#endif

// equivalent to std::log(2.0)/5568.0
#define CONR  0.0001244876401867718


class Crit3DRothCMaps
{
public:
    Crit3DRothCMaps() {}
    ~Crit3DRothCMaps() { clear(); }

    void clear();
    void initialize(const gis::Crit3DRasterGrid& DEM);

    void setClay(double clay, int row, int col) {
        clayMap->value[row][col] = (float)clay;
    }

    double getClay(int row, int col) const {
        return clayMap->value[row][col];
    }

    double getAvgBIC(int row, int col, int month) const;
    double getAvgTemp(int row, int col, int month) const;

    std::vector<double> getAvgBICVector(int row, int col) const;
    std::vector<double> getAvgTempVector(int row, int col) const;

    gis::Crit3DRasterGrid* getDPM() { return decomposablePlantMaterial; }
    gis::Crit3DRasterGrid* getRPM() { return resistantPlantMaterial; }
    gis::Crit3DRasterGrid* getBIO() { return microbialBiomass; }
    gis::Crit3DRasterGrid* getHUM() { return humifiedOrganicMatter; }
    gis::Crit3DRasterGrid* getIOM() { return inertOrganicMatter; }
    gis::Crit3DRasterGrid* getSOC() { return soilOrganicMatter; }

    void setDPMRowCol(double myDPM, int row, int col) { decomposablePlantMaterial->value[row][col] = (float)myDPM; }
    void setRPMRowCol(double myRPM, int row, int col) { resistantPlantMaterial->value[row][col] = (float)myRPM; }
    void setBIORowCol(double myBIO, int row, int col) { microbialBiomass->value[row][col] = (float)myBIO; }
    void setHUMRowCol(double myHUM, int row, int col) { humifiedOrganicMatter->value[row][col] = (float)myHUM; }
    void setIOMRowCol(double myIOM, int row, int col) { inertOrganicMatter->value[row][col] = (float)myIOM; }
    void setSOCRowCol(double mySOC, int row, int col) { soilOrganicMatter->value[row][col] = (float)mySOC; }

    gis::Crit3DRasterGrid* decomposablePlantMaterial;   // [tC/ha]
    gis::Crit3DRasterGrid* resistantPlantMaterial;      // [tC/ha]
    gis::Crit3DRasterGrid* microbialBiomass;            // [tC/ha]
    gis::Crit3DRasterGrid* humifiedOrganicMatter;       // [tC/ha]
    gis::Crit3DRasterGrid* inertOrganicMatter;          // [tC/ha]
    gis::Crit3DRasterGrid* soilOrganicMatter;           // [tC/ha]

    std::vector<gis::Crit3DRasterGrid*> avgBICMap;      // [mm]
    std::vector<gis::Crit3DRasterGrid*> avgTempMap;     // [C°]

private:

    gis::Crit3DRasterGrid* clayMap;                     // [%]
};


struct Crit3DRothCRadioCarbon
{
public:
    double decomposablePlantMatter_age;
    double resistantPlantMatter_age;
    double microbialBiomass_age;
    double humifiedOrganicMatter_age;
    double IOM_age;
    double modernC;

    bool isActive = false;
};


struct TAnnualYield
{
    std::string name;
    double carbon;          // annual carbon biomass
};


class Crit3DRothC
{
public:
    Crit3DRothCMaps maps;
    Crit3DRothCRadioCarbon radioCarbon;

    std::string BICMapFolderName;
    std::string temperatureMapFolderName;

    std::vector<int> conversionTableVector;
    std::vector<TAnnualYield> tableYield;

    Crit3DRothC();

    void initialize();

    void setInputC(double myInputC){ _inputC = myInputC; }
    void setClay(double myClay) { _clay = myClay; }

    void setPlantCover(double myPlantCover) { _plantCover = myPlantCover;}

    void setAvgTemperature (double myTemperature) { _avgTemp = myTemperature; }
    void setFractionAW(double myFractionAW) { _fractionAW = myFractionAW;}
    void setSixMonthBIC(double myBIC) { _sixMonthBIC = myBIC; }

    double getDPM() const {return decomposablePlantMatter;}
    double getRPM() const {return resistantPlantMatter;}
    double getBIO() const {return microbialBiomass;}
    double getHUM() const {return humifiedOrganicMatter;}
    double getIOM() const {return inorganicMatter;}
    double getSOC() const {return soilOrganicCarbon;}

    void setDPM(double myDPM) {decomposablePlantMatter = myDPM;}
    void setRPM(double myRPM) {resistantPlantMatter = myRPM;}
    void setBIO(double myBIO) {microbialBiomass = myBIO; }
    void setHUM(double myHUM) {humifiedOrganicMatter = myHUM;}
    void setIOM(double myIOM) {inorganicMatter = myIOM;}
    void setSOC(double mySOC) {soilOrganicCarbon = mySOC;}

    void setStateVariables(int row, int col);
    void storeStateVariables(int row, int col);
    bool checkCellPools ();

    bool initializeFromClimate(const std::vector<double> &monthlyAvgT,
                               const std::vector<double> &monthlyBIC);

    bool RothC();


private:
    double decomposablePlantMatter;     //[t C /ha]
    double resistantPlantMatter;        //[t C /ha]
    double microbialBiomass;            //[t C /ha]
    double humifiedOrganicMatter;       //[t C /ha]
    double inorganicMatter;             //[t C /ha]
    double soilOrganicCarbon;           //[t C /ha]

    double decomposablePMResistantPMRatio;  //[-]
    double totalRage;

    double _inputC;                     //[t C /ha]
    double _inputFYM;                   //[t C /ha]

    double _plantCover;                 // [0-1]

    double _avgTemp;                    // [°C]
    double _sixMonthBIC;                // [mm] 6-month hydro climatic balance (precipitation - ET0)
    double _fractionAW;                 // [-] fraction of Available Water (0: wilting point, 1: field capacity)
    double _clay;                       // [%]

    double RMF_plantCover(bool plantCover);
    double RMF_plantCover(double plantCover);
    double RMF_Tmp(double monthlyAvgT);
    double RMF_Moist_FractionAW(double monthlyAvgFAW);
    double RMF_Moist_BIC(double sixMonthBIC);

    //double RMF_Moist(double RAIN, double PEVAP, bool PC);
    //double RMF_Moist(double monthlyBIC, bool PC);

    void decomp(int timeFact, double modifyingRate);

};


#endif // ROTHCPLUSPLUS_H
