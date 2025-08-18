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

/*
#ifndef COMMONCONSTANTS_H
#include "commonConstants.h"
#endif
#ifndef CRIT3DDATE_H
#include "crit3dDate.h"
#endif
*/
#ifndef GIS_H
#include "gis.h"
#endif

#define CONR  0.0001244876401867718 // equivalent to std::log(2.0)/5568.0;
#define EXP_DECAY(FACTOR, EXT_COEF,RAGE) ((FACTOR) * std::exp(-(EXT_COEF) * (RAGE)))
#define LOG_RAGE(PLANT_MATERIAL, PLAT_MATERIAL_ACT) (std::log((PLANT_MATERIAL)/(PLAT_MATERIAL_ACT)) ) / CONR

class Crit3DRothCMeteoVariable {

public:
    Crit3DRothCMeteoVariable();
    void initialize();

    void setTemperature (double myTemperature);
    double getTemperature();
    void setPrecipitation(double myPrecipitation);
    void cumulatePrec(double myPrec);
    double getPrecipitation();
    void setBIC(double myBIC);
    void cumulateBIC(double myBIC);
    double getBIC();
    void setAvgBIC(double myAvgBIC);
    double getAvgBIC();
    void setWaterLoss(double myWaterLoss);
    void cumulateWaterLoss(double myWaterLoss);
    double getWaterLoss();

private:
    double temp;
    double prec;
    double BIC;
    double avgBIC;
    double waterLoss; //hourly water loss is temporarily stored here, then cumulated BIC is calculated
};

class Crit3DRothCplusplusMaps
{
private:
    //
    gis::Crit3DRasterGrid* _depthMap; //[?]
    gis::Crit3DRasterGrid* _clayMap; // [-]


public:
    gis::Crit3DRasterGrid* decomposablePlantMaterial; //[tC/ha]
    gis::Crit3DRasterGrid* resistantPlantMaterial; //[tC/ha]
    gis::Crit3DRasterGrid* microbialBiomass; //[tC/ha]
    gis::Crit3DRasterGrid* humifiedOrganicMatter; //[tC/ha]
    gis::Crit3DRasterGrid* inertOrganicMatter; //[tC/ha]
    gis::Crit3DRasterGrid* soilOrganicMatter; //[tC/ha]

    gis::Crit3DRasterGrid* avgYearlyTemp; //[°C]
    gis::Crit3DRasterGrid* avgBIC; //[mm?]
    bool isInitialized;


    gis::Crit3DRasterGrid* getDPM() { return decomposablePlantMaterial; };
    gis::Crit3DRasterGrid* getRPM() { return resistantPlantMaterial; };
    gis::Crit3DRasterGrid* getBIO() { return microbialBiomass; };
    gis::Crit3DRasterGrid* getHUM() { return humifiedOrganicMatter; };
    gis::Crit3DRasterGrid* getSOC() { return soilOrganicMatter; };

    void setDPMRowCol(double myDPM, int row, int col) { decomposablePlantMaterial->value[row][col] = myDPM; };
    void setRPMRowCol(double myRPM, int row, int col) { resistantPlantMaterial->value[row][col] = myRPM; };
    void setBIORowCol(double myBIO, int row, int col) { microbialBiomass->value[row][col] = myBIO; };
    void setHUMRowCol(double myHUM, int row, int col) { humifiedOrganicMatter->value[row][col] = myHUM; };
    void setIOMRowCol(double myIOM, int row, int col) { inertOrganicMatter->value[row][col] = myIOM; };
    void setSOCRowCol(double mySOC, int row, int col) { soilOrganicMatter->value[row][col] = mySOC; };

    Crit3DRothCplusplusMaps() {};
    //~Crit3DRothCplusplusMaps();

    void initialize(const gis::Crit3DRasterGrid& DEM);
    void clear();

    void setClay(double myClay, int row, int col);
    double getClay(int row, int col);
    void setDepth(double myDepth, int row, int col);
    double getDepth(int row, int col);

    double getAvgBIC(int row, int col);
};

struct Crit3DRothCRadioCarbon {
public:
    double decomposablePlantMatter_age;
    double resistantPlantMatter_age;
    double microbialBiomass_age;
    double humifiedOrganicMatter_age;
    double IOM_age;
    double modernC;

    bool isActive = false;

};

class Crit3DRothCplusplus{

public:

    Crit3DRothCplusplus();
    //~Crit3DRothCplusplus();

    void initialize();
    bool computeRothCPoint();
    int main();
    bool loadAvgBIC(std::string errorStr);

    double getInputC();
    void setInputC(double myInputC);

    void setClay(double myClay) {clay = myClay;};
    double getClay() {return clay;};

    void setDepth(double myDepth) {depth=myDepth;};
    double getDepth() {return depth;};

    void setSWC(double mySWC) {SWC = mySWC;};
    double getSWC() {return SWC; };

    void setPlantCover(double myPC) {plantCover = myPC; };
    double getPlantCover() { return plantCover; };

    double getDPM() {return decomposablePlantMatter;};
    double getRPM() {return resistantPlantMatter;};
    double getBIO() {return microbialBiomass;};
    double getHUM() {return humifiedOrganicMatter;};
    double getIOM() {return inorganicMatter;};
    double getSOC() {return soilOrganicCarbon;};

    void setDPM(double myDPM) {decomposablePlantMatter = myDPM;};
    void setRPM(double myRPM) {resistantPlantMatter = myRPM;};
    void setBIO(double myBIO) {microbialBiomass = myBIO;};
    void setHUM(double myHUM) {humifiedOrganicMatter = myHUM;};
    void setIOM(double myIOM) {inorganicMatter = myIOM;};
    void setSOC(double mySOC) {soilOrganicCarbon = mySOC;};

    void resetInputVariables();
    void setStateVariables(int row, int col);
    void getStateVariables(int row, int col);
    bool checkCell();

    void scrivi_csv(const std::string& nome_file, const std::vector<std::vector<double>>& dati) ;

    Crit3DRothCMeteoVariable meteoVariable;
    Crit3DRothCRadioCarbon radioCarbon;
    Crit3DRothCplusplusMaps map;

    bool isInitializing;

    std::string BICMapFileName;


private:
    double decomposablePlantMatter; //[t C /ha]
    double resistantPlantMatter; //[t C /ha]
    double microbialBiomass; //[t C /ha]
    double humifiedOrganicMatter; //[t C /ha]
    double inorganicMatter; //[t C /ha]
    double soilOrganicCarbon; //[t C /ha]
    double inputC; //[t C /ha]
    double inputFYM; //[t C /ha]
    double plantCover; // formerly bool [-]

    double decomposablePMResistantPMRatio; //[-]
    double totalRage;

    bool isUpdate;


    double SWC;
    double clay;
    double depth;

    double RMF_plantCover(bool plantCover);
    double RMF_plantCover(double plantCover);
    double RMF_Moist(double RAIN, double PEVAP, bool PC);
    double RMF_Moist(double monthlyBIC, bool PC);
    double RMF_Moist_Simplified(double monthlyBIC, double avgBIC);
    double RMF_Tmp(double TEMP);
    void decomp(int timeFact,
                double &modifyingRate);
    void RothC(int timeFact, double &PC);


};





#endif // ROTHCPLUSPLUS_H
