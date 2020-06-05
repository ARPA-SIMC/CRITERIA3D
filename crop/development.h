#ifndef LEAFDEVELOPMENT_H
#define LEAFDEVELOPMENT_H

    #define MIN_EMERGENCE_DAYS 7
    class Crit3DCrop;

    namespace leafDevelopment {

        double getNewLai(double fractionTranspirableSoilWater, double lai, double a, double b,double laiMIN,double laiMAX,double growthDD,double emergenceDD,double currentDD,double thermalUnits,bool *isSenescence,double* actualLaiMax);
        double getLaiStressCoefficient(double fractionTranspirableSoilWaterAverage);
        double getDDfromLAIGrowth(double lai, double a, double b,double laiMIN,double laiMAX);
        double getTheoreticalLAIGrowth(double DD, double a, double b,double laiMIN,double laiMAX);
        double getLAISenescence(double LaiMin, double LAIStartSenescence, int daysFromStartSenescence);
        double getLAICriteria(Crit3DCrop* myCrop, double myDegreeDays);
    }


#endif // LEAFDEVELOPMENT_H
