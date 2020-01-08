#ifndef LEAFDEVELOPMENT_H
#define LEAFDEVELOPMENT_H

    class Crit3DCrop;

    namespace leafDevelopment {

        double getNewLai(double fractionTranspirableSoilWater, double lai, double a, double b,double laiMIN,double laiMAX,double growthDD,double emergenceDD,double currentDD,double thermalUnits,bool *isSenescence,double* actualLaiMax);
        double getLaiStressCoefficient(double fractionTranspirableSoilWaterAverage);
        double getDDfromLAIGrowth(double lai, double a, double b,double laiMIN,double laiMAX);
        double getTheoreticalLAIGrowth(double DD, double a, double b,double laiMIN,double laiMAX);
        double getLAICriteria(Crit3DCrop* myCrop, double myDegreeDays);
        double getLAISenescence(double LaiMin, double LAIStartSenescence, int daysFromStartSenescence);
    }

#endif // LEAFDEVELOPMENT_H
