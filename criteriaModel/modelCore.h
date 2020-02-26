#ifndef MODELCORE_H
#define MODELCORE_H

    class Crit3DDate;
    class CriteriaModel;
    class CriteriaUnit;
    class QString;

    bool runModel(CriteriaModel* myCase, CriteriaUnit *myUnit, QString *myError);
    bool computeModel(CriteriaModel* myCase, const Crit3DDate& firstDate, const Crit3DDate& lastDate, QString *myError);

    bool updateCrop(CriteriaModel* myCase, Crit3DDate myDate,
                    float tmin, float tmax, double waterTableDepth, QString *myError);

    double getCropReadilyAvailableWater(CriteriaModel* myCase);

    double getSoilWaterDeficit(CriteriaModel* myCase);


#endif // MODELCORE_H
