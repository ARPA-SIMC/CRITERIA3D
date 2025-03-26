#ifndef CLIMATE_H
#define CLIMATE_H

    #ifndef METEO_H
        #include "meteo.h"
    #endif
    #ifndef DBMETEOPOINTS_H
        #include "dbMeteoPointsHandler.h"
    #endif
    #ifndef DBMETEOGRID_H
        #include "dbMeteoGrid.h"
    #endif
    #ifndef CRIT3DCLIMATE_H
        #include "crit3dClimate.h"
    #endif
    #ifndef CRIT3DELABLIST_H
        #include "crit3dElabList.h"
    #endif
    #ifndef CRIT3DANOMALYLIST_H
        #include "crit3dAnomalyList.h"
    #endif
    #ifndef CRIT3DDROUGHTLIST_H
        #include "crit3dDroughtList.h"
    #endif
    #ifndef CRIT3DPHENOLOGYLIST_H
        #include "crit3dPhenologyList.h"
    #endif


    const std::map<std::string, int> MapElabWithParam = {
      { "differenceWithThreshold", 1 },
      { "lastDayBelowThreshold", 1 },
      { "sumAbove", 1 },
      { "avgAbove", 1 },
      { "stdDevAbove", 1 },
      { "percentile", 1 },
      { "daysAbove", 1 },
      { "daysBelow", 1 },
      { "consecutiveDaysAbove", 1 },
      { "consecutiveDaysBelow", 1 },
      { "correctedDegreeDaysSum", 1 }
    };

    bool elaborationOnPoint(QString *myError, Crit3DMeteoPointsDbHandler* meteoPointsDbHandler,
                            Crit3DMeteoGridDbHandler* meteoGridDbHandler, Crit3DMeteoPoint* meteoPointTemp,
                            Crit3DClimate* clima, bool isMeteoGrid, QDate startDate, QDate endDate,
                            bool isAnomaly, Crit3DMeteoSettings *meteoSettings, bool dataAlreadyLoaded);

    bool elaborationOnPointHourly(Crit3DMeteoPointsDbHandler* meteoPointsDbHandler,
                                  Crit3DMeteoGridDbHandler* meteoGridDbHandler, Crit3DMeteoPoint* meteoPointTemp,
                                  bool isMeteoGrid, Crit3DClimate* climate, Crit3DMeteoSettings* meteoSettings, QString &myError);

    frequencyType getAggregationFrequency(meteoVariable myVar);

    bool elaborateDailyAggregatedVar(meteoVariable myVar, Crit3DMeteoPoint meteoPoint, std::vector<float> &outputValues, float* percValue, Crit3DMeteoSettings *meteoSettings);
    bool elaborateDailyAggrVarFromStartDate(meteoVariable myVar, Crit3DMeteoPoint meteoPoint, QDate first, QDate last, std::vector<float> &outputValues, float* percValue, Crit3DMeteoSettings* meteoSettings);
    bool elaborateDailyAggregatedVarFromDaily(meteoVariable myVar, Crit3DMeteoPoint meteoPoint, Crit3DMeteoSettings *meteoSettings, std::vector<float> &outputValues, float* percValue);
    bool elaborateDailyAggrVarFromDailyFromStartDate(meteoVariable myVar, Crit3DMeteoPoint meteoPoint, Crit3DMeteoSettings* meteoSettings, QDate first, QDate last,
                                              std::vector<float> &outputValues, float* percValue);
    bool elaborateDailyAggregatedVarFromHourly(meteoVariable myVar, Crit3DMeteoPoint meteoPoint, std::vector<float> &outputValues, Crit3DMeteoSettings *meteoSettings);
    bool aggregatedHourlyToDaily(meteoVariable myVar, Crit3DMeteoPoint *meteoPoint, Crit3DDate dateIni, Crit3DDate dateFin, Crit3DMeteoSettings *meteoSettings);
    std::vector<float> aggregatedHourlyToDailyList(meteoVariable myVar, Crit3DMeteoPoint* meteoPoint, Crit3DDate dateIni, Crit3DDate dateFin, Crit3DMeteoSettings *meteoSettings);

    bool anomalyOnPoint(Crit3DMeteoPoint* meteoPoint, float refValue);

    bool passingClimateToAnomaly(QString *myError, Crit3DMeteoPoint* meteoPointTemp, Crit3DClimate* clima, Crit3DMeteoPoint *meteoPoints, int nrMeteoPoints, Crit3DElaborationSettings *elabSettings);

    bool passingClimateToAnomalyGrid(QString *myError, Crit3DMeteoPoint* meteoPointTemp, Crit3DClimate* clima);

    bool climateOnPoint(QString *myError, Crit3DMeteoPointsDbHandler* meteoPointsDbHandler, Crit3DMeteoGridDbHandler* meteoGridDbHandler, Crit3DClimate* clima, Crit3DMeteoPoint *meteoPointTemp, std::vector<float> &outputValues, bool isMeteoGrid, QDate startDate, QDate endDate, bool changeDataSet, Crit3DMeteoSettings *meteoSettings);

    bool climateTemporalCycle(QString *myError, Crit3DClimate* clima, std::vector<float> &outputValues, Crit3DMeteoPoint* meteoPoint, meteoComputation elab1, meteoComputation elab2, Crit3DMeteoSettings *meteoSettings);

    bool dailyCumulatedClimate(QString *myError, std::vector<float> &inputValues, Crit3DClimate* clima, Crit3DMeteoPoint* meteoPoint, meteoComputation elab2, Crit3DMeteoSettings* meteoSettings);

    float thomDayTime(float tempMax, float relHumMinAir);

    float thomNightTime(float tempMin, float relHumMaxAir);

    float thomH(float tempAvg, float relHumAvgAir);

    int thomDailyNHoursAbove(TObsDataH *hourlyValues, float thomthreshold, float minimumPercentage);

    int temperatureDailyNHoursAbove(TObsDataH *hourlyValues, float temperaturethreshold, float minimumPercentage);

    float thomDailyMax(TObsDataH *hourlyValues, float minimumPercentage);

    float thomDailyMean(TObsDataH *hourlyValues, float minimumPercentage);

    float dailyLeafWetnessComputation(TObsDataH *hourlyValues, float minimumPercentage);

    float computeLastDayBelowThreshold(std::vector<float> &inputValues, Crit3DDate firstDateDailyVar, Crit3DDate firstDate, Crit3DDate finishDate, float param1);

    float computeWinkler(Crit3DMeteoPoint* meteoPoint, Crit3DDate firstDate, Crit3DDate finishDate, float minimumPercentage);

    float computeHuglin(Crit3DMeteoPoint* meteoPoint, Crit3DDate firstDate, Crit3DDate finishDate, float minimumPercentage);

    float computeFregoni(Crit3DMeteoPoint* meteoPoint, Crit3DDate firstDate, Crit3DDate finishDate, float minimumPercentage);

    float computeCorrectedSum(Crit3DMeteoPoint* meteoPoint, Crit3DDate firstDate, Crit3DDate finishDate, float param, float minimumPercentage);

    bool preElaboration(Crit3DMeteoPointsDbHandler* meteoPointsDbHandler, Crit3DMeteoGridDbHandler* meteoGridDbHandler,
                        Crit3DMeteoPoint* meteoPoint, bool isMeteoGrid, meteoVariable variable, meteoComputation elab1,
                        QDate startDate, QDate endDate, std::vector<float> &outputValues, float* percValue,
                        Crit3DMeteoSettings *meteoSettings, QString &myError);

    float loadDailyVarSeries(Crit3DMeteoPointsDbHandler *meteoPointsDbHandler,
                        Crit3DMeteoGridDbHandler *meteoGridDbHandler, Crit3DMeteoPoint* meteoPoint,
                        bool isMeteoGrid, meteoVariable variable, QDate first, QDate last, QString &myError);

    float loadDailyVarSeries_SaveOutput(Crit3DMeteoPointsDbHandler *meteoPointsDbHandler,
                        Crit3DMeteoGridDbHandler *meteoGridDbHandler, Crit3DMeteoPoint &meteoPoint, bool isMeteoGrid,
                        meteoVariable variable, QDate first, QDate last, std::vector<float> &outputValues, QString &myError);

    float loadHourlyVarSeries_SaveOutput(Crit3DMeteoPointsDbHandler *meteoPointsDbHandler, Crit3DMeteoGridDbHandler *meteoGridDbHandler,
                                         const QString &meteoPointId, bool isMeteoGrid, meteoVariable variable, const QDateTime &firstTime,
                                         const QDateTime &lastTime, std::vector<float> &outputValues, QString &myError);

    float loadHourlyVarSeries(Crit3DMeteoPointsDbHandler *meteoPointsDbHandler,
                        Crit3DMeteoGridDbHandler *meteoGridDbHandler, Crit3DMeteoPoint* meteoPoint,
                        bool isMeteoGrid, meteoVariable variable, const QDateTime &firstTime, const QDateTime &lastTime, QString &myError);

    void extractValidValuesCC(std::vector<float> &outputValues);

    void extractValidValuesWithThreshold(std::vector<float> &outputValues, float myThreshold);

    float computeStatistic(std::vector<float> &inputValues, Crit3DMeteoPoint* meteoPoint, Crit3DClimate* clima, 
						Crit3DDate firstDate, Crit3DDate lastDate, int nYears, meteoComputation elab1, meteoComputation elab2, 
						Crit3DMeteoSettings *meteoSettings, bool dataAlreadyLoaded);

    QString getTable(QString elab);

    int getClimateIndexFromElab(QDate myDate, QString elab);

    int getNumberClimateIndexFromElab(QString elab);

    period getPeriodTypeFromString(QString periodStr);

    int nParameters(meteoComputation elab);

    bool parseXMLElaboration(Crit3DElabList *listXMLElab, Crit3DAnomalyList *listXMLAnomaly, Crit3DDroughtList *listXMLDrought, 
						Crit3DPhenologyList *listXMLPhenology, QString xmlFileName, QString *myError);

    bool parseXMLPeriodType(QDomNode ancestor, QString attributePeriod, Crit3DElabList *listXMLElab, Crit3DAnomalyList *listXMLAnomaly, 
						bool isAnomaly, bool isRefPeriod, QString* period, QString *myError);
    
    bool parseXMLPeriodTag(QDomNode child, Crit3DElabList *listXMLElab, Crit3DAnomalyList *listXMLAnomaly, bool isAnomaly, bool isRefPeriod,
                        QString period, QString *myError);

    bool checkYears(QString firstYear, QString lastYear);

    bool checkElabParam(QString elab, QString param);

    bool checkDataType(QString xmlFileName, bool isMeteoGrid, QString *myError);

    bool appendXMLElaboration(Crit3DElabList *listXMLElab, QString xmlFileName, QString *myError);

    bool appendXMLAnomaly(Crit3DAnomalyList *listXMLAnomaly, QString xmlFileName, QString *myError);

    void createXMLFile(QString xmlFileName, QString *myError);

    bool monthlyAggregateDataGrid(Crit3DMeteoGridDbHandler* meteoGridDbHandler, QDate firstDate, QDate lastDate,
                    std::vector<meteoVariable> dailyMeteoVar, Crit3DMeteoSettings* meteoSettings,
                    Crit3DQuality *qualityCheck, Crit3DClimateParameters *climateParam, QString &myError);

    int computeAnnualSeriesOnPointFromDaily(QString *myError, Crit3DMeteoPointsDbHandler* meteoPointsDbHandler, Crit3DMeteoGridDbHandler* meteoGridDbHandler,
                    Crit3DMeteoPoint* meteoPointTemp, Crit3DClimate* clima, bool isMeteoGrid, bool isAnomaly,
                    Crit3DMeteoSettings* meteoSettings, std::vector<float> &outputValues, std::vector<int> &vectorYears, bool dataAlreadyLoaded);
    
	void computeClimateOnDailyData(Crit3DMeteoPoint meteoPoint, meteoVariable var, QDate firstDate, QDate lastDate,
                    int smooth, float* dataPresence, Crit3DQuality* qualityCheck, Crit3DClimateParameters* climateParam,
                    Crit3DMeteoSettings* meteoSettings, std::vector<float> &dailyClima, std::vector<float> &decadalClima, std::vector<float> &monthlyClima);
    
	bool preElaborationWithoutLoad(Crit3DMeteoPoint* meteoPoint, meteoVariable variable, QDate startDate, QDate endDate, 
					std::vector<float> &outputValues, float* percValue, Crit3DMeteoSettings* meteoSettings);
    
	float loadFromMp_SaveOutput(Crit3DMeteoPoint* meteoPoint,
					meteoVariable variable, QDate first, QDate last, std::vector<float> &outputValues);

	void setMpValues(Crit3DMeteoPoint meteoPointGet, Crit3DMeteoPoint* meteoPointSet, QDate myDate, meteoVariable myVar, Crit3DMeteoSettings* meteoSettings);

    meteoComputation getMeteoCompFromString(const std::map<std::string, meteoComputation> &map, const std::string &computationStr);


#endif // CLIMATE_H
