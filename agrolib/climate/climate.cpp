#include <QString>
#include <QDate>
#include <QtSql>
#include <math.h>       /* ceil */

#include "commonConstants.h"
#include "basicMath.h"
#include "climate.h"
#include "crit3dDate.h"
#include "utilities.h"
#include "statistics.h"
#include "quality.h"
#include "dbClimate.h"
#include "qdebug.h"

using namespace std;

bool elaborationOnPoint(QString *myError, Crit3DMeteoPointsDbHandler* meteoPointsDbHandler, Crit3DMeteoGridDbHandler* meteoGridDbHandler,
    Crit3DMeteoPoint* meteoPointTemp, Crit3DClimate* clima, bool isMeteoGrid, QDate startDate, QDate endDate, bool isAnomaly, Crit3DMeteoSettings* meteoSettings, bool dataAlreadyLoaded)
{
    float percValue;
    float result;

    std::vector<float> outputValues;

    meteoComputation elab1MeteoComp = getMeteoCompFromString(MapMeteoComputation, clima->elab1().toStdString());
    meteoComputation elab2MeteoComp = getMeteoCompFromString(MapMeteoComputation, clima->elab2().toStdString());

    if (clima->param1IsClimate())
    {
        QString table = getTable(clima->param1ClimateField());
        int index = clima->getParam1ClimateIndex();
        if (index != NODATA)
        {
            float param = readClimateElab(clima->db(), table, index, QString::fromStdString(meteoPointTemp->id), clima->param1ClimateField(), myError);
            clima->setParam1(param);
        }
        else
        {
            clima->setParam1(NODATA);
        }
    }

    bool dataLoaded;
    if (dataAlreadyLoaded)
    {
        dataLoaded = preElaborationWithoutLoad(meteoPointTemp, clima->variable(), startDate, endDate, outputValues, &percValue, meteoSettings);
    }
    else
    {
        dataLoaded = preElaboration(myError, meteoPointsDbHandler, meteoGridDbHandler, meteoPointTemp, isMeteoGrid, clima->variable(), elab1MeteoComp, startDate, endDate, outputValues, &percValue, meteoSettings);
    }

    if (dataLoaded)
    {
        // check
        Crit3DDate startD(startDate.day(), startDate.month(), clima->yearStart());
        Crit3DDate endD(endDate.day(), endDate.month(), clima->yearStart());

        if ( clima->nYears() < 0)
        {
            startD.year = clima->yearStart() + clima->nYears();
        }
        else if ( clima->nYears() > 0)
        {
            endD.year = clima->yearStart() + clima->nYears();
        }

        if (difference (startD, endD) < 0)
        {
            *myError = "Wrong dates!";
            return false;
        }
        if (clima->elab1() == "")
        {
            *myError = "Missing elaboration";
            return false;
        }

        result = computeStatistic(outputValues, meteoPointTemp, clima, startD, endD, clima->nYears(), elab1MeteoComp, elab2MeteoComp, meteoSettings, dataAlreadyLoaded);

        if (isAnomaly)
        {
            return anomalyOnPoint(meteoPointTemp, result);
        }
        else
        {
            meteoPointTemp->elaboration = result;
            if (meteoPointTemp->elaboration != NODATA)
            {
                return true;
            }
            else
            {
                return false;
            }
        }
    }
    else if (isAnomaly)
    {
        meteoPointTemp->anomaly = NODATA;
        meteoPointTemp->anomalyPercentage = NODATA;
    }

    return false;

}

bool anomalyOnPoint(Crit3DMeteoPoint* meteoPoint, float refValue)
{

    bool anomalyOnPoint = (refValue != NODATA && meteoPoint->elaboration != NODATA);

    if (anomalyOnPoint)
    {
        meteoPoint->anomaly = meteoPoint->elaboration - refValue;
        if (refValue != 0)
        {
            meteoPoint->anomalyPercentage = (meteoPoint->elaboration - refValue) / abs(refValue) * 100;
        }
        else
        {
            meteoPoint->anomalyPercentage = NODATA;
        }
    }
    else
    {
        meteoPoint->anomaly = NODATA;
        meteoPoint->anomalyPercentage = NODATA;
    }
    return anomalyOnPoint;

}


bool passingClimateToAnomaly(QString *myError, Crit3DMeteoPoint* meteoPointTemp, Crit3DClimate* clima, Crit3DMeteoPoint* meteoPoints, int nrMeteoPoints, Crit3DElaborationSettings *elabSettings)
{
    float valueClimate;
    QString table = getTable(clima->climateElab());
    int index = clima->getParam1ClimateIndex();
    valueClimate = readClimateElab(clima->db(), table, index, QString::fromStdString(meteoPointTemp->id), clima->climateElab(), myError);
    if (index != NODATA && valueClimate != NODATA)
    {
        // MP found
        return anomalyOnPoint(meteoPointTemp, valueClimate);
    }
    else
    {
        float maxVerticalDist = elabSettings->getAnomalyPtsMaxDeltaZ();
        float maxHorizontalDist = elabSettings->getAnomalyPtsMaxDistance();

        QList<QString> idList;
        if (table == "climate_generic" || table == "climate_annual")
        {
            idList = getIdListFromElab(clima->db(), table, myError, clima->climateElab());
        }
        else
        {
            idList = getIdListFromElab(clima->db(), table, myError, clima->climateElab(), index);
        }

        float minDist = NODATA;
        float currentDist = NODATA;
        bool noHeigth = false;
        QString idNearMP = "";

        for (int i = 0; i < idList.size(); i++)
        {
            for (int j = 0; j < nrMeteoPoints; j++)
            {
                if ( QString::fromStdString(meteoPoints[j].id) == idList[i])
                {

                    currentDist = gis::computeDistance(meteoPointTemp->point.utm.x, meteoPointTemp->point.utm.y, meteoPoints[j].point.utm.x, meteoPoints[j].point.utm.y);
                    if (currentDist < maxHorizontalDist)
                    {
                        if (minDist == NODATA || currentDist < minDist)
                        {
                            if (meteoPointTemp->point.z == NODATA && meteoPoints[j].point.z == NODATA)
                            {
                                noHeigth = true;
                            }
                            if (noHeigth || abs(meteoPointTemp->point.z - meteoPoints[j].point.z) < maxVerticalDist)
                            {
                                minDist = currentDist;
                                idNearMP = QString::fromStdString(meteoPoints[j].id);
                            }

                        }
                    }
                }
            }
        }

        if (minDist != NODATA)
        {
            valueClimate = readClimateElab(clima->db(), table, index, idNearMP, clima->climateElab(), myError);
            if (index != NODATA && valueClimate != NODATA)
            {
                return anomalyOnPoint(meteoPointTemp, valueClimate);
            }
            else
            {
                return anomalyOnPoint(meteoPointTemp, NODATA);
            }
        }
        else return false;
    }
}


bool passingClimateToAnomalyGrid(QString *myError, Crit3DMeteoPoint* meteoPointTemp, Crit3DClimate* clima)
{

    float valueClimate;
    QString table = getTable(clima->climateElab());
    int index = clima->getParam1ClimateIndex();
    valueClimate = readClimateElab(clima->db(), table, index, QString::fromStdString(meteoPointTemp->id), clima->climateElab(), myError);
    if (index != NODATA && valueClimate != NODATA)
    {
        // MP found
        return anomalyOnPoint(meteoPointTemp, valueClimate);
    }
    else
        return false;
}

bool climateOnPoint(QString *myError, Crit3DMeteoPointsDbHandler* meteoPointsDbHandler, Crit3DMeteoGridDbHandler* meteoGridDbHandler,
                    Crit3DClimate* clima, Crit3DMeteoPoint* meteoPointTemp, std::vector<float> &outputValues, bool isMeteoGrid, QDate startDate, QDate endDate, bool changeDataSet, Crit3DMeteoSettings* meteoSettings)
{
    float percValue;
    bool dataLoaded = true;

    if (isMeteoGrid)
    {
        clima->setDb(meteoGridDbHandler->db());
    }
    else
    {
        clima->setDb(meteoPointsDbHandler->getDb());
    }

    meteoComputation elab1MeteoComp = getMeteoCompFromString(MapMeteoComputation, clima->elab1().toStdString());
    meteoComputation elab2MeteoComp = getMeteoCompFromString(MapMeteoComputation, clima->elab2().toStdString());

    // check id points
    if (changeDataSet == false)
    {
        // check download data
        if ( (clima->variable() != clima->getCurrentVar() || clima->yearStart() < clima->getCurrentYearStart() || clima->yearEnd() > clima->getCurrentYearEnd()) ||
                (clima->elab1() != clima->getCurrentElab1() && (elab1MeteoComp == correctedDegreeDaysSum || elab1MeteoComp == huglin || elab1MeteoComp == winkler ||  elab1MeteoComp == fregoni) ) )
        {
            changeDataSet = true;
        }
        else
        {
            if (meteoPointTemp->nrObsDataDaysD == 0)
            {
                dataLoaded = false;
            }
        }
    }

    if (changeDataSet)
    {
        clima->setCurrentVar(clima->variable());
        clima->setCurrentElab1(clima->elab1());
        clima->setCurrentYearStart(clima->yearStart());
        clima->setCurrentYearEnd(clima->yearEnd());

        outputValues.clear();
        meteoPointTemp->nrObsDataDaysD = 0;
        meteoPointTemp->nrObsDataDaysH = 0;

        dataLoaded = preElaboration(myError, meteoPointsDbHandler, meteoGridDbHandler, meteoPointTemp, isMeteoGrid, clima->variable(), elab1MeteoComp, startDate, endDate, outputValues, &percValue, meteoSettings);
    }

    if (dataLoaded)
    {
        if (climateTemporalCycle(myError, clima, outputValues, meteoPointTemp, elab1MeteoComp, elab2MeteoComp, meteoSettings))
        {
            return true;
        }
    }

    return false;
}

bool climateTemporalCycle(QString *myError, Crit3DClimate* clima, std::vector<float> &outputValues, Crit3DMeteoPoint* meteoPoint, meteoComputation elab1, meteoComputation elab2, Crit3DMeteoSettings* meteoSettings)
{
    QSqlDatabase db = clima->db();
    bool dataAlreadyLoaded = false;

    float result;
    float paramValue;

    switch(clima->periodType())
    {

    case dailyPeriod:
    {
        clima->setCurrentPeriodType(clima->periodType());
        if ( clima->dailyCumulated() == true)
        {
            return dailyCumulatedClimate(myError, outputValues, clima, meteoPoint, elab2, meteoSettings);
        }

        bool okAtLeastOne = false;
        int nLeapYears = 0;
        int totYears = 0;
        int nDays = 366;
        int leapYear;
        std::vector<float> allResults;

        Crit3DDate startD;
        Crit3DDate endD;

        for (int i = clima->yearStart(); i<=clima->yearEnd(); i++)
        {
            if (isLeapYear(i))
            {
                nLeapYears = nLeapYears + 1;
                leapYear = i;
            }
            totYears = totYears + 1;
        }

        float minPerc = meteoSettings->getMinimumPercentage();

        if (nLeapYears == 0)
        {
            nDays = nDays - 1;
        }

        for (int i = 1; i<=nDays; i++)
        {
            if (nLeapYears == 0)
            {
                startD = getDateFromDoy(clima->yearStart(), i);
            }
            else
            {
                startD = getDateFromDoy(leapYear, i);
            }
            endD = startD;

            if (i == 366)
            {
                meteoSettings->setMinimumPercentage(minPerc * nLeapYears/totYears);
            }

            if (clima->param1IsClimate())
            {
                QString table = getTable(clima->param1ClimateField());
                int climateIndex = getClimateIndexFromElab(getQDate(startD), clima->param1ClimateField());
                if (climateIndex != NODATA)
                {
                    paramValue = readClimateElab(db, table, climateIndex, QString::fromStdString(meteoPoint->id), clima->param1ClimateField(), myError);
                    clima->setParam1(paramValue);
                }
                else
                {
                    clima->setParam1(NODATA);
                }
            }

            result = computeStatistic(outputValues, meteoPoint, clima, startD, endD, clima->nYears(), elab1, elab2, meteoSettings, dataAlreadyLoaded);

            if (result != NODATA)
            {
                okAtLeastOne = true;
            }
            allResults.push_back(result);
        }

        // if there are no leap years, save NODATA into 366row
        if (nLeapYears == 0)
        {
            allResults.push_back(NODATA);
        }

        meteoSettings->setMinimumPercentage(minPerc);

        // reset currentPeriod
        clima->setCurrentPeriodType(noPeriodType);

        if (okAtLeastOne)
        {
            return saveDailyElab(db, myError, QString::fromStdString(meteoPoint->id), allResults, clima->climateElab());
        }
        else
        {
            *myError = "no results to save";
            return false;
        }

    }
    case decadalPeriod:
    {
        bool okAtLeastOne = false;
        std::vector<float> allResults;

        for (int i = 1; i<=36; i++)
        {
            int dayStart;
            int dayEnd;
            int month;

            intervalDecade(i, clima->yearStart(), &dayStart, &dayEnd, &month);

            Crit3DDate startD (dayStart, month, clima->yearStart());
            Crit3DDate endD (dayEnd, month, clima->yearStart());

            if (clima->param1IsClimate())
            {
                QString table = getTable(clima->param1ClimateField());
                int climateIndex = getClimateIndexFromElab(getQDate(startD), clima->param1ClimateField());
                if (climateIndex != NODATA)
                {
                    paramValue = readClimateElab(db, table, climateIndex, QString::fromStdString(meteoPoint->id), clima->param1ClimateField(), myError);
                    clima->setParam1(paramValue);
                }
                else
                {
                    clima->setParam1(NODATA);
                }
            }

            result = computeStatistic(outputValues, meteoPoint, clima, startD, endD, clima->nYears(), elab1, elab2, meteoSettings, dataAlreadyLoaded);

            if (result != NODATA)
            {
                okAtLeastOne = true;
            }
            allResults.push_back(result);
        }
        if (okAtLeastOne)
        {
            return saveDecadalElab(db, myError, QString::fromStdString(meteoPoint->id), allResults, clima->climateElab());
        }
        else
        {
            *myError = "no results to save";
            return false;
        }
    }

    case monthlyPeriod:
    {
        bool okAtLeastOne = false;
        std::vector<float> allResults;

        for (int i = 1; i<=12; i++)
        {

            Crit3DDate startD (1, i, clima->yearStart());
            QDate temp(clima->yearEnd(), i, 1);
            int dayEnd = temp.daysInMonth();

            Crit3DDate endD (dayEnd, i, clima->yearStart());

            if (clima->param1IsClimate())
            {
                QString table = getTable(clima->param1ClimateField());
                int climateIndex = getClimateIndexFromElab(getQDate(startD), clima->param1ClimateField());

                if (climateIndex != NODATA)
                {
                    paramValue = readClimateElab(db, table, climateIndex, QString::fromStdString(meteoPoint->id), clima->param1ClimateField(), myError);
                    clima->setParam1(paramValue);
                }
                else
                {
                    clima->setParam1(NODATA);
                }
            }

            result = computeStatistic(outputValues, meteoPoint, clima, startD, endD, clima->nYears(), elab1, elab2, meteoSettings, dataAlreadyLoaded);

            if (result != NODATA)
            {
                okAtLeastOne = true;
            }
            allResults.push_back(result);
        }
        if (okAtLeastOne)
        {
            return saveMonthlyElab(db, myError, QString::fromStdString(meteoPoint->id), allResults, clima->climateElab());
        }
        else
        {
            *myError = "no results to save";
            return false;
        }
    }

    case seasonalPeriod:
    {
        bool okAtLeastOne = false;
        std::vector<float> allResults;

        for (int i = 1; i<=4; i++)
        {

            int monthEnd;
            int dayEnd;
            int seasonalNPeriodYears ;

            if (i<4)
            {
                monthEnd = i*3+2;
                seasonalNPeriodYears = 0;
            }
            else
            {
                monthEnd = 2;
                seasonalNPeriodYears = -1;
            }

            QDate temp(clima->yearEnd(), monthEnd, 1);

            dayEnd = temp.daysInMonth();

            Crit3DDate startD (1, i*3, clima->yearStart());
            Crit3DDate endD (dayEnd, monthEnd, clima->yearEnd());

            if (clima->param1IsClimate())
            {
                QString table = getTable(clima->param1ClimateField());
                int climateIndex = getClimateIndexFromElab(getQDate(startD), clima->param1ClimateField());

                if (climateIndex != NODATA)
                {
                    paramValue = readClimateElab(db, table, climateIndex, QString::fromStdString(meteoPoint->id), clima->param1ClimateField(), myError);
                    clima->setParam1(paramValue);
                }
                else
                {
                    clima->setParam1(NODATA);
                }
            }

            result = computeStatistic(outputValues, meteoPoint, clima, startD, endD, seasonalNPeriodYears, elab1, elab2, meteoSettings, dataAlreadyLoaded);

            if (result != NODATA)
            {
                okAtLeastOne = true;
            }
            allResults.push_back(result);
        }
        if (okAtLeastOne)
        {
            return saveSeasonalElab(db, myError, QString::fromStdString(meteoPoint->id), allResults, clima->climateElab());
        }
        else
        {
            *myError = "no results to save";
            return false;
        }

    }

    case annualPeriod:
    {

        Crit3DDate startD (1, 1, clima->yearStart());
        Crit3DDate endD (31, 12, clima->yearStart());

        if (clima->param1IsClimate())
        {
            QString table = getTable(clima->param1ClimateField());
            int climateIndex = getClimateIndexFromElab(getQDate(startD), clima->param1ClimateField());

            if (climateIndex != NODATA)
            {
                paramValue = readClimateElab(db, table, climateIndex, QString::fromStdString(meteoPoint->id), clima->param1ClimateField(), myError);
                clima->setParam1(paramValue);
            }
            else
            {
                clima->setParam1(NODATA);
            }
        }

        result = computeStatistic(outputValues, meteoPoint, clima, startD, endD, clima->nYears(), elab1, elab2, meteoSettings, dataAlreadyLoaded);

        if (result != NODATA)
        {
            return saveAnnualElab(db, myError, QString::fromStdString(meteoPoint->id), result, clima->climateElab());
        }
        else
        {
            *myError = "no results to save";
            return false;
        }
    }

    case genericPeriod:
    {

        Crit3DDate startD = getCrit3DDate(clima->genericPeriodDateStart());
        Crit3DDate endD = getCrit3DDate(clima->genericPeriodDateEnd());

        if (clima->param1IsClimate())
        {
            QString table = getTable(clima->param1ClimateField());
            int climateIndex = getClimateIndexFromElab(getQDate(startD), clima->param1ClimateField());
            if (climateIndex != NODATA)
            {
                paramValue = readClimateElab(db, table, climateIndex, QString::fromStdString(meteoPoint->id), clima->param1ClimateField(), myError);
                clima->setParam1(paramValue);
            }
            else
            {
                clima->setParam1(NODATA);
            }
        }
        result = computeStatistic(outputValues, meteoPoint, clima, startD, endD, clima->nYears(), elab1, elab2, meteoSettings, dataAlreadyLoaded);

        if (result != NODATA)
        {
            return saveGenericElab(db, myError, QString::fromStdString(meteoPoint->id), result, clima->climateElab());
        }
        else
        {
            *myError = "no results to save";
            return false;
        }
    }

    default:
    {
        *myError = "period not valid";
        return false;
    }

    }
}

bool dailyCumulatedClimate(QString *myError, std::vector<float> &inputValues, Crit3DClimate* clima, Crit3DMeteoPoint* meteoPoint, meteoComputation elab2, Crit3DMeteoSettings* meteoSettings)
{
    bool okAtLeastOne = false;
    int nLeapYears = 0;
    int totYears = 0;
    int nDays;
    float result;
    unsigned int index;
    std::vector<float> allResults;
    float cumulatedValue = 0;
    std::vector<float> cumulatedValues;
    std::vector< std::vector<float> > cumulatedAllDaysAllYears;
    std::vector<int> valuesYears;

    Crit3DDate presentDate;

    QSqlDatabase db = clima->db();
    float minPerc = meteoSettings->getMinimumPercentage();
    float param2 = clima->param2();
    for (int year = clima->yearStart(); year <= clima->yearEnd(); year++)
    {
        valuesYears.push_back(year);
        if (isLeapYear(year))
        {
            nLeapYears = nLeapYears + 1;
            nDays = 366;
            meteoSettings->setMinimumPercentage(minPerc * nLeapYears/totYears);
        }
        else
        {
            nDays = 365;
            meteoSettings->setMinimumPercentage(minPerc);
        }

        for (int n = 1; n<=nDays; n++)
        {
            presentDate = getDateFromDoy(year, n);
            float value = NODATA;

            if (presentDate >= meteoPoint->obsDataD[0].date)
            {
                index = difference(meteoPoint->obsDataD[0].date, presentDate);
                if (index < inputValues.size())
                {
                    value = inputValues.at(index);
                    if (value != NODATA) cumulatedValue = cumulatedValue + value;

                    cumulatedValues.push_back(cumulatedValue);
                }
            }
        }
        float validPercentage = (float(cumulatedValues.size()) / float(nDays)) * 100;
        if (validPercentage > meteoSettings->getMinimumPercentage())
        {
            cumulatedAllDaysAllYears.push_back(cumulatedValues);
        }
        cumulatedValues.clear();
        cumulatedValue = 0;
        totYears = totYears + 1;
    }

    if (nLeapYears == 0)
    {
        nDays = 365;
    }
    else
    {
        nDays = 366;
    }

    std::vector<float> cumulatedValuesPerDay;
    for (int i = 1; i<=nDays; i++)
    {
        for (int j=0; j<cumulatedAllDaysAllYears.size(); j++)
        {
            if (i <= cumulatedAllDaysAllYears[j].size())
            {
                cumulatedValuesPerDay.push_back(cumulatedAllDaysAllYears[j][i-1]);
            }
        }
        switch(elab2)
        {
            case yearMax: case yearMin:
            {
                int index = statisticalElab(elab2, clima->yearStart(), cumulatedValuesPerDay, cumulatedValuesPerDay.size(), meteoSettings->getRainfallThreshold());
                if (index != NODATA && index < valuesYears.size())
                {
                    result = valuesYears[index];
                }
                else
                {
                    result = NODATA;
                }
                break;
            }
            case trend:
                result = statisticalElab(elab2, clima->yearStart(), cumulatedValuesPerDay, cumulatedValuesPerDay.size(), meteoSettings->getRainfallThreshold());
                break;
            default:
                result = statisticalElab(elab2, param2, cumulatedValuesPerDay, cumulatedValuesPerDay.size(), meteoSettings->getRainfallThreshold());
        }
        cumulatedValuesPerDay.clear();

        if (result != NODATA)
        {
            okAtLeastOne = true;
        }
        allResults.push_back(result);
    }

    // if there are no leap years, save NODATA into 366row
    if (nLeapYears == 0)
    {
        allResults.push_back(NODATA);
    }

    // reset currentPeriod
    clima->setCurrentPeriodType(noPeriodType);

    if (okAtLeastOne)
    {
        return saveDailyElab(db, myError, QString::fromStdString(meteoPoint->id), allResults, clima->climateElab());
    }
    else
    {
        *myError = "no results to save";
        return false;
    }
}


float loadDailyVarSeries(QString *myError, Crit3DMeteoPointsDbHandler *meteoPointsDbHandler,
        Crit3DMeteoGridDbHandler *meteoGridDbHandler, Crit3DMeteoPoint* meteoPoint, bool isMeteoGrid,
        meteoVariable variable, QDate first, QDate last)
{

    std::vector<float> dailyValues;
    QDate firstDateDB;
    Crit3DQuality qualityCheck;
    int nrValidValues = 0;
    int nrRequestedValues = first.daysTo(last) +1;

    // meteoGrid
    if (isMeteoGrid)
    {
        if (meteoGridDbHandler->gridStructure().isFixedFields())
        {
            dailyValues = meteoGridDbHandler->loadGridDailyVarFixedFields(myError, QString::fromStdString(meteoPoint->id), variable, first, last, &firstDateDB);
        }
        else
        {
            dailyValues = meteoGridDbHandler->loadGridDailyVar(myError, QString::fromStdString(meteoPoint->id), variable, first, last, &firstDateDB);
        }
    }
    // meteoPoint
    else
    {
        dailyValues = meteoPointsDbHandler->loadDailyVar(myError, variable, getCrit3DDate(first), getCrit3DDate(last), &firstDateDB, meteoPoint );
    }


    if ( dailyValues.empty() )
    {
        //qDebug() << "myError: " << *myError;
        return 0;
    }
    else
    {
        if (meteoPoint->nrObsDataDaysD == 0)
        {
            meteoPoint->initializeObsDataD(int(dailyValues.size()), getCrit3DDate(firstDateDB));
        }

        Crit3DDate currentDate = getCrit3DDate(firstDateDB);
        for (unsigned int i = 0; i < dailyValues.size(); i++)
        {
            quality::qualityType qualityT = qualityCheck.syntacticQualitySingleValue(variable, dailyValues[i]);
            if (qualityT == quality::accepted)
            {
                nrValidValues = nrValidValues + 1;
            }
            meteoPoint->setMeteoPointValueD(currentDate, variable, dailyValues[i]);
            currentDate = currentDate.addDays(1);
        }

        float percValue = float(nrValidValues) / float(nrRequestedValues);
        return percValue;
    }
}


float loadDailyVarSeries_SaveOutput(QString *myError, Crit3DMeteoPointsDbHandler *meteoPointsDbHandler,
        Crit3DMeteoGridDbHandler *meteoGridDbHandler, Crit3DMeteoPoint* meteoPoint, bool isMeteoGrid,
        meteoVariable variable, QDate first, QDate last, std::vector<float> &outputValues)
{
    std::vector<float> dailyValues;
    QDate firstDateDB;
    Crit3DQuality qualityCheck;
    int nrValidValues = 0;
    int nrRequestedValues = first.daysTo(last) +1;

    // meteoGrid
    if (isMeteoGrid)
    {
        if (meteoGridDbHandler->gridStructure().isFixedFields())
        {
            dailyValues = meteoGridDbHandler->loadGridDailyVarFixedFields(myError, QString::fromStdString(meteoPoint->id), variable, first, last, &firstDateDB);
        }
        else
        {
            dailyValues = meteoGridDbHandler->loadGridDailyVar(myError, QString::fromStdString(meteoPoint->id), variable, first, last, &firstDateDB);
        }

    }
    // meteoPoint
    else
    {
        dailyValues = meteoPointsDbHandler->loadDailyVar(myError, variable, getCrit3DDate(first), getCrit3DDate(last), &firstDateDB, meteoPoint );
    }


    if ( dailyValues.empty() )
    {
        return 0;
    }
    else
    {
        if (meteoPoint->nrObsDataDaysD == 0)
        {
            meteoPoint->initializeObsDataD(int(dailyValues.size()), getCrit3DDate(firstDateDB));
        }

        Crit3DDate currentDate = getCrit3DDate(firstDateDB);
        for (unsigned int i = 0; i < dailyValues.size(); i++)
        {
            quality::qualityType qualityT = qualityCheck.syntacticQualitySingleValue(variable, dailyValues[i]);
            if (qualityT == quality::accepted)
            {
                nrValidValues = nrValidValues + 1;
                meteoPoint->setMeteoPointValueD(currentDate, variable, dailyValues[i]);
                outputValues.push_back(dailyValues[i]);
            }
            else
            {
                meteoPoint->setMeteoPointValueD(currentDate, variable, NODATA);
                outputValues.push_back(NODATA);
            }

            currentDate = currentDate.addDays(1);
        }

        float percValue = float(nrValidValues) / float(nrRequestedValues);
        return percValue;
    }
}

float loadFromMp_SaveOutput(Crit3DMeteoPoint* meteoPoint,
        meteoVariable variable, QDate first, QDate last, std::vector<float> &outputValues)
{
    Crit3DQuality qualityCheck;
    int nrValidValues = 0;
    int nrRequestedValues = first.daysTo(last) +1;


    for (QDate myDate = first; myDate<=last; myDate=myDate.addDays(1))
    {
        float value = meteoPoint->getMeteoPointValueD(getCrit3DDate(myDate), variable);
        quality::qualityType qualityT = qualityCheck.syntacticQualitySingleValue(variable, value);
        if (qualityT == quality::accepted)
        {
            nrValidValues = nrValidValues + 1;
        }
        outputValues.push_back(value);
    }

    float percValue = float(nrValidValues) / float(nrRequestedValues);
    return percValue;
}

float loadHourlyVarSeries(QString *myError, Crit3DMeteoPointsDbHandler* meteoPointsDbHandler,
           Crit3DMeteoGridDbHandler* meteoGridDbHandler, Crit3DMeteoPoint* meteoPoint, bool isMeteoGrid,
           meteoVariable variable, QDateTime first, QDateTime last)
{
    std::vector<float> hourlyValues;
    QDateTime firstDateDB;
    Crit3DQuality qualityCheck;
    int nrValidValues = 0;
    int nrRequestedDays = first.daysTo(last);
    int nrRequestedValues = nrRequestedDays * 24 * meteoPoint->hourlyFraction;

    // meteoGrid
    if (isMeteoGrid)
    {
        if (meteoGridDbHandler->gridStructure().isFixedFields())
        {
            hourlyValues = meteoGridDbHandler->loadGridHourlyVarFixedFields(myError, QString::fromStdString(meteoPoint->id), variable, first, last, &firstDateDB);
        }
        else
        {
            hourlyValues = meteoGridDbHandler->loadGridHourlyVar(myError, QString::fromStdString(meteoPoint->id), variable, first, last, &firstDateDB);
        }
    }
    // meteoPoint
    else
    {
        hourlyValues = meteoPointsDbHandler->loadHourlyVar(myError, variable, getCrit3DDate(first.date()), getCrit3DDate(last.date()), &firstDateDB, meteoPoint );
    }


    if ( hourlyValues.empty() )
    {
        return 0;
    }
    else
    {
        if (meteoPoint->nrObsDataDaysH == 0)
        {
            int nrOfDays = ceil(float(hourlyValues.size()) / float(24 * meteoPoint->hourlyFraction));
            meteoPoint->initializeObsDataH(meteoPoint->hourlyFraction, nrOfDays,getCrit3DDate(firstDateDB.date()));
            meteoPoint->initializeObsDataD(nrOfDays, getCrit3DDate(firstDateDB.date()));
        }

        for (unsigned int i = 0; i < hourlyValues.size(); i++)
        {
            quality::qualityType qualityT = qualityCheck.syntacticQualitySingleValue(variable, hourlyValues[i]);
            if (qualityT == quality::accepted)
            {
                nrValidValues = nrValidValues + 1;
            }

            meteoPoint->setMeteoPointValueH(Crit3DDate(firstDateDB.date().day(), firstDateDB.date().month(), firstDateDB.date().year()), firstDateDB.time().hour(), firstDateDB.time().minute(), variable, hourlyValues[i]);

            firstDateDB = firstDateDB.addSecs(3600);
        }

        float percValue = float(nrValidValues) / float(nrRequestedValues);
        return percValue;
    }

}


// compute daily Thom using Tmax e RHmin
float thomDayTime(float tempMax, float relHumMinAir)
{

    Crit3DQuality qualityCheck;
    quality::qualityType qualityT = qualityCheck.syntacticQualitySingleValue(dailyAirTemperatureMax, tempMax);
    quality::qualityType qualityRelHumMinAir = qualityCheck.syntacticQualitySingleValue(dailyAirRelHumidityMin, relHumMinAir);


    // TODO nella versione vb ammessi anche i qualitySuspectData, questo tipo per ora non è stato implementato
    if ( qualityT == quality::accepted && qualityRelHumMinAir == quality::accepted )
    {
            return computeThomIndex(tempMax, relHumMinAir);
    }
    else
        return NODATA;

}

// compute daily Thom using Tmin e RHmax
float thomNightTime(float tempMin, float relHumMaxAir)
{

    Crit3DQuality qualityCheck;
    quality::qualityType qualityT = qualityCheck.syntacticQualitySingleValue(dailyAirTemperatureMin, tempMin);
    quality::qualityType qualityRelHumMaxAir = qualityCheck.syntacticQualitySingleValue(dailyAirRelHumidityMax, relHumMaxAir);

    // TODO nella versione vb ammessi anche i qualitySuspectData, questo tipo per ora non è stato implementato
    if ( qualityT == quality::accepted && qualityRelHumMaxAir == quality::accepted )
    {
            return computeThomIndex(tempMin, relHumMaxAir);
    }
    else
        return NODATA;

}

// compuote hourly thom
float thomH(float tempAvg, float relHumAvgAir)
{

    Crit3DQuality qualityCheck;
    quality::qualityType qualityT = qualityCheck.syntacticQualitySingleValue(dailyAirTemperatureAvg, tempAvg);
    quality::qualityType qualityRelHumAvgAir = qualityCheck.syntacticQualitySingleValue(dailyAirRelHumidityAvg, relHumAvgAir);

    // TODO nella versione vb ammessi anche i qualitySuspectData, questo tipo per ora non è stato implementato
    if ( qualityT == quality::accepted && qualityRelHumAvgAir == quality::accepted )
    {
            return computeThomIndex(tempAvg, relHumAvgAir);
    }
    else
        return NODATA;
}


// compute # hours thom >  threshold per day
int thomDailyNHoursAbove(TObsDataH* hourlyValues, float thomthreshold, float minimumPercentage)
{

    int nData = 0;
    int nrHours = NODATA;
    for (int hour = 0; hour < 24; hour++)
    {
        float thom = thomH(hourlyValues->tAir[hour], hourlyValues->rhAir[hour]);
        if (fabs(thom - NODATA) > EPSILON)
        {
            nData++;// = nData + 1;
            if (nrHours == NODATA)
                nrHours = 0;
            if (thom > thomthreshold)
                nrHours++;
        }
    }
    if ( (float(nData) / 24 * 100) < minimumPercentage)
        nrHours = NODATA;

    return nrHours;


}

// compute daily max thom value
float thomDailyMax(TObsDataH* hourlyValues, float minimumPercentage)
{
    int nData = 0;
    float thomMax = NODATA;
    for (int hour = 0; hour < 24; hour++)
    {
        float thom = thomH(hourlyValues->tAir[hour], hourlyValues->rhAir[hour]);
        if (fabs(thom - NODATA) > EPSILON)
        {
            nData++;// = nData + 1;
            if (thom > thomMax)
                thomMax = thom;
        }
    }
    if ( (float(nData) / 24 * 100) < minimumPercentage)
        thomMax = NODATA;

    return thomMax;
}

// compute daily avg thom value
float thomDailyMean(TObsDataH* hourlyValues, float minimumPercentage)
{

    int nData = 0;
    std::vector<float> thomValues;
    float thomDailyMean;

    for (int hour = 0; hour < 24; hour++)
    {
        float thom = thomH(hourlyValues->tAir[hour], hourlyValues->rhAir[hour]);
        if (fabs(thom - NODATA) > EPSILON)
        {
            thomValues.push_back(thom);
            nData++; //nData = nData + 1;
        }
    }
    if ( (float(nData) / 24 * 100) < minimumPercentage)
        thomDailyMean = NODATA;
    else
        thomDailyMean = statistics::mean(thomValues);


    return thomDailyMean;

}

// compute # hours per day where temperature >  threshold
int temperatureDailyNHoursAbove(TObsDataH* hourlyValues, float temperaturethreshold, float minimumPercentage)
{

    int nData = 0;
    int nrHours = NODATA;
    for (int hour = 0; hour < 24; hour++)
    {
        if (fabs(hourlyValues->tAir[hour] - NODATA) > EPSILON)
        {
            nData++;
            if (nrHours == NODATA)
                nrHours = 0;
            if (hourlyValues->tAir[hour] > temperaturethreshold)
                nrHours++;
        }
    }
    if ( (float(nData) / 24 * 100) < minimumPercentage)
        nrHours = NODATA;
    return nrHours;
}

float dailyLeafWetnessComputation(TObsDataH* hourlyValues, float minimumPercentage)
{

    int nData = 0;
    float dailyLeafWetnessRes = 0;

    for (int hour = 0; hour < 24; hour++)
    {
        if (hourlyValues->leafW[hour] == 0 || hourlyValues->leafW[hour] == 1)
        {
                dailyLeafWetnessRes = dailyLeafWetnessRes + hourlyValues->leafW[hour];
                nData++; //nData = nData + 1;
        }
    }
    if ( (float(nData) / 24 * 100) < minimumPercentage)
        dailyLeafWetnessRes = NODATA;

    return dailyLeafWetnessRes;

}



float computeWinkler(Crit3DMeteoPoint* meteoPoint, Crit3DDate firstDate, Crit3DDate finishDate, float minimumPercentage)
{

    float computeWinkler = 0;

    Crit3DQuality qualityCheck;
    int index;
    int count = 0;
    bool checkData;
    float Tavg;


    int numberOfDays = difference(firstDate, finishDate) +1;

    Crit3DDate presentDate = firstDate;
    for (int i = 0; i < numberOfDays; i++)
    {
        index = difference(meteoPoint->obsDataD[0].date, presentDate);
        checkData = false;
        if (index >= 0 && index < meteoPoint->nrObsDataDaysD)
        {

            // TO DO nella versione vb il check prevede anche l'immissione del parametro height
            quality::qualityType qualityTavg = qualityCheck.syntacticQualitySingleValue(dailyAirTemperatureAvg, meteoPoint->obsDataD[index].tAvg);
            if (qualityTavg == quality::accepted)
            {
                Tavg = meteoPoint->obsDataD[index].tAvg;
                checkData = true;
            }
            else
            {
                quality::qualityType qualityTmin = qualityCheck.syntacticQualitySingleValue(dailyAirTemperatureMin, meteoPoint->obsDataD[index].tMin);
                quality::qualityType qualityTmax = qualityCheck.syntacticQualitySingleValue(dailyAirTemperatureMax, meteoPoint->obsDataD[index].tMax);
                if (qualityTmin  == quality::accepted && qualityTmax == quality::accepted)
                {
                    Tavg = (meteoPoint->obsDataD[index].tMin + meteoPoint->obsDataD[index].tMax)/2;
                    checkData = true;
                }

            }

        }
        if (checkData)
        {
            if (Tavg > WINKLERTHRESHOLD)
            {
                Tavg = Tavg - WINKLERTHRESHOLD;
            }
            else
            {
                Tavg = 0;
            }
            computeWinkler = computeWinkler + Tavg;
            count = count + 1;
        }
        presentDate = presentDate.addDays(1);
    }
    if (numberOfDays != 0)
    {
        if ( (float(count) / float(numberOfDays) * 100.f) < minimumPercentage )
        {
            computeWinkler = NODATA;
        }
    }

    return computeWinkler;

}

float computeLastDayBelowThreshold(std::vector<float> &inputValues, Crit3DDate firstDateDailyVar, Crit3DDate firstDate, Crit3DDate finishDate, float param1)
{
    unsigned int index;
    float lastDay = NODATA;

    int numberOfDays = difference(firstDate, finishDate) +1;
    Crit3DDate presentDate = finishDate;
    for (int i = 0; i < numberOfDays; i++)
    {
        index = difference(firstDateDailyVar, presentDate);
        if (index >= 0 && index < inputValues.size())
        {
            if (inputValues.at(index) != NODATA && inputValues.at(index) < param1)
            {
                lastDay = getDoyFromDate(presentDate);
            }
        }
        presentDate = presentDate.addDays(-1);
    }


    return lastDay;
}

float computeHuglin(Crit3DMeteoPoint* meteoPoint, Crit3DDate firstDate, Crit3DDate finishDate, float minimumPercentage)
{
    float computeHuglin = 0;

    const int threshold = 10;
    const float K = 1.04f;                      //coeff. K di Huglin lunghezza giorno (=1.04 per ER)

    Crit3DQuality qualityCheck;
    int index;
    int count = 0;
    bool checkData;
    float Tavg;
    float Tmax;


    int numberOfDays = difference(firstDate, finishDate) +1;

    Crit3DDate presentDate = firstDate;
    for (int i = 0; i < numberOfDays; i++)
    {
        index = difference(meteoPoint->obsDataD[0].date, presentDate);
        checkData = false;
        if (index >= 0 && index < meteoPoint->nrObsDataDaysD)
        {

            // TO DO nella versione vb il check prevede anche l'immissione del parametro height
            quality::qualityType qualityTavg = qualityCheck.syntacticQualitySingleValue(dailyAirTemperatureAvg, meteoPoint->obsDataD[index].tAvg);
            quality::qualityType qualityTmax = qualityCheck.syntacticQualitySingleValue(dailyAirTemperatureMax, meteoPoint->obsDataD[index].tMax);
            if (qualityTavg == quality::accepted && qualityTmax == quality::accepted)
            {
                Tmax = meteoPoint->obsDataD[index].tMax;
                Tavg = meteoPoint->obsDataD[index].tAvg;
                checkData = true;
            }
            else
            {
                quality::qualityType qualityTmin = qualityCheck.syntacticQualitySingleValue(dailyAirTemperatureMin, meteoPoint->obsDataD[index].tMin);
                if (qualityTmin  == quality::accepted && qualityTmax == quality::accepted)
                {
                    Tmax = meteoPoint->obsDataD[index].tMax;
                    Tavg = (meteoPoint->obsDataD[index].tMin + Tmax)/2;
                    checkData = true;
                }

            }

        }
        if (checkData)
        {
            computeHuglin = computeHuglin + K * ((Tavg - threshold) + (Tmax - threshold)) / 2;
            count = count + 1;
        }
        presentDate = presentDate.addDays(1);
    }
    if (numberOfDays != 0)
    {
        if ( (float(count) / float(numberOfDays) * 100.f) < minimumPercentage )
        {
            computeHuglin = NODATA;
        }
    }

    return computeHuglin;
}


float computeFregoni(Crit3DMeteoPoint* meteoPoint, Crit3DDate firstDate, Crit3DDate finishDate, float minimumPercentage)
{
    const int threshold = 10;
    float computeFregoni = 0;

    Crit3DQuality qualityCheck;
    int index;
    int count = 0;
    int myDaysBelow = 0;
    bool checkData;
    float tMin, tMax;
    float tRange;
    float sumTRange = 0;


    int numberOfDays = difference(firstDate, finishDate) +1;

    Crit3DDate presentDate = firstDate;
    for (int i = 0; i < numberOfDays; i++)
    {
        index = difference(meteoPoint->obsDataD[0].date, presentDate);
        checkData = false;
        if (index >= 0 && index < meteoPoint->nrObsDataDaysD)
        {

            // TO DO nella versione vb il check prevede anche l'immissione del parametro height
            quality::qualityType qualityTmin = qualityCheck.syntacticQualitySingleValue(dailyAirTemperatureMin, meteoPoint->obsDataD[index].tMin);
            quality::qualityType qualityTmax = qualityCheck.syntacticQualitySingleValue(dailyAirTemperatureMax, meteoPoint->obsDataD[index].tMax);
            if (qualityTmin == quality::accepted && qualityTmax == quality::accepted)
            {
                tMin = meteoPoint->obsDataD[index].tMin;
                tMax = meteoPoint->obsDataD[index].tMax;
                tRange = tMax - tMin;
                checkData = true;
            }

        }
        if (checkData)
        {
            sumTRange = sumTRange + tRange;
            if (tMin < threshold)
            {
                myDaysBelow = myDaysBelow + 1;
            }
            count = count + 1;
        }
        presentDate = presentDate.addDays(1);
    }
    if (numberOfDays != 0)
    {
        if ( (float(count) / float(numberOfDays) * 100.f) < minimumPercentage )
        {
            computeFregoni = NODATA;
        }
        else
        {
            computeFregoni = sumTRange * myDaysBelow;
        }
    }

    return computeFregoni;
}


float computeCorrectedSum(Crit3DMeteoPoint* meteoPoint, Crit3DDate firstDate, Crit3DDate finishDate, float param, float minimumPercentage)
{
    float sum = 0;

    Crit3DQuality qualityCheck;
    int index;
    int count = 0;
    bool checkData;
    float tMin, tMax, tAvg;
    float numTmp, numerator, denominator;

    int numberOfDays = difference(firstDate, finishDate)+1;

    Crit3DDate presentDate = firstDate;
    index = difference(meteoPoint->obsDataD[0].date, presentDate);

    for (int i = 0; i < numberOfDays; i++)
    {
        checkData = false;
        if (index >= 0 && index < meteoPoint->nrObsDataDaysD)
        {
            // TO DO nella versione vb il check prevede anche l'immissione del parametro height
            quality::qualityType qualityTmin = qualityCheck.syntacticQualitySingleValue(dailyAirTemperatureMin, meteoPoint->obsDataD[index].tMin);
            quality::qualityType qualityTmax = qualityCheck.syntacticQualitySingleValue(dailyAirTemperatureMax, meteoPoint->obsDataD[index].tMax);
            if (qualityTmin == quality::accepted && qualityTmax == quality::accepted)
            {
                tMax = meteoPoint->obsDataD[index].tMax;
                tMin = meteoPoint->obsDataD[index].tMin;
                checkData = true;
            }
        }

        if (checkData)
        {
            if (param < tMax)
            {
                if (param <= tMin)
                {
                    tAvg = (tMax + tMin) / 2;
                    sum += (tAvg - param);
                }
                else
                {
                    numTmp = tMax - param;
                    numerator = numTmp * numTmp;
                    denominator = 2 * (tMax - tMin);
                    if (denominator != 0)
                    {
                        sum += (numerator/denominator);
                    }
                }
            }
            count = count + 1;
        }
        ++presentDate;
        index++;
    }

    if (numberOfDays != 0)
    {
        if ( (float(count) / float(numberOfDays) * 100.f) < minimumPercentage )
        {
            sum = NODATA;
        }
    }

    return sum;
}


bool elaborateDailyAggregatedVar(meteoVariable myVar, Crit3DMeteoPoint meteoPoint, std::vector<float> &outputValues, float* percValue, Crit3DMeteoSettings* meteoSettings)
{
    outputValues.clear();

    frequencyType aggregationFrequency = getAggregationFrequency(myVar);

    if (aggregationFrequency == hourly)
    {
            return elaborateDailyAggregatedVarFromHourly(myVar, meteoPoint, outputValues, meteoSettings);
    }
    else if (aggregationFrequency == daily)
    {
            return elaborateDailyAggregatedVarFromDaily(myVar, meteoPoint, meteoSettings, outputValues, percValue);
    }
    else
        return false;

}

bool elaborateDailyAggrVarFromStartDate(meteoVariable myVar, Crit3DMeteoPoint meteoPoint, QDate first, QDate last, std::vector<float> &outputValues, float* percValue, Crit3DMeteoSettings* meteoSettings)
{
    outputValues.clear();
    return elaborateDailyAggrVarFromDailyFromStartDate(myVar, meteoPoint, meteoSettings, first, last, outputValues, percValue);

}


frequencyType getAggregationFrequency(meteoVariable myVar)
{

    if (myVar == dailyThomHoursAbove || myVar == dailyThomAvg || myVar == dailyThomMax || myVar == dailyLeafWetness || myVar == dailyTemperatureHoursAbove)
    {
        return hourly;
    }
    else if (myVar != noMeteoVar)
    {
        return daily;
    }
    else
    {
        return noFrequency;
    }

}

bool elaborateDailyAggregatedVarFromDaily(meteoVariable myVar, Crit3DMeteoPoint meteoPoint, Crit3DMeteoSettings* meteoSettings,
                                          std::vector<float> &outputValues, float* percValue)
{

    float res;
    int nrValidValues = 0;
    Crit3DDate date = meteoPoint.obsDataD[0].date;
    Crit3DQuality qualityCheck;

    for (unsigned int index = 0; index < unsigned(meteoPoint.nrObsDataDaysD); index++)
    {
        switch(myVar)
        {
        case dailyThomDaytime:
                res = thomDayTime(meteoPoint.obsDataD[index].tMax, meteoPoint.obsDataD[index].rhMin);
            break;
        case dailyThomNighttime:
                res = thomNightTime(meteoPoint.obsDataD[index].tMin, meteoPoint.obsDataD[index].rhMax);
                break;
        case dailyBIC:
                res = computeDailyBIC(meteoPoint.obsDataD[index].prec, meteoPoint.obsDataD[index].et0_hs);
                break;
        case dailyAirTemperatureRange:
                res = dailyThermalRange(meteoPoint.obsDataD[index].tMin, meteoPoint.obsDataD[index].tMax);
                break;
        case dailyAirTemperatureAvg:
                {
                    quality::qualityType qualityTavg = qualityCheck.syntacticQualitySingleValue(dailyAirTemperatureAvg, meteoPoint.obsDataD[index].tAvg);
                    if (qualityTavg == quality::accepted)
                    {
                        res = meteoPoint.obsDataD[index].tAvg;
                    }
                    else
                    {
                        res = dailyAverageT(meteoPoint.obsDataD[index].tMin, meteoPoint.obsDataD[index].tMax);
                        meteoPoint.obsDataD[index].tAvg = res;
                    }
                    break;
                }
        case dailyReferenceEvapotranspirationHS:
        {
            quality::qualityType qualityEtp = qualityCheck.syntacticQualitySingleValue(dailyReferenceEvapotranspirationHS, meteoPoint.obsDataD[index].et0_hs);
            if (qualityEtp == quality::accepted)
            {
                res = meteoPoint.obsDataD[index].et0_hs;
            }
            else
            {
                res = dailyEtpHargreaves(meteoPoint.obsDataD[index].tMin, meteoPoint.obsDataD[index].tMax, date, meteoPoint.latitude, meteoSettings);
                meteoPoint.obsDataD[index].et0_hs = res;
            }
            break;
        }
        case dailyHeatingDegreeDays:
        {
            quality::qualityType qualityTavg = qualityCheck.syntacticQualitySingleValue(dailyAirTemperatureAvg, meteoPoint.obsDataD[index].tAvg);
            if (qualityTavg == quality::accepted)
            {
                res = 0;
                if ( meteoPoint.obsDataD[index].tAvg < DDHEATING_THRESHOLD)
                {
                    res = DDHEATING_THRESHOLD - meteoPoint.obsDataD[index].tAvg;
                }
            }
            else
            {
                meteoPoint.obsDataD[index].tAvg = dailyAverageT(meteoPoint.obsDataD[index].tMin, meteoPoint.obsDataD[index].tMax);
                qualityTavg = qualityCheck.syntacticQualitySingleValue(dailyAirTemperatureAvg, meteoPoint.obsDataD[index].tAvg);
                if (qualityTavg == quality::accepted)
                {
                    res = 0;
                    if ( meteoPoint.obsDataD[index].tAvg < DDHEATING_THRESHOLD)
                    {
                        res = DDHEATING_THRESHOLD - meteoPoint.obsDataD[index].tAvg;
                    }
                }
                else
                {
                    res = NODATA;
                }
            }
            meteoPoint.obsDataD[index].dd_heating = res;
            break;
        }
        case dailyCoolingDegreeDays:
        {
            quality::qualityType qualityTavg = qualityCheck.syntacticQualitySingleValue(dailyAirTemperatureAvg, meteoPoint.obsDataD[index].tAvg);
            if (qualityTavg == quality::accepted)
            {
                res = 0;
                if ( meteoPoint.obsDataD[index].tAvg > DDCOOLING_THRESHOLD)
                {
                    res = meteoPoint.obsDataD[index].tAvg - DDCOOLING_SUBTRACTION;
                }
            }
            else
            {
                meteoPoint.obsDataD[index].tAvg = dailyAverageT(meteoPoint.obsDataD[index].tMin, meteoPoint.obsDataD[index].tMax);
                qualityTavg = qualityCheck.syntacticQualitySingleValue(dailyAirTemperatureAvg, meteoPoint.obsDataD[index].tAvg);
                if (qualityTavg == quality::accepted)
                {
                    res = 0;
                    if ( meteoPoint.obsDataD[index].tAvg > DDCOOLING_THRESHOLD)
                    {
                        res = meteoPoint.obsDataD[index].tAvg - DDCOOLING_SUBTRACTION;
                    }
                }
                else
                {
                    res = NODATA;
                }
            }
            meteoPoint.obsDataD[index].dd_heating = res;
            break;
        }
        default:
                res = NODATA;
                break;
        }

        if (res != NODATA)
        {
            nrValidValues += 1;
        }

        outputValues.push_back(res);
        date = date.addDays(1);
    }

    *percValue = nrValidValues / meteoPoint.nrObsDataDaysD;
    if (nrValidValues > 0)
        return true;
    else
        return false;

}


bool elaborateDailyAggregatedVarFromHourly(meteoVariable myVar, Crit3DMeteoPoint meteoPoint, std::vector<float> &outputValues, Crit3DMeteoSettings *meteoSettings)
{

    float res;
    int nrValidValues = 0;

    TObsDataH* hourlyValues;

    Crit3DDate date;

    for (int index = 0; index < meteoPoint.nrObsDataDaysH; index++)
    {
        date = meteoPoint.getMeteoPointHourlyValuesDate(index);
        if (meteoPoint.getMeteoPointValueDayH(date, hourlyValues))
        {
            switch(myVar)
            {
                case dailyThomHoursAbove:
                    res = thomDailyNHoursAbove(hourlyValues, meteoSettings->getThomThreshold(), meteoSettings->getMinimumPercentage());
                    break;
                case dailyThomMax:
                    res = thomDailyMax(hourlyValues, meteoSettings->getMinimumPercentage());
                    break;
                case dailyThomAvg:
                    res = thomDailyMean(hourlyValues, meteoSettings->getMinimumPercentage());
                    break;
                case dailyLeafWetness:
                    res = dailyLeafWetnessComputation(hourlyValues, meteoSettings->getMinimumPercentage());
                    break;
                case dailyTemperatureHoursAbove:
                    res = temperatureDailyNHoursAbove(hourlyValues, meteoSettings->getTemperatureThreshold(), meteoSettings->getMinimumPercentage());
                    break;
                default:
                    res = NODATA;
                    break;
            }

            if (! isEqual(res, NODATA)) nrValidValues += 1;
        }

        outputValues.push_back(res);
    }

    if (nrValidValues > 0)
        return true;
    else
        return false;

}

bool elaborateDailyAggrVarFromDailyFromStartDate(meteoVariable myVar, Crit3DMeteoPoint meteoPoint, Crit3DMeteoSettings* meteoSettings, QDate first, QDate last,
                                          std::vector<float> &outputValues, float* percValue)
{

    float res;
    int nrValidValues = 0;
    QDate firstDate = getQDate(meteoPoint.obsDataD[0].date);
    Crit3DQuality qualityCheck;

    for (QDate myDate = first; myDate<=last; myDate=myDate.addDays(1))
    {
        unsigned long index = firstDate.daysTo(myDate);
        if (index >= meteoPoint.obsDataD.size())
        {
            break;
        }

        switch(myVar)
        {
        case dailyThomDaytime:
                res = thomDayTime(meteoPoint.obsDataD[index].tMax, meteoPoint.obsDataD[index].rhMin);
            break;
        case dailyThomNighttime:
                res = thomNightTime(meteoPoint.obsDataD[index].tMin, meteoPoint.obsDataD[index].rhMax);
                break;
        case dailyBIC:
                res = computeDailyBIC(meteoPoint.obsDataD[index].prec, meteoPoint.obsDataD[index].et0_hs);
                break;
        case dailyAirTemperatureRange:
                res = dailyThermalRange(meteoPoint.obsDataD[index].tMin, meteoPoint.obsDataD[index].tMax);
                break;
        case dailyAirTemperatureAvg:
                {
                    quality::qualityType qualityTavg = qualityCheck.syntacticQualitySingleValue(dailyAirTemperatureAvg, meteoPoint.obsDataD[index].tAvg);
                    if (qualityTavg == quality::accepted)
                    {
                        res = meteoPoint.obsDataD[index].tAvg;
                    }
                    else
                    {
                        res = dailyAverageT(meteoPoint.obsDataD[index].tMin, meteoPoint.obsDataD[index].tMax);
                        meteoPoint.obsDataD[index].tAvg = res;
                    }
                    break;
                }
        case dailyReferenceEvapotranspirationHS:
        {
            quality::qualityType qualityEtp = qualityCheck.syntacticQualitySingleValue(dailyReferenceEvapotranspirationHS, meteoPoint.obsDataD[index].et0_hs);
            if (qualityEtp == quality::accepted)
            {
                res = meteoPoint.obsDataD[index].et0_hs;
            }
            else
            {
                res = dailyEtpHargreaves(meteoPoint.obsDataD[index].tMin, meteoPoint.obsDataD[index].tMax, getCrit3DDate(myDate), meteoPoint.latitude, meteoSettings);
                meteoPoint.obsDataD[index].et0_hs = res;
            }
            break;
        }
        case dailyHeatingDegreeDays:
        {
            quality::qualityType qualityTavg = qualityCheck.syntacticQualitySingleValue(dailyAirTemperatureAvg, meteoPoint.obsDataD[index].tAvg);
            if (qualityTavg == quality::accepted)
            {
                res = 0;
                if ( meteoPoint.obsDataD[index].tAvg < DDHEATING_THRESHOLD)
                {
                    res = DDHEATING_THRESHOLD - meteoPoint.obsDataD[index].tAvg;
                }
            }
            else
            {
                meteoPoint.obsDataD[index].tAvg = dailyAverageT(meteoPoint.obsDataD[index].tMin, meteoPoint.obsDataD[index].tMax);
                qualityTavg = qualityCheck.syntacticQualitySingleValue(dailyAirTemperatureAvg, meteoPoint.obsDataD[index].tAvg);
                if (qualityTavg == quality::accepted)
                {
                    res = 0;
                    if ( meteoPoint.obsDataD[index].tAvg < DDHEATING_THRESHOLD)
                    {
                        res = DDHEATING_THRESHOLD - meteoPoint.obsDataD[index].tAvg;
                    }
                }
                else
                {
                    res = NODATA;
                }
            }
            meteoPoint.obsDataD[index].dd_heating = res;
            break;
        }
        case dailyCoolingDegreeDays:
        {
            quality::qualityType qualityTavg = qualityCheck.syntacticQualitySingleValue(dailyAirTemperatureAvg, meteoPoint.obsDataD[index].tAvg);
            if (qualityTavg == quality::accepted)
            {
                res = 0;
                if ( meteoPoint.obsDataD[index].tAvg > DDCOOLING_THRESHOLD)
                {
                    res = meteoPoint.obsDataD[index].tAvg - DDCOOLING_SUBTRACTION;
                }
            }
            else
            {
                meteoPoint.obsDataD[index].tAvg = dailyAverageT(meteoPoint.obsDataD[index].tMin, meteoPoint.obsDataD[index].tMax);
                qualityTavg = qualityCheck.syntacticQualitySingleValue(dailyAirTemperatureAvg, meteoPoint.obsDataD[index].tAvg);
                if (qualityTavg == quality::accepted)
                {
                    res = 0;
                    if ( meteoPoint.obsDataD[index].tAvg > DDCOOLING_THRESHOLD)
                    {
                        res = meteoPoint.obsDataD[index].tAvg - DDCOOLING_SUBTRACTION;
                    }
                }
                else
                {
                    res = NODATA;
                }
            }
            meteoPoint.obsDataD[index].dd_heating = res;
            break;
        }
        default:
                res = NODATA;
                break;
        }

        if (res != NODATA)
        {
            nrValidValues += 1;
        }

        outputValues.push_back(res);
    }

    *percValue = nrValidValues / meteoPoint.nrObsDataDaysD;
    if (nrValidValues > 0)
        return true;
    else
        return false;
}


bool aggregatedHourlyToDaily(meteoVariable myVar, Crit3DMeteoPoint* meteoPoint, Crit3DDate dateIni, Crit3DDate dateFin, Crit3DMeteoSettings *meteoSettings)
{
    Crit3DDate date;
    std::vector <float> values;
    float value, dailyValue;
    short hour;
    meteoVariable hourlyVar = noMeteoVar;
    meteoComputation elab = noMeteoComp;
    float param = NODATA;
    int nValidValues;

    if (meteoPoint->nrObsDataDaysD == 0)
        meteoPoint->initializeObsDataD(dateIni.daysTo(dateFin)+1, dateIni);

    switch(myVar)
    {
        case dailyAirTemperatureAvg:
            hourlyVar = airTemperature;
            elab = average;
            break;

        case dailyAirTemperatureMax:
            hourlyVar = airTemperature;
            elab = maxInList;
            break;

        case dailyAirTemperatureMin:
            hourlyVar = airTemperature;
            elab = minInList;
            break;

        case dailyPrecipitation:
            hourlyVar = precipitation;
            elab = sum;
            break;

        case dailyAirRelHumidityAvg:
            hourlyVar = airRelHumidity;
            elab = average;
            break;

        case dailyAirRelHumidityMax:
            hourlyVar = airRelHumidity;
            elab = maxInList;
            break;

        case dailyAirRelHumidityMin:
            hourlyVar = airRelHumidity;
            elab = minInList;
            break;

        case dailyGlobalRadiation:
            hourlyVar = globalIrradiance;
            elab = timeIntegration;
            param = float(0.003600);
            break;

        case dailyWindScalarIntensityAvg:
            hourlyVar = windScalarIntensity;
            elab = average;
            break;

        case dailyWindScalarIntensityMax:
            hourlyVar = windScalarIntensity;
            elab = maxInList;
            break;

        case dailyReferenceEvapotranspirationPM:
            hourlyVar = referenceEvapotranspiration;
            elab = sum;
            break;

        case dailyLeafWetness:
            hourlyVar = leafWetness;
            elab = sum;
            break;

        default:
            hourlyVar = noMeteoVar;
            break;
    }

    if (hourlyVar == noMeteoVar || elab == noMeteoComp) return false;

    for (date = dateIni; date <= dateFin; date = date.addDays(1))
    {
        dailyValue = NODATA;
        value = NODATA;
        values.clear();
        nValidValues = 0;

        for (hour = 1; hour <= 24; hour++)
        {
            value = meteoPoint->getMeteoPointValueH(date, hour, 0, hourlyVar);
            if (int(value) != NODATA)
            {
                values.push_back(value);
                nValidValues = nValidValues + 1;
            }
        }

        float validPercentage = (float(nValidValues) / float(24)) * 100;
        if (validPercentage < meteoSettings->getMinimumPercentage())
        {
            dailyValue = NODATA;
        }
        else
        {
            dailyValue = statisticalElab(elab, param, values, values.size(), NODATA);
        }
        meteoPoint->setMeteoPointValueD(date, myVar, dailyValue);

        if (myVar == dailyLeafWetness && dailyValue > 24)
        {
            // todo warning
        }

    }

    return true;

}

std::vector<float> aggregatedHourlyToDailyList(meteoVariable myVar, Crit3DMeteoPoint* meteoPoint, Crit3DDate dateIni, Crit3DDate dateFin, Crit3DMeteoSettings *meteoSettings)
{

    Crit3DDate date;
    std::vector <float> values;
    std::vector<float> dailyData;
    float value, dailyValue;
    short hour;
    meteoVariable hourlyVar = noMeteoVar;
    meteoComputation elab = noMeteoComp;
    float param = NODATA;
    int nValidValues;

    if (meteoPoint->nrObsDataDaysD == 0)
        meteoPoint->initializeObsDataD(dateIni.daysTo(dateFin)+1, dateIni);

    switch(myVar)
    {
        case dailyAirTemperatureAvg:
            hourlyVar = airTemperature;
            elab = average;
            break;

        case dailyAirTemperatureMax:
            hourlyVar = airTemperature;
            elab = maxInList;
            break;

        case dailyAirTemperatureMin:
            hourlyVar = airTemperature;
            elab = minInList;
            break;

        case dailyPrecipitation:
            hourlyVar = precipitation;
            elab = sum;
            break;

        case dailyAirRelHumidityAvg:
            hourlyVar = airRelHumidity;
            elab = average;
            break;

        case dailyAirRelHumidityMax:
            hourlyVar = airRelHumidity;
            elab = maxInList;
            break;

        case dailyAirRelHumidityMin:
            hourlyVar = airRelHumidity;
            elab = minInList;
            break;

        case dailyGlobalRadiation:
            hourlyVar = globalIrradiance;
            elab = timeIntegration;
            param = float(0.003600);
            break;

        case dailyWindScalarIntensityAvg:
            hourlyVar = windScalarIntensity;
            elab = average;
            break;

        case dailyWindScalarIntensityMax:
            hourlyVar = windScalarIntensity;
            elab = maxInList;
            break;

        case dailyReferenceEvapotranspirationPM:
            hourlyVar = referenceEvapotranspiration;
            elab = sum;
            break;

        case dailyLeafWetness:
            hourlyVar = leafWetness;
            elab = sum;
            break;

        default:
            hourlyVar = noMeteoVar;
            break;
    }

    if (hourlyVar == noMeteoVar || elab == noMeteoComp) return dailyData;

    for (date = dateIni; date <= dateFin; date = date.addDays(1))
    {
        dailyValue = NODATA;
        value = NODATA;
        values.clear();
        nValidValues = 0;

        for (hour = 1; hour <= 24; hour++)
        {
            value = meteoPoint->getMeteoPointValueH(date, hour, 0, hourlyVar);
            if (int(value) != NODATA)
            {
                values.push_back(value);
                nValidValues = nValidValues + 1;
            }
        }

        float validPercentage = (float(nValidValues) / float(24)) * 100;
        if (validPercentage < meteoSettings->getMinimumPercentage())
        {
            dailyData.push_back(NODATA);
        }
        else
        {
            dailyValue = statisticalElab(elab, param, values, values.size(), NODATA);
            dailyData.push_back(dailyValue);
        }

        if (myVar == dailyLeafWetness && dailyValue > 24)
        {
            // todo warning
        }

    }

    return dailyData;

}

bool preElaboration(QString *myError, Crit3DMeteoPointsDbHandler* meteoPointsDbHandler, Crit3DMeteoGridDbHandler* meteoGridDbHandler, Crit3DMeteoPoint* meteoPoint, bool isMeteoGrid, meteoVariable variable, meteoComputation elab1,
    QDate startDate, QDate endDate, std::vector<float> &outputValues, float* percValue, Crit3DMeteoSettings* meteoSettings)
{
    bool preElaboration = false;
    bool automaticETP = meteoSettings->getAutomaticET0HS();
    bool automaticTmed = meteoSettings->getAutomaticTavg();

    switch(variable)
    {

        case dailyLeafWetness:
        {
            if ( loadHourlyVarSeries(myError, meteoPointsDbHandler, meteoGridDbHandler, meteoPoint, isMeteoGrid, leafWetness, QDateTime(startDate,QTime(1,0,0),Qt::UTC), QDateTime(endDate.addDays(1),QTime(0,0,0),Qt::UTC)) > 0)
            {
                preElaboration = elaborateDailyAggregatedVar(dailyLeafWetness, *meteoPoint, outputValues, percValue, meteoSettings);
            }
            break;
        }
        case dailyTemperatureHoursAbove:
        {
            if ( loadHourlyVarSeries(myError, meteoPointsDbHandler, meteoGridDbHandler, meteoPoint, isMeteoGrid, airTemperature, QDateTime(startDate,QTime(1,0,0),Qt::UTC), QDateTime(endDate.addDays(1),QTime(0,0,0),Qt::UTC)) > 0)
            {
                preElaboration = elaborateDailyAggregatedVar(dailyTemperatureHoursAbove, *meteoPoint, outputValues, percValue, meteoSettings);
            }
            break;
        }
        case dailyThomDaytime:
        {
            if ( loadDailyVarSeries(myError, meteoPointsDbHandler, meteoGridDbHandler, meteoPoint, isMeteoGrid, dailyAirRelHumidityMin, startDate, endDate) > 0)
            {
                if (loadDailyVarSeries(myError, meteoPointsDbHandler, meteoGridDbHandler, meteoPoint, isMeteoGrid, dailyAirTemperatureMax, startDate, endDate) > 0)
                {
                    preElaboration = elaborateDailyAggregatedVar(dailyThomDaytime, *meteoPoint, outputValues, percValue, meteoSettings);
                }
            }
            break;
        }

        case dailyThomNighttime:
        {
            if (loadDailyVarSeries(myError, meteoPointsDbHandler, meteoGridDbHandler, meteoPoint, isMeteoGrid, dailyAirRelHumidityMax, startDate, endDate) > 0)
            {
                if (loadDailyVarSeries(myError, meteoPointsDbHandler, meteoGridDbHandler, meteoPoint, isMeteoGrid, dailyAirTemperatureMin, startDate, endDate) > 0)
                {
                    preElaboration = elaborateDailyAggregatedVar(dailyThomNighttime, *meteoPoint, outputValues, percValue, meteoSettings);
                }
            }
            break;
        }
        case dailyThomAvg: case dailyThomMax: case dailyThomHoursAbove:
        {

            if (loadHourlyVarSeries(myError, meteoPointsDbHandler, meteoGridDbHandler, meteoPoint, isMeteoGrid, airTemperature, QDateTime(startDate,QTime(1,0,0),Qt::UTC), QDateTime(endDate.addDays(1),QTime(0,0,0),Qt::UTC)) > 0)
            {
                if (loadHourlyVarSeries(myError, meteoPointsDbHandler, meteoGridDbHandler, meteoPoint, isMeteoGrid, airRelHumidity, QDateTime(startDate,QTime(1,0,0),Qt::UTC), QDateTime(endDate.addDays(1),QTime(0,0,0),Qt::UTC))  > 0)
                {
                    preElaboration = elaborateDailyAggregatedVar(variable, *meteoPoint, outputValues, percValue, meteoSettings);
                }
            }
            break;
        }
        case dailyBIC:
        {

            if (loadDailyVarSeries(myError, meteoPointsDbHandler, meteoGridDbHandler, meteoPoint, isMeteoGrid, dailyReferenceEvapotranspirationHS, startDate, endDate) > 0)
            {
                preElaboration = true;
            }
            else if (automaticETP)
            {
                if (loadDailyVarSeries(myError, meteoPointsDbHandler, meteoGridDbHandler, meteoPoint, isMeteoGrid, dailyAirTemperatureMin, startDate, endDate) > 0)
                {
                    if (loadDailyVarSeries(myError, meteoPointsDbHandler, meteoGridDbHandler, meteoPoint, isMeteoGrid, dailyAirTemperatureMax, startDate, endDate) > 0)
                    {
                        preElaboration = elaborateDailyAggregatedVar(dailyReferenceEvapotranspirationHS, *meteoPoint, outputValues, percValue, meteoSettings);
                        for (int outputIndex = 0; outputIndex<outputValues.size(); outputIndex++)
                        {
                            meteoPoint->obsDataD[outputIndex].et0_hs = outputValues[outputIndex];
                        }
                    }
                }
            }

            if (preElaboration)
            {
                preElaboration = false;
                if (loadDailyVarSeries(myError, meteoPointsDbHandler, meteoGridDbHandler, meteoPoint, isMeteoGrid, dailyPrecipitation, startDate, endDate) > 0)
                {
                    preElaboration = elaborateDailyAggregatedVar(dailyBIC, *meteoPoint, outputValues, percValue, meteoSettings);
                }
            }
            break;
        }

        case dailyAirTemperatureRange:
        {
            if (loadDailyVarSeries(myError, meteoPointsDbHandler, meteoGridDbHandler, meteoPoint, isMeteoGrid, dailyAirTemperatureMin, startDate, endDate) > 0)
            {
                if (loadDailyVarSeries(myError, meteoPointsDbHandler, meteoGridDbHandler, meteoPoint, isMeteoGrid, dailyAirTemperatureMax, startDate, endDate) > 0)
                {
                    preElaboration = elaborateDailyAggregatedVar(dailyAirTemperatureRange, *meteoPoint, outputValues, percValue, meteoSettings);
                }
            }
            break;
        }

        case dailyAirTemperatureAvg:
        {
            if (loadDailyVarSeries_SaveOutput(myError, meteoPointsDbHandler, meteoGridDbHandler, meteoPoint, isMeteoGrid, dailyAirTemperatureAvg, startDate, endDate, outputValues) > 0)
            {
                preElaboration = true;
            }
            else if (automaticTmed)
            {
                if (loadDailyVarSeries(myError, meteoPointsDbHandler, meteoGridDbHandler, meteoPoint, isMeteoGrid, dailyAirTemperatureMin, startDate, endDate) > 0 )
                {
                    if (loadDailyVarSeries(myError, meteoPointsDbHandler, meteoGridDbHandler, meteoPoint, isMeteoGrid, dailyAirTemperatureMax, startDate, endDate) > 0)
                    {
                        preElaboration = elaborateDailyAggregatedVar(dailyAirTemperatureAvg, *meteoPoint, outputValues, percValue, meteoSettings);
                    }
                }
            }
            break;
        }

        case dailyReferenceEvapotranspirationHS:
        {
            if (loadDailyVarSeries_SaveOutput(myError, meteoPointsDbHandler, meteoGridDbHandler, meteoPoint, isMeteoGrid, dailyReferenceEvapotranspirationHS, startDate, endDate, outputValues) > 0)
            {
                preElaboration = true;
            }
            else if (automaticETP)
            {
                if (loadDailyVarSeries(myError, meteoPointsDbHandler, meteoGridDbHandler, meteoPoint, isMeteoGrid, dailyAirTemperatureMin, startDate, endDate) > 0)
                {
                    if (loadDailyVarSeries(myError, meteoPointsDbHandler, meteoGridDbHandler, meteoPoint, isMeteoGrid, dailyAirTemperatureMax, startDate, endDate) > 0)
                    {
                        preElaboration = elaborateDailyAggregatedVar(dailyReferenceEvapotranspirationHS, *meteoPoint, outputValues, percValue, meteoSettings);
                    }
                }
            }
            break;
        }
        case dailyHeatingDegreeDays:
        {
            if (loadDailyVarSeries_SaveOutput(myError, meteoPointsDbHandler, meteoGridDbHandler, meteoPoint, isMeteoGrid, dailyHeatingDegreeDays, startDate, endDate, outputValues) > 0)
            {
                preElaboration = true;
            }
            else
            {
                if (loadDailyVarSeries(myError, meteoPointsDbHandler, meteoGridDbHandler, meteoPoint, isMeteoGrid, dailyAirTemperatureAvg, startDate, endDate) > 0)
                {
                    preElaboration = elaborateDailyAggregatedVar(dailyHeatingDegreeDays, *meteoPoint, outputValues, percValue, meteoSettings);
                }
                else
                {
                    if (loadDailyVarSeries(myError, meteoPointsDbHandler, meteoGridDbHandler, meteoPoint, isMeteoGrid, dailyAirTemperatureMin, startDate, endDate) > 0)
                    {
                        if (loadDailyVarSeries(myError, meteoPointsDbHandler, meteoGridDbHandler, meteoPoint, isMeteoGrid, dailyAirTemperatureMax, startDate, endDate) > 0)
                        {
                            preElaboration = elaborateDailyAggregatedVar(dailyHeatingDegreeDays, *meteoPoint, outputValues, percValue, meteoSettings);
                        }
                    }
                }
            }
            break;
        }
        case dailyCoolingDegreeDays:
        {
            if (loadDailyVarSeries_SaveOutput(myError, meteoPointsDbHandler, meteoGridDbHandler, meteoPoint, isMeteoGrid, dailyCoolingDegreeDays, startDate, endDate, outputValues) > 0)
            {
                preElaboration = true;
            }
            else
            {
                if (loadDailyVarSeries(myError, meteoPointsDbHandler, meteoGridDbHandler, meteoPoint, isMeteoGrid, dailyAirTemperatureAvg, startDate, endDate) > 0)
                {
                    preElaboration = elaborateDailyAggregatedVar(dailyCoolingDegreeDays, *meteoPoint, outputValues, percValue, meteoSettings);
                }
                else
                {
                    if (loadDailyVarSeries(myError, meteoPointsDbHandler, meteoGridDbHandler, meteoPoint, isMeteoGrid, dailyAirTemperatureMin, startDate, endDate) > 0)
                    {
                        if (loadDailyVarSeries(myError, meteoPointsDbHandler, meteoGridDbHandler, meteoPoint, isMeteoGrid, dailyAirTemperatureMax, startDate, endDate) > 0)
                        {
                            preElaboration = elaborateDailyAggregatedVar(dailyCoolingDegreeDays, *meteoPoint, outputValues, percValue, meteoSettings);
                        }
                    }
                }
            }
            break;
        }

        default:
        {
            switch(elab1)
            {
                case huglin:
                {
                    if (loadDailyVarSeries(myError, meteoPointsDbHandler, meteoGridDbHandler, meteoPoint, isMeteoGrid, dailyAirTemperatureMin, startDate, endDate) > 0 )
                    {
                        if (loadDailyVarSeries(myError, meteoPointsDbHandler, meteoGridDbHandler, meteoPoint, isMeteoGrid, dailyAirTemperatureMax, startDate, endDate) > 0 )
                        {
                            if (loadDailyVarSeries(myError, meteoPointsDbHandler, meteoGridDbHandler, meteoPoint, isMeteoGrid, dailyAirTemperatureAvg, startDate, endDate) > 0 )
                            {
                                preElaboration = true;
                            }
                            else if (automaticTmed)
                            {
                                preElaboration = elaborateDailyAggregatedVar(dailyAirTemperatureAvg, *meteoPoint, outputValues, percValue, meteoSettings);
                            }
                        }
                    }
                    break;
                }

            case winkler: case correctedDegreeDaysSum: case fregoni:
            {
                if (loadDailyVarSeries(myError, meteoPointsDbHandler, meteoGridDbHandler, meteoPoint, isMeteoGrid, dailyAirTemperatureMin, startDate, endDate) > 0 )
                {
                    if (loadDailyVarSeries(myError, meteoPointsDbHandler, meteoGridDbHandler, meteoPoint, isMeteoGrid, dailyAirTemperatureMax, startDate, endDate) > 0 )
                    {
                        preElaboration = true;
                    }
                }
                break;
            }

            case phenology:
            {
                if (loadDailyVarSeries(myError, meteoPointsDbHandler, meteoGridDbHandler, meteoPoint, isMeteoGrid, dailyAirTemperatureMin, startDate, endDate) > 0 )
                {
                    if (loadDailyVarSeries(myError, meteoPointsDbHandler, meteoGridDbHandler, meteoPoint, isMeteoGrid, dailyAirTemperatureMax, startDate, endDate) > 0 )
                    {
                        if (loadDailyVarSeries(myError, meteoPointsDbHandler, meteoGridDbHandler, meteoPoint, isMeteoGrid, dailyPrecipitation, startDate, endDate) > 0 )
                        {
                            preElaboration = true;
                        }
                    }
                }
                break;
             }

            default:
            {
                *percValue = loadDailyVarSeries_SaveOutput(myError, meteoPointsDbHandler, meteoGridDbHandler, meteoPoint, isMeteoGrid, variable, startDate, endDate, outputValues);

                preElaboration = ((*percValue) > 0);
                break;
            }

            }
            break;
        }
    }

    return preElaboration;
}

bool preElaborationWithoutLoad(Crit3DMeteoPoint* meteoPoint, meteoVariable variable, QDate startDate, QDate endDate, std::vector<float> &outputValues, float* percValue, Crit3DMeteoSettings* meteoSettings)
{

    bool preElaboration = false;
    bool automaticETP = meteoSettings->getAutomaticET0HS();
    bool automaticTmed = meteoSettings->getAutomaticTavg();

    switch(variable)
    {

        case dailyLeafWetness:
        {
            preElaboration = elaborateDailyAggrVarFromStartDate(dailyLeafWetness, *meteoPoint, startDate, endDate, outputValues, percValue, meteoSettings);
            break;
        }

        case dailyThomDaytime:
        {
            preElaboration = elaborateDailyAggrVarFromStartDate(dailyThomDaytime, *meteoPoint, startDate, endDate, outputValues, percValue, meteoSettings);
            break;
        }

        case dailyThomNighttime:
        {
            preElaboration = elaborateDailyAggrVarFromStartDate(dailyThomNighttime, *meteoPoint, startDate, endDate, outputValues, percValue, meteoSettings);
            break;
        }
        case dailyThomAvg: case dailyThomMax: case dailyThomHoursAbove:
        {
            preElaboration = elaborateDailyAggrVarFromStartDate(variable, *meteoPoint, startDate, endDate, outputValues, percValue, meteoSettings);
            break;
        }
        case dailyBIC:
        {
            if (automaticETP)
            {
                preElaboration = elaborateDailyAggrVarFromStartDate(dailyReferenceEvapotranspirationHS, *meteoPoint, startDate, endDate, outputValues, percValue, meteoSettings);
                for (int outputIndex = 0; outputIndex<outputValues.size(); outputIndex++)
                {
                    meteoPoint->obsDataD[outputIndex].et0_hs = outputValues[outputIndex];
                }
            }

            if (preElaboration)
            {
                preElaboration = elaborateDailyAggrVarFromStartDate(dailyBIC, *meteoPoint, startDate, endDate, outputValues, percValue, meteoSettings);
            }
            break;
        }

        case dailyAirTemperatureRange:
        {
            preElaboration = elaborateDailyAggrVarFromStartDate(dailyAirTemperatureRange, *meteoPoint, startDate, endDate, outputValues, percValue, meteoSettings);
            break;
        }

        case dailyAirTemperatureAvg:
        {
            if (loadFromMp_SaveOutput(meteoPoint, dailyAirTemperatureAvg, startDate, endDate, outputValues) > 0)
            {
                preElaboration = true;
            }
            else if (automaticTmed)
            {
                outputValues.clear();
                preElaboration = elaborateDailyAggrVarFromStartDate(dailyAirTemperatureAvg, *meteoPoint, startDate, endDate, outputValues, percValue, meteoSettings);
            }
            break;
        }

        case dailyReferenceEvapotranspirationHS:
        {
            if (loadFromMp_SaveOutput(meteoPoint, dailyReferenceEvapotranspirationHS, startDate, endDate, outputValues) > 0)
            {
                preElaboration = true;
            }
            else if (automaticETP)
            {
                outputValues.clear();
                preElaboration = elaborateDailyAggrVarFromStartDate(dailyReferenceEvapotranspirationHS, *meteoPoint, startDate, endDate, outputValues, percValue, meteoSettings);
            }
            break;
        }
        case dailyHeatingDegreeDays:
        {
            if (loadFromMp_SaveOutput(meteoPoint, dailyHeatingDegreeDays, startDate, endDate, outputValues) > 0)
            {
                preElaboration = true;
            }
            else
            {
                outputValues.clear();
                preElaboration = elaborateDailyAggrVarFromStartDate(dailyHeatingDegreeDays, *meteoPoint, startDate, endDate, outputValues, percValue, meteoSettings);
            }
            break;
        }
        case dailyCoolingDegreeDays:
        {
            if (loadFromMp_SaveOutput(meteoPoint, dailyCoolingDegreeDays, startDate, endDate, outputValues) > 0)
            {
                preElaboration = true;
            }
            else
            {
                outputValues.clear();
                preElaboration = elaborateDailyAggrVarFromStartDate(dailyCoolingDegreeDays, *meteoPoint, startDate, endDate, outputValues, percValue, meteoSettings);
            }
            break;
        }

        default:
        {
            *percValue = loadFromMp_SaveOutput(meteoPoint, variable, startDate, endDate, outputValues);

            preElaboration = ((*percValue) > 0);
            break;
        }
    }

    return preElaboration;
}

void extractValidValuesCC(std::vector<float> &outputValues)
{

    for (unsigned int i = 0;  i < outputValues.size(); i++)
    {
        if (outputValues[i] == NODATA)
        {
            outputValues.erase(outputValues.begin()+i);
        }
    }

}

void extractValidValuesWithThreshold(std::vector<float> &outputValues, float myThreshold)
{

    for (unsigned int i = 0;  i < outputValues.size(); i++)
    {
        if (outputValues[i] == NODATA || outputValues[i] < myThreshold)
        {
            outputValues.erase(outputValues.begin()+i);
        }
    }

}


//nYears   = 0         same year
//nYears   = 1,2,3...   betweend years 1,2,3...
float computeStatistic(std::vector<float> &inputValues, Crit3DMeteoPoint* meteoPoint, Crit3DClimate *clima, Crit3DDate firstDate, Crit3DDate lastDate, int nYears, meteoComputation elab1, meteoComputation elab2, Crit3DMeteoSettings* meteoSettings, bool dataAlreadyLoaded)
{

    std::vector<float> values;
    std::vector<float> valuesSecondElab;
    std::vector<int> valuesYearsPrimaryElab;
    Crit3DDate presentDate;
    int numberOfDays;
    int nValidValues = 0;
    int nValues = 0;
    unsigned int index;

    float primary = NODATA;

    int firstYear = clima->yearStart();
    int lastYear = clima->yearEnd();
    float param1 = clima->param1();
    float param2 = clima->param2();

    // no secondary elab
    if (elab2 == noMeteoComp)
    {
        switch(elab1)
        {
            case lastDayBelowThreshold:
            {
                return computeLastDayBelowThreshold(inputValues, meteoPoint->obsDataD[0].date ,firstDate, lastDate, param1);
            }
            case winkler:
            {
                return computeWinkler(meteoPoint, firstDate, lastDate, meteoSettings->getMinimumPercentage());
            }
            case huglin:
            {
                return computeHuglin(meteoPoint, firstDate, lastDate, meteoSettings->getMinimumPercentage());
            }
            case fregoni:
            {
                return computeFregoni(meteoPoint, firstDate, lastDate, meteoSettings->getMinimumPercentage());
            }
            case correctedDegreeDaysSum:
            {
                return computeCorrectedSum(meteoPoint, firstDate, lastDate, param1, meteoSettings->getMinimumPercentage());
            }
            default:
            {
                int dayOfYear = getDoyFromDate(firstDate);

                for (int presentYear = firstYear; presentYear <= lastYear; presentYear++)
                {
                    if ( (clima->getCurrentPeriodType() == dailyPeriod) )
                    {
                        firstDate = getDateFromDoy(presentYear, dayOfYear);
                    }
                    firstDate.year = presentYear;
                    lastDate.year = presentYear;

                    if (nYears != NODATA)
                    {
                        if (nYears < 0)
                        {
                            firstDate.year = (presentYear + nYears);
                        }
                        else if (nYears > 0)
                        {
                            lastDate.year = (presentYear + nYears);
                        }
                    }

                    if ( (clima->getCurrentPeriodType() == dailyPeriod) )
                    {
                        numberOfDays = 1;
                    }
                    else
                    {
                        numberOfDays = difference(firstDate, lastDate) +1;
                    }

                    presentDate = firstDate;
                    for (int i = 0; i < numberOfDays; i++)
                    {

                        float value = NODATA;

                        if (meteoPoint->obsDataD[0].date > presentDate)
                        {
                            value = NODATA;
                        }
                        else
                        {
                            if (dataAlreadyLoaded)
                            {
                                index = difference(firstDate, presentDate);
                            }
                            else
                            {
                                index = difference(meteoPoint->obsDataD[0].date, presentDate);
                            }
                            if (index >= 0 && index < inputValues.size())
                            {
                                value = inputValues.at(index);
                            }
                        }

                        if (int(value) != NODATA)
                        {
                            values.push_back(value);
                            nValidValues = nValidValues + 1;
                        }

                        nValues = nValues + 1;

                        presentDate = presentDate.addDays(1);

                    }
                }

                if (nValidValues == 0)
                    return NODATA;

                float validPercentage = (float(nValidValues) / float(nValues)) * 100;
                if (validPercentage < meteoSettings->getMinimumPercentage())
                    return NODATA;

                return statisticalElab(elab1, param1, values, nValidValues, meteoSettings->getRainfallThreshold());
            }
        }
    }
    // secondary elab
    else
    {
        int nTotYears = 0;
        int nValidYears = 0;
        valuesSecondElab.clear();

        int dayOfYear = getDoyFromDate(firstDate);

        for (int presentYear = firstYear; presentYear <= lastYear; presentYear++)
        {

            if ( (clima->getCurrentPeriodType() == dailyPeriod) )
            {
                firstDate = getDateFromDoy(presentYear, dayOfYear);
            }

            firstDate.year = presentYear;
            lastDate.year = presentYear;

            if (nYears < 0)
            {
                firstDate.year = (presentYear + nYears);
            }
            else if (nYears > 0)
            {
                lastDate.year = (presentYear + nYears);
            }
            primary = NODATA;

            nValues = 0;
            nValidValues = 0;
            values.clear();

            switch(elab1)
            {
                case lastDayBelowThreshold:
                {
                    primary = computeLastDayBelowThreshold(inputValues, meteoPoint->obsDataD[0].date ,firstDate, lastDate, param1);
                    break;
                }
                case winkler:
                {
                    primary = computeWinkler(meteoPoint, firstDate, lastDate, meteoSettings->getMinimumPercentage());
                    break;
                }
                case huglin:
                {
                    primary = computeHuglin(meteoPoint, firstDate, lastDate, meteoSettings->getMinimumPercentage());
                    break;
                }
                case fregoni:
                {
                    primary = computeFregoni(meteoPoint, firstDate, lastDate, meteoSettings->getMinimumPercentage());
                    break;
                }
                case correctedDegreeDaysSum:
                {
                    primary = computeCorrectedSum(meteoPoint, firstDate, lastDate, param1, meteoSettings->getMinimumPercentage());
                    break;
                }
                default:
                {

                    if ( (clima->getCurrentPeriodType() == dailyPeriod) )
                    {
                        numberOfDays = 1;
                    }
                    else
                    {
                        numberOfDays = difference(firstDate, lastDate) +1;
                    }
                    presentDate = firstDate;
                    for (int i = 0; i < numberOfDays; i++)
                    {
                        float value = NODATA;

                        if (meteoPoint->obsDataD[0].date > presentDate)
                        {
                            value = NODATA;
                        }
                        else
                        {
                            index = difference(meteoPoint->obsDataD[0].date, presentDate);
                            if (index >= 0 && index < inputValues.size())
                            {
                                value = inputValues.at(index);
                            }
                        }

                        if (int(value) != NODATA)
                        {
                            values.push_back(value);
                            nValidValues = nValidValues + 1;
                        }


                        nValues = nValues + 1;
                        presentDate = presentDate.addDays(1);

                    }

                    if (nValidValues > 0)
                    {
                        if (float(nValidValues) / float(nValues) * 100.f >= meteoSettings->getMinimumPercentage())
                        {
                            primary = statisticalElab(elab1, param1, values, nValidValues, meteoSettings->getRainfallThreshold());
                        }
                    }

                    break;

                }
            }

            if (primary != NODATA)
            {
                valuesSecondElab.push_back(primary);
                valuesYearsPrimaryElab.push_back(presentYear);
                nValidYears = nValidYears + 1;
            }

            nTotYears = nTotYears + 1;

        } // end for

        if (nTotYears == 0)
        {
            return NODATA;
        }
        else if (float(nValidYears) / float(nTotYears) * 100.f < meteoSettings->getMinimumPercentage())
        {
            return NODATA;
        }
        else
        {
            switch(elab2)
            {
                case yearMax: case yearMin:
                {
                    int index = statisticalElab(elab2, firstYear, valuesSecondElab, nValidYears, meteoSettings->getRainfallThreshold());
                    if (index != NODATA && index < valuesYearsPrimaryElab.size())
                    {
                        return valuesYearsPrimaryElab[index];
                    }
                    else
                        return NODATA;
                }
                case trend:
                    return statisticalElab(elab2, firstYear, valuesSecondElab, nValidYears, meteoSettings->getRainfallThreshold());
                default:
                    return statisticalElab(elab2, param2, valuesSecondElab, nValidYears, meteoSettings->getRainfallThreshold());
            }
        }
    }
}

QString getTable(QString elab)
{
    QList<QString> words = elab.split('_');
    QString periodTypeStr = words[2];
    QString tableName = "climate_"+periodTypeStr.toLower();

    return tableName;
}

int getClimateIndexFromElab(QDate myDate, QString elab)
{

    QList<QString> words = elab.split('_');
    QString periodTypeStr = words[2];

    period periodType = getPeriodTypeFromString(periodTypeStr);

    switch(periodType)
    {
    case annualPeriod: case genericPeriod:
            return 1;
    case decadalPeriod:
            return decadeFromDate(myDate);
    case monthlyPeriod:
            return myDate.month();
    case seasonalPeriod:
            return getSeasonFromDate(myDate);
    case dailyPeriod:
            return myDate.dayOfYear();
    default:
            return NODATA;
    }
}

int getNumberClimateIndexFromElab(QString elab)
{

    QList<QString> words = elab.split('_');
    QString periodTypeStr = words[2];

    period periodType = getPeriodTypeFromString(periodTypeStr);

    switch(periodType)
    {
    case annualPeriod: case genericPeriod:
            return 1;
    case decadalPeriod:
            return 36;
    case monthlyPeriod:
            return 12;
    case seasonalPeriod:
            return 4;
    case dailyPeriod:
            return 366;
    default:
            return NODATA;
    }
}

period getPeriodTypeFromString(QString periodStr)
{

    if (periodStr == "Daily")
        return dailyPeriod;
    if (periodStr == "Decadal")
        return decadalPeriod;
    if (periodStr == "Monthly")
        return monthlyPeriod;
    if (periodStr == "Seasonal")
        return seasonalPeriod;
    if (periodStr == "Annual")
        return annualPeriod;
    if (periodStr == "Generic")
        return genericPeriod;

    return noPeriodType;

}

int nParameters(meteoComputation elab)
{
    switch(elab)
    {
    case average:
        return 0;
    case maxInList:
        return 0;
    case minInList:
        return 0;
    case sum:
        return 0;
    case avgAbove:
        return 1;
    case stdDevAbove:
        return 1;
    case sumAbove:
        return 1;
    case daysAbove:
        return 1;
    case daysBelow:
        return 1;
    case consecutiveDaysAbove:
        return 1;
    case consecutiveDaysBelow:
        return 1;
    case percentile:
        return 1;
    case prevailingWindDir:
        return 0;
    case correctedDegreeDaysSum:
        return 1;
    case trend:
        return 0;
    case mannKendall:
        return 0;
    case differenceWithThreshold:
        return 1;
    case lastDayBelowThreshold:
        return 1;
    case yearMax:
        return 0;
    case yearMin:
        return 0;
    default:
        return 0;
    }

}

bool parseXMLElaboration(Crit3DElabList *listXMLElab, Crit3DAnomalyList *listXMLAnomaly, Crit3DDroughtList *listXMLDrought, Crit3DPhenologyList *listXMLPhenology, QString xmlFileName, QString *myError)
{

    QDomDocument xmlDoc;

    // check
    if (xmlFileName == "")
    {
        *myError = "Missing XML file.";
        return false;
    }

    QFile myFile(xmlFileName);
    if (!myFile.open(QIODevice::ReadOnly))
    {
        *myError = "Open XML failed:\n" + xmlFileName + "\n" + myFile.errorString();
        return (false);
    }

    int myErrLine, myErrColumn;
    if (!xmlDoc.setContent(&myFile, myError, &myErrLine, &myErrColumn))
    {
        QString completeError = "Parse xml failed:" + xmlFileName
                + " Row: " + QString::number(myErrLine)
                + " - Column: " + QString::number(myErrColumn)
                + "\n" + *myError;
       *myError = completeError;
        myFile.close();
        return(false);
    }

    myFile.close();

    QDomNode child;
    QDomNode secondChild;
    QDomNodeList secElab;
    QDomNodeList primaryElab;
    QDomNodeList anomalySecElab;
    QDomNodeList anomalyRefSecElab;
    TXMLvar varTable;

    QDomNode ancestor = xmlDoc.documentElement().firstChild();
    QString myTag;
    QString mySecondTag;

    QString firstYear;
    QString lastYear;
    QString refLastYear;
    QString refFirstYear;
    QString variable;
    QString period;
    QString refPeriod;

    QString elab;
    QString elabParam1;
    QString refElab;
    QString refElab2;
    QString refElabParam1;
    QString refElabParam2;
    bool param1IsClimate = false;
    bool anomalyIsClimate;
    QString elabParam2;
    QString filename;

    int nElab = 0;
    int nAnomaly = 0;
    int nDrought = 0;
    int nPhenology = 0;
    bool errorElab = false;
    bool errorAnomaly = false;
    bool errorDrought = false;
    bool errorPhenology = false;
    bool periodPresent = false;

    while(!ancestor.isNull())
    {
        if (ancestor.toElement().tagName().toUpper() == "ELABORATION")
        {
            qDebug() << "ELABORATION ";
            QString dataTypeAttribute = ancestor.toElement().attribute("Datatype").toUpper();
            if ( dataTypeAttribute == "GRID")
            {
                listXMLElab->setIsMeteoGrid(true);
            }
            else if (dataTypeAttribute == "POINT")
            {
                listXMLElab->setIsMeteoGrid(false);
            }
            else if (dataTypeAttribute.isEmpty() || (dataTypeAttribute != "GRID" && dataTypeAttribute != "POINT"))
            {
                ancestor = ancestor.nextSibling(); // something is wrong, go to next elab
                continue;
            }

            QString cumulated = ancestor.toElement().attribute("dailyCumulated").toUpper();
            if ( cumulated == "TRUE")
            {
                listXMLElab->insertDailyCumulated(true);
            }
            else
            {
                listXMLElab->insertDailyCumulated(false);
            }

            if (parseXMLPeriodType(ancestor, "PeriodType", listXMLElab, listXMLAnomaly, false, false, &period, myError) == false)
            {
                listXMLElab->eraseElement(nElab);
                ancestor = ancestor.nextSibling(); // something is wrong, go to next elab
                qDebug() << "parseXMLPeriodType ";
                continue;
            }

            secElab = ancestor.toElement().elementsByTagName("SecondaryElaboration");
            if (secElab.size() == 0)
            {
                listXMLElab->insertElab2("");       // there is not secondary elab
                listXMLElab->insertParam2(NODATA);
            }

            primaryElab = ancestor.toElement().elementsByTagName("PrimaryElaboration");
            if (primaryElab.size() == 0)
            {
                qDebug() << "NO PRIMARY ELAB ";
                if (listXMLElab->listPeriodType().back() != dailyPeriod)
                {
                    listXMLElab->eraseElement(nElab);
                    ancestor = ancestor.nextSibling(); // something is wrong, go to next elab
                    qDebug() << "no primary elab, period type != dailyPeriod ";
                    continue;
                }
                listXMLElab->insertElab1("");       // there is not primary elab
                listXMLElab->insertParam1(NODATA);
                param1IsClimate = false;
                listXMLElab->insertParam1IsClimate(false);
                listXMLElab->insertParam1ClimateField("");
            }

            child = ancestor.firstChild();
            while( !child.isNull())
            {
                myTag = child.toElement().tagName().toUpper();
                if (myTag == "VARIABLE")
                {
                    variable = child.toElement().text();
                    meteoVariable var = getKeyMeteoVarMeteoMap(MapDailyMeteoVarToString, variable.toStdString());
                    if (var == noMeteoVar)
                    {
                        listXMLElab->eraseElement(nElab);
                        qDebug() << "noMeteoVar ";
                        errorElab = true;
                    }
                    else
                    {
                        listXMLElab->insertVariable(var);
                    }

                }
                if (myTag == "YEARINTERVAL")
                {
                    lastYear = child.toElement().attribute("fin");
                    firstYear = child.toElement().attribute("ini");
                    listXMLElab->insertYearStart(firstYear.toInt());
                    listXMLElab->insertYearEnd(lastYear.toInt());
                    if (checkYears(firstYear, lastYear) == false)
                    {
                        listXMLElab->eraseElement(nElab);
                        qDebug() << "checkYears ";
                        errorElab = true;
                    }
                }
                if (myTag == "PERIOD")
                {
                    periodPresent = true;
                    if (parseXMLPeriodTag(child, listXMLElab, listXMLAnomaly, false, false, period, myError) == false)
                    {
                        listXMLElab->eraseElement(nElab);
                        qDebug() << "parseXMLPeriodTag ";
                        errorElab = true;
                    }
                }
                if (myTag == "PRIMARYELABORATION")
                {
                    if (ancestor.toElement().attribute("readParamFromClimate").toUpper() == "TRUE" || ancestor.toElement().attribute("readParamFromClimate").toUpper() == "YES")
                    {
                        param1IsClimate = true;
                        listXMLElab->insertParam1IsClimate(true);
                        listXMLElab->insertParam1(NODATA);
                    }
                    else
                    {
                        param1IsClimate = false;
                        listXMLElab->insertParam1IsClimate(false);
                        listXMLElab->insertParam1ClimateField("");
                    }
                    elabParam1 = child.toElement().attribute("Param1");

                    if (param1IsClimate)
                    {
                        if (elabParam1.isEmpty())
                        {
                            listXMLElab->eraseElement(nElab);
                            qDebug() << "elabParam1 ";
                            errorElab = true;
                        }
                        else
                        {
                            listXMLElab->insertParam1ClimateField(elabParam1);
                        }
                    }
                    else
                    {
                        if (checkElabParam(child.toElement().text(), elabParam1) == false)
                        {
                            listXMLElab->eraseElement(nElab);
                            qDebug() << "checkElabParam ";
                            errorElab = true;
                        }
                        else if (elabParam1.isEmpty())
                        {
                            listXMLElab->insertParam1(NODATA);
                        }
                        else
                        {
                            listXMLElab->insertParam1(elabParam1.toFloat());
                        }
                    }

                    elab = child.toElement().text();
                    if ( (elab.toUpper() == "HUGLIN" || elab.toUpper() == "WINKLER" || elab.toUpper() == "FREGONI") && secElab.size() == 0 )
                    {
                        listXMLElab->eraseElement(nElab);
                        errorElab = true;
                    }
                    else
                    {
                        if (elab.toUpper() == "MEAN")
                        {
                            elab = "average";
                        }
                        listXMLElab->insertElab1(elab.toLower());
                    }

                }
                if (myTag == "SECONDARYELABORATION")
                {
                    elabParam2 = child.toElement().attribute("Param2");
                    if (checkElabParam(child.toElement().text(), elabParam2) == false)
                    {
                        listXMLElab->eraseElement(nElab);
                        errorElab = true;
                    }
                    else if (elabParam2.isEmpty())
                    {
                        listXMLElab->insertParam2(NODATA);
                    }
                    else
                    {
                        listXMLElab->insertParam2(elabParam2.toFloat());
                    }
                    elab = child.toElement().text();
                    if (elab.toUpper() == "MEAN")
                    {
                        elab = "average";
                    }
                    listXMLElab->insertElab2(elab.toLower());
                }
                if (myTag == "EXPORT")
                {
                    secondChild = child.firstChild();
                    while( !secondChild.isNull())
                    {
                        mySecondTag = secondChild.toElement().tagName().toUpper();
                        if (mySecondTag == "FILENAME")
                        {
                            filename = secondChild.toElement().text();
                            if (filename.isEmpty())
                            {
                                listXMLElab->insertFileName("");
                            }
                            else
                            {
                                listXMLElab->insertFileName(filename);
                            }
                        }
                        secondChild = secondChild.nextSibling();
                    }
                }
                if (errorElab)
                {
                    errorElab = false;
                    child = child.lastChild();
                    child = child.nextSibling();
                    nElab = nElab - 1;
                }
                else
                {
                    child = child.nextSibling();
                }
            }
            if (periodPresent == false)
            {
                listXMLElab->insertDateStart(QDate(firstYear.toInt(), 0, 0));
                listXMLElab->insertDateEnd(QDate(lastYear.toInt(), 0, 0));
                listXMLElab->insertNYears(0);
            }
            nElab = nElab + 1;
            qDebug() << "nElab " << nElab;
        }
        else if (ancestor.toElement().tagName().toUpper() == "ANOMALY")
        {
            qDebug() << "ANOMALY ";
            if (ancestor.toElement().attribute("AnomalyType").toUpper() == "PERCENTAGE")
            {
                listXMLAnomaly->insertIsPercentage(true);
            }
            else if ( !ancestor.toElement().hasAttribute("AnomalyType") || ancestor.toElement().attribute("AnomalyType").toUpper() != "PERCENTAGE")
            {
                listXMLAnomaly->insertIsPercentage(false);
            }
            QString anomalyDataTypeAttribute = ancestor.toElement().attribute("Datatype").toUpper();
            if ( anomalyDataTypeAttribute == "GRID")
            {
                listXMLAnomaly->setIsMeteoGrid(true);
            }
            else if (anomalyDataTypeAttribute == "POINT")
            {
                listXMLAnomaly->setIsMeteoGrid(false);
            }
            else if (anomalyDataTypeAttribute.isEmpty() || (anomalyDataTypeAttribute != "GRID" && anomalyDataTypeAttribute != "POINT"))
            {
                ancestor = ancestor.nextSibling(); // something is wrong, go to next elab/anomaly
                continue;
            }

            if (ancestor.toElement().attribute("RefType").toUpper() == "PERIOD")
            {
                anomalyIsClimate = false;
                listXMLAnomaly->insertIsAnomalyFromDb(false);
                listXMLAnomaly->insertAnomalyClimateField("");
            }
            else if (ancestor.toElement().attribute("RefType").toUpper() == "CLIMA")
            {
                anomalyIsClimate = true;
                listXMLAnomaly->insertIsAnomalyFromDb(true);
            }
            else
            {
                ancestor = ancestor.nextSibling(); // something is wrong, go to next elab/anomaly
                continue;
            }

            if (parseXMLPeriodType(ancestor, "PeriodType", listXMLElab, listXMLAnomaly, true, false, &period, myError) == false)
            {
                ancestor = ancestor.nextSibling(); // something is wrong, go to next elab/anomaly
                continue;
            }

            if (anomalyIsClimate)
            {
                QString anomalyClimateField = ancestor.toElement().attribute("ClimateField");
                if (anomalyClimateField.isEmpty())
                {
                    ancestor = ancestor.nextSibling(); // something is wrong, go to next elab/anomaly
                    continue;
                }
                else
                {
                    listXMLAnomaly->insertAnomalyClimateField(anomalyClimateField);
                }
                listXMLAnomaly->insertRefPeriodStr("");
                listXMLAnomaly->insertRefPeriodType(noPeriodType);
                listXMLAnomaly->insertRefYearStart(NODATA);
                listXMLAnomaly->insertRefYearEnd(NODATA);
                listXMLAnomaly->insertRefDateStart(QDate(1800,1,1));
                listXMLAnomaly->insertRefDateEnd(QDate(1800,1,1));
                listXMLAnomaly->insertRefNYears(NODATA);
                listXMLAnomaly->insertRefParam1IsClimate(false);
                listXMLAnomaly->insertRefParam1ClimateField("");
                listXMLAnomaly->insertRefParam1(NODATA);
                listXMLAnomaly->insertRefElab1("");
                listXMLAnomaly->insertRefParam2(NODATA);
                listXMLAnomaly->insertRefElab2("");

            }
            else
            {
                if (parseXMLPeriodType(ancestor, "RefPeriodType", listXMLElab, listXMLAnomaly, true, true, &refPeriod, myError) == false)
                {
                    ancestor = ancestor.nextSibling(); // something is wrong, go to next elab/anomaly
                    continue;
                }
                anomalySecElab = ancestor.toElement().elementsByTagName("SecondaryElaboration");
                if (anomalySecElab.size() == 0)
                {
                    listXMLAnomaly->insertElab2("");       // there is not secondary elab
                    listXMLAnomaly->insertParam2(NODATA);
                }
                anomalyRefSecElab = ancestor.toElement().elementsByTagName("RefSecondaryElaboration");
                if (anomalyRefSecElab.size() == 0)
                {
                    listXMLAnomaly->insertRefElab2("");     // there is not ref secondary elab
                    listXMLAnomaly->insertRefParam2(NODATA);
                }
            }
            child = ancestor.firstChild();

            while( !child.isNull())
            {

                myTag = child.toElement().tagName().toUpper();
                if (myTag == "VARIABLE")
                {
                    variable = child.toElement().text();
                    meteoVariable var = getKeyMeteoVarMeteoMap(MapDailyMeteoVarToString, variable.toStdString());
                    if (var == noMeteoVar)
                    {
                        listXMLAnomaly->eraseElement(nAnomaly);
                        errorAnomaly = true;
                    }
                    else
                    {
                        listXMLAnomaly->insertVariable(var);
                    }
                }
                if (myTag == "YEARINTERVAL")
                {
                    lastYear = child.toElement().attribute("fin");
                    firstYear = child.toElement().attribute("ini");
                    listXMLAnomaly->insertYearStart(firstYear.toInt());
                    listXMLAnomaly->insertYearEnd(lastYear.toInt());
                    if (checkYears(firstYear, lastYear) == false)
                    {
                        listXMLAnomaly->eraseElement(nAnomaly);
                        errorAnomaly = true;
                    }
                }
                if (!anomalyIsClimate)
                {
                    if (myTag == "REFYEARINTERVAL")
                    {
                        refLastYear = child.toElement().attribute("fin");
                        refFirstYear = child.toElement().attribute("ini");
                        listXMLAnomaly->insertRefYearStart(refFirstYear.toInt());
                        listXMLAnomaly->insertRefYearEnd(refLastYear.toInt());
                        if (checkYears(refFirstYear, refLastYear) == false)
                        {
                            listXMLAnomaly->eraseElement(nAnomaly);
                            errorAnomaly = true;
                        }
                    }
                }

                if (myTag == "PERIOD")
                {
                    if (parseXMLPeriodTag(child, listXMLElab, listXMLAnomaly, true, false, period, myError) == false)
                    {
                        listXMLAnomaly->eraseElement(nAnomaly);
                        errorAnomaly = true;
                    }
                }
                if (!anomalyIsClimate)
                {
                    if (myTag == "REFPERIOD")
                    {
                        if (parseXMLPeriodTag(child, listXMLElab, listXMLAnomaly, true, true, refPeriod, myError) == false)
                        {
                            listXMLAnomaly->eraseElement(nAnomaly);
                            errorAnomaly = true;
                        }
                    }
                }

                if (myTag == "PRIMARYELABORATION")
                {
                    if (ancestor.toElement().attribute("readParamFromClimate").toUpper() == "TRUE" || ancestor.toElement().attribute("readParamFromClimate").toUpper() == "YES")
                    {
                        param1IsClimate = true;
                        listXMLAnomaly->insertParam1IsClimate(true);
                        listXMLAnomaly->insertParam1(NODATA);
                    }
                    else
                    {
                        param1IsClimate = false;
                        listXMLAnomaly->insertParam1IsClimate(false);
                        listXMLAnomaly->insertParam1ClimateField("");
                    }
                    elabParam1 = child.toElement().attribute("Param1");

                    if (param1IsClimate)
                    {
                        if (elabParam1.isEmpty())
                        {
                            listXMLAnomaly->eraseElement(nAnomaly);
                            errorAnomaly = true;
                        }
                        else
                        {
                            listXMLAnomaly->insertParam1ClimateField(elabParam1);
                        }
                    }
                    else
                    {
                        if (checkElabParam(child.toElement().text(), elabParam1) == false)
                        {
                            listXMLAnomaly->eraseElement(nAnomaly);
                            errorAnomaly = true;
                        }
                        else if (elabParam1.isEmpty())
                        {
                            listXMLAnomaly->insertParam1(NODATA);
                        }
                        else
                        {
                            listXMLAnomaly->insertParam1(elabParam1.toFloat());
                        }

                    }

                    elab = child.toElement().text();
                    if ( (elab == "huglin" || elab == "winkler" || elab == "fregoni") && anomalySecElab.size() == 0 )
                    {
                        listXMLAnomaly->eraseElement(nAnomaly);
                        errorAnomaly = true;
                    }
                    else
                    {
                        if (elab.toUpper() == "MEAN")
                        {
                            elab = "average";
                        }
                        listXMLAnomaly->insertElab1(elab.toLower());
                    }
                }
                if (myTag == "SECONDARYELABORATION")
                {
                    elabParam2 = child.toElement().attribute("Param2");
                    if (checkElabParam(child.toElement().text(), elabParam2) == false)
                    {
                        listXMLAnomaly->eraseElement(nAnomaly);
                        errorAnomaly = true;
                    }
                    else if (elabParam2.isEmpty())
                    {
                        listXMLAnomaly->insertParam2(NODATA);
                    }
                    else
                    {
                        listXMLAnomaly->insertParam2(elabParam2.toFloat());
                    }
                    elab = child.toElement().text();
                    if (elab.toUpper() == "MEAN")
                    {
                        elab = "average";
                    }
                    listXMLAnomaly->insertElab2(elab.toLower());
                }


                if (!anomalyIsClimate)
                {
                    if (myTag == "REFPRIMARYELABORATION")
                    {
                        if (ancestor.toElement().attribute("readParamFromClimate").toUpper() == "TRUE" || ancestor.toElement().attribute("readParamFromClimate").toUpper() == "YES")
                        {
                            listXMLAnomaly->insertRefParam1IsClimate(true);
                            listXMLAnomaly->insertRefParam1(NODATA);
                        }
                        else
                        {
                            listXMLAnomaly->insertRefParam1IsClimate(false);
                            listXMLAnomaly->insertRefParam1ClimateField("");
                        }
                        refElabParam1 = child.toElement().attribute("Param1");

                        if (param1IsClimate)
                        {
                            if (refElabParam1.isEmpty())
                            {
                                listXMLAnomaly->eraseElement(nAnomaly);
                                errorAnomaly = true;
                            }
                            else
                            {
                                listXMLAnomaly->insertRefParam1ClimateField(refElabParam1);
                            }
                        }
                        else
                        {
                            if (checkElabParam(child.toElement().text(), refElabParam1) == false)
                            {
                                listXMLAnomaly->eraseElement(nAnomaly);
                                errorAnomaly = true;
                            }
                            else if (refElabParam1.isEmpty())
                            {
                                listXMLAnomaly->insertRefParam1(NODATA);
                            }
                            else
                            {
                                listXMLAnomaly->insertRefParam1(refElabParam1.toFloat());
                            }

                        }

                        refElab = child.toElement().text();
                        if ( (refElab == "huglin" || refElab == "winkler" || refElab == "fregoni") && anomalyRefSecElab.size() == 0 )
                        {
                            listXMLAnomaly->eraseElement(nAnomaly);
                            errorAnomaly = true;
                        }
                        else
                        {
                            if (refElab.toUpper() == "MEAN")
                            {
                                refElab = "average";
                            }
                            listXMLAnomaly->insertRefElab1(refElab.toLower());
                        }

                    }
                    if (myTag == "REFSECONDARYELABORATION")
                    {
                        refElabParam2 = child.toElement().attribute("Param2");
                        if (checkElabParam(child.toElement().text(), refElabParam2) == false)
                        {
                            listXMLAnomaly->eraseElement(nAnomaly);
                            errorAnomaly = true;
                        }
                        else if (refElabParam2.isEmpty())
                        {
                            listXMLAnomaly->insertRefParam2(NODATA);
                        }
                        else
                        {
                            listXMLAnomaly->insertRefParam2(refElabParam2.toFloat());
                        }

                        refElab2 = child.toElement().text();
                        if (refElab2.toUpper() == "MEAN")
                        {
                            refElab2 = "average";
                        }
                        listXMLAnomaly->insertRefElab2(refElab2.toLower());
                    }
                }
                if (myTag == "EXPORT")
                {
                    secondChild = child.firstChild();
                    while( !secondChild.isNull())
                    {
                        mySecondTag = secondChild.toElement().tagName().toUpper();
                        if (mySecondTag == "FILENAME")
                        {
                            filename = secondChild.toElement().text();
                            if (filename.isEmpty())
                            {
                                listXMLAnomaly->insertFileName("");
                            }
                            else
                            {
                                listXMLAnomaly->insertFileName(filename);
                            }
                        }
                        secondChild = secondChild.nextSibling();
                    }
                }
                if (errorAnomaly)
                {
                    errorAnomaly = false;
                    child = child.lastChild();
                    child = child.nextSibling();
                    nAnomaly = nAnomaly - 1;
                }
                else
                {
                    child = child.nextSibling();
                }
            }
            nAnomaly = nAnomaly + 1;
            qDebug() << "nAnomaly " << nAnomaly;
        }

        else if (ancestor.toElement().tagName().toUpper() == "PHENOLOGY")
        {
            qDebug() << "PHENOLOGY ";
            QString dataTypeAttribute = ancestor.toElement().attribute("Datatype").toUpper();
            if ( dataTypeAttribute == "GRID")
            {
                listXMLPhenology->setIsMeteoGrid(true);
            }
            else if (dataTypeAttribute == "POINT")
            {
                listXMLPhenology->setIsMeteoGrid(false);
            }
            else if (dataTypeAttribute.isEmpty() || (dataTypeAttribute != "GRID" && dataTypeAttribute != "POINT"))
            {
                ancestor = ancestor.nextSibling(); // something is wrong, go to next elab
                continue;
            }
            QString computationType = ancestor.toElement().attribute("computationType").toUpper();
            if ( computationType == "CURRENTSTAGE" || computationType == "CURRENT")
            {
                listXMLPhenology->insertComputation(currentStage);
            }
            else if (computationType == "ANOMALYDAYS")
            {
                listXMLPhenology->insertComputation(anomalyDays);
            }
            else if (computationType == "DIFFERENCESTAGES")
            {
                listXMLPhenology->insertComputation(differenceStages);
            }
            else
            {
                ancestor = ancestor.nextSibling(); // something is wrong, go to next elab
                continue;
            }
            QString cropStr = ancestor.toElement().attribute("crop").toUpper();
            phenoCrop crop = getKeyMapPhenoCrop(MapPhenoCropToString, cropStr.toStdString());
            if (crop == invalidCrop)
            {
                ancestor = ancestor.nextSibling(); // something is wrong, go to next elab
                continue;
            }
            else
            {
                listXMLPhenology->insertCrop(crop);
            }
            QString variety = ancestor.toElement().attribute("variety").toUpper();
            // LC Classe 500 a cosa corrisponde?
            if ( variety == "PRECOCISSIMA" || variety == "CLASSE 500")
            {
                listXMLPhenology->insertVariety(precocissima);
            }
            else if (variety == "PRECOCE")
            {
                listXMLPhenology->insertVariety(precoce);
            }
            else if (variety == "MEDIA")
            {
                listXMLPhenology->insertVariety(media);
            }
            else if (variety == "TARDIVE")
            {
                listXMLPhenology->insertVariety(tardive);
            }
            else
            {
                ancestor = ancestor.nextSibling(); // something is wrong, go to next elab
                continue;
            }
            bool ok;
            int vernalization = ancestor.toElement().attribute("vernalization").toUpper().toInt(&ok);
            if (ok)
            {
                listXMLPhenology->insertVernalization(vernalization);
            }
            else
            {
                ancestor = ancestor.nextSibling(); // something is wrong, go to next elab
                continue;
            }
            QString scale = ancestor.toElement().attribute("scale").toUpper();
            if (scale == "BBCH")
            {
                listXMLPhenology->insertScale(BBCH);
            }
            else if (scale == "ARPA")
            {
                listXMLPhenology->insertScale(ARPA);
            }
            else
            {
                ancestor = ancestor.nextSibling(); // something is wrong, go to next elab
                continue;
            }
            child = ancestor.firstChild();
            while( !child.isNull())
            {
                myTag = child.toElement().tagName().toUpper();
                if (myTag == "PERIOD")
                {
                    QDate startDate = QDate::fromString(child.toElement().attribute("ini"), "dd/MM/yyyy");
                    QDate endDate = QDate::fromString(child.toElement().attribute("fin"), "dd/MM/yyyy");
                    if (!endDate.isValid() || !startDate.isValid() || endDate<startDate)
                    {
                        listXMLPhenology->eraseElement(nPhenology);
                        qDebug() << "Invalid date";
                        errorPhenology = true;
                    }
                    else
                    {
                        listXMLPhenology->insertDateStart(startDate);
                        listXMLPhenology->insertDateEnd(endDate);
                    }
                }
                if (myTag == "EXPORT")
                {
                    secondChild = child.firstChild();
                    while( !secondChild.isNull())
                    {
                        mySecondTag = secondChild.toElement().tagName().toUpper();
                        if (mySecondTag == "FILENAME")
                        {
                            filename = secondChild.toElement().text();
                            if (filename.isEmpty())
                            {
                                listXMLPhenology->insertFileName("");
                            }
                            else
                            {
                                listXMLPhenology->insertFileName(filename);
                            }
                        }
                        secondChild = secondChild.nextSibling();
                    }
                }
                if (errorPhenology)
                {
                    errorPhenology = false;
                    child = child.lastChild();
                    child = child.nextSibling();
                    nPhenology = nPhenology - 1;
                }
                else
                {
                    child = child.nextSibling();
                }
            }
            nPhenology = nPhenology + 1;
            qDebug() << "nPhenology " << nPhenology;

        }

        else if (ancestor.toElement().tagName().toUpper() == "DROUGHT")
        {
            qDebug() << "DROUGHT ";
            QString dataTypeAttribute = ancestor.toElement().attribute("Datatype").toUpper();
            if ( dataTypeAttribute == "GRID")
            {
                listXMLDrought->setIsMeteoGrid(true);
            }
            else if (dataTypeAttribute == "POINT")
            {
                listXMLDrought->setIsMeteoGrid(false);
            }
            else if (dataTypeAttribute.isEmpty() || (dataTypeAttribute != "GRID" && dataTypeAttribute != "POINT"))
            {
                ancestor = ancestor.nextSibling(); // something is wrong, go to next elab
                continue;
            }
            child = ancestor.firstChild();
            while( !child.isNull())
            {
                myTag = child.toElement().tagName().toUpper();
                if (myTag == "INDEX")
                {
                    QString index = child.toElement().text().toUpper();
                    if (index == "SPI")
                    {
                        listXMLDrought->insertIndex(INDEX_SPI);
                        listXMLDrought->insertVariable(noMeteoVar); // SPI has not variable
                    }
                    else if (index == "SPEI")
                    {
                        listXMLDrought->insertIndex(INDEX_SPEI);
                        listXMLDrought->insertVariable(noMeteoVar); // SPEI has not variable
                    }
                    else if (index == "DECILES")
                    {
                        listXMLDrought->insertIndex(INDEX_DECILES);
                        listXMLDrought->insertTimescale(0);  // Deciles has not timescale
                        listXMLDrought->insertVariable(noMeteoVar);
                    }
                    else
                    {
                        listXMLDrought->eraseElement(nDrought);
                        qDebug() << "noIndex ";
                        errorDrought = true;
                    }

                }
                if (myTag == "REFINTERVAL")
                {
                    firstYear = child.toElement().attribute("yearini");
                    lastYear = child.toElement().attribute("yearfin");
                    listXMLDrought->insertYearStart(firstYear.toInt());
                    listXMLDrought->insertYearEnd(lastYear.toInt());
                    if (checkYears(firstYear, lastYear) == false)
                    {
                        listXMLDrought->eraseElement(nDrought);
                        qDebug() << "checkYears ";
                        errorDrought = true;
                    }
                }
                if (myTag == "DATE")
                {
                    QString dateStr = child.toElement().text();
                    QDate date = QDate::fromString(dateStr,"dd/MM/yyyy");
                    if (date.isValid())
                    {
                        listXMLDrought->insertDate(date);
                    }
                    else
                    {
                        listXMLDrought->eraseElement(nDrought);
                        qDebug() << "invalid date ";
                        errorDrought = true;
                    }
                }
                if (myTag == "TIMESCALE")
                {
                    bool ok;
                    QString timescaleStr = child.toElement().text();
                    int timeScale = timescaleStr.toInt(&ok);
                    if (ok)
                    {
                        listXMLDrought->insertTimescale(timeScale);
                    }
                    else
                    {
                        listXMLDrought->eraseElement(nDrought);
                        qDebug() << "invalid timescale ";
                        errorDrought = true;
                    }
                }
                if (myTag == "VARIABLE")
                {
                    QString variable = child.toElement().text();
                    meteoVariable var = getKeyMeteoVarMeteoMap(MapMonthlyMeteoVarToString, variable.toStdString());
                    if (var != noMeteoVar)
                    {
                        listXMLDrought->updateVariable(var, int(listXMLDrought->listVariable().size()) - 1);   //change var
                    }
                }
                if (myTag == "EXPORT")
                {
                    secondChild = child.firstChild();
                    while( !secondChild.isNull())
                    {
                        mySecondTag = secondChild.toElement().tagName().toUpper();
                        if (mySecondTag == "FILENAME")
                        {
                            filename = secondChild.toElement().text();
                            if (filename.isEmpty())
                            {
                                listXMLDrought->insertFileName("");
                            }
                            else
                            {
                                listXMLDrought->insertFileName(filename);
                            }
                        }
                        secondChild = secondChild.nextSibling();
                    }
                }
                if (errorDrought)
                {
                    errorDrought = false;
                    child = child.lastChild();
                    child = child.nextSibling();
                    nDrought = nDrought - 1;
                }
                else
                {
                    child = child.nextSibling();
                }
            }
            nDrought = nDrought + 1;
            qDebug() << "nDrought " << nDrought;
        }

        ancestor = ancestor.nextSibling();
    }
    xmlDoc.clear();

    for (int i = 0; i < nElab; i++)
    {
        if (listXMLElab->addElab(i))
        {
            qDebug() << "elab: " << listXMLElab->listAll().back();
        }
    }
    for (int i = 0; i < nAnomaly; i++)
    {
        if (listXMLAnomaly->addAnomaly(i))
        {
            qDebug() << "anomaly: " << listXMLAnomaly->listAll().back();
        }
    }
    for (int i = 0; i < nDrought; i++)
    {
        if (listXMLDrought->addDrought(i))
        {
            qDebug() << "drought: " << listXMLDrought->listAll().back();
        }
    }
    for (int i = 0; i < nPhenology; i++)
    {
        if (listXMLPhenology->addPhenology(i))
        {
            qDebug() << "phenology: " << listXMLPhenology->listAll().back();
        }
    }
    return true;
}

bool parseXMLPeriodType(QDomNode ancestor, QString attributePeriod, Crit3DElabList *listXMLElab, Crit3DAnomalyList *listXMLAnomaly, bool isAnomaly, bool isRefPeriod,
                        QString* period, QString *myError)
{

    enum period periodType;
    if (ancestor.toElement().attribute(attributePeriod).toUpper() == "GENERIC")
    {
        *period = "Generic";
        periodType = genericPeriod;
        if (isAnomaly)
        {
            if (isRefPeriod)
            {
                listXMLAnomaly->insertRefPeriodStr(*period);
                listXMLAnomaly->insertRefPeriodType(periodType);
            }
            else
            {
                listXMLAnomaly->insertPeriodStr(*period);
                listXMLAnomaly->insertPeriodType(periodType);
            }

        }
        else
        {
            listXMLElab->insertPeriodStr(*period);
            listXMLElab->insertPeriodType(periodType);
        }

    }
    else if (ancestor.toElement().attribute(attributePeriod).toUpper() == "DAILY")
    {
        *period = "Daily";
        periodType = dailyPeriod;
        if (isAnomaly)
        {
            if (isRefPeriod)
            {
                listXMLAnomaly->insertRefPeriodStr(*period);
                listXMLAnomaly->insertRefPeriodType(periodType);
            }
            else
            {
                listXMLAnomaly->insertPeriodStr(*period);
                listXMLAnomaly->insertPeriodType(periodType);
            }

        }
        else
        {
            listXMLElab->insertPeriodStr(*period);
            listXMLElab->insertPeriodType(periodType);
        }
    }
    else if (ancestor.toElement().attribute(attributePeriod).toUpper() == "DECADAL")
    {
        *period = "Decadal";
        periodType = decadalPeriod;
        if (isAnomaly)
        {
            if (isRefPeriod)
            {
                listXMLAnomaly->insertRefPeriodStr(*period);
                listXMLAnomaly->insertRefPeriodType(periodType);
            }
            else
            {
                listXMLAnomaly->insertPeriodStr(*period);
                listXMLAnomaly->insertPeriodType(periodType);
            }

        }
        else
        {
            listXMLElab->insertPeriodStr(*period);
            listXMLElab->insertPeriodType(periodType);
        }
    }
    else if (ancestor.toElement().attribute(attributePeriod).toUpper() == "MONTHLY")
    {
        *period = "Monthly";
        periodType = monthlyPeriod;
        if (isAnomaly)
        {
            if (isRefPeriod)
            {
                listXMLAnomaly->insertRefPeriodStr(*period);
                listXMLAnomaly->insertRefPeriodType(periodType);
            }
            else
            {
                listXMLAnomaly->insertPeriodStr(*period);
                listXMLAnomaly->insertPeriodType(periodType);
            }

        }
        else
        {
            listXMLElab->insertPeriodStr(*period);
            listXMLElab->insertPeriodType(periodType);
        }
    }
    else if (ancestor.toElement().attribute(attributePeriod).toUpper() == "SEASONAL")
    {
        *period = "Seasonal";
        periodType = seasonalPeriod;
        if (isAnomaly)
        {
            if (isRefPeriod)
            {
                listXMLAnomaly->insertRefPeriodStr(*period);
                listXMLAnomaly->insertRefPeriodType(periodType);
            }
            else
            {
                listXMLAnomaly->insertPeriodStr(*period);
                listXMLAnomaly->insertPeriodType(periodType);
            }

        }
        else
        {
            listXMLElab->insertPeriodStr(*period);
            listXMLElab->insertPeriodType(periodType);
        }
    }
    else if (ancestor.toElement().attribute(attributePeriod).toUpper() == "ANNUAL")
    {
        *period = "Annual";
        periodType = annualPeriod;
        if (isAnomaly)
        {
            if (isRefPeriod)
            {
                listXMLAnomaly->insertRefPeriodStr(*period);
                listXMLAnomaly->insertRefPeriodType(periodType);
            }
            else
            {
                listXMLAnomaly->insertPeriodStr(*period);
                listXMLAnomaly->insertPeriodType(periodType);
            }

        }
        else
        {
            listXMLElab->insertPeriodStr(*period);
            listXMLElab->insertPeriodType(periodType);
        }
    }
    else
    {
        *myError = "Invalid PeriodType attribute";
        return false;
    }
    return true;
}

bool parseXMLPeriodTag(QDomNode child, Crit3DElabList *listXMLElab, Crit3DAnomalyList *listXMLAnomaly, bool isAnomaly, bool isRefPeriod,
                        QString period, QString *myError)
{
    QDate dateStart;
    QDate dateEnd;
    QString nYears = "0";
    if (period == "Generic")
    {
        QString periodEnd = child.toElement().attribute("fin");
        QString periodStart = child.toElement().attribute("ini");
        periodEnd = periodEnd+"/2000";
        periodStart = periodStart+"/2000";
        dateStart = QDate::fromString(periodStart, "dd/MM/yyyy");
        dateEnd = QDate::fromString(periodEnd, "dd/MM/yyyy");
        nYears = child.toElement().attribute("nyears");
        bool ok = true;
        int nY = nYears.toInt(&ok);

        if (isAnomaly)
        {
            if (isRefPeriod)
            {
                listXMLAnomaly->insertRefDateStart(dateStart);
                listXMLAnomaly->insertRefDateEnd(dateEnd);
                listXMLAnomaly->insertRefNYears(nY);
            }
            else
            {
                listXMLAnomaly->insertDateStart(dateStart);
                listXMLAnomaly->insertDateEnd(dateEnd);
                listXMLAnomaly->insertNYears(nY);
            }

        }
        else
        {
            listXMLElab->insertDateStart(dateStart);
            listXMLElab->insertDateEnd(dateEnd);
            listXMLElab->insertNYears(nY);
        }

        if (!dateStart.isValid() || !dateEnd.isValid() || ok == false)
        {
            *myError = "Invalid period";
            return false;
        }
        else if (nY == 0 && dateStart > dateEnd)
        {
            *myError = "Invalid period";
            return false;
        }
        else
        {
            return true;
        }

    }
    if (period == "Daily")
    {
        int dayOfYear = child.toElement().attribute("doy").toInt();
        dateStart = QDate(2000, 1, 1).addDays(dayOfYear - 1);
        dateEnd = dateStart;

        if (isAnomaly)
        {
            if (isRefPeriod)
            {
                listXMLAnomaly->insertRefDateStart(dateStart);
                listXMLAnomaly->insertRefDateEnd(dateEnd);
                listXMLAnomaly->insertRefNYears(nYears.toInt());
            }
            else
            {
                listXMLAnomaly->insertDateStart(dateStart);
                listXMLAnomaly->insertDateEnd(dateEnd);
                listXMLAnomaly->insertNYears(nYears.toInt());
            }
        }
        else
        {
            listXMLElab->insertDateStart(dateStart);
            listXMLElab->insertDateEnd(dateEnd);
            listXMLElab->insertNYears(nYears.toInt());
        }

        if (dayOfYear < 1 || dayOfYear > 366)
        {
            *myError = "Invalid period";
            return false;
        }
        else if (!dateStart.isValid() || !dateEnd.isValid() || dateStart > dateEnd)
        {
            *myError = "Invalid period";
            return false;
        }
        else
        {
            return true;
        }
    }
    if (period == "Decadal")
    {
        int decade = child.toElement().attribute("decade").toInt();
        int dayStart;
        int dayEnd;
        int month;

        intervalDecade(decade, 2000, &dayStart, &dayEnd, &month);
        dateStart.setDate(2000, month, dayStart);
        dateEnd.setDate(2000, month, dayEnd);

        if (isAnomaly)
        {
            if (isRefPeriod)
            {
                listXMLAnomaly->insertRefDateStart(dateStart);
                listXMLAnomaly->insertRefDateEnd(dateEnd);
                listXMLAnomaly->insertRefNYears(nYears.toInt());
            }
            else
            {
                listXMLAnomaly->insertDateStart(dateStart);
                listXMLAnomaly->insertDateEnd(dateEnd);
                listXMLAnomaly->insertNYears(nYears.toInt());
            }
        }
        else
        {
            listXMLElab->insertDateStart(dateStart);
            listXMLElab->insertDateEnd(dateEnd);
            listXMLElab->insertNYears(nYears.toInt());
        }
        if (decade < 1 || decade > 36)
        {
            *myError = "Invalid period";
            return false;
        }
        else if (!dateStart.isValid() || !dateEnd.isValid() || dateStart > dateEnd)
        {
            *myError = "Invalid period";
            return false;
        }
        else
        {
            return true;
        }
    }
    if (period == "Monthly")
    {
        int month = child.toElement().attribute("month").toInt();
        dateStart.setDate(2000, month, 1);
        dateEnd = dateStart;
        dateEnd.setDate(2000, month, dateEnd.daysInMonth());

        if (isAnomaly)
        {
            if (isRefPeriod)
            {
                listXMLAnomaly->insertRefDateStart(dateStart);
                listXMLAnomaly->insertRefDateEnd(dateEnd);
                listXMLAnomaly->insertRefNYears(nYears.toInt());
            }
            else
            {
                listXMLAnomaly->insertDateStart(dateStart);
                listXMLAnomaly->insertDateEnd(dateEnd);
                listXMLAnomaly->insertNYears(nYears.toInt());
            }
        }
        else
        {
            listXMLElab->insertDateStart(dateStart);
            listXMLElab->insertDateEnd(dateEnd);
            listXMLElab->insertNYears(nYears.toInt());
        }
        if (month < 1 || month > 12)
        {
            *myError = "Invalid period";
            return false;
        }
        else
        if (!dateStart.isValid() || !dateEnd.isValid() || dateStart > dateEnd)
        {
            *myError = "Invalid period";
            return false;
        }
        else
        {
            return true;
        }
    }
    if (period == "Seasonal")
    {
        QString seasonString = child.toElement().attribute("season");
        int season = getSeasonFromString(seasonString);
        if (season == 4)
        {
            dateStart.setDate(1999, 12, 1);
            QDate temp(2000, 2, 1);
            dateEnd.setDate(2000, 2, temp.daysInMonth());
        }
        else
        {
            dateStart.setDate(2000, season*3, 1);
            QDate temp(2000, season*3+2, 1);
            dateEnd.setDate(2000, season*3+2, temp.daysInMonth());
        }

        if (isAnomaly)
        {
            if (isRefPeriod)
            {
                listXMLAnomaly->insertRefDateStart(dateStart);
                listXMLAnomaly->insertRefDateEnd(dateEnd);
                listXMLAnomaly->insertRefNYears(nYears.toInt());
            }
            else
            {
                listXMLAnomaly->insertDateStart(dateStart);
                listXMLAnomaly->insertDateEnd(dateEnd);
                listXMLAnomaly->insertNYears(nYears.toInt());
            }
        }
        else
        {
            listXMLElab->insertDateStart(dateStart);
            listXMLElab->insertDateEnd(dateEnd);
            listXMLElab->insertNYears(nYears.toInt());
        }
        if (season < 1 || season > 4)
        {
            *myError = "Invalid period";
            return false;
        }
        if (!dateStart.isValid() || !dateEnd.isValid() || dateStart > dateEnd)
        {
            *myError = "Invalid period";
            return false;
        }
        else
        {
            return true;
        }
    }
    if (period == "Annual")
    {
        dateStart.setDate(2000, 1, 1);
        dateEnd.setDate(2000, 12, 31);

        if (isAnomaly)
        {
            if (isRefPeriod)
            {
                listXMLAnomaly->insertRefDateStart(dateStart);
                listXMLAnomaly->insertRefDateEnd(dateEnd);
                listXMLAnomaly->insertRefNYears(nYears.toInt());
            }
            else
            {
                listXMLAnomaly->insertDateStart(dateStart);
                listXMLAnomaly->insertDateEnd(dateEnd);
                listXMLAnomaly->insertNYears(nYears.toInt());
            }
        }
        else
        {
            listXMLElab->insertDateStart(dateStart);
            listXMLElab->insertDateEnd(dateEnd);
            listXMLElab->insertNYears(nYears.toInt());
        }


        if (!dateStart.isValid() || !dateEnd.isValid() || dateStart > dateEnd)
        {
            *myError = "Invalid period";
            return false;
        }
        else
        {
            return true;
        }
    }

    *myError = "Invalid period";
    return false;
}

bool checkYears(QString firstYear, QString lastYear)
{
    bool okFirst = true;
    bool okLast = true;
    int fY = firstYear.toInt(&okFirst);
    int lY = lastYear.toInt(&okLast);
    if (okFirst == false || okLast == false)
    {
        return false;
    }
    else if (firstYear.size() != 4 || lastYear.size() != 4 || fY > lY)
    {
        return false;
    }
    else
    {
        return true;
    }
}

bool checkElabParam(QString elab, QString param)
{
    if ( MapElabWithParam.find(elab.toStdString()) != MapElabWithParam.end())
    {
        if(param.isEmpty())
        {
            return false;
        }
    }
    return true;
}

bool checkDataType(QString xmlFileName, bool isMeteoGrid, QString *myError)
{

    QDomDocument xmlDoc;

    QFile myFile(xmlFileName);
    if (!myFile.open(QIODevice::ReadOnly))
    {
        *myError = "Open XML failed:\n" + xmlFileName + "\n" + myFile.errorString();
        return (false);
    }

    int myErrLine, myErrColumn;
    if (!xmlDoc.setContent(&myFile, myError, &myErrLine, &myErrColumn))
    {
       QString completeError = "Parse xml failed:" + xmlFileName
               + " Row: " + QString::number(myErrLine)
               + " - Column: " + QString::number(myErrColumn)
               + "\n" + *myError;
       *myError = completeError;
        myFile.close();
        return(false);
    }

    myFile.close();
    QDomNode ancestor = xmlDoc.documentElement().firstChild();

    while(!ancestor.isNull())
    {
        if (ancestor.toElement().tagName().toUpper() == "ELABORATION" || ancestor.toElement().tagName().toUpper() == "ANOMALY")
        {
            QString dataTypeAttribute = ancestor.toElement().attribute("Datatype").toUpper();
            if ( dataTypeAttribute == "GRID")
            {
                if (isMeteoGrid)
                {
                    xmlDoc.clear();
                    return true;
                }
                else
                {
                    xmlDoc.clear();
                    *myError = "XML Datatype is GRID";
                    return false;
                }
            }
            else if (dataTypeAttribute == "POINT")
            {
                if (isMeteoGrid)
                {
                    xmlDoc.clear();
                    *myError = "XML Datatype is POINT";
                    return false;
                }
                else
                {
                    xmlDoc.clear();
                    return true;
                }
            }
            else if (dataTypeAttribute.isEmpty() || (dataTypeAttribute != "GRID" && dataTypeAttribute != "POINT"))
            {
                ancestor = ancestor.nextSibling(); // something is wrong, go to next elab
                continue;
            }
        }
        ancestor = ancestor.nextSibling();
    }
    return false;

}

bool appendXMLElaboration(Crit3DElabList *listXMLElab, QString xmlFileName, QString *myError)
{

    QDomDocument xmlDoc;

    // check
    if (xmlFileName == "")
    {
        *myError = "Missing XML file.";
        return false;
    }

    QFile myFile(xmlFileName);

    if (!myFile.open(QIODevice::ReadWrite))
    {
        *myError = "Open XML failed:\n" + xmlFileName + "\n" + myFile.errorString();
        myFile.close();
        return false;
    }

    int myErrLine, myErrColumn;
    if (!xmlDoc.setContent(&myFile, myError, &myErrLine, &myErrColumn))
    {
       QString completeError = "Parse xml failed:" + xmlFileName
               + " Row: " + QString::number(myErrLine)
               + " - Column: " + QString::number(myErrColumn)
               + "\n" + *myError;
       *myError = completeError;
        myFile.close();
        return false;
    }

    QDomElement root = xmlDoc.documentElement();
    if( root.tagName() != "xml" )
    {
        *myError = "missing xml root tag";
        myFile.close();
        return false;
    }

    QString dataType;
    QString periodElab = listXMLElab->listPeriodStr()[0];
    if (listXMLElab->isMeteoGrid())
    {
        dataType = "Grid";
    }
    else
    {
        dataType = "Point";
    }
    meteoVariable var = listXMLElab->listVariable()[0];
    std::string variableString = MapDailyMeteoVarToString.at(var);
    QDomElement elaborationTag = xmlDoc.createElement(QString("Elaboration"));
    elaborationTag.setAttribute("Datatype", dataType);
    elaborationTag.setAttribute("PeriodType", periodElab);
    QDomElement variableTag = xmlDoc.createElement(QString("Variable"));
    QDomText variableText = xmlDoc.createTextNode(QString::fromStdString(variableString));
    variableTag.appendChild(variableText);

    QDomElement yearIntervalTag = xmlDoc.createElement(QString("YearInterval"));
    yearIntervalTag.setAttribute("ini", listXMLElab->listYearStart()[0]);
    yearIntervalTag.setAttribute("fin", listXMLElab->listYearEnd()[0]);

    QDomElement periodTag = xmlDoc.createElement(QString("period"));

    QDate dateStart = listXMLElab->listDateStart()[0];
    QDate dateEnd = listXMLElab->listDateEnd()[0];
    if (periodElab == "Generic")
    {
        int month = dateStart.month();
        int day = dateStart.day();
        QString monthStr;
        QString dayStr;
        if (month < 10)
        {
            monthStr = "0"+QString::number(month);
        }
        else
        {
            monthStr = QString::number(month);
        }
        if (day < 10)
        {
            dayStr = "0"+QString::number(day);
        }
        else
        {
            dayStr = QString::number(day);
        }

        periodTag.setAttribute("ini", dayStr+"/"+monthStr);

        month = dateEnd.month();
        day = dateEnd.day();
        if (month < 10)
        {
            monthStr = "0"+QString::number(month);
        }
        else
        {
            monthStr = QString::number(month);
        }
        if (day < 10)
        {
            dayStr = "0"+QString::number(day);
        }
        else
        {
            dayStr = QString::number(day);
        }

        periodTag.setAttribute("fin", dayStr+"/"+monthStr);
        periodTag.setAttribute("nyears", QString::number(listXMLElab->listNYears()[0]));
    }
    else if (periodElab == "Daily")
    {
        periodTag.setAttribute("doy", QString::number(dateStart.dayOfYear()));
    }
    else if (periodElab == "Monthly")
    {
        periodTag.setAttribute("month", QString::number(dateStart.month()));
    }
    else if (periodElab == "Decadal")
    {
        periodTag.setAttribute("decade", QString::number(decadeFromDate(dateStart)));
    }
    else if (periodElab == "Seasonal")
    {
        periodTag.setAttribute("season", QString::number(getSeasonFromDate(dateStart)));
    }

    QDomElement elab1Tag = xmlDoc.createElement(QString("PrimaryElaboration"));
    if (listXMLElab->listParam1()[0] != NODATA)
    {
        elab1Tag.setAttribute("Param1", QString::number(listXMLElab->listParam1()[0]));
    }
    else if (listXMLElab->listParam1IsClimate()[0])
    {
        elab1Tag.setAttribute("readParamFromClimate", "TRUE");
        elab1Tag.setAttribute("Param1", listXMLElab->listParam1ClimateField()[0]);
    }
    QDomText elab1Text = xmlDoc.createTextNode(listXMLElab->listElab1()[0]);
    elab1Tag.appendChild(elab1Text);

    elaborationTag.appendChild(variableTag);
    elaborationTag.appendChild(yearIntervalTag);
    elaborationTag.appendChild(periodTag);
    elaborationTag.appendChild(elab1Tag);

    if (!listXMLElab->listElab2()[0].isEmpty())
    {
        QDomElement elab2Tag = xmlDoc.createElement(QString("SecondaryElaboration"));
        if (listXMLElab->listParam2()[0] != NODATA)
        {
            elab2Tag.setAttribute("Param2", QString::number(listXMLElab->listParam2()[0]));
        }
        QDomText elab2Text = xmlDoc.createTextNode(listXMLElab->listElab2()[0]);
        elab2Tag.appendChild(elab2Text);
        elaborationTag.appendChild(elab2Tag);
    }

    root.appendChild(elaborationTag);
    // Remove old file and save the new one with same name
    myFile.remove();
    QFile outputFile(xmlFileName);
    outputFile.open(QIODevice::ReadWrite);
    QTextStream output(&outputFile);
    output << xmlDoc.toString();
    outputFile.close();
    return true;

}

bool appendXMLAnomaly(Crit3DAnomalyList *listXMLAnomaly, QString xmlFileName, QString *myError)
{

    QDomDocument xmlDoc;

    // check
    if (xmlFileName == "")
    {
        *myError = "Missing XML file.";
        return false;
    }

    QFile myFile(xmlFileName);

    if (!myFile.open(QIODevice::ReadWrite))
    {
        *myError = "Open XML failed:\n" + xmlFileName + "\n" + myFile.errorString();
        myFile.close();
        return false;
    }

    int myErrLine, myErrColumn;
    if (!xmlDoc.setContent(&myFile, myError, &myErrLine, &myErrColumn))
    {
       QString completeError = "Parse xml failed:" + xmlFileName
               + " Row: " + QString::number(myErrLine)
               + " - Column: " + QString::number(myErrColumn)
               + "\n" + *myError;
       *myError = completeError;
        myFile.close();
        return false;
    }

    QDomElement root = xmlDoc.documentElement();
    if( root.tagName() != "xml" )
    {
        *myError = "missing xml root tag";
        myFile.close();
        return false;
    }

    QString dataType;
    QString periodElab = listXMLAnomaly->listPeriodStr()[0];

    if (listXMLAnomaly->isMeteoGrid())
    {
        dataType = "Grid";
    }
    else
    {
        dataType = "Point";
    }
    meteoVariable var = listXMLAnomaly->listVariable()[0];
    std::string variableString = MapDailyMeteoVarToString.at(var);
    QDomElement elaborationTag = xmlDoc.createElement(QString("Anomaly"));
    elaborationTag.setAttribute("Datatype", dataType);
    elaborationTag.setAttribute("PeriodType", periodElab);
    bool clima = listXMLAnomaly->isAnomalyFromDb()[0];
    if (clima)
    {
        elaborationTag.setAttribute("RefType", "Clima");
        elaborationTag.setAttribute("ClimateField", listXMLAnomaly->listAnomalyClimateField()[0]);
    }
    else
    {
        elaborationTag.setAttribute("RefType", "Period");
        elaborationTag.setAttribute("RefPeriodType", listXMLAnomaly->listRefPeriodStr()[0]);
    }

    QDomElement variableTag = xmlDoc.createElement(QString("Variable"));
    QDomText variableText = xmlDoc.createTextNode(QString::fromStdString(variableString));
    variableTag.appendChild(variableText);

    QDomElement yearIntervalTag = xmlDoc.createElement(QString("YearInterval"));
    yearIntervalTag.setAttribute("ini", listXMLAnomaly->listYearStart()[0]);
    yearIntervalTag.setAttribute("fin", listXMLAnomaly->listYearEnd()[0]);

    QDomElement periodTag = xmlDoc.createElement(QString("period"));

    QDate dateStart = listXMLAnomaly->listDateStart()[0];
    QDate dateEnd = listXMLAnomaly->listDateEnd()[0];
    if (periodElab == "Generic")
    {
        int month = dateStart.month();
        int day = dateStart.day();
        QString monthStr;
        QString dayStr;
        if (month < 10)
        {
            monthStr = "0"+QString::number(month);
        }
        else
        {
            monthStr = QString::number(month);
        }
        if (day < 10)
        {
            dayStr = "0"+QString::number(day);
        }
        else
        {
            dayStr = QString::number(day);
        }

        periodTag.setAttribute("ini", dayStr+"/"+monthStr);

        month = dateEnd.month();
        day = dateEnd.day();
        if (month < 10)
        {
            monthStr = "0"+QString::number(month);
        }
        else
        {
            monthStr = QString::number(month);
        }
        if (day < 10)
        {
            dayStr = "0"+QString::number(day);
        }
        else
        {
            dayStr = QString::number(day);
        }

        periodTag.setAttribute("fin", dayStr+"/"+monthStr);
        periodTag.setAttribute("nyears", QString::number(listXMLAnomaly->listNYears()[0]));
    }
    else if (periodElab == "Daily")
    {
        periodTag.setAttribute("doy", QString::number(dateStart.dayOfYear()));
    }
    else if (periodElab == "Monthly")
    {
        periodTag.setAttribute("month", QString::number(dateStart.month()));
    }
    else if (periodElab == "Decadal")
    {
        periodTag.setAttribute("decade", QString::number(decadeFromDate(dateStart)));
    }
    else if (periodElab == "Seasonal")
    {
        periodTag.setAttribute("season", QString::number(getSeasonFromDate(dateStart)));
    }

    QDomElement elab1Tag = xmlDoc.createElement(QString("PrimaryElaboration"));
    if (listXMLAnomaly->listParam1()[0] != NODATA)
    {
        elab1Tag.setAttribute("Param1", QString::number(listXMLAnomaly->listParam1()[0]));
    }
    else if (listXMLAnomaly->listParam1IsClimate()[0])
    {
        elab1Tag.setAttribute("readParamFromClimate", "TRUE");
        elab1Tag.setAttribute("Param1", listXMLAnomaly->listParam1ClimateField()[0]);
    }
    QDomText elab1Text = xmlDoc.createTextNode(listXMLAnomaly->listElab1()[0]);
    elab1Tag.appendChild(elab1Text);

    QDomElement elab2Tag;
    if (!listXMLAnomaly->listElab2()[0].isEmpty())
    {
        elab2Tag = xmlDoc.createElement(QString("SecondaryElaboration"));
        if (listXMLAnomaly->listParam2()[0] != NODATA)
        {
            elab2Tag.setAttribute("Param2", QString::number(listXMLAnomaly->listParam2()[0]));
        }
        QDomText elab2Text = xmlDoc.createTextNode(listXMLAnomaly->listElab2()[0]);
        elab2Tag.appendChild(elab2Text);
    }

    // reference
    if (!clima)
    {
        QDomElement refYearIntervalTag = xmlDoc.createElement(QString("RefYearInterval"));
        refYearIntervalTag.setAttribute("ini", listXMLAnomaly->listRefYearStart()[0]);
        refYearIntervalTag.setAttribute("fin", listXMLAnomaly->listRefYearEnd()[0]);

        QString periodRefElab = listXMLAnomaly->listRefPeriodStr()[0];
        QDomElement refPeriodTag = xmlDoc.createElement(QString("Refperiod"));

        QDate refDateStart = listXMLAnomaly->listRefDateStart()[0];
        QDate refdateEnd = listXMLAnomaly->listRefDateEnd()[0];
        if (periodRefElab == "Generic")
        {

            int month = refDateStart.month();
            int day = refDateStart.day();
            QString monthStr;
            QString dayStr;
            if (month < 10)
            {
                monthStr = "0"+QString::number(month);
            }
            else
            {
                monthStr = QString::number(month);
            }
            if (day < 10)
            {
                dayStr = "0"+QString::number(day);
            }
            else
            {
                dayStr = QString::number(day);
            }

            refPeriodTag.setAttribute("ini", dayStr+"/"+monthStr);

            month = refdateEnd.month();
            day = refdateEnd.day();
            if (month < 10)
            {
                monthStr = "0"+QString::number(month);
            }
            else
            {
                monthStr = QString::number(month);
            }
            if (day < 10)
            {
                dayStr = "0"+QString::number(day);
            }
            else
            {
                dayStr = QString::number(day);
            }

            refPeriodTag.setAttribute("fin", dayStr+"/"+monthStr);
            refPeriodTag.setAttribute("nyears", QString::number(listXMLAnomaly->listRefNYears()[0]));
        }
        else if (periodRefElab == "Daily")
        {
            refPeriodTag.setAttribute("doy", QString::number(refDateStart.dayOfYear()));
        }
        else if (periodRefElab == "Monthly")
        {
            refPeriodTag.setAttribute("month", QString::number(refDateStart.month()));
        }
        else if (periodRefElab == "Decadal")
        {
            refPeriodTag.setAttribute("decade", QString::number(decadeFromDate(refDateStart)));
        }
        else if (periodRefElab == "Seasonal")
        {
            refPeriodTag.setAttribute("season", QString::number(getSeasonFromDate(refDateStart)));
        }
        QDomElement refElab1Tag = xmlDoc.createElement(QString("RefPrimaryElaboration"));
        if (listXMLAnomaly->listRefParam1()[0] != NODATA)
        {
            refElab1Tag.setAttribute("Param1", QString::number(listXMLAnomaly->listRefParam1()[0]));
        }
        else if (listXMLAnomaly->listRefParam1IsClimate()[0])
        {
            refElab1Tag.setAttribute("readParamFromClimate", "TRUE");
            refElab1Tag.setAttribute("Param1", listXMLAnomaly->listRefParam1ClimateField()[0]);
        }
        QDomText refElab1Text = xmlDoc.createTextNode(listXMLAnomaly->listRefElab1()[0]);
        refElab1Tag.appendChild(refElab1Text);

        elaborationTag.appendChild(variableTag);
        elaborationTag.appendChild(yearIntervalTag);
        elaborationTag.appendChild(refYearIntervalTag);
        elaborationTag.appendChild(periodTag);
        elaborationTag.appendChild(refPeriodTag);
        elaborationTag.appendChild(elab1Tag);
        elaborationTag.appendChild(refElab1Tag);

        if (!listXMLAnomaly->listElab2()[0].isEmpty())
        {
            elaborationTag.appendChild(elab2Tag);
        }

        if (!listXMLAnomaly->listRefElab2()[0].isEmpty())
        {
            QDomElement refElab2Tag = xmlDoc.createElement(QString("RefSecondaryElaboration"));
            if (listXMLAnomaly->listRefParam2()[0] != NODATA)
            {
                refElab2Tag.setAttribute("Param2", QString::number(listXMLAnomaly->listRefParam2()[0]));
            }
            QDomText refElab2Text = xmlDoc.createTextNode(listXMLAnomaly->listRefElab2()[0]);
            refElab2Tag.appendChild(refElab2Text);
            elaborationTag.appendChild(refElab2Tag);
        }
    }
    else
    {
        elaborationTag.appendChild(variableTag);
        elaborationTag.appendChild(yearIntervalTag);
        elaborationTag.appendChild(periodTag);
        elaborationTag.appendChild(elab1Tag);
        if (!listXMLAnomaly->listElab2()[0].isEmpty())
        {
            elaborationTag.appendChild(elab2Tag);
        }
    }

    root.appendChild(elaborationTag);
    // Remove old file and save the new one with same name
    myFile.remove();
    QFile outputFile(xmlFileName);
    outputFile.open(QIODevice::ReadWrite);
    QTextStream output(&outputFile);
    output << xmlDoc.toString();
    outputFile.close();
    return true;
}


bool monthlyAggregateDataGrid(Crit3DMeteoGridDbHandler* meteoGridDbHandler, QDate firstDate, QDate lastDate,
                              std::vector<meteoVariable> dailyMeteoVar,
                              Crit3DMeteoSettings* meteoSettings, Crit3DQuality* qualityCheck,
                              Crit3DClimateParameters* climateParam, QString &myError)
{
    int nrMonths = (lastDate.year()-firstDate.year())*12+lastDate.month()-(firstDate.month()-1);
    bool isMeteoGrid = true;
    Crit3DMeteoPoint* meteoPointTemp = new Crit3DMeteoPoint;
    float percValue;
    std::vector<float> outputValues;
    QList<meteoVariable> meteoVariableList;

    for (unsigned row = 0; row < unsigned(meteoGridDbHandler->meteoGrid()->gridStructure().header().nrRows); row++)
    {
        for (unsigned col = 0; col < unsigned(meteoGridDbHandler->meteoGrid()->gridStructure().header().nrCols); col++)
        {
            meteoVariableList.clear();
            if (meteoGridDbHandler->meteoGrid()->meteoPointPointer(row,col)->active)
            {
                meteoGridDbHandler->meteoGrid()->meteoPointPointer(row,col)->initializeObsDataM(nrMonths, firstDate.month(), firstDate.year());
                meteoPointTemp->initializeObsDataM(nrMonths, firstDate.month(), firstDate.year());
                // copy id to MPTemp
                meteoPointTemp->id = meteoGridDbHandler->meteoGrid()->meteoPointPointer(row,col)->id;
                meteoPointTemp->latitude = meteoGridDbHandler->meteoGrid()->meteoPointPointer(row,col)->latitude;
                // meteoPointTemp should be init
                meteoPointTemp->nrObsDataDaysH = 0;
                meteoPointTemp->nrObsDataDaysD = 0;

                for(int i = 0; i < dailyMeteoVar.size(); i++)
                {
                    if (preElaboration(&myError, nullptr, meteoGridDbHandler, meteoPointTemp, isMeteoGrid, dailyMeteoVar[i], noMeteoComp, firstDate, lastDate, outputValues, &percValue, meteoSettings))
                    {
                        if (meteoPointTemp->computeMonthlyAggregate(getCrit3DDate(firstDate), getCrit3DDate(lastDate), dailyMeteoVar[i], meteoSettings, qualityCheck, climateParam))
                        {
                            meteoVariable monthlyVar = updateMeteoVariable(dailyMeteoVar[i], monthly);
                            if (monthlyVar != noMeteoVar)
                            {
                                meteoVariableList.append(monthlyVar);
                            }
                        }
                    }
                }
                // copy meteoPoint Temp values
                meteoGridDbHandler->meteoGrid()->meteoPointPointer(row,col)->obsDataM = meteoPointTemp->obsDataM ;
                if (! meteoVariableList.isEmpty())
                {
                    if (! meteoGridDbHandler->saveCellGridMonthlyData(&myError, QString::fromStdString(meteoGridDbHandler->meteoGrid()->meteoPointPointer(row,col)->id), row, col, firstDate, lastDate, meteoVariableList))
                    {
                        delete meteoPointTemp;
                        return false;
                    }
                }
            }
        }
    }

    delete meteoPointTemp;
    return true;
}


int computeAnnualSeriesOnPointFromDaily(QString *myError, Crit3DMeteoPointsDbHandler* meteoPointsDbHandler, Crit3DMeteoGridDbHandler* meteoGridDbHandler,
                                         Crit3DMeteoPoint* meteoPointTemp, Crit3DClimate* clima, bool isMeteoGrid, bool isAnomaly, Crit3DMeteoSettings* meteoSettings,
                                        std::vector<float> &outputValues, std::vector<int> &vectorYears, bool dataAlreadyLoaded)
{
    int validYears = 0;
    if (clima->param1IsClimate())
    {
        clima->param1();
    }

    QDate startDate(clima->yearStart(), clima->genericPeriodDateStart().month(), clima->genericPeriodDateStart().day());
    QDate endDate(clima->yearEnd(), clima->genericPeriodDateEnd().month(), clima->genericPeriodDateEnd().day());
    int yearStart = clima->yearStart();
    int yearEnd = clima->yearEnd();

    for (int myYear = yearStart; myYear <= yearEnd; myYear++)
    {
        startDate.setDate(myYear, startDate.month(), startDate.day());
        endDate.setDate(myYear, endDate.month(), endDate.day());
        clima->setYearStart(myYear);
        clima->setYearEnd(myYear);
        if (!dataAlreadyLoaded)
        {
            meteoPointTemp->nrObsDataDaysD = 0; // should be init
        }
        if (clima->nYears() < 0)
        {
            startDate.setDate(myYear + clima->nYears(), startDate.month(), startDate.day());
        }
        else if (clima->nYears() > 0)
        {
            endDate.setDate(myYear + clima->nYears(), endDate.month(), endDate.day());
        }
        if (isMeteoGrid)
        {
            if ( elaborationOnPoint(myError, nullptr, meteoGridDbHandler, meteoPointTemp, clima, isMeteoGrid, startDate, endDate, isAnomaly, meteoSettings, dataAlreadyLoaded))
            {
                validYears = validYears + 1;
                if(isAnomaly)
                {
                    outputValues.push_back(meteoPointTemp->anomaly);
                }
                else
                {
                    outputValues.push_back(meteoPointTemp->elaboration);
                }
            }
            else
            {
                outputValues.push_back(NODATA);
            }
        }
        else
        {
            if ( elaborationOnPoint(myError, meteoPointsDbHandler, nullptr, meteoPointTemp, clima, isMeteoGrid, startDate, endDate, isAnomaly, meteoSettings, dataAlreadyLoaded))
            {
                validYears = validYears + 1;
                vectorYears.push_back(myYear);
                if(isAnomaly)
                {
                    outputValues.push_back(meteoPointTemp->anomaly);
                }
                else
                {
                    outputValues.push_back(meteoPointTemp->elaboration);
                }
            }
            else
            {
                outputValues.push_back(NODATA);
            }
        }
    }
    return validYears;
}

void computeClimateOnDailyData(Crit3DMeteoPoint meteoPoint, meteoVariable var, QDate firstDate, QDate lastDate,
                              int smooth, float* dataPresence, Crit3DQuality* qualityCheck, Crit3DClimateParameters* climateParam,
                               Crit3DMeteoSettings* meteoSettings, std::vector<float> &dailyClima, std::vector<float> &decadalClima, std::vector<float> &monthlyClima)
{

    int nrDays = int(firstDate.daysTo(lastDate) + 1);
    Crit3DDate mpFirstDate = meteoPoint.obsDataD[0].date;
    QDate myDate;
    int month;
    int decade;
    int dayOfYear;
    vector<float> monthly;
    vector<float> decadal;
    vector<float> daily;
    vector<float> numMonthlyData;
    vector<float> numDecadeData;
    vector<float> numDailyData;
    vector<float> maxMonthlyData;
    vector<float> maxDecadeData;
    vector<float> maxDailyData;
    for (int fill = 0; fill <= 12; fill++)
    {
        monthly.push_back(0);
        numMonthlyData.push_back(0);
        maxMonthlyData.push_back(0);
    }
    for (int fill = 0; fill <= 36; fill++)
    {
        decadal.push_back(0);
        numDecadeData.push_back(0);
        maxDecadeData.push_back(0);
    }
    for (int fill = 0; fill <= 366; fill++)
    {
        daily.push_back(0);
        numDailyData.push_back(0);
        maxDailyData.push_back(0);
    }

    quality::qualityType check;
    for (int day = 0; day < nrDays; day++)
    {
        myDate = firstDate.addDays(day);

        month = myDate.month();
        decade = decadeFromDate(myDate);
        dayOfYear = myDate.dayOfYear();

        maxMonthlyData[month] = maxMonthlyData[month] + 1;
        maxDecadeData[decade] = maxDecadeData[decade] + 1;
        maxDailyData[dayOfYear] = maxDailyData[dayOfYear] + 1;

        int i = getQDate(mpFirstDate).daysTo(myDate);
        float myDailyValue = meteoPoint.getMeteoPointValueD(getCrit3DDate(myDate), var, meteoSettings);
        if (i<0 || i>meteoPoint.nrObsDataDaysD)
        {
            check = quality::missing_data;
        }
        else
        {
            check = qualityCheck->checkFastValueDaily_SingleValue(var, climateParam, myDailyValue, month, meteoPoint.point.z);
        }
        if (check == quality::accepted)
        {
            if (numMonthlyData[month] == 0)
            {
                monthly[month] = myDailyValue;
            }
            else
            {
                monthly[month] = monthly[month] + myDailyValue;
            }

            if (numDecadeData[decade] == 0)
            {
                decadal[decade] = myDailyValue;
            }
            else
            {
                decadal[decade] = decadal[decade] + myDailyValue;
            }

            if (numDailyData[dayOfYear] == 0)
            {
                daily[dayOfYear] = myDailyValue;
            }
            else
            {
                daily[dayOfYear] = daily[dayOfYear] + myDailyValue;
            }

            numMonthlyData[month] = numMonthlyData[month] + 1;
            numDecadeData[decade] = numDecadeData[decade] + 1;
            numDailyData[dayOfYear] = numDailyData[dayOfYear] + 1;
        }
    }
    // consistenza
    float numDati = 0;
    float numDatiMax = 0;
    for (int myMonth = 1; myMonth <= 12; myMonth++)
    {
        numDati = numDati + numMonthlyData[myMonth];
        numDatiMax = numDatiMax + maxMonthlyData[myMonth];
    }
    *dataPresence = numDati / numDatiMax * 100;
    float minPerc = meteoSettings->getMinimumPercentage();

    for (int day = 1; day <= 366; day++)
    {
        myDate = QDate(2000, 1, 1).addDays(day - 1);
        // daily
        if (maxDailyData[day] > 0)
        {
            if (numDailyData[day] / maxDailyData[day] >= (minPerc/100))
            {
                dailyClima[day] = daily[day] / numDailyData[day];
            }
            else
            {
                dailyClima[day] = NODATA;
            }
        }
        else
        {
            dailyClima[day] = NODATA;
        }

        // decadal
        decade = decadeFromDate(myDate);
        if (maxDecadeData[decade] > 0)
        {
            if (numDecadeData[decade] / maxDecadeData[decade] >= (minPerc/100))
            {
                decadalClima[decade] = decadal[decade] / numDecadeData[decade];
            }
            else
            {
                decadalClima[decade] = NODATA;
            }
        }
        else
        {
            decadalClima[decade] = NODATA;
        }

        // monthly
        month = myDate.month();
        if (maxMonthlyData[month] > 0)
        {
            if (numMonthlyData[month] / maxMonthlyData[month] >= (minPerc/100))
            {
                monthlyClima[month] = monthly[month] / numMonthlyData[month];
            }
            else
            {
                monthlyClima[month] = NODATA;
            }
        }
        else
        {
            monthlyClima[month] = NODATA;
        }
    }
    // smooth
    if (smooth > 0)
    {
        std::vector<float> myClimaTmp = dailyClima;
        int doy;
        int nDays;
        float dSum;
        for (int day = 1; day <= 366; day++)
        {
            dSum = 0;
            nDays = 0;
            for (int d = day-smooth; d <= day+smooth; d++)
            {
                doy = d;
                if (doy < 1)
                {
                    doy = 366 + doy;
                }
                else if (doy > 366)
                {
                    doy = doy - 366;
                }
                if (myClimaTmp[doy] != NODATA)
                {
                    dSum = dSum + myClimaTmp[doy];
                    nDays = nDays + 1;
                }
            }
            if (nDays > 0)
            {
                dailyClima[day] = dSum / nDays;
            }
            else
            {
                dailyClima[day] = NODATA;
            }
        }
    }
}


void setMpValues(Crit3DMeteoPoint meteoPointGet, Crit3DMeteoPoint* meteoPointSet, QDate myDate, meteoVariable myVar, Crit3DMeteoSettings* meteoSettings)
{
    bool automaticETP = meteoSettings->getAutomaticET0HS();
    Crit3DQuality qualityCheck;

    switch(myVar)
    {
        case dailyLeafWetness:
        {
            QDateTime myDateTime(myDate,QTime(1,0,0));
            QDateTime endDateTime(myDate.addDays(1),QTime(0,0,0));
            while(myDateTime<=endDateTime)
            {
                float value = meteoPointGet.getMeteoPointValueH(getCrit3DDate(myDateTime.date()), myDateTime.time().hour(), 0, leafWetness);
                meteoPointSet->setMeteoPointValueH(getCrit3DDate(myDateTime.date()), myDateTime.time().hour(), 0, leafWetness, value);
                myDateTime = myDateTime.addSecs(3600);
            }
            break;
        }

        case dailyThomDaytime:
        {
            float value = meteoPointGet.getMeteoPointValueD(getCrit3DDate(myDate), dailyAirRelHumidityMin, meteoSettings);
            meteoPointSet->setMeteoPointValueD(getCrit3DDate(myDate), dailyAirRelHumidityMin, value);
            value = meteoPointGet.getMeteoPointValueD(getCrit3DDate(myDate), dailyAirTemperatureMax, meteoSettings);
            meteoPointSet->setMeteoPointValueD(getCrit3DDate(myDate), dailyAirTemperatureMax, value);
            break;
        }

        case dailyThomNighttime:
        {
            float value = meteoPointGet.getMeteoPointValueD(getCrit3DDate(myDate), dailyAirRelHumidityMax, meteoSettings);
            meteoPointSet->setMeteoPointValueD(getCrit3DDate(myDate), dailyAirRelHumidityMax, value);
            value = meteoPointGet.getMeteoPointValueD(getCrit3DDate(myDate), dailyAirTemperatureMin, meteoSettings);
            meteoPointSet->setMeteoPointValueD(getCrit3DDate(myDate), dailyAirTemperatureMin, value);
            break;
        }
        case dailyThomAvg: case dailyThomMax: case dailyThomHoursAbove:
        {
            QDateTime myDateTime(myDate,QTime(1,0,0));
            QDateTime endDateTime(myDate.addDays(1),QTime(0,0,0));
            while(myDateTime<=endDateTime)
            {
                float value = meteoPointGet.getMeteoPointValueH(getCrit3DDate(myDateTime.date()), myDateTime.time().hour(), 0, airTemperature);
                meteoPointSet->setMeteoPointValueH(getCrit3DDate(myDateTime.date()), myDateTime.time().hour(), 0, airTemperature, value);
                value = meteoPointGet.getMeteoPointValueH(getCrit3DDate(myDateTime.date()), myDateTime.time().hour(), 0, airRelHumidity);
                meteoPointSet->setMeteoPointValueH(getCrit3DDate(myDateTime.date()), myDateTime.time().hour(), 0, airRelHumidity, value);
                myDateTime = myDateTime.addSecs(3600);
            }
            break;
        }
        case dailyBIC:
        {
            float value = meteoPointGet.getMeteoPointValueD(getCrit3DDate(myDate), dailyReferenceEvapotranspirationHS, meteoSettings);
            meteoPointSet->setMeteoPointValueD(getCrit3DDate(myDate), dailyReferenceEvapotranspirationHS, value);
            if (automaticETP)
            {
                float value = meteoPointGet.getMeteoPointValueD(getCrit3DDate(myDate), dailyAirTemperatureMin, meteoSettings);
                meteoPointSet->setMeteoPointValueD(getCrit3DDate(myDate), dailyAirTemperatureMin, value);
                value = meteoPointGet.getMeteoPointValueD(getCrit3DDate(myDate), dailyAirTemperatureMax, meteoSettings);
                meteoPointSet->setMeteoPointValueD(getCrit3DDate(myDate), dailyAirTemperatureMax, value);
            }
            value = meteoPointGet.getMeteoPointValueD(getCrit3DDate(myDate), dailyPrecipitation, meteoSettings);
            meteoPointSet->setMeteoPointValueD(getCrit3DDate(myDate), dailyPrecipitation, value);
            break;
        }

        case dailyAirTemperatureRange:
        {
            float value = meteoPointGet.getMeteoPointValueD(getCrit3DDate(myDate), dailyAirTemperatureMin, meteoSettings);
            meteoPointSet->setMeteoPointValueD(getCrit3DDate(myDate), dailyAirTemperatureMin, value);
            value = meteoPointGet.getMeteoPointValueD(getCrit3DDate(myDate), dailyAirTemperatureMax, meteoSettings);
            meteoPointSet->setMeteoPointValueD(getCrit3DDate(myDate), dailyAirTemperatureMax, value);
            break;
        }

        case dailyAirTemperatureAvg:
        {
            float valueTavg = meteoPointGet.getMeteoPointValueD(getCrit3DDate(myDate), dailyAirTemperatureAvg, meteoSettings);
            quality::qualityType qualityTavg = qualityCheck.syntacticQualitySingleValue(dailyAirTemperatureAvg, valueTavg);
            if (qualityTavg == quality::accepted)
            {
                meteoPointSet->setMeteoPointValueD(getCrit3DDate(myDate), dailyAirTemperatureAvg, valueTavg);
            }
            break;
        }

        case dailyReferenceEvapotranspirationHS:
        {
            if (automaticETP)
            {
                float value = meteoPointGet.getMeteoPointValueD(getCrit3DDate(myDate), dailyAirTemperatureMin, meteoSettings);
                meteoPointSet->setMeteoPointValueD(getCrit3DDate(myDate), dailyAirTemperatureMin, value);
                value = meteoPointGet.getMeteoPointValueD(getCrit3DDate(myDate), dailyAirTemperatureMax, meteoSettings);
                meteoPointSet->setMeteoPointValueD(getCrit3DDate(myDate), dailyAirTemperatureMax, value);
            }
            break;
        }
        case dailyHeatingDegreeDays: case dailyCoolingDegreeDays:
        {
            float value = meteoPointGet.getMeteoPointValueD(getCrit3DDate(myDate), dailyAirTemperatureAvg, meteoSettings);
            meteoPointSet->setMeteoPointValueD(getCrit3DDate(myDate), dailyAirTemperatureAvg, value);
            value = meteoPointGet.getMeteoPointValueD(getCrit3DDate(myDate), dailyAirTemperatureMin, meteoSettings);
            meteoPointSet->setMeteoPointValueD(getCrit3DDate(myDate), dailyAirTemperatureMin, value);
            value = meteoPointGet.getMeteoPointValueD(getCrit3DDate(myDate), dailyAirTemperatureMax, meteoSettings);
            meteoPointSet->setMeteoPointValueD(getCrit3DDate(myDate), dailyAirTemperatureMax, value);
            break;
        }

        default:
        {
            float value = meteoPointGet.getMeteoPointValueD(getCrit3DDate(myDate), myVar, meteoSettings);
            meteoPointSet->setMeteoPointValueD(getCrit3DDate(myDate), myVar, value);
            break;
        }
    }
}


meteoComputation getMeteoCompFromString(std::map<std::string, meteoComputation> map, std::string value)
{

    std::map<std::string, meteoComputation>::const_iterator it;
    meteoComputation meteoValue = noMeteoComp;
    QString valueLower = QString::fromStdString(value).toLower();
    QString mapStringToLower;

    for (it = map.begin(); it != map.end(); ++it)
    {
        mapStringToLower = QString::fromStdString(it->first).toLower();
        if (mapStringToLower == valueLower)
        {
            meteoValue = it->second;
            break;
        }
    }
    return meteoValue;
}
