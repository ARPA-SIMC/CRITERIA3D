/*!
    \copyright 2020 Fausto Tomei, Gabriele Antolini,
    Alberto Pistocchi, Marco Bittelli, Antonio Volta, Laura Costantini

    This file is part of AGROLIB.
    AGROLIB has been developed under contract issued by ARPAE Emilia-Romagna

    AGROLIB is free software: you can redistribute it and/or modify
    it under the terms of the GNU Lesser General Public License as published by
    the Free Software Foundation, either version 3 of the License, or
    (at your option) any later version.

    AGROLIB is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
    GNU Lesser General Public License for more details.

    You should have received a copy of the GNU Lesser General Public License
    along with AGROLIB.  If not, see <http://www.gnu.org/licenses/>.

    contacts:
    ftomei@arpae.it
    gantolini@arpae.it
*/

#include "meteo.h"
#include "synchronicityWidget.h"
#include "synchronicityChartView.h"
#include "utilities.h"
#include "interpolation.h"
#include "spatialControl.h"
#include "commonConstants.h"
#include "basicMath.h"
#include "climate.h"
#include "dialogChangeAxis.h"
#include "gammaFunction.h"
#include "furtherMathFunctions.h"

#include <QProgressDialog>
#include <QLayout>
#include <QDate>


Crit3DSynchronicityWidget::~Crit3DSynchronicityWidget()
{}

void Crit3DSynchronicityWidget::closeEvent(QCloseEvent *event)
{
    emit closeSynchWidget();
    event->accept();
}

void Crit3DSynchronicityWidget::setReferencePointId(const std::string &value)
{
    if (referencePointId != value)
    {
        mpRef.cleanObsDataD();
        mpRef.clear();
        referencePointId = value;
        firstRefDaily = meteoPointsDbHandler->getFirstDate(daily, value).date();
        lastRefDaily = meteoPointsDbHandler->getLastDate(daily, value).date();
        bool hasDailyData = !(firstRefDaily.isNull() || lastRefDaily.isNull());

        if (!hasDailyData)
        {
            QMessageBox::information(nullptr, "Error", "Reference point has not daiy data");
            stationAddGraph.setEnabled(false);
            return;
        }
        QString errorString;
        meteoPointsDbHandler->getPropertiesGivenId(QString::fromStdString(value), mpRef, gisSettings, errorString);
        nameRefLabel.setText("Id "+ QString::fromStdString(referencePointId) + " - " + QString::fromStdString(mpRef.name));
        stationAddGraph.setEnabled(true);
    }
}


void Crit3DSynchronicityWidget::changeVar(const QString varName)
{
    myVar = getKeyMeteoVarMeteoMap(MapDailyMeteoVarToString, varName.toStdString());
}

void Crit3DSynchronicityWidget::changeYears()
{
    stationClearAndReload = true;
}

void Crit3DSynchronicityWidget::addStationGraph()
{

    if (referencePointId == "")
    {
        QMessageBox::information(nullptr, "Error", "Select a reference point on the map");
        return;
    }
    if (stationClearAndReload)
    {
        clearStationGraph();
        stationClearAndReload = false;
    }
    QDate myStartDate(stationYearFrom.currentText().toInt(), 1, 1);
    QDate myEndDate(stationYearTo.currentText().toInt(), 12, 31);
    int myLag = stationLag.text().toInt();

    std::vector<float> dailyValues;
    QString myError;
    dailyValues = meteoPointsDbHandler->loadDailyVar(myVar, getCrit3DDate(myStartDate.addDays(myLag)), getCrit3DDate(myEndDate.addDays(myLag)), mp, firstDaily);
    if (dailyValues.empty())
    {
        QMessageBox::information(nullptr, "Error", "No data for active station");
        return;
    }
    std::vector<float> dailyRefValues;
    dailyRefValues = meteoPointsDbHandler->loadDailyVar(myVar, getCrit3DDate(myStartDate), getCrit3DDate(myEndDate), mpRef, firstRefDaily);
    if (dailyRefValues.empty())
    {
        QMessageBox::information(nullptr, "Error", "No data for reference station");
        return;
    }

    if (firstDaily.addDays(std::min(0,myLag)) > firstRefDaily)
    {
        if (firstDaily.addDays(std::min(0,myLag)) > QDate(stationYearFrom.currentText().toInt(), 1, 1))
        {
            myStartDate = firstDaily.addDays(std::min(0,myLag));
        }
        else
        {
            myStartDate = QDate(stationYearFrom.currentText().toInt(), 1, 1);
        }
    }
    else
    {
        if (firstRefDaily > QDate(stationYearFrom.currentText().toInt(), 1, 1))
        {
            myStartDate = firstRefDaily;
        }
        else
        {
            myStartDate = QDate(stationYearFrom.currentText().toInt(), 1, 1);
        }

    }

    if (firstDaily.addDays(dailyValues.size()-1-std::max(0,myLag)) < firstRefDaily.addDays(dailyRefValues.size()-1))
    {
        if (firstDaily.addDays(dailyValues.size()-1-std::max(0,myLag)) < QDate(stationYearTo.currentText().toInt(), 12, 31))
        {
            myEndDate = firstDaily.addDays(dailyValues.size()-1-std::max(0,myLag));
        }
        else
        {
            myEndDate = QDate(stationYearTo.currentText().toInt(), 12, 31);
        }

    }
    else
    {
        if (firstRefDaily.addDays(dailyRefValues.size()-1) < QDate(stationYearTo.currentText().toInt(), 12, 31))
        {
            myEndDate = firstRefDaily.addDays(dailyRefValues.size()-1);
        }
        else
        {
            myEndDate =  QDate(stationYearTo.currentText().toInt(), 12, 31);
        }
    }
    QDate currentDate = myStartDate;
    int currentYear = currentDate.year();
    std::vector<float> myX;
    std::vector<float> myY;
    QList<QPointF> pointList;
    float minPerc = meteoSettings->getMinimumPercentage();
    while (currentDate <= myEndDate)
    {
        float myValue1 = dailyValues[firstDaily.daysTo(currentDate)+myLag];
        float myValue2 = dailyRefValues[firstRefDaily.daysTo(currentDate)];
        if (myValue1 != NODATA && myValue2 != NODATA)
        {
            myX.push_back(myValue1);
            myY.push_back(myValue2);
        }
        if ( currentDate == QDate(currentYear, 12, 31)  || currentDate == myEndDate)
        {
            float days = 365;
            if (isLeapYear(currentYear))
            {
                days = 366;
            }
            float r2, y_intercept, trend;

            if ((float)myX.size() / days * 100.0 > minPerc)
            {
                statistics::linearRegression(myX, myY, int(myX.size()), false, &y_intercept, &trend, &r2);
            }
            else
            {
                r2 = NODATA;
            }
            if (r2 != NODATA)
            {
                pointList.append(QPointF(currentYear,r2));
            }
            myX.clear();
            myY.clear();
            currentYear = currentYear + 1;
        }
        currentDate = currentDate.addDays(1);
    }
    // draw
    synchronicityChartView->drawGraphStation(pointList, variable.currentText(), myLag);

}

void Crit3DSynchronicityWidget::clearStationGraph()
{
    synchronicityChartView->clearStationGraphSeries();
}

void Crit3DSynchronicityWidget::clearInterpolationGraph()
{
    interpolationChartView->clearInterpolationGraphSeries();
    interpolationChartView->setVisible(false);
}

void Crit3DSynchronicityWidget::addInterpolationGraph()
{
    if (interpolationClearAndReload)
    {
        clearInterpolationGraph();
        interpolationClearAndReload = false;
    }
    interpolationStartDate = interpolationDateFrom.date();
    QDate myEndDate = interpolationDateTo.date();
    int myLag = interpolationLag.text().toInt();
    int mySmooth = smooth.text().toInt();
    QString elabType = interpolationElab.currentText();

    if (interpolationChangeSmooth)
    {
        // draw without compute all series
        interpolationChartView->setVisible(true);
        smoothSeries();
        // draw
        interpolationChartView->drawGraphInterpolation(smoothInterpDailySeries, interpolationStartDate, variable.currentText(), myLag, mySmooth, elabType);
        interpolationChangeSmooth = false;
        return;
    }
    interpolationDailySeries.clear();

    std::vector<float> dailyValues;
    dailyValues = meteoPointsDbHandler->loadDailyVar(myVar, getCrit3DDate(interpolationStartDate.addDays(myLag)), getCrit3DDate(myEndDate.addDays(myLag)), mp, firstDaily);
    if (dailyValues.empty())
    {
        QMessageBox::information(nullptr, "Error", "No data for active station");
        return;
    }

    if (firstDaily.addDays(std::min(0,myLag)) > interpolationStartDate)
    {
        interpolationStartDate = firstDaily.addDays(std::min(0,myLag));
    }
    if (firstDaily.addDays(dailyValues.size()-1-std::max(0,myLag)) < myEndDate)
    {
        myEndDate = firstDaily.addDays(dailyValues.size()-1-std::max(0,myLag));
    }

    std::vector <Crit3DInterpolationDataPoint> interpolationPoints;

    // load data
    QProgressDialog progress("Loading daily data...", "Abort", 0, nrMeteoPoints, this);
    progress.setWindowModality(Qt::WindowModal);
    progress.show();

    for (int i = 0; i<nrMeteoPoints; i++)
    {
        progress.setValue(i+1);
        if (progress.wasCanceled())
        {
            break;
        }
        meteoPointsDbHandler->loadDailyData(getCrit3DDate(interpolationStartDate), getCrit3DDate(myEndDate), meteoPoints[i]);
    }
    progress.close();

    std::string errorStdStr;
    for (QDate currentDate = interpolationStartDate; currentDate <= myEndDate; currentDate = currentDate.addDays(1))
    {
        float myValue1 = dailyValues[firstDaily.daysTo(currentDate)+myLag];
        // check quality and pass data to interpolation
        if (!checkAndPassDataToInterpolation(quality, myVar, meteoPoints, nrMeteoPoints, getCrit3DTime(currentDate, 1),
                                             &qualityInterpolationSettings, &interpolationSettings, meteoSettings, climateParameters, interpolationPoints,
                                             checkSpatialQuality, errorStdStr))
        {
            QMessageBox::critical(nullptr, "Error", "No data available");
            return;
        }
        if (! preInterpolation(interpolationPoints, &interpolationSettings, meteoSettings, climateParameters,
                              meteoPoints, nrMeteoPoints, myVar, getCrit3DTime(currentDate, 1), errorStdStr))
        {
            QMessageBox::critical(nullptr, "Error", "Error in function preInterpolation: " + QString::fromStdString(errorStdStr));
            return;
        }
        float interpolatedValue = interpolate(interpolationPoints, &interpolationSettings, meteoSettings, myVar,
                                              float(mp.point.utm.x),
                                              float(mp.point.utm.y),
                                              float(mp.point.z),
                                              mp.getProxyValues(), false);

        if (myValue1 != NODATA && interpolatedValue != NODATA)
        {
            if (elabType == "Difference")
            {
                interpolationDailySeries.push_back(interpolatedValue - myValue1);
            }
            else if (elabType == "Square difference")
            {
                interpolationDailySeries.push_back((interpolatedValue - myValue1) * (interpolatedValue - myValue1));
            }
            else if (elabType == "Absolute difference")
            {
                interpolationDailySeries.push_back(abs(interpolatedValue - myValue1));
            }
        }
        else
        {
            interpolationDailySeries.push_back(NODATA);
        }

    }
    if (interpolationDailySeries.empty())
    {
        QMessageBox::information(nullptr, "Error", "No data available");
        return;
    }
    else
    {
        interpolationChartView->setVisible(true);
        // smooth
        smoothSeries();
        // draw
        interpolationChartView->drawGraphInterpolation(smoothInterpDailySeries, interpolationStartDate, variable.currentText(), myLag, mySmooth, elabType);
    }

}

void Crit3DSynchronicityWidget::smoothSeries()
{
    int mySmooth = smooth.text().toInt();
    QString elabType = interpolationElab.currentText();
    smoothInterpDailySeries.clear();

    if (mySmooth > 0)
    {
        for (int i = 0; i < interpolationDailySeries.size(); i++)
        {
            float dSum = 0;
            int nDays = 0;
            for (int j = i-mySmooth; j<=i+mySmooth; j++)
            {
                if (j >= 0 && j < interpolationDailySeries.size())
                {
                    if (interpolationDailySeries[j] != NODATA)
                    {
                        dSum = dSum + interpolationDailySeries[j];
                        nDays = nDays + 1;
                    }
                }
            }
            if (nDays / (mySmooth * 2 + 1) > meteoSettings->getMinimumPercentage() / 100)
            {
                smoothInterpDailySeries.push_back(dSum/nDays);
            }
            else
            {
                smoothInterpDailySeries.push_back(NODATA);
            }
        }
    }
    else
    {
        smoothInterpDailySeries = interpolationDailySeries;
    }

}

void Crit3DSynchronicityWidget::changeSmooth()
{
    interpolationChangeSmooth = true;
}

void Crit3DSynchronicityWidget::changeInterpolationDate()
{
    interpolationClearAndReload = true;
}

void Crit3DSynchronicityWidget::on_actionChangeLeftSynchAxis()
{
    DialogChangeAxis changeAxisDialog(1, false);
    if (changeAxisDialog.result() == QDialog::Accepted)
    {
        synchronicityChartView->setYmax(changeAxisDialog.getMaxVal());
        synchronicityChartView->setYmin(changeAxisDialog.getMinVal());
    }
}

void Crit3DSynchronicityWidget::on_actionChangeLeftInterpolationAxis()
{
    DialogChangeAxis changeAxisDialog(1, false);
    if (changeAxisDialog.result() == QDialog::Accepted)
    {
        interpolationChartView->setYmax(changeAxisDialog.getMaxVal());
        interpolationChartView->setYmin(changeAxisDialog.getMinVal());
    }
}


