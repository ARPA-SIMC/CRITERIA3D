/*!
    CRITERIA3D
    \copyright 2016 Fausto Tomei, Gabriele Antolini, Laura Costantini
    Alberto Pistocchi, Marco Bittelli, Antonio Volta
    You should have received a copy of the GNU General Public License
    along with Nome-Programma.  If not, see <http://www.gnu.org/licenses/>.
    This file is part of CRITERIA3D.
    CRITERIA3D has been developed under contract issued by A.R.P.A. Emilia-Romagna
    CRITERIA3D is free software: you can redistribute it and/or modify
    it under the terms of the GNU Lesser General Public License as published by
    the Free Software Foundation, either version 3 of the License, or
    (at your option) any later version.
    CRITERIA3D is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
    GNU Lesser General Public License for more details.
    You should have received a copy of the /NU Lesser General Public License
    along with CRITERIA3D.  If not, see <http://www.gnu.org/licenses/>.
    contacts:
    fausto.tomei@gmail.com
    ftomei@arpae.it
*/

#include "meteo.h"
#include "pointStatisticsWidget.h"
#include "utilities.h"
#include "interpolation.h"
#include "spatialControl.h"
#include "commonConstants.h"
#include "formInfo.h"

#include <QLayout>
#include <QDate>

Crit3DPointStatisticsWidget::Crit3DPointStatisticsWidget(bool isGrid, QList<Crit3DMeteoPoint> meteoPoints, frequencyType currentFrequency)
:isGrid(isGrid), meteoPoints(meteoPoints), currentFrequency(currentFrequency)
{
    this->setWindowTitle("Point statistics");
    this->resize(1240, 700);
    this->setSizePolicy(QSizePolicy::Expanding, QSizePolicy::Expanding);
    this->setAttribute(Qt::WA_DeleteOnClose);
    
    // layout
    QVBoxLayout *mainLayout = new QVBoxLayout();
    QHBoxLayout *upperLayout = new QHBoxLayout();
    QVBoxLayout *rightLayout = new QVBoxLayout();
    QVBoxLayout *leftLayout = new QVBoxLayout();

    QGroupBox *horizontalGroupBox = new QGroupBox();
    QHBoxLayout *variableLayout = new QHBoxLayout;
    QGroupBox *referencePeriodGroupBox = new QGroupBox();
    QHBoxLayout *referencePeriodChartLayout = new QHBoxLayout;
    QHBoxLayout *dateChartLayout = new QHBoxLayout;

    QGroupBox *jointStationsGroupBox = new QGroupBox();
    QHBoxLayout *jointStationsLayout = new QHBoxLayout;
    QVBoxLayout *jointStationsSelectLayout = new QVBoxLayout;

    QVBoxLayout *plotLayout = new QVBoxLayout;

    QLabel *variableLabel = new QLabel(tr("Variable"));
    std::map<meteoVariable, std::string>::const_iterator it;
    if (currentFrequency == daily)
    {
        for(it = MapDailyMeteoVarToString.begin(); it != MapDailyMeteoVarToString.end(); ++it)
        {
            variable.addItem(QString::fromStdString(it->second));
        }
        myVar = getKeyMeteoVarMeteoMap(MapDailyMeteoVarToString, variable.currentText().toStdString());
    }
    else if (currentFrequency == hourly)
    {
        for(it = MapHourlyMeteoVarToString.begin(); it != MapHourlyMeteoVarToString.end(); ++it)
        {
            variable.addItem(QString::fromStdString(it->second));
        }
        myVar = getKeyMeteoVarMeteoMap(MapHourlyMeteoVarToString, variable.currentText().toStdString());
    }
    variable.setMinimumWidth(130);
    variable.setSizeAdjustPolicy(QComboBox::AdjustToContents);
    elaboration.setText("Elaboration");
    
    variableLayout->addWidget(variableLabel);
    variableLayout->addWidget(&variable);
    variableLayout->addWidget(&elaboration);
    referencePeriodGroupBox->setTitle("Reference period");
    referencePeriodGroupBox->setLayout(variableLayout);

    QLabel *yearFromLabel = new QLabel(tr("From"));
    referencePeriodChartLayout->addWidget(yearFromLabel);
    referencePeriodChartLayout->addWidget(&yearFrom);
    QLabel *yearToLabel = new QLabel(tr("To"));
    referencePeriodChartLayout->addWidget(yearToLabel);
    referencePeriodChartLayout->addWidget(&yearTo);

    QLabel *dayFromLabel = new QLabel(tr("Day from"));
    dateChartLayout->addWidget(dayFromLabel);
    dayFrom.setDisplayFormat("dd/MM");
    dateChartLayout->addWidget(&dayFrom);
    QLabel *dayToLabel = new QLabel(tr("Day to"));
    dateChartLayout->addWidget(dayToLabel);
    dayTo.setDisplayFormat("dd/MM");
    dateChartLayout->addWidget(&dayTo);
    QLabel *hourLabel = new QLabel(tr("Hour"));
    hour.setRange(1,24);
    hour.setSingleStep(1);
    dateChartLayout->addWidget(hourLabel);
    dateChartLayout->addWidget(&hour);
    compute.setText("Compute");

    jointStationsSelectLayout->addWidget(&jointStationsList);
    QHBoxLayout *addDeleteStationLayout = new QHBoxLayout;
    addDeleteStationLayout->addWidget(&addStation);
    addStation.setText("Add");
    deleteStation.setText("Delete");
    saveToDb.setText("Save to DB");
    addDeleteStationLayout->addWidget(&deleteStation);
    jointStationsSelectLayout->addLayout(addDeleteStationLayout);
    jointStationsSelectLayout->addWidget(&saveToDb);
    jointStationsLayout->addLayout(jointStationsSelectLayout);
    jointStationsSelected.setMaximumWidth(this->width()/4);
    jointStationsLayout->addWidget(&jointStationsSelected);
    jointStationsGroupBox->setTitle("Joint stations");
    jointStationsGroupBox->setLayout(jointStationsLayout);

    chartView = new PointStatisticsChartView();
    plotLayout->addWidget(chartView);

    horizontalGroupBox->setMaximumSize(1240, 130);
    horizontalGroupBox->setLayout(variableLayout);
    rightLayout->addWidget(horizontalGroupBox);
    referencePeriodGroupBox->setLayout(referencePeriodChartLayout);
    rightLayout->addWidget(referencePeriodGroupBox);
    rightLayout->addLayout(dateChartLayout);
    rightLayout->addWidget(&compute);

    leftLayout->addWidget(jointStationsGroupBox);
    QLabel *selectGraphLabel = new QLabel(tr("Select graph"));
    leftLayout->addWidget(selectGraphLabel);
    leftLayout->addWidget(&graph);
    QLabel *availabilityLabel = new QLabel(tr("availability [%]"));
    leftLayout->addWidget(availabilityLabel);
    availability.setEnabled(false);
    leftLayout->addWidget(&availability);


    upperLayout->addLayout(rightLayout);
    upperLayout->addLayout(leftLayout);
    mainLayout->addLayout(upperLayout);
    mainLayout->addLayout(plotLayout);
    setLayout(mainLayout);

    if (currentFrequency != noFrequency)
    {
        //plot();
    }

    show();
}


Crit3DPointStatisticsWidget::~Crit3DPointStatisticsWidget()
{

}

void Crit3DPointStatisticsWidget::closeEvent(QCloseEvent *event)
{
    emit closePointStatistics();
    event->accept();

}

/*
void Crit3DPointStatisticsWidget::changeProxyPos(const QString proxyName)
{
    for (int pos=0; pos < int(interpolationSettings->getProxyNr()); pos++)
    {
        QString myProxy = QString::fromStdString(interpolationSettings->getProxy(pos)->getName());
        if (myProxy == proxyName)
        {
            proxyPos = pos;
            break;
        }
    }
    plot();
}

void Crit3DPointStatisticsWidget::changeVar(const QString varName)
{
    if (currentFrequency == daily)
    {
        myVar = getKeyMeteoVarMeteoMap(MapDailyMeteoVarToString, varName.toStdString());
    }
    else if (currentFrequency == hourly)
    {
        myVar = getKeyMeteoVarMeteoMap(MapHourlyMeteoVarToString, varName.toStdString());
    }
    plot();
}

void Crit3DPointStatisticsWidget::updateDateTime(QDate newDate, int newHour)
{
    currentDate = newDate;
    currentHour = newHour;
    plot();
}

void Crit3DPointStatisticsWidget::updateFrequency(frequencyType newFrequency)
{
    currentFrequency = newFrequency;
    variable.clear();

    std::map<meteoVariable, std::string>::const_iterator it;
    if (currentFrequency == daily)
    {
        for(it = MapDailyMeteoVarToString.begin(); it != MapDailyMeteoVarToString.end(); ++it)
        {
            variable.addItem(QString::fromStdString(it->second));
        }
        myVar = getKeyMeteoVarMeteoMap(MapDailyMeteoVarToString, variable.currentText().toStdString());
    }
    else if (currentFrequency == hourly)
    {
        for(it = MapHourlyMeteoVarToString.begin(); it != MapHourlyMeteoVarToString.end(); ++it)
        {
            variable.addItem(QString::fromStdString(it->second));
        }
        myVar = getKeyMeteoVarMeteoMap(MapHourlyMeteoVarToString, variable.currentText().toStdString());
    }
    variable.adjustSize();

    plot();
}

void Crit3DPointStatisticsWidget::plot()
{
    chartView->cleanScatterSeries();
    outInterpolationPoints.clear();
    if (detrended.isChecked())
    {
        outInterpolationPoints.clear();
        checkAndPassDataToInterpolation(quality, myVar, meteoPoints, nrMeteoPoints, getCrit3DTime(currentDate, currentHour), SQinterpolationSettings, interpolationSettings, meteoSettings, climateParam, outInterpolationPoints, checkSpatialQuality);
        detrending(outInterpolationPoints, interpolationSettings->getSelectedCombination(), interpolationSettings, climateParam, myVar,
                   getCrit3DTime(currentDate, currentHour));
    }
    else
    {
        checkAndPassDataToInterpolation(quality, myVar, meteoPoints, nrMeteoPoints, getCrit3DTime(currentDate, currentHour), SQinterpolationSettings, interpolationSettings, meteoSettings, climateParam, outInterpolationPoints, checkSpatialQuality);
    }
    QList<QPointF> point_vector;
    QList<QPointF> point_vector2;
    QList<QPointF> point_vector3;
    QMap< QString, QPointF > idPointMap;
    QMap< QString, QPointF > idPointMap2;
    QMap< QString, QPointF > idPointMap3;

    QPointF point;
    for (int i = 0; i < int(outInterpolationPoints.size()); i++)
    {
        if (outInterpolationPoints[i].lapseRateCode == primary)
        {
            float proxyVal = outInterpolationPoints[i].getProxyValue(proxyPos);
            float varVal = outInterpolationPoints[i].value;
            if (proxyVal != NODATA && varVal != NODATA)
            {
                point.setX(proxyVal);
                point.setY(varVal);
                point_vector.append(point);
                idPointMap.insert("id: "+QString::fromStdString(meteoPoints[outInterpolationPoints[i].index].id) + "\nname: "+QString::fromStdString(meteoPoints[outInterpolationPoints[i].index].name), point);
            }
        }
        else if (outInterpolationPoints[i].lapseRateCode == secondary)
        {
            float proxyVal = outInterpolationPoints[i].getProxyValue(proxyPos);
            float varVal = outInterpolationPoints[i].value;
            if (proxyVal != NODATA && varVal != NODATA)
            {
                point.setX(proxyVal);
                point.setY(varVal);
                point_vector2.append(point);
                idPointMap2.insert("id: "+QString::fromStdString(meteoPoints[outInterpolationPoints[i].index].id) + "\nname: "+QString::fromStdString(meteoPoints[outInterpolationPoints[i].index].name), point);
            }
        }
        else if (outInterpolationPoints[i].lapseRateCode == supplemental)
        {
            float proxyVal = outInterpolationPoints[i].getProxyValue(proxyPos);
            float varVal = outInterpolationPoints[i].value;
            if (proxyVal != NODATA && varVal != NODATA)
            {
                point.setX(proxyVal);
                point.setY(varVal);
                point_vector3.append(point);
                idPointMap3.insert("id: "+QString::fromStdString(meteoPoints[outInterpolationPoints[i].index].id) + "\nname: "+QString::fromStdString(meteoPoints[outInterpolationPoints[i].index].name), point);
            }
        }
    }
    chartView->setIdPointMap(idPointMap,idPointMap2,idPointMap3);
    chartView->drawScatterSeries(point_vector, point_vector2, point_vector3);
    if (axisX.currentText() != "elevation")
    {
        chartView->cleanClimLapseRate();
        climatologicalLR.setVisible(false);
    }
    else
    {
        climatologicalLR.setVisible(true);
        if (climatologicalLR.isChecked())
        {
            climatologicalLRClicked(1);
        }
    }
    if (modelLR.isChecked())
    {
        modelLRClicked(1);
    }

}

void Crit3DPointStatisticsWidget::climatologicalLRClicked(int toggled)
{
    chartView->cleanClimLapseRate();
    if (toggled && outInterpolationPoints.size() != 0)
    {
        float zMax = getZmax(outInterpolationPoints);
        float zMin = getZmin(outInterpolationPoints);
        float firstIntervalHeightValue = getFirstIntervalHeightValue(outInterpolationPoints, interpolationSettings->getUseLapseRateCode());
        float lapseRate = climateParam->getClimateLapseRate(myVar, getCrit3DTime(currentDate, currentHour));
        if (lapseRate == NODATA)
        {
            return;
        }
        QPointF firstPoint(zMin, firstIntervalHeightValue);
        QPointF lastPoint(zMax, firstIntervalHeightValue + lapseRate*(zMax - zMin));
        chartView->drawClimLapseRate(firstPoint, lastPoint);
    }
}

void Crit3DPointStatisticsWidget::modelLRClicked(int toggled)
{
    chartView->cleanModelLapseRate();
    r2.clear();
    lapseRate.clear();
    QList<QPointF> point_vector;
    QPointF point;
    float xMin;
    float xMax;
    if (toggled && outInterpolationPoints.size() != 0)
    {
        if (axisX.currentText() == "elevation")
        {
            xMin = getZmin(outInterpolationPoints);
            xMax = getZmax(outInterpolationPoints);

            if (!regressionOrography(outInterpolationPoints,interpolationSettings->getSelectedCombination(), interpolationSettings, climateParam,
                                                               getCrit3DTime(currentDate, currentHour), myVar, proxyPos))
            {
                return;
            }

            float lapseRateH0 = interpolationSettings->getProxy(proxyPos)->getLapseRateH0();
            float lapseRateH1 = interpolationSettings->getProxy(proxyPos)->getLapseRateH1();
            float lapseRateT0 = interpolationSettings->getProxy(proxyPos)->getLapseRateT0();
            float lapseRateT1 = interpolationSettings->getProxy(proxyPos)->getLapseRateT1();
            float regressionSlope = interpolationSettings->getProxy(proxyPos)->getRegressionSlope();

            if (interpolationSettings->getProxy(proxyPos)->getInversionIsSignificative())
            {
                if (xMin < interpolationSettings->getProxy(proxyPos)->getLapseRateH0())
                {
                    point.setX(xMin);
                    point.setY(lapseRateT0);
                    point_vector.append(point);
                }
                point.setX(lapseRateH0);
                point.setY(lapseRateT0);
                point_vector.append(point);

                point.setX(lapseRateH1);
                point.setY(lapseRateT1);
                point_vector.append(point);

                float myY = lapseRateT1 + regressionSlope * (xMax - lapseRateH1);
                point.setX(xMax);
                point.setY(myY);
                point_vector.append(point);
            }
            else
            {
                float myY = lapseRateT0 + regressionSlope * xMin;
                point.setX(xMin);
                point.setY(myY);
                point_vector.append(point);

                myY = lapseRateT0 + regressionSlope * xMax;
                point.setX(xMax);
                point.setY(myY);
                point_vector.append(point);
            }
            if (interpolationSettings->getProxy(proxyPos)->getRegressionR2() != NODATA)
            {
                r2.setText(QString("%1").arg(interpolationSettings->getProxy(proxyPos)->getRegressionR2(), 0, 'f', 4));
            }
            lapseRate.setText(QString("%1").arg(regressionSlope*1000, 0, 'f', 2));
        }
        else
        {
            xMin = getProxyMinValue(outInterpolationPoints, proxyPos);
            xMax = getProxyMaxValue(outInterpolationPoints, proxyPos);
            bool isZeroIntercept = false;
            if (!regressionGeneric(outInterpolationPoints, interpolationSettings, proxyPos, isZeroIntercept))
            {
                return;
            }
            float regressionSlope = interpolationSettings->getProxy(proxyPos)->getRegressionSlope();
            float regressionIntercept = interpolationSettings->getProxy(proxyPos)->getRegressionIntercept();
            point.setX(xMin);
            point.setY(regressionIntercept + regressionSlope * xMin);
            point_vector.append(point);
            point.setX(xMax);
            point.setY(regressionIntercept + regressionSlope * xMax);
            point_vector.append(point);

            if (interpolationSettings->getProxy(proxyPos)->getRegressionR2() != NODATA)
            {
                r2.setText(QString("%1").arg(interpolationSettings->getProxy(proxyPos)->getRegressionR2(), 0, 'f', 4));
            }
            lapseRate.setText(QString("%1").arg(regressionSlope, 0, 'f', 2));
        }
        chartView->drawModelLapseRate(point_vector);
    }
}
*/
