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
#include "proxyWidget.h"
#include "utilities.h"
#include "interpolation.h"
#include "spatialControl.h"
#include "commonConstants.h"
#include "formInfo.h"

#include <QLayout>
#include <QDate>

Crit3DProxyWidget::Crit3DProxyWidget(Crit3DInterpolationSettings* interpolationSettings, Crit3DMeteoPoint *meteoPoints, int nrMeteoPoints, frequencyType currentFrequency, QDate currentDate, int currentHour, Crit3DQuality *quality, Crit3DInterpolationSettings* SQinterpolationSettings, Crit3DMeteoSettings *meteoSettings, Crit3DClimateParameters *climateParam, bool checkSpatialQuality)
:interpolationSettings(interpolationSettings), meteoPoints(meteoPoints), nrMeteoPoints(nrMeteoPoints), currentFrequency(currentFrequency), currentDate(currentDate), currentHour(currentHour), quality(quality), SQinterpolationSettings(SQinterpolationSettings), meteoSettings(meteoSettings), climateParam(climateParam), checkSpatialQuality(checkSpatialQuality)
{
    this->setWindowTitle("Proxy analysis over " + QString::number(nrMeteoPoints) +  " points");
    this->resize(1024, 700);
    this->setSizePolicy(QSizePolicy::Preferred, QSizePolicy::Preferred);
    this->setAttribute(Qt::WA_DeleteOnClose);
    
    // layout
    QVBoxLayout *mainLayout = new QVBoxLayout();
    QGroupBox *horizontalGroupBox = new QGroupBox();
    QHBoxLayout *selectionLayout = new QHBoxLayout;
    QHBoxLayout *selectionChartLayout = new QHBoxLayout;
    QVBoxLayout *selectionOptionLayout = new QVBoxLayout;
    QHBoxLayout *selectionOptionBoxLayout = new QHBoxLayout;
    QHBoxLayout *selectionOptionEditLayout = new QHBoxLayout;

    detrended.setText("Detrended data");
    climatologicalLR.setText("Climatological lapse rate");
    modelLR.setText("Model lapse rate");
    
    QLabel *r2Label = new QLabel(tr("R2"));
    QLabel *lapseRateLabel = new QLabel(tr("Lapse rate"));
    
    r2.setMaximumWidth(60);
    r2.setMaximumHeight(30);
    r2.setEnabled(false);
    lapseRate.setMaximumWidth(60);
    lapseRate.setMaximumHeight(30);
    lapseRate.setEnabled(false);
    
    QLabel *variableLabel = new QLabel(tr("Variable"));
    QLabel *axisXLabel = new QLabel(tr("Axis X"));

    std::vector<Crit3DProxy> proxy = interpolationSettings->getCurrentProxy();

    for(int i=0; i<int(proxy.size()); i++)
    {
        axisX.addItem(QString::fromStdString(proxy[i].getName()));
    }
    proxyPos = 0;
    axisX.setSizeAdjustPolicy(QComboBox::AdjustToContents);
    if (axisX.currentText() != "elevation")
    {
        climatologicalLR.setVisible(false);
    }
    else
    {
        climatologicalLR.setVisible(true);
    }

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
    variable.setSizeAdjustPolicy(QComboBox::AdjustToContents);
    
    selectionChartLayout->addWidget(variableLabel);
    selectionChartLayout->addWidget(&variable);
    selectionChartLayout->addWidget(axisXLabel);
    selectionChartLayout->addWidget(&axisX);
    
    selectionOptionBoxLayout->addWidget(&detrended);
    selectionOptionBoxLayout->addWidget(&modelLR);
    selectionOptionBoxLayout->addWidget(&climatologicalLR);

    selectionOptionEditLayout->addWidget(r2Label);
    selectionOptionEditLayout->addWidget(&r2);
    selectionOptionEditLayout->addStretch(150);
    selectionOptionEditLayout->addWidget(lapseRateLabel);
    selectionOptionEditLayout->addWidget(&lapseRate);
    selectionOptionEditLayout->addStretch(150);
    selectionOptionEditLayout->addStretch(150);

    selectionOptionLayout->addLayout(selectionOptionBoxLayout);
    selectionOptionLayout->addLayout(selectionOptionEditLayout);

    selectionLayout->addLayout(selectionChartLayout);
    selectionLayout->addStretch(30);
    selectionLayout->addLayout(selectionOptionLayout);
    horizontalGroupBox->setMaximumSize(1240, 130);
    horizontalGroupBox->setLayout(selectionLayout);

    chartView = new ChartView();
    chartView->setMinimumHeight(200);
    QStatusBar* statusBar = new QStatusBar();

    mainLayout->addWidget(horizontalGroupBox);
    mainLayout->addWidget(chartView);
    mainLayout->addWidget(statusBar);

    setLayout(mainLayout);
    
    connect(&axisX, &QComboBox::currentTextChanged, [=](const QString &newProxy){ this->changeProxyPos(newProxy); });
    connect(&variable, &QComboBox::currentTextChanged, [=](const QString &newVariable){ this->changeVar(newVariable); });
    connect(&climatologicalLR, &QCheckBox::toggled, [=](int toggled){ this->climatologicalLRClicked(toggled); });
    connect(&modelLR, &QCheckBox::toggled, [=](int toggled){ this->modelLRClicked(toggled); });
    connect(&detrended, &QCheckBox::toggled, [=](){ this->plot(); });

    if (currentFrequency != noFrequency)
    {
        plot();
    }

    show();
}


Crit3DProxyWidget::~Crit3DProxyWidget()
{

}

void Crit3DProxyWidget::changeProxyPos(const QString proxyName)
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

void Crit3DProxyWidget::changeVar(const QString varName)
{
    if (varName == "ELABORATION")
    {
        myVar = elaboration;
    }
    else if (varName == "ANOMALY")
    {
        myVar = anomaly;
    }
    else
    {
        if (currentFrequency == daily)
        {
            myVar = getKeyMeteoVarMeteoMap(MapDailyMeteoVarToString, varName.toStdString());
        }
        else if (currentFrequency == hourly)
        {
            myVar = getKeyMeteoVarMeteoMap(MapHourlyMeteoVarToString, varName.toStdString());
        }
    }
    plot();
}

void Crit3DProxyWidget::updateDateTime(QDate newDate, int newHour)
{
    currentDate = newDate;
    currentHour = newHour;
    plot();
}

void Crit3DProxyWidget::updateFrequency(frequencyType newFrequency)
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

void Crit3DProxyWidget::closeEvent(QCloseEvent *event)
{
    emit closeProxyWidget();
    event->accept();

}

void Crit3DProxyWidget::plot()
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

void Crit3DProxyWidget::climatologicalLRClicked(int toggled)
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

void Crit3DProxyWidget::modelLRClicked(int toggled)
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

