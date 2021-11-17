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
    
    this->setWindowTitle("Proxy analysis");
    this->resize(1240, 700);
    this->setSizePolicy(QSizePolicy::Expanding, QSizePolicy::Expanding);
    this->setAttribute(Qt::WA_DeleteOnClose);
    

    // layout
    QVBoxLayout *mainLayout = new QVBoxLayout();
    QGroupBox *horizontalGroupBox = new QGroupBox();
    QHBoxLayout *selectionLayout = new QHBoxLayout;
    QHBoxLayout *selectionChartLayout = new QHBoxLayout;
    QVBoxLayout *selectionOptionLayout = new QVBoxLayout;
    QHBoxLayout *selectionOptionBoxLayout = new QHBoxLayout;
    QHBoxLayout *selectionOptionEditLayout = new QHBoxLayout;
    QVBoxLayout *plotLayout = new QVBoxLayout;

    detrended.setText("Detrended data");
    climatologicalLR.setText("Climatological lapse rate");
    modelLR.setText("Model lapse rate");
    
    QLabel *r2Label = new QLabel(tr("R2"));
    QLabel *lapseRateLabel = new QLabel(tr("Lapse rate"));
    
    r2.setMaximumWidth(50);
    r2.setMaximumHeight(30);
    lapseRate.setMaximumWidth(50);
    lapseRate.setMaximumHeight(30);
    
    QLabel *variableLabel = new QLabel(tr("Variable"));
    QLabel *axisXLabel = new QLabel(tr("Axis X"));

    std::vector<Crit3DProxy> proxy = interpolationSettings->getCurrentProxy();

    for(int i=0; i<proxy.size(); i++)
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
    variable.setMinimumWidth(130);
    variable.setSizeAdjustPolicy(QComboBox::AdjustToContents);
    
    selectionChartLayout->addWidget(variableLabel);
    selectionChartLayout->addWidget(&variable);
    selectionChartLayout->addWidget(axisXLabel);
    selectionChartLayout->addWidget(&axisX);
    
    selectionOptionBoxLayout->addWidget(&detrended);
    selectionOptionBoxLayout->addWidget(&climatologicalLR);
    selectionOptionBoxLayout->addWidget(&modelLR);

    selectionOptionEditLayout->addWidget(r2Label);
    selectionOptionEditLayout->addWidget(&r2);
    selectionOptionEditLayout->addStretch(200);
    selectionOptionEditLayout->addWidget(lapseRateLabel);
    selectionOptionEditLayout->addWidget(&lapseRate);
    selectionOptionEditLayout->addStretch(200);
    selectionOptionEditLayout->addStretch(200);

    selectionOptionLayout->addLayout(selectionOptionBoxLayout);
    selectionOptionLayout->addLayout(selectionOptionEditLayout);

    selectionLayout->addLayout(selectionChartLayout);
    selectionLayout->addStretch(50);
    selectionLayout->addLayout(selectionOptionLayout);
    
    connect(&axisX, &QComboBox::currentTextChanged, [=](const QString &newProxy){ this->changeProxyPos(newProxy); });
    connect(&variable, &QComboBox::currentTextChanged, [=](const QString &newVariable){ this->changeVar(newVariable); });
    connect(&climatologicalLR, &QCheckBox::toggled, [=](int toggled){ this->climatologicalLRClicked(toggled); });
    connect(&modelLR, &QCheckBox::toggled, [=](int toggled){ this->modelLRClicked(toggled); });

    // init
    zMin = NODATA;
    zMax = NODATA;

    // menu
    QMenuBar* menuBar = new QMenuBar();
    QMenu *editMenu = new QMenu("Edit");

    menuBar->addMenu(editMenu);
    mainLayout->setMenuBar(menuBar);

    /*
    QAction* changeLeftAxis = new QAction(tr("&Change axis left"), this);
    QAction* changeRightAxis = new QAction(tr("&Change axis right"), this);
    QAction* exportGraph = new QAction(tr("&Export graph"), this);

    editMenu->addAction(changeLeftAxis);
    editMenu->addAction(changeRightAxis);
    editMenu->addAction(exportGraph);
    */

    chartView = new ChartView();
    plotLayout->addWidget(chartView);

    horizontalGroupBox->setMaximumSize(1240, 130);
    horizontalGroupBox->setLayout(selectionLayout);
    mainLayout->addWidget(horizontalGroupBox);
    mainLayout->addLayout(plotLayout);
    setLayout(mainLayout);

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
    for (int pos=0; pos < interpolationSettings->getProxyNr(); pos++)
    {
        QString myProxy = QString::fromStdString(interpolationSettings->getProxy(pos)->getName());
        if (myProxy == proxyName)
        {
            proxyPos = pos;
            break;
        }
    }
    if (proxyName != "elevation")
    {
        climatologicalLR.setVisible(false);
    }
    else
    {
        climatologicalLR.setVisible(true);
    }
    plot();
}

void Crit3DProxyWidget::changeVar(const QString varName)
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
    if (climatologicalLR.isChecked())
    {
        climatologicalLRClicked(1);
    }
    if (modelLR.isChecked())
    {
        modelLRClicked(1);
    }
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
    outInterpolationPoints.clear();
    checkAndPassDataToInterpolation(quality, myVar, meteoPoints, nrMeteoPoints, getCrit3DTime(currentDate, currentHour), SQinterpolationSettings, interpolationSettings, meteoSettings, climateParam, outInterpolationPoints, checkSpatialQuality);

    QList<QPointF> point_vector;
    QList<QPointF> point_vector2;
    QList<QPointF> point_vector3;
    QMap< QString, QPointF > idPointMap;
    QMap< QString, QPointF > idPointMap2;
    QMap< QString, QPointF > idPointMap3;

    QPointF point;
    for (int i = 0; i<outInterpolationPoints.size(); i++)
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

}

void Crit3DProxyWidget::climatologicalLRClicked(int toggled)
{
    chartView->cleanClimLapseRate();
    if (toggled && outInterpolationPoints.size() != 0)
    {
        if (zMax == NODATA || zMin == NODATA)
        {
            zMax = getZmax(outInterpolationPoints);
            zMin = getZmin(outInterpolationPoints);
        }
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
    QList<QPointF> point_vector;
    QPointF point;
    float xMin;
    float xMax;
    if (toggled && outInterpolationPoints.size() != 0)
    {
        if (axisX.currentText() == "elevation")
        {
            if (zMax == NODATA || zMin == NODATA)
            {
                zMax = getZmax(outInterpolationPoints);
                zMin = getZmin(outInterpolationPoints);
            }
            xMin = zMin;
            xMax = zMax;

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
            xMin = 0;
            xMax = 360;
            bool isZeroIntercept = false;
            if (!regressionGeneric(outInterpolationPoints, interpolationSettings, proxyPos, isZeroIntercept))
            {
                return;
            }
            point.setX(xMin);
            //point.setY(Interpolation.AspectIntercept + Interpolation.AspectCoefficient * xMin);
            point_vector.append(point);
            point.setX(xMax);
            //point.setY(Interpolation.AspectIntercept + Interpolation.AspectCoefficient * xMax);
            point_vector.append(point);

            // Me.TxtR2.Text = format(Interpolation.AspectR2, "0.0000") corrisponde a sotto? Il controllo sul NODATA va aggiunto?
            /*
            if (interpolationSettings->getProxy(proxyPos)->getRegressionR2() != NODATA)
            {
                r2.setText(QString("%1").arg(interpolationSettings->getProxy(proxyPos)->getRegressionR2(), 0, 'f', 4));
            }
            */
        }
        chartView->drawModelLapseRate(point_vector);
    }
}


