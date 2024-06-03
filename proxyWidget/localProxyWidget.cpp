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
#include "localProxyWidget.h"
#include "proxyWidget.h"
#include "utilities.h"
#include "interpolation.h"
#include "spatialControl.h"
#include "commonConstants.h"
#include "formInfo.h"
#include "math.h"
#include "furtherMathFunctions.h"

#include <QLayout>
#include <QDate>


Crit3DLocalProxyWidget::Crit3DLocalProxyWidget(double x, double y, std::vector<std::vector<double>> parameters, gis::Crit3DGisSettings gisSettings, Crit3DInterpolationSettings* interpolationSettings, Crit3DMeteoPoint *meteoPoints, int nrMeteoPoints, meteoVariable currentVariable, frequencyType currentFrequency, QDate currentDate, int currentHour, Crit3DQuality *quality, Crit3DInterpolationSettings* SQinterpolationSettings, Crit3DMeteoSettings *meteoSettings, Crit3DClimateParameters *climateParam, bool checkSpatialQuality)
    :x(x), y(y), parameters(parameters), gisSettings(gisSettings), interpolationSettings(interpolationSettings), meteoPoints(meteoPoints), nrMeteoPoints(nrMeteoPoints), currentVariable(currentVariable), currentFrequency(currentFrequency), currentDate(currentDate), currentHour(currentHour), quality(quality), SQinterpolationSettings(SQinterpolationSettings), meteoSettings(meteoSettings), climateParam(climateParam), checkSpatialQuality(checkSpatialQuality)
{
    gis::Crit3DGeoPoint localGeoPoint;
    gis::Crit3DUtmPoint localUtmPoint;
    localUtmPoint.x = x;
    localUtmPoint.y = y;
    gis::getLatLonFromUtm(gisSettings, localUtmPoint, localGeoPoint);

    this->setWindowTitle("Local proxy analysis for point of coordinates (" + QString::number(localGeoPoint.latitude) + ", " + QString::number(localGeoPoint.longitude) + ")");
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
    climatologicalLR.setText("Climate lapserate");
    modelLR.setText("Model lapse rate");


    QLabel *r2Label = new QLabel(tr("R2"));
    QLabel *lapseRateLabel = new QLabel(tr("Lapse rate"));

    r2.setMaximumWidth(60);
    r2.setMinimumHeight(25);
    r2.setMaximumHeight(25);
    r2.setEnabled(false);
    lapseRate.setMaximumWidth(60);
    lapseRate.setMinimumHeight(25);
    lapseRate.setMaximumHeight(25);
    lapseRate.setEnabled(false);

    QLabel *variableLabel = new QLabel(tr("Variable"));
    QLabel *axisXLabel = new QLabel(tr("Axis X"));

    std::vector<Crit3DProxy> proxy = interpolationSettings->getCurrentProxy();

    for(int i=0; i<int(proxy.size()); i++)
    {
        comboAxisX.addItem(QString::fromStdString(proxy[i].getName()));
    }
    proxyPos = 0;
    comboAxisX.setSizeAdjustPolicy(QComboBox::AdjustToContents);
    if (comboAxisX.currentText() != "elevation")
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
            comboVariable.addItem(QString::fromStdString(it->second));
        }
        myVar = getKeyMeteoVarMeteoMap(MapDailyMeteoVarToString, comboVariable.currentText().toStdString());
    }
    else if (currentFrequency == hourly)
    {
        for(it = MapHourlyMeteoVarToString.begin(); it != MapHourlyMeteoVarToString.end(); ++it)
        {
            comboVariable.addItem(QString::fromStdString(it->second));
        }
        myVar = getKeyMeteoVarMeteoMap(MapHourlyMeteoVarToString, comboVariable.currentText().toStdString());
    }
    //comboVariable.setSizeAdjustPolicy(QComboBox::AdjustToContents);
    comboVariable.setMinimumWidth(100);

    selectionChartLayout->addWidget(variableLabel);
    selectionChartLayout->addWidget(&comboVariable);
    selectionChartLayout->addWidget(axisXLabel);
    selectionChartLayout->addWidget(&comboAxisX);

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

    if (!parameters.empty() && interpolationSettings->getProxy(proxyPos)->getFittingFunctionName() == piecewiseThree && parameters[proxyPos].size() == 5)
    {
        QVBoxLayout *parametriLayout = new QVBoxLayout();

        QLabel *H0Lab = new QLabel(QString("H0: %1").arg(parameters[proxyPos][0]));
        QLabel *T0Lab = new QLabel(QString("T0: %1").arg(parameters[proxyPos][1]));
        QLabel *H1Lab = new QLabel(QString("H1: %1").arg(parameters[proxyPos][0]+parameters[proxyPos][2]));
        QLabel *T1Lab = new QLabel(QString("T1: %1").arg(parameters[proxyPos][1]+parameters[proxyPos][3]));
        QLabel *slopeLab = new QLabel(QString("slope: %1").arg(parameters[proxyPos][4]));

        parametriLayout->addWidget(H0Lab);
        parametriLayout->addWidget(T0Lab);
        parametriLayout->addWidget(H1Lab);
        parametriLayout->addWidget(T1Lab);
        parametriLayout->addWidget(slopeLab);

        selectionLayout->addLayout(parametriLayout);
    }
    else if (!parameters.empty() && interpolationSettings->getProxy(proxyPos)->getFittingFunctionName() == piecewiseTwo && parameters[proxyPos].size() == 4)
    {
        QVBoxLayout *parametriLayout = new QVBoxLayout();

        QLabel *H0Lab = new QLabel(QString("H0: %1").arg(parameters[proxyPos][0]));
        QLabel *T0Lab = new QLabel(QString("T0: %1").arg(parameters[proxyPos][1]));
        QLabel *slope1Lab = new QLabel(QString("slope1: %1").arg(parameters[proxyPos][2]));
        QLabel *slope2Lab = new QLabel(QString("slope2: %1").arg(parameters[proxyPos][3]));

        parametriLayout->addWidget(H0Lab);
        parametriLayout->addWidget(T0Lab);
        parametriLayout->addWidget(slope1Lab);
        parametriLayout->addWidget(slope2Lab);

        selectionLayout->addLayout(parametriLayout);
    }
    else if (!parameters.empty() && interpolationSettings->getProxy(proxyPos)->getFittingFunctionName() == piecewiseThreeFree && parameters[proxyPos].size() == 6)
    {
        QVBoxLayout *parametriLayout = new QVBoxLayout();

        QLabel *H0Lab = new QLabel(QString("H0: %1").arg(parameters[proxyPos][0]));
        QLabel *T0Lab = new QLabel(QString("T0: %1").arg(parameters[proxyPos][1]));
        QLabel *H1Lab = new QLabel(QString("H1: %1").arg(parameters[proxyPos][0]+parameters[proxyPos][2]));
        QLabel *slope1Lab = new QLabel(QString("Slope1: %1").arg(parameters[proxyPos][4]));
        QLabel *slope2Lab = new QLabel(QString("Slope2: %1").arg(parameters[proxyPos][3]));
        QLabel *slope3Lab = new QLabel(QString("Slope3: %1").arg(parameters[proxyPos][5]));

        parametriLayout->addWidget(H0Lab);
        parametriLayout->addWidget(T0Lab);
        parametriLayout->addWidget(H1Lab);
        parametriLayout->addWidget(slope1Lab);
        parametriLayout->addWidget(slope2Lab);
        parametriLayout->addWidget(slope3Lab);


        selectionLayout->addLayout(parametriLayout);
    }
    else if (!parameters.empty() && interpolationSettings->getProxy(proxyPos)->getFittingFunctionName() == piecewiseThree && parameters[proxyPos].size() == 5)
    {
        QVBoxLayout *parametriLayout = new QVBoxLayout();

        QLabel *H0Lab = new QLabel(QString("H0: %1").arg(parameters[proxyPos][0]));
        QLabel *T0Lab = new QLabel(QString("T0: %1").arg(parameters[proxyPos][1]));
        QLabel *H1Lab = new QLabel(QString("H1: %1").arg(parameters[proxyPos][0]+parameters[proxyPos][2]));
        QLabel *slope1Lab = new QLabel(QString("Slope1: %1").arg(parameters[proxyPos][4]));
        QLabel *slope2Lab = new QLabel(QString("Slope2: %1").arg(parameters[proxyPos][3]));

        parametriLayout->addWidget(H0Lab);
        parametriLayout->addWidget(T0Lab);
        parametriLayout->addWidget(H1Lab);
        parametriLayout->addWidget(slope2Lab);
        parametriLayout->addWidget(slope1Lab);


        selectionLayout->addLayout(parametriLayout);
    }
    horizontalGroupBox->setMaximumSize(1240, 130);
    horizontalGroupBox->setLayout(selectionLayout);

    mainLayout->addWidget(horizontalGroupBox);

    chartView = new ChartView();
    chartView->setMinimumHeight(200);
    mainLayout->addWidget(chartView);

    QStatusBar* statusBar = new QStatusBar();
    mainLayout->addWidget(statusBar);

    // menu
    QMenuBar* menuBar = new QMenuBar();
    QMenu *editMenu = new QMenu("Edit");
    QAction* updateStations = new QAction(tr("&Update"), this);
    editMenu->addAction(updateStations);

    menuBar->addMenu(editMenu);
    mainLayout->setMenuBar(menuBar);

    setLayout(mainLayout);

    connect(&comboAxisX, &QComboBox::currentTextChanged, [=](const QString &newProxy){ this->changeProxyPos(newProxy); });
    connect(&comboVariable, &QComboBox::currentTextChanged, [=](const QString &newVariable){ this->changeVar(newVariable); });
    connect(&climatologicalLR, &QCheckBox::toggled, [=](int toggled){ this->climatologicalLRClicked(toggled); });
    connect(&modelLR, &QCheckBox::toggled, [=](int toggled){ this->modelLRClicked(toggled); });
    connect(&detrended, &QCheckBox::toggled, [=](){ this->plot(); });
    connect(updateStations, &QAction::triggered, this, [=](){ this->plot(); });

    if (currentFrequency != noFrequency)
    {
        plot();
    }

    show();
}


Crit3DLocalProxyWidget::~Crit3DLocalProxyWidget()
{

}

void Crit3DLocalProxyWidget::changeProxyPos(const QString proxyName)
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

void Crit3DLocalProxyWidget::changeVar(const QString varName)
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

void Crit3DLocalProxyWidget::updateDateTime(QDate newDate, int newHour)
{
    currentDate = newDate;
    currentHour = newHour;
    plot();
}

void Crit3DLocalProxyWidget::updateFrequency(frequencyType newFrequency)
{
    currentFrequency = newFrequency;
    meteoVariable newVar = updateMeteoVariable(myVar, newFrequency);
    int cmbIndex = -1;
    std::string newVarString ;

    comboVariable.clear();

    std::map<meteoVariable, std::string>::const_iterator it;
    if (currentFrequency == daily)
    {
        newVarString = getKeyStringMeteoMap(MapDailyMeteoVar, newVar);

        for(it = MapDailyMeteoVarToString.begin(); it != MapDailyMeteoVarToString.end(); ++it)
        {
            comboVariable.addItem(QString::fromStdString(it->second));
        }
        cmbIndex = comboVariable.findText(QString::fromStdString(newVarString));
        if (cmbIndex != -1) comboVariable.setCurrentIndex(cmbIndex);
        myVar = getKeyMeteoVarMeteoMap(MapDailyMeteoVarToString, comboVariable.currentText().toStdString());
    }
    else if (currentFrequency == hourly)
    {
        newVarString = getKeyStringMeteoMap(MapHourlyMeteoVar, newVar);

        for(it = MapHourlyMeteoVarToString.begin(); it != MapHourlyMeteoVarToString.end(); ++it)
        {
            comboVariable.addItem(QString::fromStdString(it->second));
        }
        cmbIndex = comboVariable.findText(QString::fromStdString(newVarString));
        if (cmbIndex != -1) comboVariable.setCurrentIndex(cmbIndex);
        myVar = getKeyMeteoVarMeteoMap(MapHourlyMeteoVarToString, comboVariable.currentText().toStdString());
    }
    comboVariable.adjustSize();

    plot();
}

void Crit3DLocalProxyWidget::closeEvent(QCloseEvent *event)
{
    emit closeLocalProxyWidget();
    event->accept();

}

Crit3DTime Crit3DLocalProxyWidget::getCurrentTime()
{
    Crit3DTime myTime;
    if (currentFrequency == hourly)
    {
        myTime = getCrit3DTime(currentDate, currentHour);
    }
    else
    {
        myTime = getCrit3DTime(currentDate, 0);
    }

    return myTime;
}


void Crit3DLocalProxyWidget::plot()
{
    chartView->cleanScatterSeries();
    outInterpolationPoints.clear();
    subsetInterpolationPoints.clear();
    std::string errorStdStr;
    if (detrended.isChecked())
    {
        outInterpolationPoints.clear();

        checkAndPassDataToInterpolation(quality, myVar, meteoPoints, nrMeteoPoints, getCurrentTime(), SQinterpolationSettings,
                                        interpolationSettings, meteoSettings, climateParam,
                                        outInterpolationPoints, checkSpatialQuality, errorStdStr);

        localSelection(outInterpolationPoints, subsetInterpolationPoints, x, y, *interpolationSettings);
        detrending(subsetInterpolationPoints, interpolationSettings->getSelectedCombination(), interpolationSettings, climateParam, myVar, getCurrentTime());
    }
    else
    {
        checkAndPassDataToInterpolation(quality, myVar, meteoPoints, nrMeteoPoints, getCurrentTime(), SQinterpolationSettings,
                                        interpolationSettings, meteoSettings, climateParam,
                                        outInterpolationPoints, checkSpatialQuality, errorStdStr);
        localSelection(outInterpolationPoints, subsetInterpolationPoints, x, y, *interpolationSettings);
    }
    QList<QPointF> pointListPrimary, pointListSecondary, pointListSupplemental, pointListMarked;
    QMap< QString, QPointF > idPointMap1;
    QMap< QString, QPointF > idPointMap2;
    QMap< QString, QPointF > idPointMap3;
    QMap< QString, QPointF > idPointMapMarked;

    QPointF point;
    for (int i = 0; i < int(subsetInterpolationPoints.size()); i++)
    {
        float proxyVal = subsetInterpolationPoints[i].getProxyValue(proxyPos);
        float varValue = subsetInterpolationPoints[i].value;

        if (proxyVal != NODATA && varValue != NODATA)
        {
            point.setX(proxyVal);
            point.setY(varValue);
            QString text = "id: " + QString::fromStdString(meteoPoints[subsetInterpolationPoints[i].index].id) + "\n"
                           + "name: " + QString::fromStdString(meteoPoints[subsetInterpolationPoints[i].index].name);
            if (subsetInterpolationPoints[i].isMarked)
            {
                pointListMarked.append(point);
                idPointMapMarked.insert(text, point);
            }
            if (subsetInterpolationPoints[i].lapseRateCode == primary)
            {
                pointListPrimary.append(point);
                idPointMap1.insert(text, point);
            }
            else if (subsetInterpolationPoints[i].lapseRateCode == secondary)
            {
                pointListSecondary.append(point);
                idPointMap2.insert(text, point);
            }
            else if (subsetInterpolationPoints[i].lapseRateCode == supplemental)
            {
                pointListSupplemental.append(point);
                idPointMap3.insert(text, point);
            }
        }
    }

    chartView->setIdPointMap(idPointMap1, idPointMap2, idPointMap3, idPointMapMarked);
    chartView->drawScatterSeries(pointListPrimary, pointListSecondary, pointListSupplemental, pointListMarked);

    chartView->axisX->setTitleText(comboAxisX.currentText());
    chartView->axisY->setTitleText(comboVariable.currentText());

    chartView->axisY->setMin(floor(chartView->axisY->min()));
    chartView->axisY->setMax(ceil(chartView->axisY->max()));

    if (comboAxisX.currentText() == "elevation")
    /*    {
        chartView->cleanClimLapseRate();
        climatologicalLR.setVisible(false);

        // set minumum and maximum
        if (comboAxisX.currentText() == "urban")
        {
            chartView->axisX->setMin(-0.1);
            chartView->axisX->setMax(1.1);
            chartView->axisX->setTickCount(13);
        }
        if (comboAxisX.currentText() == "seaProximity")
        {
            chartView->axisX->setMin(0.0);
            chartView->axisX->setMax(1.1);
            chartView->axisX->setTickCount(12);
        }
        else if (comboAxisX.currentText() == "orogIndex")
        {
            chartView->axisX->setMin(-1);
            chartView->axisX->setMax(1.0);
            chartView->axisX->setTickCount(11);
        }
    }
    else */
    {
        climatologicalLR.setVisible(true);
        if (climatologicalLR.isChecked())
        {
            climatologicalLRClicked(1);
        }

        // set minumum and maximum
        double maximum = chartView->axisX->max();
        int nrStep = floor(maximum / 100) + 1;
        chartView->axisX->setMin(-100);
        chartView->axisX->setMax(nrStep * 100);
        chartView->axisX->setTickCount(nrStep+2);
    }

    if (modelLR.isChecked())
    {
        modelLRClicked(1);
    }
}


void Crit3DLocalProxyWidget::climatologicalLRClicked(int toggled)
{
    chartView->cleanClimLapseRate();
    if (toggled && outInterpolationPoints.size() != 0)
    {
        float zMax = getZmax(outInterpolationPoints);
        float zMin = getZmin(outInterpolationPoints);
        float firstIntervalHeightValue = getFirstIntervalHeightValue(outInterpolationPoints, interpolationSettings->getUseLapseRateCode());
        float lapseRate = climateParam->getClimateLapseRate(myVar, getCurrentTime());
        if (lapseRate == NODATA)
        {
            return;
        }
        QPointF firstPoint(zMin, firstIntervalHeightValue);
        QPointF lastPoint(zMax, firstIntervalHeightValue + lapseRate*(zMax - zMin));
        chartView->drawClimLapseRate(firstPoint, lastPoint);
    }
}

void Crit3DLocalProxyWidget::modelLRClicked(int toggled)
{
    chartView->cleanModelLapseRate();
    r2.clear();
    lapseRate.clear();
    QList<QPointF> point_vector;
    QPointF point;
    float xMin;
    float xMax;

    if (parameters.empty())
        return;

    if (toggled && subsetInterpolationPoints.size() != 0 && currentVariable == myVar)
    {
        if (comboAxisX.currentText() == "elevation")
        {
            xMin = getZmin(subsetInterpolationPoints);
            xMax = getZmax(subsetInterpolationPoints);
            float myY;

            if (interpolationSettings->getUseMultipleDetrending())
            {
                if (parameters.empty() || (parameters[proxyPos].size() != 5 && parameters[proxyPos].size() != 6 && parameters[proxyPos].size() != 4))
                    return;

                if (interpolationSettings->getProxy(proxyPos)->getFittingFunctionName() == piecewiseThree)
                {
                    std::vector <double> xVector;
                    for (int m = xMin; m < xMax; m += 5)
                        xVector.push_back(m);

                    for (int p = 0; p < xVector.size(); p++)
                    {
                        point.setX(xVector[p]);
                        point.setY(lapseRatePiecewiseThree_withSlope(xVector[p], parameters[proxyPos]));
                        point_vector.append(point);
                    }
                }
                else if (interpolationSettings->getProxy(proxyPos)->getFittingFunctionName() == piecewiseTwo)
                {
                    float lapseRateH0 = parameters[proxyPos][0];
                    float lapseRateT0 = parameters[proxyPos][1];
                    float slope1 = parameters[proxyPos][2];
                    float slope2 = parameters[proxyPos][3];

                    if (xMin < lapseRateH0)
                    {
                        myY = lapseRateT0 + slope1 * (xMin - lapseRateH0);
                        point.setX(xMin);
                        point.setY(myY);
                        point_vector.append(point);
                    }

                    point.setX(lapseRateH0);
                    point.setY(lapseRateT0);
                    point_vector.append(point);

                    myY = lapseRateT0 + slope2 * (xMax - lapseRateH0);
                    point.setX(xMax);
                    point.setY(myY);
                    point_vector.append(point);
                }
                else if (interpolationSettings->getProxy(proxyPos)->getFittingFunctionName() == piecewiseThreeFree)
                {
                    std::vector <double> xVector;
                    for (int m = xMin; m < xMax; m += 5)
                        xVector.push_back(m);

                    for (int p = 0; p < xVector.size(); p++)
                    {
                        point.setX(xVector[p]);
                        point.setY(lapseRatePiecewiseFree(xVector[p], parameters[proxyPos]));
                        point_vector.append(point);
                    }

                }
                /*if (interpolationSettings->getProxy(proxyPos)->getInversionIsSignificative())
                {
                    if (xMin < interpolationSettings->getProxy(proxyPos)->getLapseRateH0())
                    {
                        point.setX(xMin);
                        point.setY(lapseRateT0);
                        point_vector.append(point);
                    }*/


                /*}
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
                }*/

            }

            /*if (interpolationSettings->getProxy(proxyPos)->getRegressionR2() != NODATA)
            {
                r2.setText(QString("%1").arg(interpolationSettings->getProxy(proxyPos)->getRegressionR2(), 0, 'f', 2));
            }
            lapseRate.setText(QString("%1").arg(regressionSlope*1000, 0, 'f', 2));*/
        }
        else
        {
            //TODO lineari
            /*xMin = getProxyMinValue(subsetInterpolationPoints, proxyPos);
            xMax = getProxyMaxValue(subsetInterpolationPoints, proxyPos);

            float slope = parameters[proxyPos][0];
            float intercept = parameters[proxyPos][1];

            float myY = intercept + slope * xMin;
            point.setX(xMin);
            point.setY(myY);
            point_vector.append(point);

            myY = intercept + slope * xMax;
            point.setX(xMax);
            point.setY(myY);
            point_vector.append(point);*/
        }
        chartView->drawModelLapseRate(point_vector);
    }
}
