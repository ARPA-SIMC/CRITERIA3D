#include "meteo.h"
#include "localProxyWidget.h"
#include "utilities.h"
#include "interpolation.h"
#include "spatialControl.h"
#include "commonConstants.h"
#include "math.h"
#include "furtherMathFunctions.h"

#include <QLayout>
#include <QDate>


Crit3DLocalProxyWidget::Crit3DLocalProxyWidget(double x, double y, gis::Crit3DGisSettings gisSettings, Crit3DInterpolationSettings* interpolationSettings, Crit3DMeteoPoint *meteoPoints, int nrMeteoPoints, meteoVariable currentVariable, frequencyType currentFrequency, QDate currentDate, int currentHour, Crit3DQuality *quality, Crit3DInterpolationSettings* SQinterpolationSettings, Crit3DMeteoSettings *meteoSettings, Crit3DClimateParameters *climateParam, bool checkSpatialQuality)
    :x(x), y(y), gisSettings(gisSettings), interpolationSettings(interpolationSettings), meteoPoints(meteoPoints), nrMeteoPoints(nrMeteoPoints), currentVariable(currentVariable), currentFrequency(currentFrequency), currentDate(currentDate), currentHour(currentHour), quality(quality), SQinterpolationSettings(SQinterpolationSettings), meteoSettings(meteoSettings), climateParam(climateParam), checkSpatialQuality(checkSpatialQuality)
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
    stationWeights.setText("See weight of stations");

    //temporaneamente disattivati
    detrended.setVisible(false);
    climatologicalLR.setVisible(false);
    climatologicalLR.setEnabled(false);

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
    selectionOptionBoxLayout->addWidget(&stationWeights);

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
    connect(&stationWeights, &QCheckBox::toggled, [=] () {this->plot();});
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

    for (QGraphicsTextItem* label : weightLabels)
    {
        chartView->scene()->removeItem(label);
        delete label;
    }
    weightLabels.clear();

    if (detrended.isChecked())
    {
        outInterpolationPoints.clear();

        checkAndPassDataToInterpolation(quality, myVar, meteoPoints, nrMeteoPoints, getCurrentTime(), SQinterpolationSettings,
                                        interpolationSettings, meteoSettings, climateParam,
                                        outInterpolationPoints, checkSpatialQuality, errorStdStr);

        localSelection(outInterpolationPoints, subsetInterpolationPoints, x, y, z, *interpolationSettings);
        detrending(subsetInterpolationPoints, interpolationSettings->getSelectedCombination(), interpolationSettings, climateParam, myVar, getCurrentTime());
    }
    else
    {
        checkAndPassDataToInterpolation(quality, myVar, meteoPoints, nrMeteoPoints, getCurrentTime(), SQinterpolationSettings,
                                        interpolationSettings, meteoSettings, climateParam,
                                        outInterpolationPoints, checkSpatialQuality, errorStdStr);
        localSelection(outInterpolationPoints, subsetInterpolationPoints, x, y, z, *interpolationSettings);
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
                           + "name: " + QString::fromStdString(meteoPoints[subsetInterpolationPoints[i].index].name) + "\n"
                           + "weight: " + QString::number(subsetInterpolationPoints[i].regressionWeight, 'f', 5);
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

    if (stationWeights.isChecked())
    {
        QChart* chart = chartView->chart();
        QRectF chartRect = chart->plotArea();
        double xMin = chartView->axisX->min();
        double xMax = chartView->axisX->max();
        double yMin = chartView->axisY->min();
        double yMax = chartView->axisY->max();

        for (int i = 0; i < int(subsetInterpolationPoints.size()); i++)
        {
            float proxyVal = subsetInterpolationPoints[i].getProxyValue(proxyPos);
            float varValue = subsetInterpolationPoints[i].value;

            if (proxyVal != NODATA && varValue != NODATA)
            {
                double xRatio = (proxyVal - xMin) / (xMax - xMin);
                double yRatio = (varValue - yMin) / (yMax - yMin);

                QPointF scenePos;
                scenePos.setX(chartRect.left() + xRatio * chartRect.width());
                scenePos.setY(chartRect.bottom() - yRatio * chartRect.height());

                QGraphicsTextItem* weightLabel = new QGraphicsTextItem(QString::number(subsetInterpolationPoints[i].regressionWeight, 'f', 3));
                weightLabel->setPos(scenePos);
                chartView->scene()->addItem(weightLabel);
                weightLabels.push_back(weightLabel);
            }
        }
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


    if (toggled && subsetInterpolationPoints.size() != 0)
    {
        std::string errorStr;
        setHeightTemperatureRange(interpolationSettings->getSelectedCombination(), interpolationSettings);
        interpolationSettings->setCurrentCombination(interpolationSettings->getSelectedCombination());
        interpolationSettings->clearFitting();
        if (! multipleDetrendingElevationFitting(proxyPos, outInterpolationPoints, interpolationSettings, myVar, errorStr)) return;

        std::vector<std::vector<double>> parameters = interpolationSettings->getFittingParameters();

        if (!parameters.empty())
        {
            if (parameters.front().size() > 2)
            {
                xMin = getZmin(subsetInterpolationPoints);
                xMax = getZmax(subsetInterpolationPoints);

                if (interpolationSettings->getUseMultipleDetrending())
                {
                    if ((parameters.front().size() != 5 && parameters.front().size() != 6 && parameters.front().size() != 4))
                        return;

                    if (parameters.size() > proxyPos)
                    {
                        std::vector <double> xVector;
                        for (int m = xMin; m < xMax; m += 5)
                            xVector.push_back(m);

                        for (int p = 0; p < int(xVector.size()); p++)
                        {
                            point.setX(xVector[p]);
                            if (parameters.front().size() == 4)
                                point.setY(lapseRatePiecewise_two(xVector[p], parameters.front()));
                            else if (parameters.front().size() == 5)
                                point.setY(lapseRatePiecewise_three(xVector[p], parameters.front()));
                            else if (parameters.front().size() == 6)
                                point.setY(lapseRatePiecewise_three_free(xVector[p], parameters.front()));
                            point_vector.append(point);
                        }
                    }
                }
            }
            else
            {
                xMin = getProxyMinValue(subsetInterpolationPoints, proxyPos);
                xMax = getProxyMaxValue(subsetInterpolationPoints, proxyPos);

                if (parameters[proxyPos].empty())
                    return;

                float slope = parameters[proxyPos][0];
                float intercept = parameters[proxyPos][1];

                float myY = intercept + slope * xMin;
                point.setX(xMin);
                point.setY(myY);
                point_vector.append(point);

                myY = intercept + slope * xMax;
                point.setX(xMax);
                point.setY(myY);
                point_vector.append(point);
            }
        }
        chartView->drawModelLapseRate(point_vector);
    }
}
