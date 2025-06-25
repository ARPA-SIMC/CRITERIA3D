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


Crit3DLocalProxyWidget::Crit3DLocalProxyWidget(double x, double y, double zDEM, double zGrid, gis::Crit3DGisSettings gisSettings,
                                               Crit3DInterpolationSettings* interpolationSettings, Crit3DMeteoPoint *meteoPoints,
                                               int nrMeteoPoints, meteoVariable currentVariable, frequencyType currentFrequency,
                                               QDate currentDate, int currentHour, Crit3DQuality *quality,
                                               Crit3DInterpolationSettings* SQinterpolationSettings, Crit3DMeteoSettings *meteoSettings,
                                               Crit3DClimateParameters *climateParameters, bool checkSpatialQuality)
    :_x(x), _y(y), _zDEM(zDEM), _zGrid(zGrid), _gisSettings(gisSettings), _interpolationSettings(interpolationSettings),
    _meteoPoints(meteoPoints), _nrMeteoPoints(nrMeteoPoints), _currentVariable(currentVariable), _currentFrequency(currentFrequency),
    _currentDate(currentDate), _currentHour(currentHour), _quality(quality), _SQinterpolationSettings(SQinterpolationSettings),
    _meteoSettings(meteoSettings), _climateParameters(climateParameters), _checkSpatialQuality(checkSpatialQuality)
{
    gis::Crit3DGeoPoint localGeoPoint;
    gis::Crit3DUtmPoint localUtmPoint;
    localUtmPoint.x = x;
    localUtmPoint.y = y;
    gis::getLatLonFromUtm(gisSettings, localUtmPoint, localGeoPoint);

    QString windowTitle = "Local proxy analysis for point of coordinates (" + QString::number(localGeoPoint.latitude) + ", " + QString::number(localGeoPoint.longitude) + ")." + " z value: " + QString::number(zDEM) + " (DEM)";

    if (_interpolationSettings->getUseGlocalDetrending())
        windowTitle += ", macro area nr. " + QString::number(gis::getValueFromXY(*_interpolationSettings->getMacroAreasMap(), x, y));

    this->setWindowTitle(windowTitle); // + QString::number(zGrid) + " (Grid)");
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
    QHBoxLayout *parametersListLayoutUp = new QHBoxLayout;
    QHBoxLayout *parametersListLayoutMiddle = new QHBoxLayout;
    QHBoxLayout *parametersListLayoutDown = new QHBoxLayout;
    QVBoxLayout *parametersListLayout = new QVBoxLayout;

    detrended.setText("Detrended data");
    modelLR.setText("Model lapse rate");
    stationWeights.setText("See weight of stations");

    //temporaneamente disattivati
    detrended.setVisible(true);
    QLabel *r2Label = new QLabel(tr("R2"));

    r2.setMaximumWidth(60);
    r2.setMinimumHeight(25);
    r2.setMaximumHeight(25);
    r2.setEnabled(false);

    QLabel *variableLabel = new QLabel(tr("Variable"));
    QLabel *axisXLabel = new QLabel(tr("Axis X"));

    QLabel *par0Label = new QLabel(tr("par0"));
    QLabel *par1Label = new QLabel(tr("par1"));
    QLabel *par2Label = new QLabel(tr("par2"));
    QLabel *par3Label = new QLabel(tr("par3"));
    QLabel *par4Label = new QLabel(tr("par4"));
    QLabel *par5Label = new QLabel(tr("par5"));

    par0.setMaximumWidth(90);
    par0.setMinimumHeight(25);
    par0.setMaximumHeight(25);
    par0.setEnabled(false);

    par1.setMaximumWidth(90);
    par1.setMinimumHeight(25);
    par1.setMaximumHeight(25);
    par1.setEnabled(false);

    par2.setMaximumWidth(90);
    par2.setMinimumHeight(25);
    par2.setMaximumHeight(25);
    par2.setEnabled(false);

    par3.setMaximumWidth(90);
    par3.setMinimumHeight(25);
    par3.setMaximumHeight(25);
    par3.setEnabled(false);

    par4.setMaximumWidth(90);
    par4.setMinimumHeight(25);
    par4.setMaximumHeight(25);
    par4.setEnabled(false);

    par5.setMaximumWidth(90);
    par5.setMinimumHeight(25);
    par5.setMaximumHeight(25);
    par5.setEnabled(false);

    std::vector<Crit3DProxy> proxy = _interpolationSettings->getCurrentProxy();

    for(int i=0; i<int(proxy.size()); i++)
    {
        comboAxisX.addItem(QString::fromStdString(proxy[i].getName()));
    }
    proxyPos = 0;
    comboAxisX.setSizeAdjustPolicy(QComboBox::AdjustToContents);

    std::map<meteoVariable, std::string>::const_iterator it;
    if (_currentFrequency == daily)
    {
        for(it = MapDailyMeteoVarToString.begin(); it != MapDailyMeteoVarToString.end(); ++it)
        {
            comboVariable.addItem(QString::fromStdString(it->second));
        }
        myVar = getKeyMeteoVarMeteoMap(MapDailyMeteoVarToString, comboVariable.currentText().toStdString());
    }
    else if (_currentFrequency == hourly)
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

    parametersListLayoutUp->addWidget(par0Label);
    parametersListLayoutUp->addWidget(&par0);
    parametersListLayoutUp->addWidget(par3Label);
    parametersListLayoutUp->addWidget(&par3);
    parametersListLayoutMiddle->addWidget(par1Label);
    parametersListLayoutMiddle->addWidget(&par1);
    parametersListLayoutMiddle->addWidget(par4Label);
    parametersListLayoutMiddle->addWidget(&par4);
    parametersListLayoutDown->addWidget(par2Label);
    parametersListLayoutDown->addWidget(&par2);
    parametersListLayoutDown->addWidget(par5Label);
    parametersListLayoutDown->addWidget(&par5);

    parametersListLayout->addStretch(150);
    parametersListLayout->addStretch(150);
    parametersListLayout->addStretch(150);
    parametersListLayout->addLayout(parametersListLayoutUp);
    parametersListLayout->addLayout(parametersListLayoutMiddle);
    parametersListLayout->addLayout(parametersListLayoutDown);

    selectionOptionBoxLayout->addWidget(&detrended);
    selectionOptionBoxLayout->addWidget(&modelLR);
    selectionOptionBoxLayout->addWidget(&stationWeights);

    selectionOptionEditLayout->addWidget(r2Label);
    selectionOptionEditLayout->addWidget(&r2);
    selectionOptionEditLayout->addStretch(150);
    selectionOptionEditLayout->addStretch(150);
    selectionOptionEditLayout->addStretch(150);

    selectionOptionLayout->addLayout(selectionOptionBoxLayout);
    selectionOptionLayout->addLayout(selectionOptionEditLayout);

    selectionLayout->addLayout(selectionChartLayout);
    selectionLayout->addStretch(30);
    selectionLayout->addLayout(parametersListLayout);
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
    connect(&modelLR, &QCheckBox::toggled, [=](int toggled){ this->modelLRClicked(toggled); });
    connect(&detrended, &QCheckBox::toggled, [=](){ this->plot(); });
    connect(&stationWeights, &QCheckBox::toggled, [=] () {this->plot();});
    connect(updateStations, &QAction::triggered, this, [=](){ this->plot(); });

    if (_currentFrequency != noFrequency)
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
    for (int pos=0; pos < int(_interpolationSettings->getProxyNr()); pos++)
    {
        QString myProxy = QString::fromStdString(_interpolationSettings->getProxy(pos)->getName());
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
        if (_currentFrequency == daily)
        {
            myVar = getKeyMeteoVarMeteoMap(MapDailyMeteoVarToString, varName.toStdString());
        }
        else if (_currentFrequency == hourly)
        {
            myVar = getKeyMeteoVarMeteoMap(MapHourlyMeteoVarToString, varName.toStdString());
        }
    }
    plot();
}

void Crit3DLocalProxyWidget::updateDateTime(QDate newDate, int newHour)
{
    _currentDate = newDate;
    _currentHour = newHour;
    plot();
}

void Crit3DLocalProxyWidget::updateFrequency(frequencyType newFrequency)
{
    _currentFrequency = newFrequency;
    meteoVariable newVar = updateMeteoVariable(myVar, newFrequency);
    int cmbIndex = -1;
    std::string newVarString ;

    comboVariable.clear();

    std::map<meteoVariable, std::string>::const_iterator it;
    if (_currentFrequency == daily)
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
    else if (_currentFrequency == hourly)
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
    if (_currentFrequency == hourly)
    {
        myTime = getCrit3DTime(_currentDate, _currentHour);
    }
    else
    {
        myTime = getCrit3DTime(_currentDate, 0);
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
    int areaCode = NODATA;

    if (_interpolationSettings->getUseLocalDetrending())
    {
    if (detrended.isChecked())
    {
        outInterpolationPoints.clear();

        checkAndPassDataToInterpolation(_quality, myVar, _meteoPoints, _nrMeteoPoints, getCurrentTime(), _SQinterpolationSettings,
                                        _interpolationSettings, _meteoSettings, _climateParameters,
                                        outInterpolationPoints, _checkSpatialQuality, errorStdStr);

        localSelection(outInterpolationPoints, subsetInterpolationPoints, _x, _y, *_interpolationSettings, false);
        //detrending(subsetInterpolationPoints, _interpolationSettings->getSelectedCombination(), _interpolationSettings, _climateParameters, myVar, getCurrentTime());

    }
    else
    {
        checkAndPassDataToInterpolation(_quality, myVar, _meteoPoints, _nrMeteoPoints, getCurrentTime(), _SQinterpolationSettings,
                                        _interpolationSettings, _meteoSettings, _climateParameters,
                                        outInterpolationPoints, _checkSpatialQuality, errorStdStr);
        localSelection(outInterpolationPoints, subsetInterpolationPoints, _x, _y, *_interpolationSettings, false);
    }
    }
    else if (_interpolationSettings->getUseGlocalDetrending())
    {

        areaCode = gis::getValueFromXY(*_interpolationSettings->getMacroAreasMap(), _x, _y);
        if (areaCode < (int)_interpolationSettings->getMacroAreas().size())
        {
            Crit3DMacroArea myArea = _interpolationSettings->getMacroAreas()[areaCode];
            std::vector<int> stations = myArea.getMeteoPoints();
            if (detrended.isChecked())
            {
                outInterpolationPoints.clear();

                checkAndPassDataToInterpolation(_quality, myVar, _meteoPoints, _nrMeteoPoints, getCurrentTime(), _SQinterpolationSettings,
                                                _interpolationSettings, _meteoSettings, _climateParameters,
                                                outInterpolationPoints, _checkSpatialQuality, errorStdStr);

                for (int k = 0; k < (int)stations.size(); k++)
                {
                    for (int j = 0; j < (int)outInterpolationPoints.size(); j++)
                        if (outInterpolationPoints[j].index == stations[k])
                        {
                            subsetInterpolationPoints.push_back(outInterpolationPoints[j]);
                        }
                }

                detrending(subsetInterpolationPoints, _interpolationSettings->getSelectedCombination(), _interpolationSettings, _climateParameters, myVar, getCurrentTime());
            }
            else
            {
                outInterpolationPoints.clear();
                checkAndPassDataToInterpolation(_quality, myVar, _meteoPoints, _nrMeteoPoints, getCurrentTime(), _SQinterpolationSettings,
                                                _interpolationSettings, _meteoSettings, _climateParameters,
                                                outInterpolationPoints, _checkSpatialQuality, errorStdStr);

                for (int k = 0; k < (int)stations.size(); k++)
                {
                    for (int j = 0; j < (int)outInterpolationPoints.size(); j++)
                        if (outInterpolationPoints[j].index == stations[k])
                        {
                            subsetInterpolationPoints.push_back(outInterpolationPoints[j]);
                        }
                }
            }
        }
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
            QString text = "id: " + QString::fromStdString(_meteoPoints[subsetInterpolationPoints[i].index].id) + "\n"
                           + "name: " + QString::fromStdString(_meteoPoints[subsetInterpolationPoints[i].index].name) + "\n"
                           + "province: " + QString::fromStdString(_meteoPoints[subsetInterpolationPoints[i].index].province) + "\n"
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
    else
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
    }*/

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
        float firstIntervalHeightValue = getFirstIntervalHeightValue(outInterpolationPoints, _interpolationSettings->getUseLapseRateCode());
        float lapseRate = _climateParameters->getClimateLapseRate(myVar, getCurrentTime());
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
    par0.clear();
    par1.clear();
    par2.clear();
    par3.clear();
    par4.clear();
    par5.clear();
    QList<QPointF> point_vector;
    QPointF point;
    float xMin;
    float xMax;


    if (toggled && subsetInterpolationPoints.size() != 0)
    {
        int elevationPos = NODATA;
        for (unsigned int pos=0; pos < _interpolationSettings->getSelectedCombination().getProxySize(); pos++)
        {
            if (getProxyPragaName(_interpolationSettings->getProxy(pos)->getName()) == proxyHeight)
                elevationPos = pos;
        }
            std::string errorStr;
            setMultipleDetrendingHeightTemperatureRange(_interpolationSettings);
            _interpolationSettings->setCurrentCombination(_interpolationSettings->getSelectedCombination());
            _interpolationSettings->clearFitting();
            if (_interpolationSettings->getUseLocalDetrending())
            {
                if (! multipleDetrendingElevationFitting(elevationPos, subsetInterpolationPoints, _interpolationSettings, myVar, errorStr, true)) return;
            }
            else if (_interpolationSettings->getUseGlocalDetrending())
                if (! multipleDetrendingElevationFitting(elevationPos, subsetInterpolationPoints, _interpolationSettings, myVar, errorStr, false)) return;

            std::vector<std::vector<double>> parameters = _interpolationSettings->getFittingParameters();

            if (comboAxisX.currentText() == "elevation")
            {
                if (!parameters.empty())
                {
                    if (parameters.front().size() > 3 && parameters.front().size() < 7)
                    {
                        xMin = getZmin(subsetInterpolationPoints);
                        xMax = getZmax(subsetInterpolationPoints);

                        if (_interpolationSettings->getUseMultipleDetrending())
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

                            if (_interpolationSettings->getProxy(elevationPos)->getRegressionR2() != NODATA)
                                r2.setText(QString("%1").arg(_interpolationSettings->getProxy(elevationPos)->getRegressionR2(), 0, 'f', 3));

                            if (parameters.front().size() > 3)
                            {
                                par0.setText(QString("%1").arg(parameters.front()[0], 0, 'f', 4));
                                par1.setText(QString("%1").arg(parameters.front()[1], 0, 'f', 4));
                                par2.setText(QString("%1").arg(parameters.front()[2], 0, 'f', 4));
                                par3.setText(QString("%1").arg(parameters.front()[3], 0, 'f', 4));
                            }
                            if (parameters.front().size() > 4)
                                par4.setText(QString("%1").arg(parameters.front()[4], 0, 'f', 4));
                            if (parameters.front().size() > 5)
                                par5.setText(QString("%1").arg(parameters.front()[5], 0, 'f', 4));
                        }
                    }
                }
            }
            else
            {
                //TODO: OTHER PROXIES
                /*std::string errorStr;
                //detrendingElevation(elevationPos, outInterpolationPoints, _interpolationSettings);
                if (! multipleDetrendingOtherProxiesFitting(elevationPos, subsetInterpolationPoints, _interpolationSettings, myVar, errorStr)) return;

                parameters = _interpolationSettings->getFittingParameters();
                Crit3DProxyCombination myCombination = _interpolationSettings->getCurrentCombination();

                int pos = 0;
                for (int i = 0; i < proxyPos + 1; i++)
                    if (myCombination.isProxyActive(i) && myCombination.isProxySignificant(i)) pos++;
                pos -= 1;

                xMin = getZmin(subsetInterpolationPoints);
                xMax = getZmax(subsetInterpolationPoints);

                if (_interpolationSettings->getUseMultipleDetrending() && pos < parameters.size())
                {
                    std::vector <double> xVector;
                    for (int m = xMin; m < xMax; m += 5)
                        xVector.push_back(m);

                    for (int p = 0; p < int(xVector.size()); p++)
                    {
                        point.setX(xVector[p]);
                        point.setY(functionLinear_intercept(xVector[p], parameters[pos]));
                        point_vector.append(point);
                    }

                    if (_interpolationSettings->getProxy(proxyPos)->getRegressionR2() != NODATA)
                        r2.setText(QString("%1").arg(_interpolationSettings->getProxy(proxyPos)->getRegressionR2(), 0, 'f', 3));

                    if (parameters[pos].size() == 2)
                    {
                        par0.setText(QString("%1").arg(parameters[pos][0], 0, 'f', 4));
                        par1.setText(QString("%1").arg(parameters[pos][1], 0, 'f', 4));
                    }
                }*/

            }

        chartView->drawModelLapseRate(point_vector);
    }
}
