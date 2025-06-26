#include "dialogChangeAxis.h"
#include "meteo.h"
#include "proxyWidget.h"
#include "utilities.h"
#include "interpolation.h"
#include "spatialControl.h"
#include "commonConstants.h"
#include "math.h"
#include "furtherMathFunctions.h"

#include <QLayout>
#include <QDate>
#include <QColorDialog>



Crit3DProxyWidget::Crit3DProxyWidget(Crit3DInterpolationSettings* _interpolationSettings, Crit3DMeteoPoint *meteoPoints, int nrMeteoPoints,
                                     frequencyType currentFrequency, QDate currentDate, int currentHour, Crit3DQuality *quality,
                                     Crit3DInterpolationSettings* SQinterpolationSettings, Crit3DMeteoSettings *meteoSettings,
                                     Crit3DClimateParameters *climateParameters, bool checkSpatialQuality, int macroAreaNumber)
    :_interpolationSettings(_interpolationSettings), _meteoPoints(meteoPoints), _nrMeteoPoints(nrMeteoPoints),
    _currentFrequency(currentFrequency), _currentDate(currentDate), _currentHour(currentHour), _quality(quality),
    _SQinterpolationSettings(SQinterpolationSettings), _meteoSettings(meteoSettings), _checkSpatialQuality(checkSpatialQuality),
    _macroAreaNumber(macroAreaNumber), _climateParameters(climateParameters)
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

    std::vector<Crit3DProxy> proxy = _interpolationSettings->getCurrentProxy();

    for(int i=0; i<int(proxy.size()); i++)
    {
        comboAxisX.addItem(QString::fromStdString(proxy[i].getName()));
    }
    _proxyPos = 0;
    comboAxisX.setSizeAdjustPolicy(QComboBox::AdjustToContents);
    if (comboAxisX.currentText() != "elevation")
    {
        climatologicalLR.setVisible(false);
    }
    else
    {
        climatologicalLR.setVisible(true);
    }

    std::map<meteoVariable, std::string>::const_iterator iterator;
    if (currentFrequency == daily)
    {
        for(iterator = MapDailyMeteoVarToString.begin(); iterator != MapDailyMeteoVarToString.end(); ++iterator)
        {
            comboVariable.addItem(QString::fromStdString(iterator->second));
        }
        myVar = getKeyMeteoVarMeteoMap(MapDailyMeteoVarToString, comboVariable.currentText().toStdString());
    }
    else if (currentFrequency == hourly)
    {
        for(iterator = MapHourlyMeteoVarToString.begin(); iterator != MapHourlyMeteoVarToString.end(); ++iterator)
        {
            comboVariable.addItem(QString::fromStdString(iterator->second));
        }
        myVar = getKeyMeteoVarMeteoMap(MapHourlyMeteoVarToString, comboVariable.currentText().toStdString());
    }
    else
    {
        QMessageBox::information(nullptr, "Warning!", "Select data frequency (daily or hourly) before.");
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

    if (macroAreaNumber != NODATA) modelLR.setEnabled(false);

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
    QAction* changeLeftAxis = new QAction(tr("&Change axis left"), this);

    editMenu->addAction(updateStations);
    editMenu->addAction(changeLeftAxis);

    menuBar->addMenu(editMenu);
    mainLayout->setMenuBar(menuBar);

    setLayout(mainLayout);
    
    connect(&comboAxisX, &QComboBox::currentTextChanged, [=](const QString &newProxy){ this->changeProxyPos(newProxy); });
    connect(&comboVariable, &QComboBox::currentTextChanged, [=](const QString &newVariable){ this->changeVar(newVariable); });
    connect(&climatologicalLR, &QCheckBox::toggled, [=](int toggled){ this->climatologicalLRClicked(toggled); });
    connect(&modelLR, &QCheckBox::toggled, [=](int toggled){ this->modelLRClicked(toggled); });
    connect(&detrended, &QCheckBox::toggled, [=](){ this->plot(); });
    connect(updateStations, &QAction::triggered, this, [=](){ this->plot(); });
    connect(changeLeftAxis, &QAction::triggered, this, &Crit3DProxyWidget::on_actionChangeLeftAxis);


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
    for (int pos=0; pos < int(_interpolationSettings->getProxyNr()); pos++)
    {
        QString myProxy = QString::fromStdString(_interpolationSettings->getProxy(pos)->getName());
        if (myProxy == proxyName)
        {
            _proxyPos = pos;
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

void Crit3DProxyWidget::updateDateTime(QDate newDate, int newHour)
{
    _currentDate = newDate;
    _currentHour = newHour;
    plot();
}

void Crit3DProxyWidget::updateFrequency(frequencyType newFrequency)
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

void Crit3DProxyWidget::closeEvent(QCloseEvent *event)
{
    emit closeProxyWidget();
    event->accept();

}

Crit3DTime Crit3DProxyWidget::getCurrentTime()
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


void Crit3DProxyWidget::plot()
{
    chartView->cleanScatterSeries();
    _outInterpolationPoints.clear();

    std::string errorStdStr;
    if (detrended.isChecked())
    {
        _outInterpolationPoints.clear();

        checkAndPassDataToInterpolation(_quality, myVar, _meteoPoints, _nrMeteoPoints, getCurrentTime(), _SQinterpolationSettings,
                                        _interpolationSettings, _meteoSettings, _climateParameters,
                                        _outInterpolationPoints, _checkSpatialQuality, errorStdStr);

        detrending(_outInterpolationPoints, _interpolationSettings->getSelectedCombination(), _interpolationSettings, _climateParameters, myVar, getCurrentTime());
    }
    else
    {
        checkAndPassDataToInterpolation(_quality, myVar, _meteoPoints, _nrMeteoPoints, getCurrentTime(), _SQinterpolationSettings,
                                        _interpolationSettings, _meteoSettings, _climateParameters,
                                        _outInterpolationPoints, _checkSpatialQuality, errorStdStr);
    }
    QList<QPointF> pointListPrimary, pointListSecondary, pointListSupplemental, pointListMarked;
    QMap< QString, QPointF > idPointMap1;
    QMap< QString, QPointF > idPointMap2;
    QMap< QString, QPointF > idPointMap3;
    QMap< QString, QPointF > idPointMapMarked;

    QPointF point;
    for (int i = 0; i < int(_outInterpolationPoints.size()); i++)
    {
        float proxyVal = _outInterpolationPoints[i].getProxyValue(_proxyPos);
        float varValue = _outInterpolationPoints[i].value;

        if (proxyVal != NODATA && varValue != NODATA)
        {
            point.setX(proxyVal);
            point.setY(varValue);
            QString text = "id: " + QString::fromStdString(_meteoPoints[_outInterpolationPoints[i].index].id) + "\n"
                           + "name: " + QString::fromStdString(_meteoPoints[_outInterpolationPoints[i].index].name) + "\n"
                           + "province: " + QString::fromStdString(_meteoPoints[_outInterpolationPoints[i].index].province);

            if (_outInterpolationPoints[i].isMarked)
            {
                pointListMarked.append(point);
                idPointMapMarked.insert(text, point);
            }
            if (_outInterpolationPoints[i].lapseRateCode == primary)
            {
                pointListPrimary.append(point);
                idPointMap1.insert(text, point);
            }
            else if (_outInterpolationPoints[i].lapseRateCode == secondary)
            {
                pointListSecondary.append(point);
                idPointMap2.insert(text, point);
            }
            else if (_outInterpolationPoints[i].lapseRateCode == supplemental)
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

    if (_macroAreaNumber != NODATA)
    {
        addMacroAreaLR();
    }
}


void Crit3DProxyWidget::climatologicalLRClicked(int toggled)
{
    chartView->cleanClimLapseRate();
    if (toggled && _outInterpolationPoints.size() != 0)
    {
        float zMax = getZmax(_outInterpolationPoints);
        float zMin = getZmin(_outInterpolationPoints);
        float firstIntervalHeightValue = getFirstIntervalHeightValue(_outInterpolationPoints, _interpolationSettings->getUseLapseRateCode());
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


void Crit3DProxyWidget::modelLRClicked(int toggled)
{
    chartView->cleanModelLapseRate();
    r2.clear();
    lapseRate.clear();

    QList<QPointF> point_vector;
    QPointF point;

    if (toggled && _outInterpolationPoints.size() != 0)
    {
        float regressionSlope = NODATA;

        if (comboAxisX.currentText() == "elevation")
        {
            float xMin = getZmin(_outInterpolationPoints);
            float xMax = getZmax(_outInterpolationPoints);

            if (! _interpolationSettings->getUseMultipleDetrending())
            {
                if (!regressionOrography(_outInterpolationPoints,_interpolationSettings->getSelectedCombination(),
                                         _interpolationSettings, _climateParameters, getCurrentTime(), myVar, _proxyPos))
                {
                    return;
                }

                float lapseRateH0 = _interpolationSettings->getProxy(_proxyPos)->getLapseRateH0();
                float lapseRateH1 = _interpolationSettings->getProxy(_proxyPos)->getLapseRateH1();
                float lapseRateT0 = _interpolationSettings->getProxy(_proxyPos)->getLapseRateT0();
                float lapseRateT1 = _interpolationSettings->getProxy(_proxyPos)->getLapseRateT1();
                regressionSlope = _interpolationSettings->getProxy(_proxyPos)->getRegressionSlope();

                if (_interpolationSettings->getProxy(_proxyPos)->getInversionIsSignificative())
                {
                    if (xMin < _interpolationSettings->getProxy(_proxyPos)->getLapseRateH0())
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

                if (_interpolationSettings->getProxy(_proxyPos)->getRegressionR2() != NODATA)
                {
                    r2.setText(QString("%1").arg(_interpolationSettings->getProxy(_proxyPos)->getRegressionR2(), 0, 'f', 2));
                }
                lapseRate.setText(QString("%1").arg(regressionSlope*1000, 0, 'f', 2));
            }
            else if (! _interpolationSettings->getUseLocalDetrending() && ! _interpolationSettings->getUseGlocalDetrending())
            {
                std::string errorStr;

                setMultipleDetrendingHeightTemperatureRange(_interpolationSettings);
                _interpolationSettings->setCurrentCombination(_interpolationSettings->getSelectedCombination());
                _interpolationSettings->clearFitting();
                if (! multipleDetrendingElevationFitting(_proxyPos, _outInterpolationPoints, _interpolationSettings, myVar, errorStr, true)) return;

                std::vector<std::vector<double>> parameters = _interpolationSettings->getFittingParameters();

                if (!parameters.empty() && !parameters.front().empty())
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
            else
            {
                QMessageBox::warning(nullptr, "Multiple detrending", "Disable multiple detrendinding to show the model lapse rate.");
            }
        }
        else
        {
            if (! _interpolationSettings->getUseMultipleDetrending())
            {
                float xMin = getProxyMinValue(_outInterpolationPoints, _proxyPos);
                float xMax = getProxyMaxValue(_outInterpolationPoints, _proxyPos);
                bool isZeroIntercept = false;
                if (!regressionGeneric(_outInterpolationPoints, _interpolationSettings, _proxyPos, isZeroIntercept))
                {
                    return;
                }
                float regressionSlope = _interpolationSettings->getProxy(_proxyPos)->getRegressionSlope();
                float regressionIntercept = _interpolationSettings->getProxy(_proxyPos)->getRegressionIntercept();
                point.setX(xMin);
                point.setY(regressionIntercept + regressionSlope * xMin);
                point_vector.append(point);
                point.setX(xMax);
                point.setY(regressionIntercept + regressionSlope * xMax);
                point_vector.append(point);

                float regressionR2 = _interpolationSettings->getProxy(_proxyPos)->getRegressionR2();
                if (regressionR2 != NODATA)
                {
                    r2.setText(QString("%1").arg(regressionR2, 0, 'f', 2));
                }
                lapseRate.setText(QString("%1").arg(regressionSlope, 0, 'f', 2));
            }
            else if (!_interpolationSettings->getUseLocalDetrending())
            {
                //TODO
            }
            else
            {
                QMessageBox::warning(nullptr, "Multiple detrending", "Disable multiple detrendinding to show the model lapse rate.");
            }
        }
        chartView->drawModelLapseRate(point_vector);
    }
}


void Crit3DProxyWidget::on_actionChangeLeftAxis()
{

    DialogChangeAxis changeAxisDialog(1, false);
    if (changeAxisDialog.result() == QDialog::Accepted)
    {
        chartView->axisY->setMin(changeAxisDialog.getMinVal());
        chartView->axisY->setMax(changeAxisDialog.getMaxVal());
    }
}


void Crit3DProxyWidget::addMacroAreaLR()
{
    //controllo is glocal ready viene fatto a monte
    chartView->cleanModelLapseRate();
    r2.clear();
    lapseRate.clear();
    if (_macroAreaNumber >= 0 && _macroAreaNumber < (int)_interpolationSettings->getMacroAreas().size())
    {
        std::string errorStr;
        setMultipleDetrendingHeightTemperatureRange(_interpolationSettings);
        glocalDetrendingFitting(_outInterpolationPoints, _interpolationSettings, myVar,errorStr);

        //plot
        std::vector<std::vector<double>> myParameters = _interpolationSettings->getMacroAreas()[_macroAreaNumber].getParameters();

        if (myParameters.empty()) return;
        if (myParameters.front().size() < 4) return; //elevation is not significant

        double xMin = getZmin(_outInterpolationPoints);
        double xMax = getZmax(_outInterpolationPoints);
        QList<QPointF> point_vector;
        QPointF point;

        std::vector <double> xVector;
        for (int m = xMin; m < xMax; m += 5)
            xVector.push_back(m);

        for (int p = 0; p < int(xVector.size()); p++)
        {
            point.setX(xVector[p]);
            if (myParameters.front().size() == 4)
                point.setY(lapseRatePiecewise_two(xVector[p], myParameters.front()));
            else if (myParameters.front().size() == 5)
                point.setY(lapseRatePiecewise_three(xVector[p], myParameters.front()));
            else if (myParameters.front().size() == 6)
                point.setY(lapseRatePiecewise_three_free(xVector[p], myParameters.front()));
            point_vector.append(point);
        }

        chartView->drawModelLapseRate(point_vector);
    }
    return;
}
