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
#include "basicMath.h"
#include "climate.h"
#include "dialogElaboration.h"
#include "gammaFunction.h"
#include "formInfo.h"

#include <QLayout>
#include <QDate>

Crit3DPointStatisticsWidget::Crit3DPointStatisticsWidget(bool isGrid, Crit3DMeteoPointsDbHandler* meteoPointsDbHandler, Crit3DMeteoGridDbHandler* meteoGridDbHandler, QList<Crit3DMeteoPoint> meteoPoints,
                                                         QDate firstDaily, QDate lastDaily, QDateTime firstHourly, QDateTime lastHourly, Crit3DMeteoSettings *meteoSettings, QSettings *settings, Crit3DClimateParameters *climateParameters, Crit3DQuality *quality)
:isGrid(isGrid), meteoPointsDbHandler(meteoPointsDbHandler), meteoGridDbHandler(meteoGridDbHandler), meteoPoints(meteoPoints), firstDaily(firstDaily),
  lastDaily(lastDaily), firstHourly(firstHourly), lastHourly(lastHourly), meteoSettings(meteoSettings), settings(settings), climateParameters(climateParameters), quality(quality)
{
    this->setWindowTitle("Point statistics Id:"+QString::fromStdString(meteoPoints[0].id)+" "+QString::fromStdString(meteoPoints[0].name));
    this->resize(1240, 700);
    this->setSizePolicy(QSizePolicy::Expanding, QSizePolicy::Expanding);
    this->setAttribute(Qt::WA_DeleteOnClose);
    
    // layout
    QVBoxLayout *mainLayout = new QVBoxLayout();
    QHBoxLayout *upperLayout = new QHBoxLayout();
    QVBoxLayout *rightLayout = new QVBoxLayout();
    QVBoxLayout *leftLayout = new QVBoxLayout();

    QGroupBox *horizontalGroupBox = new QGroupBox();
    QVBoxLayout *elabLayout = new QVBoxLayout();
    QHBoxLayout *variableLayout = new QHBoxLayout;
    QGroupBox *variableGroupBox = new QGroupBox();
    QGroupBox *referencePeriodGroupBox = new QGroupBox();
    analysisPeriodGroupBox = new QGroupBox();
    QHBoxLayout *referencePeriodChartLayout = new QHBoxLayout;
    QHBoxLayout *analysisPeriodChartLayout = new QHBoxLayout;
    QHBoxLayout *dateChartLayout = new QHBoxLayout;
    QGroupBox *gridLeftGroupBox = new QGroupBox();
    QGridLayout *gridLeftLayout = new QGridLayout;

    QGroupBox *jointStationsGroupBox = new QGroupBox();
    QHBoxLayout *jointStationsLayout = new QHBoxLayout;
    QVBoxLayout *jointStationsSelectLayout = new QVBoxLayout;
    QGridLayout *gridRightLayout = new QGridLayout;

    QVBoxLayout *plotLayout = new QVBoxLayout;

    QLabel *variableLabel = new QLabel(tr("Variable: "));

    elaboration.setText("Elaboration");
    elaboration.setEnabled(false);
    
    dailyButton.setText("Daily");
    hourlyButton.setText("Hourly");
    if (!firstDaily.isNull() || !lastDaily.isNull())
    {
        dailyButton.setChecked(true); //default
        currentFrequency = daily; //default
    }
    else
    {
        hourlyButton.setChecked(true);
        currentFrequency = hourly;
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

    variableLayout->addWidget(&dailyButton);
    variableLayout->addWidget(&hourlyButton);
    variableLayout->addWidget(variableLabel);
    variableLayout->addWidget(&variable);
    variableLayout->addWidget(&elaboration);
    variableGroupBox->setLayout(variableLayout);

    referencePeriodGroupBox->setTitle("Reference period");
    QLabel *yearFromLabel = new QLabel(tr("From"));
    referencePeriodChartLayout->addWidget(yearFromLabel);
    referencePeriodChartLayout->addWidget(&yearFrom);
    QLabel *yearToLabel = new QLabel(tr("To"));
    referencePeriodChartLayout->addWidget(yearToLabel);
    referencePeriodChartLayout->addWidget(&yearTo);
    referencePeriodGroupBox->setLayout(referencePeriodChartLayout);

    analysisPeriodGroupBox->setTitle("Analysis period");
    QLabel *analysisYearFromLabel = new QLabel(tr("From"));
    analysisPeriodChartLayout->addWidget(analysisYearFromLabel);
    analysisPeriodChartLayout->addWidget(&analysisYearFrom);
    QLabel *analysisYearToLabel = new QLabel(tr("To"));
    analysisPeriodChartLayout->addWidget(analysisYearToLabel);
    analysisPeriodChartLayout->addWidget(&analysisYearTo);
    analysisPeriodGroupBox->setLayout(analysisPeriodChartLayout);
    analysisPeriodGroupBox->setVisible(false);

    QLabel *dayFromLabel = new QLabel(tr("Day from"));
    dateChartLayout->addWidget(dayFromLabel);
    dayFrom.setDisplayFormat("dd/MM");
    dayFrom.setDate(QDate(1800,1,1));
    dateChartLayout->addWidget(&dayFrom);
    QLabel *dayToLabel = new QLabel(tr("Day to"));
    dateChartLayout->addWidget(dayToLabel);
    dayTo.setDisplayFormat("dd/MM");
    dayTo.setDate(QDate(1800,12,31));
    dateChartLayout->addWidget(&dayTo);
    QLabel *hourLabel = new QLabel(tr("Hour"));
    hour.setRange(1,24);
    hour.setSingleStep(1);
    hour.setEnabled(false);
    dateChartLayout->addWidget(hourLabel);
    dateChartLayout->addWidget(&hour);
    compute.setText("Compute");
    compute.setMaximumWidth(120);

    QLabel *jointStationsLabel = new QLabel(tr("Stations:"));
    jointStationsSelectLayout->addWidget(jointStationsLabel);
    jointStationsSelectLayout->addWidget(&jointStationsList);
    jointStationsList.setMaximumWidth(this->width()/5);
    QHBoxLayout *addDeleteStationLayout = new QHBoxLayout;
    addDeleteStationLayout->addWidget(&addStation);
    addStation.setText("Add");
    addStation.setMaximumWidth(120);
    deleteStation.setText("Delete");
    deleteStation.setMaximumWidth(120);
    saveToDb.setText("Save to DB");
    saveToDb.setMaximumWidth(120);
    addDeleteStationLayout->addWidget(&deleteStation);
    jointStationsSelectLayout->addLayout(addDeleteStationLayout);
    jointStationsSelectLayout->addWidget(&saveToDb);
    jointStationsLayout->addLayout(jointStationsSelectLayout);
    jointStationsSelected.setMaximumWidth(this->width()/4);
    jointStationsSelected.setMaximumHeight(this->height()/4);
    jointStationsLayout->addWidget(&jointStationsSelected);
    jointStationsGroupBox->setTitle("Joint stations");
    jointStationsGroupBox->setLayout(jointStationsLayout);

    chartView = new PointStatisticsChartView();
    chartView->setMinimumHeight(this->height()*2/3);
    plotLayout->addWidget(chartView);

    horizontalGroupBox->setLayout(elabLayout);
    elabLayout->addWidget(variableGroupBox);
    elabLayout->addWidget(referencePeriodGroupBox);
    elabLayout->addWidget(analysisPeriodGroupBox);
    elabLayout->addLayout(dateChartLayout);
    elabLayout->addWidget(&compute);
    leftLayout->addWidget(horizontalGroupBox);

    QLabel *classWidthLabel = new QLabel(tr("Class width"));
    gridLeftLayout->addWidget(classWidthLabel,0,0,1,1);
    QLabel *valMaxLabel = new QLabel(tr("Val max"));
    gridLeftLayout->addWidget(valMaxLabel,0,1,1,1);
    QLabel *valMinLabel = new QLabel(tr("Val min"));
    gridLeftLayout->addWidget(valMinLabel,0,2,1,1);
    QLabel *smoothingLabel = new QLabel(tr("Smoothing"));
    gridLeftLayout->addWidget(smoothingLabel,0,3,1,1);
    classWidth.setMaximumWidth(60);
    classWidth.setMaximumHeight(30);
    classWidth.setText("1");
    gridLeftLayout->addWidget(&classWidth,3,0,1,-1);
    valMax.setMaximumWidth(60);
    valMax.setMaximumHeight(30);
    gridLeftLayout->addWidget(&valMax,3,1,1,-1);
    valMin.setMaximumWidth(60);
    valMin.setMaximumHeight(30);
    gridLeftLayout->addWidget(&valMin,3,2,1,-1);
    smoothing.setMaximumWidth(60);
    smoothing.setMaximumHeight(30);
    smoothing.setValidator(new QIntValidator(0, 366));
    smoothing.setText("0");
    gridLeftLayout->addWidget(&smoothing,3,3,1,-1);
    gridLeftGroupBox->setMaximumHeight(this->height()/6);
    gridLeftGroupBox->setLayout(gridLeftLayout);
    leftLayout->addWidget(gridLeftGroupBox);

    rightLayout->addWidget(jointStationsGroupBox);
    QLabel *selectGraphLabel = new QLabel(tr("Select graph:"));
    rightLayout->addWidget(selectGraphLabel);
    if (currentFrequency == daily)
    {
        if (!firstDaily.isNull() || !lastDaily.isNull())
        {
            graph.addItem("Distribution");
            graph.addItem("Climate");
            graph.addItem("Trend");
            graph.addItem("Anomaly trend");

            for(int i = 0; i <= lastDaily.year()-firstDaily.year(); i++)
            {
                yearFrom.addItem(QString::number(firstDaily.year()+i));
                yearTo.addItem(QString::number(firstDaily.year()+i));
                analysisYearFrom.addItem(QString::number(firstDaily.year()+i));
                analysisYearTo.addItem(QString::number(firstDaily.year()+i));
            }
            yearTo.setCurrentText(QString::number(lastDaily.year()));
            analysisYearTo.setCurrentText(QString::number(lastDaily.year()));
        }
    }
    else if (currentFrequency == hourly)
    {
        if (!firstHourly.isNull() || !lastHourly.isNull())
        {
            graph.addItem("Distribution");
            for(int i = 0; i <= lastHourly.date().year() - firstHourly.date().year(); i++)
            {
                yearFrom.addItem(QString::number(firstHourly.date().year()+i));
                yearTo.addItem(QString::number(firstHourly.date().year()+i));
            }
            yearTo.setCurrentText(QString::number(lastHourly.date().year()));
        }
    }
    graph.setMaximumWidth(this->width()/5);
    graph.setSizeAdjustPolicy(QComboBox::AdjustToContents);
    rightLayout->addWidget(&graph);
    QLabel *availabilityLabel = new QLabel(tr("availability [%]"));
    gridRightLayout->addWidget(availabilityLabel,0,0,1,1);
    availability.setEnabled(false);
    availability.setMaximumWidth(80);
    availability.setMaximumHeight(30);
    gridRightLayout->addWidget(&availability,1,0,1,1);
    QLabel *rateLabel = new QLabel(tr("rate"));
    gridRightLayout->addWidget(rateLabel,2,0,1,1);
    QLabel *r2Label = new QLabel(tr("r2"));
    gridRightLayout->addWidget(r2Label,2,1,1,1);
    QLabel *significanceLabel = new QLabel(tr("significance [MK]"));
    gridRightLayout->addWidget(significanceLabel,2,2,1,1);
    rate.setEnabled(false);
    rate.setMaximumWidth(80);
    rate.setMaximumHeight(30);
    gridRightLayout->addWidget(&rate,3,0,1,1);
    r2.setEnabled(false);
    r2.setMaximumWidth(80);
    r2.setMaximumHeight(30);
    gridRightLayout->addWidget(&r2,3,1,1,1);
    significance.setEnabled(false);
    significance.setMaximumWidth(80);
    significance.setMaximumHeight(30);
    gridRightLayout->addWidget(&significance,3,2,1,1);
    QLabel *averageLabel = new QLabel(tr("average"));
    gridRightLayout->addWidget(averageLabel,4,0,1,1);
    QLabel *modeLabel = new QLabel(tr("mode"));
    gridRightLayout->addWidget(modeLabel,4,1,1,1);
    QLabel *medianLabel = new QLabel(tr("median"));
    gridRightLayout->addWidget(medianLabel,4,2,1,1);
    QLabel *sigmaLabel = new QLabel(tr("sigma"));
    gridRightLayout->addWidget(sigmaLabel,4,3,1,1);
    average.setEnabled(false);
    average.setMaximumWidth(80);
    average.setMaximumHeight(30);
    gridRightLayout->addWidget(&average,5,0,1,1);
    mode.setEnabled(false);
    mode.setMaximumWidth(80);
    mode.setMaximumHeight(30);
    gridRightLayout->addWidget(&mode,5,1,1,1);
    median.setEnabled(false);
    median.setMaximumWidth(80);
    median.setMaximumHeight(30);
    gridRightLayout->addWidget(&median,5,2,1,1);
    sigma.setEnabled(false);
    sigma.setMaximumWidth(80);
    sigma.setMaximumHeight(30);
    gridRightLayout->addWidget(&sigma,5,3,1,1);

    rightLayout->addLayout(gridRightLayout);

    upperLayout->addLayout(leftLayout);
    upperLayout->addLayout(rightLayout);
    mainLayout->addLayout(upperLayout);
    mainLayout->addLayout(plotLayout);
    setLayout(mainLayout);

    connect(&dailyButton, &QRadioButton::clicked, [=](){ dailyVar(); });
    connect(&hourlyButton, &QRadioButton::clicked, [=](){ hourlyVar(); });
    connect(&variable, &QComboBox::currentTextChanged, [=](const QString &newVariable){ this->changeVar(newVariable); });
    connect(&graph, &QComboBox::currentTextChanged, [=](const QString &newGraph){ this->changeGraph(newGraph); });
    connect(&compute, &QPushButton::clicked, [=](){ plot(); });
    connect(&elaboration, &QPushButton::clicked, [=](){ showElaboration(); });
    connect(&smoothing, &QLineEdit::textChanged, [=](){ updatePlot(); });
    connect(&valMax, &QLineEdit::textChanged, [=](){ updatePlot(); });
    connect(&valMin, &QLineEdit::textChanged, [=](){ updatePlot(); });
    connect(&classWidth, &QLineEdit::textChanged, [=](){ updatePlot(); });

    plot();
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

void Crit3DPointStatisticsWidget::dailyVar()
{
    currentFrequency = daily;
    hour.setEnabled(false);
    variable.clear();
    yearFrom.clear();
    yearTo.clear();
    analysisYearFrom.clear();
    analysisYearTo.clear();
    std::map<meteoVariable, std::string>::const_iterator it;
    for(it = MapDailyMeteoVarToString.begin(); it != MapDailyMeteoVarToString.end(); ++it)
    {
        variable.addItem(QString::fromStdString(it->second));
    }
    myVar = getKeyMeteoVarMeteoMap(MapDailyMeteoVarToString, variable.currentText().toStdString());

    graph.clear();
    if (!firstDaily.isNull() || !lastDaily.isNull())
    {
        graph.addItem("Distribution");
        graph.addItem("Climate");
        graph.addItem("Trend");
        graph.addItem("Anomaly trend");

        for(int i = 0; i <= lastDaily.year()-firstDaily.year(); i++)
        {
            yearFrom.addItem(QString::number(firstDaily.year()+i));
            yearTo.addItem(QString::number(firstDaily.year()+i));
            analysisYearFrom.addItem(QString::number(firstDaily.year()+i));
            analysisYearTo.addItem(QString::number(firstDaily.year()+i));
        }
        yearTo.setCurrentText(QString::number(lastDaily.year()));
        analysisYearTo.setCurrentText(QString::number(lastDaily.year()));
    }
    else
    {
        QMessageBox::information(nullptr, "Warning", "No daily data");
    }

}

void Crit3DPointStatisticsWidget::hourlyVar()
{
    currentFrequency = hourly;
    hour.setEnabled(true);
    variable.clear();
    yearFrom.clear();
    yearTo.clear();
    std::map<meteoVariable, std::string>::const_iterator it;
    for(it = MapHourlyMeteoVarToString.begin(); it != MapHourlyMeteoVarToString.end(); ++it)
    {
        variable.addItem(QString::fromStdString(it->second));
    }
    myVar = getKeyMeteoVarMeteoMap(MapHourlyMeteoVarToString, variable.currentText().toStdString());

    graph.clear();
    if (!firstHourly.isNull() || !lastHourly.isNull())
    {
        graph.addItem("Distribution");

        for(int i = 0; i <= lastHourly.date().year() - firstHourly.date().year(); i++)
        {
            yearFrom.addItem(QString::number(firstHourly.date().year()+i));
            yearTo.addItem(QString::number(firstHourly.date().year()+i));
        }
        yearTo.setCurrentText(QString::number(lastHourly.date().year()));
    }
    else
    {
        QMessageBox::information(nullptr, "Warning", "No hourly data");
    }

}

void Crit3DPointStatisticsWidget::changeGraph(const QString graphName)
{
    if (graphName == "Trend")
    {
        analysisPeriodGroupBox->setVisible(false);
        elaboration.setEnabled(true);
    }
    else if (graphName == "Anomaly trend")
    {
        analysisPeriodGroupBox->setVisible(true);
        elaboration.setEnabled(true);
    }
    else
    {
        analysisPeriodGroupBox->setVisible(false);
        elaboration.setEnabled(false);
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

void Crit3DPointStatisticsWidget::plot()
{
    if (currentFrequency == daily)
    {
        if (graph.currentText() == "Trend")
        {
            classWidth.setEnabled(false);
            valMax.setEnabled(false);
            valMin.setEnabled(false);
            smoothing.setEnabled(false);

            availability.clear();
            significance.clear();
            average.clear();
            r2.clear();
            rate.clear();

            int firstYear = yearFrom.currentText().toInt();
            int lastYear = yearTo.currentText().toInt();
            // check years
            if (lastYear - firstYear < 2)
            {
                QMessageBox::information(nullptr, "Error", "Number of valid years < 3");
                return;
            }
            clima.setVariable(myVar);
            if (myVar == dailyPrecipitation || myVar == dailyReferenceEvapotranspirationHS || myVar == dailyReferenceEvapotranspirationPM || myVar == dailyBIC)
            {
                clima.setElab1("sum");
            }
            else
            {
                clima.setElab1("average");
            }
            clima.setYearStart(firstYear);
            clima.setYearEnd(lastYear);
            clima.setGenericPeriodDateStart(QDate(firstYear, dayFrom.date().month(), dayFrom.date().day()));
            clima.setGenericPeriodDateEnd(QDate(lastYear, dayTo.date().month(), dayTo.date().day()));
            if (dayFrom.date()> dayTo.date())
            {
                clima.setNYears(1);
            }
            else
            {
                clima.setNYears(0);
            }
            std::vector<float> outputValues;
            std::vector<int> years;
            QString myError;
            bool isAnomaly = false;
            // copy data to MPTemp
            Crit3DMeteoPoint meteoPointTemp;
            meteoPointTemp.id = meteoPoints[0].id;
            meteoPointTemp.point.utm.x = meteoPoints[0].point.utm.x;  // LC to compute distance in passingClimateToAnomaly
            meteoPointTemp.point.utm.y = meteoPoints[0].point.utm.y;  // LC to compute distance in passingClimateToAnomaly
            meteoPointTemp.point.z = meteoPoints[0].point.z;
            meteoPointTemp.latitude = meteoPoints[0].latitude;
            meteoPointTemp.elaboration = meteoPoints[0].elaboration;

            // meteoPointTemp should be init
            meteoPointTemp.nrObsDataDaysH = 0;
            meteoPointTemp.nrObsDataDaysD = 0;

            int validYears = computeAnnualSeriesOnPointFromDaily(&myError, meteoPointsDbHandler, meteoGridDbHandler,
                                                     &meteoPointTemp, &clima, isGrid, isAnomaly, meteoSettings, outputValues);
            if (validYears < 3)
            {
                QMessageBox::information(nullptr, "Error", "Number of valid years < 3");
                return;
            }

            float sum = 0;
            int count = 0;
            for (int i = firstYear; i<=lastYear; i++)
            {
                years.push_back(i);
                if (outputValues[count] != NODATA)
                {
                    sum = sum + outputValues[count];
                }
                count = count + 1;
            }
            // draw
            chartView->drawTrend(years, outputValues);

            float availab = ((float)validYears/(float)years.size())*100.0;
            availability.setText(QString::number(availab, 'f', 3));
            float mkendall = statisticalElab(mannKendall, NODATA, outputValues, outputValues.size(), meteoSettings->getRainfallThreshold());
            significance.setText(QString::number(mkendall, 'f', 3));
            float averageValue = sum/validYears;
            average.setText(QString::number(averageValue, 'f', 1));

            float myCoeff = NODATA;
            float myIntercept = NODATA;
            float myR2 = NODATA;
            bool isZeroIntercept = false;
            std::vector<float> yearsFloat(years.begin(), years.end());
            statistics::linearRegression(yearsFloat, outputValues, outputValues.size(), isZeroIntercept,
                                             &myIntercept, &myCoeff, &myR2);
            r2.setText(QString::number(myR2, 'f', 3));
            rate.setText(QString::number(myCoeff, 'f', 3));
        }
        else if (graph.currentText() == "Anomaly trend")
        {
            classWidth.setEnabled(false);
            valMax.setEnabled(false);
            valMin.setEnabled(false);
            smoothing.setEnabled(false);

            availability.clear();
            significance.clear();
            average.clear();
            r2.clear();
            rate.clear();

            int firstYear = yearFrom.currentText().toInt();
            int lastYear = yearTo.currentText().toInt();
            // check years
            if (lastYear - firstYear < 2)
            {
                QMessageBox::information(nullptr, "Error", "Number of valid years < 3");
                return;
            }
            clima.setVariable(myVar);
            if (myVar == dailyPrecipitation || myVar == dailyReferenceEvapotranspirationHS || myVar == dailyReferenceEvapotranspirationPM || myVar == dailyBIC)
            {
                clima.setElab1("sum");
            }
            else
            {
                clima.setElab1("average");
            }
            clima.setYearStart(firstYear);
            clima.setYearEnd(lastYear);
            clima.setGenericPeriodDateStart(QDate(firstYear, dayFrom.date().month(), dayFrom.date().day()));
            clima.setGenericPeriodDateEnd(QDate(lastYear, dayTo.date().month(), dayTo.date().day()));
            if (dayFrom.date()> dayTo.date())
            {
                clima.setNYears(1);
            }
            else
            {
                clima.setNYears(0);
            }
            std::vector<float> outputValues;
            std::vector<int> years;
            QString myError;
            bool isAnomaly = false;
            // copy data to MPTemp
            Crit3DMeteoPoint meteoPointTemp;
            meteoPointTemp.id = meteoPoints[0].id;
            meteoPointTemp.point.utm.x = meteoPoints[0].point.utm.x;  // LC to compute distance in passingClimateToAnomaly
            meteoPointTemp.point.utm.y = meteoPoints[0].point.utm.y;  // LC to compute distance in passingClimateToAnomaly
            meteoPointTemp.point.z = meteoPoints[0].point.z;
            meteoPointTemp.latitude = meteoPoints[0].latitude;
            meteoPointTemp.elaboration = meteoPoints[0].elaboration;

            // meteoPointTemp should be init
            meteoPointTemp.nrObsDataDaysH = 0;
            meteoPointTemp.nrObsDataDaysD = 0;

            QDate startDate(clima.yearStart(), clima.genericPeriodDateStart().month(), clima.genericPeriodDateStart().day());
            QDate endDate(clima.yearEnd(), clima.genericPeriodDateEnd().month(), clima.genericPeriodDateEnd().day());
            if (!elaborationOnPoint(&myError, meteoPointsDbHandler, nullptr, &meteoPointTemp, &clima, isGrid, startDate, endDate, isAnomaly, meteoSettings))
            {
                QMessageBox::information(nullptr, "Error", "Data not available in the reference period");
                return;
            }

            firstYear = analysisYearFrom.currentText().toInt();
            lastYear = analysisYearTo.currentText().toInt();
            clima.setYearStart(firstYear);
            clima.setYearEnd(lastYear);
            clima.setGenericPeriodDateStart(QDate(firstYear, dayFrom.date().month(), dayFrom.date().day()));
            clima.setGenericPeriodDateEnd(QDate(lastYear, dayTo.date().month(), dayTo.date().day()));
            float elabResult = meteoPointTemp.elaboration;

            int validYears = computeAnnualSeriesOnPointFromDaily(&myError, meteoPointsDbHandler, meteoGridDbHandler,
                                                     &meteoPointTemp, &clima, isGrid, isAnomaly, meteoSettings, outputValues);
            if (validYears < 3)
            {
                QMessageBox::information(nullptr, "Error", "Number of valid years < 3");
                return;
            }

            float sum = 0;
            int count = 0;
            for (int i = firstYear; i<=lastYear; i++)
            {
                years.push_back(i);
                if (outputValues[count] != NODATA)
                {
                    outputValues[count] = outputValues[count] - elabResult;
                    sum = sum + outputValues[count];
                }
                count = count + 1;
            }
            // draw
            chartView->drawTrend(years, outputValues);

            float availab = ((float)validYears/(float)years.size())*100.0;
            availability.setText(QString::number(availab, 'f', 3));
            float mkendall = statisticalElab(mannKendall, NODATA, outputValues, outputValues.size(), meteoSettings->getRainfallThreshold());
            significance.setText(QString::number(mkendall, 'f', 3));
            float averageValue = sum/validYears;
            average.setText(QString::number(averageValue, 'f', 1));

            float myCoeff = NODATA;
            float myIntercept = NODATA;
            float myR2 = NODATA;
            bool isZeroIntercept = false;
            std::vector<float> yearsFloat(years.begin(), years.end());
            statistics::linearRegression(yearsFloat, outputValues, outputValues.size(), isZeroIntercept,
                                             &myIntercept, &myCoeff, &myR2);
            r2.setText(QString::number(myR2, 'f', 3));
            rate.setText(QString::number(myCoeff, 'f', 3));
        }
        else if (graph.currentText() == "Climate")
        {
            classWidth.setEnabled(false);
            valMax.setEnabled(false);
            valMin.setEnabled(false);
            smoothing.setEnabled(true);

            availability.clear();
            significance.clear();
            average.clear();
            r2.clear();
            rate.clear();

            bool ok = true;
            int smooth = smoothing.text().toInt(&ok);
            if (!ok || smooth < 0)
            {
                QMessageBox::information(nullptr, "Error", "Wrong smoothing factor");
                return;
            }

            int firstYear = yearFrom.currentText().toInt();
            int lastYear = yearTo.currentText().toInt();
            QDate startDate(firstYear, 1, 1);
            QDate endDate(lastYear, 12, 31);
            float dataPresence;
            std::vector<float> dailyClima;
            std::vector<float> decadalClima;
            std::vector<float> monthlyClima;
            for (int fill = 0; fill <= 12; fill++)
            {
                monthlyClima.push_back(0);
            }
            for (int fill = 0; fill <= 36; fill++)
            {
                decadalClima.push_back(0);
            }
            for (int fill = 0; fill <= 366; fill++)
            {
                dailyClima.push_back(0);
            }
            computeClimateOnDailyData(meteoPoints[0], myVar, startDate, endDate,
                                          smooth, &dataPresence, quality, climateParameters, meteoSettings, dailyClima, decadalClima, monthlyClima);
            availability.setText(QString::number(dataPresence, 'f', 3));

            QList<QPointF> dailyPointList;
            QList<QPointF> decadalPointList;
            QList<QPointF> monthlyPointList;
            for (int day = 1; day <= 366; day++)
            {
                QDate myDate = QDate(2000, 1, 1).addDays(day - 1);
                dailyPointList.append(QPointF(day,dailyClima[day]));
                int decade = decadeFromDate(myDate);
                int dayStart;
                int dayEnd;
                int month;
                intervalDecade(decade, myDate.year(), &dayStart, &dayEnd, &month);
                if (myDate.day() == (dayStart+dayEnd)/2)
                {
                    decadalPointList.append(QPointF(day,decadalClima[decade]));
                }
                if ( myDate.day() == round(getDaysInMonth(month, myDate.year())/2) )
                {
                    monthlyPointList.append(QPointF(day,monthlyClima[month]));
                }
            }
            // draw
            chartView->drawClima(dailyPointList, decadalPointList, monthlyPointList);
        }
        else if (graph.currentText() == "Distribution")
        {

            classWidth.setEnabled(true);
            valMax.setEnabled(true);
            valMin.setEnabled(true);
            smoothing.setEnabled(false);

            availability.clear();
            significance.clear();
            average.clear();
            r2.clear();
            rate.clear();
            std::vector<float> series;

            bool ok = true;
            int classWidthValue = classWidth.text().toInt(&ok);
            if (!ok || classWidthValue <= 0)
            {
                QMessageBox::information(nullptr, "Error", "Wrong class Width value");
                return;
            }
            float myMinValue = NODATA;
            float myMaxValue = NODATA;
            bool isFirstData = true;

            int firstYear = yearFrom.currentText().toInt();
            int lastYear = yearTo.currentText().toInt();
            QDate firstDate(firstYear, dayFrom.date().month(), dayFrom.date().day());
            QDate lastDate(lastYear, dayTo.date().month(), dayTo.date().day());

            bool insideInterval = true;
            QDate dateStartPeriod = firstDate;
            QDate dateEndPeriod = lastDate;
            if (firstDate.dayOfYear() <= lastDate.dayOfYear())
            {
                insideInterval = true;
                dateEndPeriod.setDate(dateStartPeriod.year(), dateEndPeriod.month(), dateEndPeriod.day());
            }
            else
            {
                insideInterval = false;
                dateEndPeriod.setDate(dateStartPeriod.year()+1, dateEndPeriod.month(), dateEndPeriod.day());
            }

            int totDays = 0;
            quality::qualityType check;
            for (QDate myDate = firstDate; myDate <= lastDate; myDate = myDate.addDays(1))
            {
                if (myDate >= dateStartPeriod && myDate <= dateEndPeriod)
                {
                    totDays = totDays + 1;
                    if (myDate >= firstDaily && myDate <= lastDaily)
                    {
                        int i = firstDaily.daysTo(myDate);
                        float myDailyValue = meteoPoints[0].getMeteoPointValueD(getCrit3DDate(myDate), myVar, meteoSettings);
                        if (i<0 || i>meteoPoints[0].nrObsDataDaysD)
                        {
                            check = quality::missing_data;
                        }
                        else
                        {
                            check = quality->checkFastValueDaily_SingleValue(myVar, climateParameters, myDailyValue, myDate.month(), meteoPoints[0].point.z);
                        }
                        if (check == quality::accepted)
                        {
                            if (myVar = dailyPrecipitation)
                            {
                                if (myDailyValue < meteoSettings->getRainfallThreshold())
                                {
                                    myDailyValue = 0;
                                }
                            }
                            series.push_back(myDailyValue);
                            if (isFirstData)
                            {
                                myMinValue = myDailyValue;
                                myMaxValue = myDailyValue;
                                isFirstData = false;
                            }
                            else if (myDailyValue < myMinValue)
                            {
                                myMinValue = myDailyValue;
                            }
                            else if (myDailyValue > myMaxValue)
                            {
                                myMaxValue = myDailyValue;
                            }
                        }
                    }
                    if (myDate == dateEndPeriod)
                    {
                        if (insideInterval)
                        {
                            dateStartPeriod.setDate(myDate.year()+1, firstDate.month(), firstDate.day());
                            dateEndPeriod.setDate(myDate.year()+1, lastDate.month(), lastDate.day());
                        }
                        else
                        {
                            dateStartPeriod.setDate(myDate.year(), firstDate.month(), firstDate.day());
                            dateEndPeriod.setDate(myDate.year()+1, lastDate.month(), lastDate.day());
                        }
                        myDate = dateStartPeriod.addDays(-1);
                    }
                }
            }
            if (myMinValue == NODATA || myMaxValue == NODATA)
            {
                return; // no data
            }
            int minValueInt = myMinValue;
            int maxValueInt = myMaxValue + 1;

            // init
            std::vector<float> bucket;
            for (int i = 0; i<= (maxValueInt - minValueInt)/classWidthValue; i++)
            {
                bucket.push_back(0);
            }

            float dev_std = NODATA;
            float millile_3Dev = NODATA;
            float millile3dev = NODATA;
            float avg = NODATA;
            float modeVal = NODATA;
            int nrValues = int(series.size());
            std::vector<float> sortedSeries = series;

            if (myVar == dailyPrecipitation)
            {
                for (int i = 0; i < nrValues; i++)
                {
                    if (series[i] > 0)
                    {
                        int index = (series[i] - minValueInt)/classWidthValue;
                        bucket[index] = bucket[index] + 1;
                    }
                }

                double beta;
                double gamma;
                double pzero;
                if (!gammaFitting(series, nrValues, &beta, &gamma,  &pzero))
                {
                    return;
                }
            }
            else
            {
                for (int i = 0; i < nrValues; i++)
                {
                    if (series[i] > 0)
                    {
                        int index = (series[i] - minValueInt)/classWidthValue;
                        bucket[index] = bucket[index] + 1;
                    }
                }
                avg = statistics::mean(series, nrValues);
                dev_std = statistics::standardDeviation(series, nrValues);
                millile3dev = sorting::percentile(sortedSeries, &nrValues, 99.73, true);
                millile_3Dev = sorting::percentile(sortedSeries, &nrValues, 0.27, false);
            }
            availability.setText(QString::number(nrValues/totDays * 100, 'f', 3));
            average.setText(QString::number(avg, 'f', 3));

            int numModeData = 0;
            for (int i = 0; i<bucket.size(); i++)
            {
                if (bucket[i] > numModeData)
                {
                    numModeData = bucket[i];
                    modeVal = i;
                }
            }

            if (modeVal != NODATA)
            {
                mode.setText(QString::number(minValueInt + (modeVal*classWidthValue) + (classWidthValue/2), 'f', 3));
            }
            if (dev_std != NODATA)
            {
                sigma.setText(QString::number(dev_std, 'f', 3));
            }
            median.setText(QString::number(sorting::percentile(sortedSeries, &nrValues, 50, false), 'f', 3));

            valMax.blockSignals(true);
            valMin.blockSignals(true);
            int valMaxValue = valMax.text().toInt(&ok);
            if (!ok || valMax.text().isEmpty() || valMaxValue == NODATA)
            {
                valMaxValue = maxValueInt;
                valMax.setText(QString::number(valMaxValue));
            }
            int valMinValue = valMin.text().toInt(&ok);
            if (!ok || valMin.text().isEmpty() || valMinValue == NODATA)
            {
                valMinValue = minValueInt;
                valMin.setText(QString::number(valMinValue));
            }
            valMax.blockSignals(false);
            valMin.blockSignals(false);


        }
    }
    else if (currentFrequency == hourly)
    {

    }
}

void Crit3DPointStatisticsWidget::showElaboration()
{
    DialogElaboration elabDialog(settings, &clima, firstDaily, lastDaily);
    if (elabDialog.result() == QDialog::Accepted)
    {
        classWidth.setEnabled(false);
        valMax.setEnabled(false);
        valMin.setEnabled(false);
        sigma.setEnabled(false);
        mode.setEnabled(false);
        median.setEnabled(false);

        smoothing.setEnabled(false);
        availability.clear();
        significance.clear();
        average.clear();
        r2.clear();
        rate.clear();

        int firstYear = clima.yearStart();
        int lastYear = clima.yearEnd();
        // check years
        if (lastYear - firstYear < 2)
        {
            QMessageBox::information(nullptr, "Error", "Number of valid years < 3");
            return;
        }
        std::vector<float> outputValues;
        std::vector<int> years;
        QString myError;
        bool isAnomaly = false;
        // copy data to MPTemp
        Crit3DMeteoPoint meteoPointTemp;
        meteoPointTemp.id = meteoPoints[0].id;
        meteoPointTemp.point.utm.x = meteoPoints[0].point.utm.x;  // LC to compute distance in passingClimateToAnomaly
        meteoPointTemp.point.utm.y = meteoPoints[0].point.utm.y;  // LC to compute distance in passingClimateToAnomaly
        meteoPointTemp.point.z = meteoPoints[0].point.z;
        meteoPointTemp.latitude = meteoPoints[0].latitude;
        meteoPointTemp.elaboration = meteoPoints[0].elaboration;

        // meteoPointTemp should be init
        meteoPointTemp.nrObsDataDaysH = 0;
        meteoPointTemp.nrObsDataDaysD = 0;

        int validYears = computeAnnualSeriesOnPointFromDaily(&myError, meteoPointsDbHandler, meteoGridDbHandler,
                                                 &meteoPointTemp, &clima, isGrid, isAnomaly, meteoSettings, outputValues);
        if (validYears < 3)
        {
            //copy to clima original value for next elab
            clima.setYearStart(firstYear);
            clima.setYearEnd(lastYear);
            QMessageBox::information(nullptr, "Error", "Number of valid years < 3");
            return;
        }

        float sum = 0;
        int count = 0;
        for (int i = firstYear; i<=lastYear; i++)
        {
            years.push_back(i);
            if (outputValues[count] != NODATA)
            {
                sum = sum + outputValues[count];
            }
            count = count + 1;
        }
        // draw
        chartView->drawTrend(years, outputValues);

        float availab = ((float)validYears/(float)years.size())*100.0;
        availability.setText(QString::number(availab, 'f', 3));
        float mkendall = statisticalElab(mannKendall, NODATA, outputValues, outputValues.size(), meteoSettings->getRainfallThreshold());
        significance.setText(QString::number(mkendall, 'f', 3));
        float averageValue = sum/validYears;
        average.setText(QString::number(averageValue, 'f', 1));

        float myCoeff = NODATA;
        float myIntercept = NODATA;
        float myR2 = NODATA;
        bool isZeroIntercept = false;
        std::vector<float> yearsFloat(years.begin(), years.end());
        statistics::linearRegression(yearsFloat, outputValues, outputValues.size(), isZeroIntercept,
                                         &myIntercept, &myCoeff, &myR2);
        r2.setText(QString::number(myR2, 'f', 3));
        rate.setText(QString::number(myCoeff, 'f', 3));

        //copy to clima original value for next elab
        clima.setYearStart(firstYear);
        clima.setYearEnd(lastYear);
    }
    return;
}

void Crit3DPointStatisticsWidget::updatePlot()
{
    plot();
}
