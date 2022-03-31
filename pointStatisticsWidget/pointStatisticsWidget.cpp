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
    gridLeftLayout->addWidget(&classWidth,3,0,1,-1);
    valMax.setMaximumWidth(60);
    valMax.setMaximumHeight(30);
    gridLeftLayout->addWidget(&valMax,3,1,1,-1);
    valMin.setMaximumWidth(60);
    valMin.setMaximumHeight(30);
    gridLeftLayout->addWidget(&valMin,3,2,1,-1);
    smoothing.setMaximumWidth(60);
    smoothing.setMaximumHeight(30);
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
    availability.setMaximumWidth(60);
    availability.setMaximumHeight(30);
    gridRightLayout->addWidget(&availability,1,0,1,1);
    QLabel *rateLabel = new QLabel(tr("rate"));
    gridRightLayout->addWidget(rateLabel,2,0,1,1);
    QLabel *r2Label = new QLabel(tr("r2"));
    gridRightLayout->addWidget(r2Label,2,1,1,1);
    QLabel *significanceLabel = new QLabel(tr("significance [MK]"));
    gridRightLayout->addWidget(significanceLabel,2,2,1,1);
    rate.setEnabled(false);
    rate.setMaximumWidth(60);
    rate.setMaximumHeight(30);
    gridRightLayout->addWidget(&rate,3,0,1,1);
    r2.setEnabled(false);
    r2.setMaximumWidth(60);
    r2.setMaximumHeight(30);
    gridRightLayout->addWidget(&r2,3,1,1,1);
    significance.setEnabled(false);
    significance.setMaximumWidth(60);
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
    average.setMaximumWidth(60);
    average.setMaximumHeight(30);
    gridRightLayout->addWidget(&average,5,0,1,1);
    mode.setEnabled(false);
    mode.setMaximumWidth(60);
    mode.setMaximumHeight(30);
    gridRightLayout->addWidget(&mode,5,1,1,1);
    median.setEnabled(false);
    median.setMaximumWidth(60);
    median.setMaximumHeight(30);
    gridRightLayout->addWidget(&median,5,2,1,1);
    sigma.setEnabled(false);
    sigma.setMaximumWidth(60);
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
    connect(&smoothing, &QTextEdit::textChanged, [=](){ changeSmooth(); });

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
            sigma.setEnabled(false);
            mode.setEnabled(false);
            median.setEnabled(false);

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
            availability.setText(QString::number(availab));
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
            sigma.setEnabled(false);
            mode.setEnabled(false);
            median.setEnabled(false);

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
            if ( elaborationOnPoint(&myError, meteoPointsDbHandler, nullptr, &meteoPointTemp, &clima, isGrid, startDate, endDate, isAnomaly, meteoSettings))
            {
                QMessageBox::information(nullptr, "Error", "Data not available in the reference period");
                return;
            }

            isAnomaly = true;
            firstYear = analysisYearFrom.currentText().toInt();
            lastYear = analysisYearTo.currentText().toInt();
            clima.setYearStart(firstYear);
            clima.setYearEnd(lastYear);
            clima.setGenericPeriodDateStart(QDate(firstYear, dayFrom.date().month(), dayFrom.date().day()));
            clima.setGenericPeriodDateEnd(QDate(lastYear, dayTo.date().month(), dayTo.date().day()));
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
            availability.setText(QString::number(availab));
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

            /*
            float stdDev = statistics::standardDeviation(outputValues, outputValues.size());
            sigma.setText(QString::number(stdDev, 'f', 3));

            int nrValues = int(outputValues.size());
            float percentile = sorting::percentile(outputValues, &nrValues, 50.0, true);
            median.setText(QString::number(percentile, 'f', 3));

            float modeVal = sorting::mode(outputValues, &nrValues, true);
            mode.setText(QString::number(modeVal, 'f', 3));
            */
        }
        else if (graph.currentText() == "Climate")
        {
            classWidth.setEnabled(false);
            valMax.setEnabled(false);
            valMin.setEnabled(false);
            sigma.setEnabled(false);
            mode.setEnabled(false);
            median.setEnabled(false);

            smoothing.setEnabled(true);
            availability.clear();
            significance.clear();
            average.clear();
            r2.clear();
            rate.clear();

            int firstYear = yearFrom.currentText().toInt();
            int lastYear = yearTo.currentText().toInt();
            QDate startDate(firstYear, 1, 1);
            QDate endDate(lastYear, 12, 31);
            float dataPresence;
            std::vector<float> dailyClima;
            std::vector<float> decadeClima;
            std::vector<float> monthlyClima;
            for (int fill = 0; fill < 12; fill++)
            {
                monthlyClima.push_back(0);
            }
            for (int fill = 0; fill < 36; fill++)
            {
                decadeClima.push_back(0);
            }
            for (int fill = 0; fill < 366; fill++)
            {
                dailyClima.push_back(0);
            }
            computeClimateOnDailyData(meteoPoints[0], myVar, startDate, endDate,
                                          smoothing.toPlainText().toInt(), &dataPresence, quality, climateParameters, meteoSettings, dailyClima, decadeClima, monthlyClima);
            availability.setText(QString::number(dataPresence));
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
        availability.setText(QString::number(availab));
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
    return;
}

void Crit3DPointStatisticsWidget::changeSmooth()
{
    //plot();
}
