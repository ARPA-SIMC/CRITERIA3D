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
#include "dialogChangeAxis.h"
#include "gammaFunction.h"
#include "furtherMathFunctions.h"
#include "formInfo.h"

#include <QLayout>
#include <QDate>

Crit3DPointStatisticsWidget::Crit3DPointStatisticsWidget(bool isGrid, Crit3DMeteoPointsDbHandler* meteoPointsDbHandler, Crit3DMeteoGridDbHandler* meteoGridDbHandler, QList<Crit3DMeteoPoint> meteoPoints,
                                                         QDate firstDaily, QDate lastDaily, QDateTime firstHourly, QDateTime lastHourly, Crit3DMeteoSettings *meteoSettings, QSettings *settings, Crit3DClimateParameters *climateParameters, Crit3DQuality *quality)
:isGrid(isGrid), meteoPointsDbHandler(meteoPointsDbHandler), meteoGridDbHandler(meteoGridDbHandler), meteoPoints(meteoPoints), firstDaily(firstDaily),
  lastDaily(lastDaily), firstHourly(firstHourly), lastHourly(lastHourly), meteoSettings(meteoSettings), settings(settings), climateParameters(climateParameters), quality(quality)
{
    this->setWindowTitle("Point statistics Id:"+QString::fromStdString(meteoPoints[0].id)+" "+QString::fromStdString(meteoPoints[0].name));
    this->resize(1000, 600);
    this->setSizePolicy(QSizePolicy::Expanding, QSizePolicy::Expanding);
    this->setAttribute(Qt::WA_DeleteOnClose);

    idPoints << meteoPoints[0].id;
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
    if (firstDaily.isNull() || lastDaily.isNull())
    {
        dailyButton.setEnabled(false);
    }
    else
    {
        dailyButton.setEnabled(true);
        dailyButton.setChecked(true); //default
        currentFrequency = daily; //default
    }

    if (firstHourly.isNull() || lastHourly.isNull())
    {
        hourlyButton.setEnabled(false);
    }
    else
    {
        hourlyButton.setEnabled(true);
        if (dailyButton.isEnabled())
        {
            hourlyButton.setChecked(false);
        }
        else
        {
            hourlyButton.setChecked(true);
            currentFrequency = hourly;
        }
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
    for (int i = 1; i<meteoPoints.size(); i++)
    {
        jointStationsList.addItem(QString::fromStdString(meteoPoints[i].id)+" "+QString::fromStdString(meteoPoints[i].name));
    }
    if (jointStationsList.count() != 0)
    {
        addStation.setEnabled(true);
    }
    else
    {
        addStation.setEnabled(false);
    }
    QHBoxLayout *addDeleteStationLayout = new QHBoxLayout;
    addDeleteStationLayout->addWidget(&addStation);
    addStation.setText("Add");
    addStation.setMaximumWidth(120);
    deleteStation.setText("Delete");
    deleteStation.setMaximumWidth(120);
    saveToDb.setText("Save to DB");
    saveToDb.setMaximumWidth(120);
    deleteStation.setEnabled(false);
    saveToDb.setEnabled(false);
    addDeleteStationLayout->addWidget(&deleteStation);
    jointStationsSelectLayout->addLayout(addDeleteStationLayout);
    jointStationsSelectLayout->addWidget(&saveToDb);
    jointStationsLayout->addLayout(jointStationsSelectLayout);
    jointStationsSelected.setMaximumWidth(this->width()/4);
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
    QLabel *valMinLabel = new QLabel(tr("Val min"));
    gridLeftLayout->addWidget(valMinLabel,0,1,1,1);
    QLabel *valMaxLabel = new QLabel(tr("Val max"));
    gridLeftLayout->addWidget(valMaxLabel,0,2,1,1);
    QLabel *smoothingLabel = new QLabel(tr("Smoothing"));
    gridLeftLayout->addWidget(smoothingLabel,0,3,1,1);
    classWidth.setMaximumWidth(60);
    classWidth.setMaximumHeight(24);
    classWidth.setText("1");
    classWidth.setValidator(new QIntValidator(1.0, 30.0));
    gridLeftLayout->addWidget(&classWidth,3,0,1,-1);

    valMin.setMaximumWidth(60);
    valMin.setMaximumHeight(24);
    valMin.setValidator(new QDoubleValidator(-999.0, 999.0, 1));
    gridLeftLayout->addWidget(&valMin,3,1,1,-1);
    valMax.setMaximumWidth(60);
    valMax.setMaximumHeight(24);
    valMax.setValidator(new QDoubleValidator(-999.0, 999.0, 1));
    gridLeftLayout->addWidget(&valMax,3,2,1,-1);
    smoothing.setMaximumWidth(60);
    smoothing.setMaximumHeight(24);
    smoothing.setValidator(new QIntValidator(0, 366));
    smoothing.setText("0");
    gridLeftLayout->addWidget(&smoothing,3,3,1,-1);
    gridLeftGroupBox->setMaximumHeight(this->height()/8);
    gridLeftGroupBox->setLayout(gridLeftLayout);
    leftLayout->addWidget(gridLeftGroupBox);

    rightLayout->addWidget(jointStationsGroupBox);

    QGroupBox *graphTypeGroupBox = new QGroupBox();
    graphTypeGroupBox->setTitle("Graph type");
    QHBoxLayout *graphTypeLayout = new QHBoxLayout();
    graphTypeLayout->setAlignment(Qt::AlignCenter);
    if (currentFrequency == daily)
    {
        if (!firstDaily.isNull() || !lastDaily.isNull())
        {
            graphType.addItem("Distribution");
            graphType.addItem("Climate");
            graphType.addItem("Trend");
            graphType.addItem("Anomaly trend");

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
            graphType.addItem("Distribution");
            for(int i = 0; i <= lastHourly.date().year() - firstHourly.date().year(); i++)
            {
                yearFrom.addItem(QString::number(firstHourly.date().year()+i));
                yearTo.addItem(QString::number(firstHourly.date().year()+i));
            }
            yearTo.setCurrentText(QString::number(lastHourly.date().year()));
        }
    }
    graphType.setMinimumWidth(200);
    graphTypeLayout->addWidget(&graphType);
    graphTypeGroupBox->setLayout(graphTypeLayout);
    rightLayout->addWidget(graphTypeGroupBox);

    QLabel *availabilityLabel = new QLabel(tr("availability [%]"));
    gridRightLayout->addWidget(availabilityLabel,0,0,1,1);
    availability.setEnabled(false);
    availability.setMaximumWidth(80);
    availability.setMaximumHeight(24);
    gridRightLayout->addWidget(&availability,0,1,1,1);
    QLabel *rateLabel = new QLabel(tr("rate"));
    gridRightLayout->addWidget(rateLabel,1,0,1,1);
    QLabel *r2Label = new QLabel(tr("r2"));
    gridRightLayout->addWidget(r2Label,1,1,1,1);
    QLabel *significanceLabel = new QLabel(tr("significance [MK]"));
    gridRightLayout->addWidget(significanceLabel,1,2,1,1);
    rate.setEnabled(false);
    rate.setMaximumWidth(80);
    rate.setMaximumHeight(24);
    gridRightLayout->addWidget(&rate,2,0,1,1);
    r2.setEnabled(false);
    r2.setMaximumWidth(80);
    r2.setMaximumHeight(24);
    gridRightLayout->addWidget(&r2,2,1,1,1);
    significance.setEnabled(false);
    significance.setMaximumWidth(80);
    significance.setMaximumHeight(24);
    gridRightLayout->addWidget(&significance,2,2,1,1);
    QLabel *averageLabel = new QLabel(tr("average"));
    gridRightLayout->addWidget(averageLabel,3,0,1,1);
    QLabel *modeLabel = new QLabel(tr("mode"));
    gridRightLayout->addWidget(modeLabel,3,1,1,1);
    QLabel *medianLabel = new QLabel(tr("median"));
    gridRightLayout->addWidget(medianLabel,3,2,1,1);
    QLabel *sigmaLabel = new QLabel(tr("sigma"));
    gridRightLayout->addWidget(sigmaLabel,3,3,1,1);
    average.setEnabled(false);
    average.setMaximumWidth(80);
    average.setMaximumHeight(24);
    gridRightLayout->addWidget(&average,4,0,1,1);
    mode.setEnabled(false);
    mode.setMaximumWidth(80);
    mode.setMaximumHeight(24);
    gridRightLayout->addWidget(&mode,4,1,1,1);
    median.setEnabled(false);
    median.setMaximumWidth(80);
    median.setMaximumHeight(24);
    gridRightLayout->addWidget(&median,4,2,1,1);
    sigma.setEnabled(false);
    sigma.setMaximumWidth(80);
    sigma.setMaximumHeight(24);
    gridRightLayout->addWidget(&sigma,4,3,1,1);

    // menu
    QMenuBar* menuBar = new QMenuBar();
    QMenu *editMenu = new QMenu("Edit");

    menuBar->addMenu(editMenu);
    mainLayout->setMenuBar(menuBar);

    QAction* changeLeftAxis = new QAction(tr("&Change axis left"), this);
    QAction* exportGraph = new QAction(tr("&Export graph"), this);
    QAction* exportData = new QAction(tr("&Export data"), this);

    editMenu->addAction(changeLeftAxis);
    editMenu->addAction(exportGraph);
    editMenu->addAction(exportData);

    rightLayout->addLayout(gridRightLayout);
    upperLayout->addLayout(leftLayout);
    upperLayout->addLayout(rightLayout);
    mainLayout->addLayout(upperLayout);
    mainLayout->addLayout(plotLayout);
    setLayout(mainLayout);

    connect(&dailyButton, &QRadioButton::clicked, [=](){ dailyVar(); });
    connect(&hourlyButton, &QRadioButton::clicked, [=](){ hourlyVar(); });
    connect(&variable, &QComboBox::currentTextChanged, [=](const QString &newVariable){ this->changeVar(newVariable); });
    connect(&graphType, &QComboBox::currentTextChanged, [=](const QString &newGraph){ this->changeGraph(newGraph); });
    connect(&compute, &QPushButton::clicked, [=](){ computePlot(); });
    connect(&elaboration, &QPushButton::clicked, [=](){ showElaboration(); });
    connect(&smoothing, &QLineEdit::editingFinished, [=](){ updatePlot(); });
    connect(&valMax, &QLineEdit::editingFinished, [=](){ updatePlotByVal(); });
    connect(&valMin, &QLineEdit::editingFinished, [=](){ updatePlotByVal(); });
    connect(&classWidth, &QLineEdit::editingFinished, [=](){ updatePlot(); });
    connect(&addStation, &QPushButton::clicked, [=](){ addStationClicked(); });
    connect(&deleteStation, &QPushButton::clicked, [=](){ deleteStationClicked(); });
    connect(&saveToDb, &QPushButton::clicked, [=](){ saveToDbClicked(); });
    connect(changeLeftAxis, &QAction::triggered, this, &Crit3DPointStatisticsWidget::on_actionChangeLeftAxis);
    connect(exportGraph, &QAction::triggered, this, &Crit3DPointStatisticsWidget::on_actionExportGraph);
    connect(exportData, &QAction::triggered, this, &Crit3DPointStatisticsWidget::on_actionExportData);

    plot();
    show();
}


Crit3DPointStatisticsWidget::~Crit3DPointStatisticsWidget()
{

}

void Crit3DPointStatisticsWidget::closeEvent(QCloseEvent *event)
{
    event->accept();
}

void Crit3DPointStatisticsWidget::dailyVar()
{
    currentFrequency = daily;

    variable.blockSignals(true);
    graphType.blockSignals(true);

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

    graphType.clear();
    if (!firstDaily.isNull() || !lastDaily.isNull())
    {
        graphType.addItem("Distribution");
        graphType.addItem("Climate");
        graphType.addItem("Trend");
        graphType.addItem("Anomaly trend");

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

    variable.blockSignals(false);
    graphType.blockSignals(false);
    computePlot();
}

void Crit3DPointStatisticsWidget::hourlyVar()
{
    currentFrequency = hourly;
    variable.blockSignals(true);
    graphType.blockSignals(true);

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

    graphType.clear();
    if (!firstHourly.isNull() || !lastHourly.isNull())
    {
        graphType.addItem("Distribution");

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
    variable.blockSignals(false);
    graphType.blockSignals(false);
    computePlot();

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
    computePlot();
}


void Crit3DPointStatisticsWidget::plot()
{
    if (currentFrequency == daily)
    {
        if (graphType.currentText() == "Trend")
        {
            classWidth.setEnabled(false);
            valMax.setEnabled(false);
            valMin.setEnabled(false);
            smoothing.setEnabled(false);

            availability.clear();
            significance.clear();
            average.clear();
            median.clear();
            mode.clear();
            sigma.clear();
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
            QDate firstDate(firstYear, dayFrom.date().month(), dayFrom.date().day());
            QDate lastDate(lastYear, dayTo.date().month(), dayTo.date().day());
            clima.setGenericPeriodDateStart(firstDate);
            clima.setGenericPeriodDateEnd(lastDate);
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

            FormInfo formInfo;
            formInfo.showInfo("compute annual series...");
            // copy data to MPTemp
            Crit3DMeteoPoint meteoPointTemp;
            meteoPointTemp.id = meteoPoints[0].id;
            meteoPointTemp.latitude = meteoPoints[0].latitude;
            meteoPointTemp.elaboration = meteoPoints[0].elaboration;
            QDate lastDateCopyed = meteoPointsDbHandler->getLastDate(daily, meteoPoints[0].id).date();

            meteoPointTemp.nrObsDataDaysD = 0;
            meteoPointTemp.nrObsDataDaysH = 0;
            bool dataAlreadyLoaded = false;
            int validYears = 0;
            std::vector<int> vectorYears;
            validYears = computeAnnualSeriesOnPointFromDaily(&myError, meteoPointsDbHandler, meteoGridDbHandler,
                                                                 &meteoPointTemp, &clima, isGrid, isAnomaly, meteoSettings, outputValues, vectorYears, dataAlreadyLoaded);
            bool yearsMissing = false;
            if (vectorYears[vectorYears.size()-1] - vectorYears[0] + 1 != vectorYears.size()-1)
            {
                yearsMissing = true;
            }
            int numberOfDays = meteoPoints[0].obsDataD[0].date.daysTo(getCrit3DDate(lastDate))+1;
            meteoPointTemp.initializeObsDataD(numberOfDays, meteoPoints[0].obsDataD[0].date);
            meteoPointTemp.initializeObsDataH(1, numberOfDays, meteoPoints[0].getMeteoPointHourlyValuesDate(0));
            meteoPointTemp.initializeObsDataDFromMp(meteoPoints[0].nrObsDataDaysD, meteoPoints[0].obsDataD[0].date, meteoPoints[0]);
            meteoPointTemp.initializeObsDataHFromMp(1,meteoPoints[0].nrObsDataDaysH, meteoPoints[0].getMeteoPointHourlyValuesDate(0), meteoPoints[0]);
            bool addedDataFromJoint = false;

            if (idPoints.size() != 1 && (lastDateCopyed < lastDate || yearsMissing))
            {
                if (yearsMissing)
                {
                    std::vector<int> vectorMissing;
                    int posMissing, n;
                    for (int y = 0; y<vectorYears.size()-1; y++)
                    {
                        posMissing = vectorYears[y+1]-vectorYears[y]-1;
                        n = 1;
                        while (posMissing != 0)
                        {
                            vectorMissing.push_back(vectorYears[y]+n);
                            posMissing = posMissing - 1;
                            n = n + 1;
                        }
                    }
                    int indexMp;
                    for (int i = 1; i<idPoints.size(); i++)
                    {
                        for (int j = 0; j<meteoPoints.size(); j++)
                        {
                            if (meteoPoints[j].id == idPoints[i])
                            {
                                indexMp = j;
                                break;
                            }
                        }
                        Crit3DMeteoPoint meteoPointTempJoint;
                        meteoPointTempJoint.id = meteoPoints[indexMp].id;
                        meteoPointTempJoint.latitude = meteoPoints[indexMp].latitude;
                        meteoPointTempJoint.elaboration = meteoPoints[indexMp].elaboration;
                        meteoPointTempJoint.nrObsDataDaysD = 0;
                        int validYearsJoint = 0;
                        std::vector<int> vectorYearsJoint;
                        std::vector<float> annualSeriesJoint;
                        clima.setYearStart(firstYear);
                        clima.setYearEnd(lastYear);
                        validYearsJoint = computeAnnualSeriesOnPointFromDaily(&myError, meteoPointsDbHandler, nullptr,
                                                                              &meteoPointTempJoint, &clima, false, isAnomaly, meteoSettings, annualSeriesJoint, vectorYearsJoint, false);
                        for (int k=0; k<vectorYearsJoint.size(); k++)
                        {
                            if(std::find(vectorMissing.begin(), vectorMissing.end(), vectorYearsJoint[k]) != vectorMissing.end())
                            {
                                // vectorMissing contains vectorYearsJoint[k]
                                QDate startYear(vectorYearsJoint[k],1,1);
                                QDate endYear(vectorYearsJoint[k],12,31);
                                // copy values
                                for (QDate myDate=startYear; myDate<=endYear; myDate=myDate.addDays(1))
                                {
                                    setMpValues(meteoPoints[indexMp], &meteoPointTemp, myDate);
                                }
                                addedDataFromJoint = true;
                                vectorMissing.erase(remove(vectorMissing.begin(), vectorMissing.end(), vectorYearsJoint[k]), vectorMissing.end());
                            }
                        }
                        if (vectorMissing.empty())
                        {
                            break;
                        }
                    }
                }
                if (lastDateCopyed < lastDate)
                {
                    for (int i = 1; i<idPoints.size(); i++)
                    {
                        QDate lastDateNew = meteoPointsDbHandler->getLastDate(daily, idPoints[i]).date();
                        if (lastDateNew > lastDateCopyed)
                        {
                            int indexMp;
                            for (int j = 0; j<meteoPoints.size(); j++)
                            {
                                if (meteoPoints[j].id == idPoints[i])
                                {
                                    indexMp = j;
                                    break;
                                }
                            }
                            for (QDate myDate=lastDateCopyed.addDays(1); myDate<=lastDateNew; myDate=myDate.addDays(1))
                            {
                                setMpValues(meteoPoints[indexMp], &meteoPointTemp, myDate);
                            }
                            addedDataFromJoint = true;
                        }
                        lastDateCopyed = lastDateNew;
                    }
                }
            }
            if(addedDataFromJoint)
            {
                dataAlreadyLoaded = true;
                vectorYears.clear();
                clima.setYearStart(firstYear);
                clima.setYearEnd(lastYear);
                validYears = computeAnnualSeriesOnPointFromDaily(&myError, meteoPointsDbHandler, meteoGridDbHandler,
                                                                 &meteoPointTemp, &clima, isGrid, isAnomaly, meteoSettings, outputValues, vectorYears, dataAlreadyLoaded);
            }

            formInfo.close();
            if (validYears < 3)
            {
                QMessageBox::information(nullptr, "Error", "Number of valid years < 3");
                return;
            }

            double sum = 0;
            int count = 0;
            int validData = 0;
            int yearsLength = lastYear - firstYear;
            int nYearsToAdd = 0;
            if (yearsLength > 20)
            {
                for (int inc = 0; inc<=3; inc++)
                {
                    if ( (yearsLength+inc) % 2 == 0 &&  (yearsLength+inc)/2 <= 20)
                    {
                        nYearsToAdd = inc;
                        break;
                    }
                    if ( (yearsLength+inc) % 3 == 0 &&  (yearsLength+inc)/3 <= 20)
                    {
                        nYearsToAdd = inc;
                        break;
                    }
                    if ( (yearsLength+inc) % 4 == 0 &&  (yearsLength+inc)/4 <= 20)
                    {
                        nYearsToAdd = inc;
                        break;
                    }
                }
                for (int i = nYearsToAdd; i> 0; i--)
                {
                    years.push_back(firstYear-i);
                    outputValues.insert(outputValues.begin(),NODATA);
                }

            }
            for (int i = firstYear; i<=lastYear; i++)
            {
                years.push_back(i);
                if (outputValues[count] != NODATA)
                {
                    sum += double(outputValues[unsigned(count)]);
                    validData++;
                }
                count = count + 1;
            }
            // draw
            chartView->drawTrend(years, outputValues);

            double availab = double(validData) / double(count) * 100.0;
            availability.setText(QString::number(availab, 'f', 3));
            double mkendall = statisticalElab(mannKendall, NODATA, outputValues, outputValues.size(), meteoSettings->getRainfallThreshold());
            significance.setText(QString::number(mkendall, 'f', 3));
            double averageValue = sum / double(validData);
            average.setText(QString::number(averageValue, 'f', 1));

            float myCoeff = NODATA;
            float myIntercept = NODATA;
            float myR2 = NODATA;
            bool isZeroIntercept = false;
            std::vector<float> yearsFloat(years.begin(), years.end());
            statistics::linearRegression(yearsFloat, outputValues, outputValues.size(), isZeroIntercept,
                                             &myIntercept, &myCoeff, &myR2);
            r2.setText(QString::number(double(myR2), 'f', 3));
            rate.setText(QString::number(double(myCoeff), 'f', 3));
        }
        else if (graphType.currentText() == "Anomaly trend")
        {
            classWidth.setEnabled(false);
            valMax.setEnabled(false);
            valMin.setEnabled(false);
            smoothing.setEnabled(false);

            availability.clear();
            significance.clear();
            average.clear();
            median.clear();
            mode.clear();
            sigma.clear();
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
            QDate firstDate(firstYear, dayFrom.date().month(), dayFrom.date().day());
            QDate lastDate(lastYear, dayTo.date().month(), dayTo.date().day());
            clima.setGenericPeriodDateStart(firstDate);
            clima.setGenericPeriodDateEnd(lastDate);
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
            bool dataAlreadyLoaded;

            // copy data to MPTemp
            Crit3DMeteoPoint meteoPointTemp;
            meteoPointTemp.id = meteoPoints[0].id;
            meteoPointTemp.latitude = meteoPoints[0].latitude;
            meteoPointTemp.elaboration = meteoPoints[0].elaboration;
            if (idPoints.size() == 1)
            {
                // meteoPointTemp should be init
                meteoPointTemp.nrObsDataDaysH = 0;
                meteoPointTemp.nrObsDataDaysD = 0;
                dataAlreadyLoaded = false;
            }
            else
            {
                int numberOfDays = meteoPoints[0].obsDataD[0].date.daysTo(getCrit3DDate(lastDate))+1;
                meteoPointTemp.initializeObsDataD(numberOfDays, meteoPoints[0].obsDataD[0].date);
                meteoPointTemp.initializeObsDataH(1, numberOfDays, meteoPoints[0].getMeteoPointHourlyValuesDate(0));
                meteoPointTemp.initializeObsDataDFromMp(meteoPoints[0].nrObsDataDaysD, meteoPoints[0].obsDataD[0].date, meteoPoints[0]);
                meteoPointTemp.initializeObsDataHFromMp(1,meteoPoints[0].nrObsDataDaysH, meteoPoints[0].getMeteoPointHourlyValuesDate(0), meteoPoints[0]);
                QDate lastDateCopyed = meteoPointsDbHandler->getLastDate(daily, meteoPoints[0].id).date();
                for (int i = 1; i<idPoints.size(); i++)
                {
                    QDate lastDateNew = meteoPointsDbHandler->getLastDate(daily, idPoints[i]).date();
                    if (lastDateNew > lastDateCopyed)
                    {
                        int indexMp;
                        for (int j = 0; j<meteoPoints.size(); j++)
                        {
                            if (meteoPoints[j].id == idPoints[i])
                            {
                                indexMp = j;
                                break;
                            }
                        }
                        for (QDate myDate=lastDateCopyed.addDays(1); myDate<=lastDateNew; myDate=myDate.addDays(1))
                        {
                            setMpValues(meteoPoints[indexMp], &meteoPointTemp, myDate);
                        }
                    }
                    lastDateCopyed = lastDateNew;
                }
                dataAlreadyLoaded = true;
            }


            if (isGrid)
            {
                if (!elaborationOnPoint(&myError, nullptr, meteoGridDbHandler, &meteoPointTemp, &clima, isGrid, firstDate, lastDate, isAnomaly, meteoSettings, dataAlreadyLoaded))
                {
                    QMessageBox::information(nullptr, "Error", "Data not available in the reference period");
                    return;
                }
            }
            else
            {
                if (!elaborationOnPoint(&myError, meteoPointsDbHandler, nullptr, &meteoPointTemp, &clima, isGrid, firstDate, lastDate, isAnomaly, meteoSettings, dataAlreadyLoaded))
                {
                    QMessageBox::information(nullptr, "Error", "Data not available in the reference period");
                    return;
                }
            }

            firstYear = analysisYearFrom.currentText().toInt();
            lastYear = analysisYearTo.currentText().toInt();
            clima.setYearStart(firstYear);
            clima.setYearEnd(lastYear);
            clima.setGenericPeriodDateStart(QDate(firstYear, dayFrom.date().month(), dayFrom.date().day()));
            clima.setGenericPeriodDateEnd(QDate(lastYear, dayTo.date().month(), dayTo.date().day()));
            float elabResult = meteoPointTemp.elaboration;

            FormInfo formInfo;
            formInfo.showInfo("compute annual series...");
            std::vector<int> vectorYears;
            int validYears = computeAnnualSeriesOnPointFromDaily(&myError, meteoPointsDbHandler, meteoGridDbHandler,
                                                     &meteoPointTemp, &clima, isGrid, isAnomaly, meteoSettings, outputValues, vectorYears, dataAlreadyLoaded);
            formInfo.close();
            if (validYears < 3)
            {
                QMessageBox::information(nullptr, "Error", "Number of valid years < 3");
                return;
            }

            float sum = 0;
            int count = 0;
            int validData = 0;

            int yearsLength = lastYear - firstYear;
            int nYearsToAdd = 0;
            if (yearsLength > 20)
            {
                for (int inc = 0; inc<=3; inc++)
                {
                    if ( (yearsLength+inc) % 2 == 0 &&  (yearsLength+inc)/2 <= 20)
                    {
                        nYearsToAdd = inc;
                        break;
                    }
                    if ( (yearsLength+inc) % 3 == 0 &&  (yearsLength+inc)/3 <= 20)
                    {
                        nYearsToAdd = inc;
                        break;
                    }
                    if ( (yearsLength+inc) % 4 == 0 &&  (yearsLength+inc)/4 <= 20)
                    {
                        nYearsToAdd = inc;
                        break;
                    }
                }
                for (int i = nYearsToAdd; i> 0; i--)
                {
                    years.push_back(firstYear-i);
                    outputValues.insert(outputValues.begin(),NODATA);
                }

            }
            for (int i = firstYear; i<=lastYear; i++)
            {
                years.push_back(i);
                if (outputValues[count] != NODATA)
                {
                    outputValues[count] = outputValues[count] - elabResult;
                    sum = sum + outputValues[count];
                    validData = validData + 1;
                }
                count = count + 1;
            }
            // draw
            chartView->drawTrend(years, outputValues);

            float availab = ((float)validData/(float)count)*100.0;
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
        else if (graphType.currentText() == "Climate")
        {
            classWidth.setEnabled(false);
            valMax.setEnabled(false);
            valMin.setEnabled(false);
            smoothing.setEnabled(true);

            availability.clear();
            significance.clear();
            average.clear();
            median.clear();
            mode.clear();
            sigma.clear();
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
            if (idPoints.size() == 1)
            {
                computeClimateOnDailyData(meteoPoints[0], myVar, startDate, endDate,
                                          smooth, &dataPresence, quality, climateParameters, meteoSettings, dailyClima, decadalClima, monthlyClima);
            }
            else
            {
                Crit3DMeteoPoint meteoPointTemp;
                meteoPointTemp.id = meteoPoints[0].id;
                meteoPointTemp.latitude = meteoPoints[0].latitude;
                meteoPointTemp.elaboration = meteoPoints[0].elaboration;
                int numberOfDays = meteoPoints[0].obsDataD[0].date.daysTo(getCrit3DDate(endDate))+1;
                meteoPointTemp.initializeObsDataD(numberOfDays, meteoPoints[0].obsDataD[0].date);
                meteoPointTemp.initializeObsDataH(1, numberOfDays, meteoPoints[0].getMeteoPointHourlyValuesDate(0));
                meteoPointTemp.initializeObsDataDFromMp(meteoPoints[0].nrObsDataDaysD, meteoPoints[0].obsDataD[0].date, meteoPoints[0]);
                meteoPointTemp.initializeObsDataHFromMp(1,meteoPoints[0].nrObsDataDaysH, meteoPoints[0].getMeteoPointHourlyValuesDate(0), meteoPoints[0]);
                QDate lastDateCopyed = meteoPointsDbHandler->getLastDate(daily, meteoPoints[0].id).date();
                for (int i = 1; i<idPoints.size(); i++)
                {
                    QDate lastDateNew = meteoPointsDbHandler->getLastDate(daily, idPoints[i]).date();
                    if (lastDateNew > lastDateCopyed)
                    {
                        int indexMp;
                        for (int j = 0; j<meteoPoints.size(); j++)
                        {
                            if (meteoPoints[j].id == idPoints[i])
                            {
                                indexMp = j;
                                break;
                            }
                        }
                        for (QDate myDate=lastDateCopyed.addDays(1); myDate<=lastDateNew; myDate=myDate.addDays(1))
                        {
                            setMpValues(meteoPoints[indexMp], &meteoPointTemp, myDate);
                        }
                    }
                    lastDateCopyed = lastDateNew;
                }
                computeClimateOnDailyData(meteoPointTemp, myVar, startDate, endDate,
                                          smooth, &dataPresence, quality, climateParameters, meteoSettings, dailyClima, decadalClima, monthlyClima);
            }
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
        else if (graphType.currentText() == "Distribution")
        {
            valMax.blockSignals(true);
            valMin.blockSignals(true);

            classWidth.setEnabled(true);
            valMax.setEnabled(true);
            valMin.setEnabled(true);
            smoothing.setEnabled(false);

            availability.clear();
            significance.clear();
            average.clear();
            r2.clear();
            median.clear();
            mode.clear();
            sigma.clear();
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
                        int indexMp = 0;
                        int i = firstDaily.daysTo(myDate);
                        if (!isGrid)
                        {
                            int nPoint;
                            for (nPoint = 0; nPoint<idPoints.size(); nPoint++)
                            {
                                if (myDate <= meteoPointsDbHandler->getLastDate(daily, idPoints[nPoint]).date())
                                {
                                    break;
                                }
                            }
                            QDate myFirstDaily = meteoPointsDbHandler->getFirstDate(daily, idPoints[nPoint]).date();
                            i = myFirstDaily.daysTo(myDate);
                            for (int j = 0; j<meteoPoints.size(); j++)
                            {
                                if (meteoPoints[j].id == idPoints[nPoint])
                                {
                                    indexMp = j;
                                    break;
                                }
                            }
                        }
                        float myDailyValue = meteoPoints[indexMp].getMeteoPointValueD(getCrit3DDate(myDate), myVar, meteoSettings);
                        if (i<0 || i>meteoPoints[indexMp].nrObsDataDaysD)
                        {
                            check = quality::missing_data;
                        }
                        else
                        {
                            check = quality->checkFastValueDaily_SingleValue(myVar, climateParameters, myDailyValue, myDate.month(), meteoPoints[indexMp].point.z);
                        }
                        if (check == quality::accepted)
                        {
                            if (myVar == dailyPrecipitation)
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

            valMaxValue = valMax.text().toInt(&ok);
            if (!ok || valMax.text().isEmpty() || valMaxValue == NODATA)
            {
                valMaxValue = maxValueInt;
                valMax.setText(QString::number(valMaxValue));
            }
            valMinValue = valMin.text().toInt(&ok);
            if (!ok || valMin.text().isEmpty() || valMinValue == NODATA)
            {
                valMinValue = minValueInt;
                valMin.setText(QString::number(valMinValue));
            }
            valMax.blockSignals(false);
            valMin.blockSignals(false);

            // init
            std::vector<float> bucket;
            for (int i = 0; i<= (valMaxValue - valMinValue)/classWidthValue; i++)
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
            double beta;
            double alpha;
            double pzero;

            int visualizedNrValues = 0;

            if (myVar == dailyPrecipitation)
            {
                for (int i = 0; i < nrValues; i++)
                {
                    if (series[i] > 0)
                    {
                        int index = (series[i] - valMinValue)/classWidthValue;
                        if( index >= 0 && index < bucket.size())
                        {
                            bucket[index] = bucket[index] + 1;
                            visualizedNrValues = visualizedNrValues + 1;
                        }
                    }
                }
                if (!generalizedGammaFitting(series, nrValues, &beta, &alpha,  &pzero))
                {
                    return;
                }
            }
            else
            {
                for (int i = 0; i < nrValues; i++)
                {
                    int index = (series[i] - valMinValue)/classWidthValue;
                    if( index >= 0 && index<bucket.size())
                    {
                        bucket[index] = bucket[index] + 1;
                        visualizedNrValues = visualizedNrValues + 1;
                    }
                }
                avg = statistics::mean(series, nrValues);
                dev_std = statistics::standardDeviation(series, nrValues);
                millile3dev = sorting::percentile(sortedSeries, nrValues, 99.73, true);
                millile_3Dev = sorting::percentile(sortedSeries, nrValues, 0.27, false);
            }

            availability.setText(QString::number((float)nrValues/(float)totDays * 100, 'f', 3));
            average.setText(QString::number(avg, 'f', 1));

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
                float myMode = minValueInt + (modeVal*classWidthValue) + (classWidthValue/2.0); // use minValueInt not the displayed minValue
                mode.setText(QString::number(myMode, 'f', 1));
            }
            if (dev_std != NODATA)
            {
                sigma.setText(QString::number(dev_std, 'f', 1));
            }
            median.setText(QString::number(sorting::percentile(sortedSeries, nrValues, 50, false), 'f', 1));

            QList<QPointF> lineValues;
            for (int i = 0; i<bucket.size(); i++)
            {
                float x = valMinValue + (i*classWidthValue) + (classWidthValue/2.0);
                if (x < float(valMaxValue))
                {
                    if (myVar == dailyPrecipitation)
                    {
                        if (x > 0)
                        {
                            if (alpha > 0 && beta > 0)
                            {
                                float probGamma = probabilityGamma(x,alpha,beta);
                                lineValues.append(QPointF(x, probGamma));
                            }
                            else
                            {
                                QMessageBox::information(nullptr, "Error", "Error in gamma distribution");
                                return;
                            }
                        }
                    }
                    else if (myVar != dailyAirRelHumidityMin && myVar != dailyAirRelHumidityMax && myVar != dailyAirRelHumidityAvg)
                    {
                        float gauss = gaussianFunction(x, avg, dev_std);
                        lineValues.append(QPointF(x,gauss));
                    }
                }
            }
            for (int i = 0; i<bucket.size(); i++)
            {
                bucket[i] = bucket[i]/visualizedNrValues;
            }
            chartView->drawDistribution(bucket, lineValues, valMinValue, valMaxValue, classWidthValue);
        }
    }
    else if (currentFrequency == hourly)
    {
        valMax.blockSignals(true);
        valMin.blockSignals(true);

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
        int myHour = hour.text().toInt();
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
                if (myDate >= firstHourly.date() && myDate <= lastHourly.date())
                {
                    int indexMp = 0;
                    int i = firstHourly.date().daysTo(myDate);
                    if (!isGrid)
                    {
                        int nPoint;
                        for (nPoint = 0; nPoint<idPoints.size(); nPoint++)
                        {
                            if (myDate <= meteoPointsDbHandler->getLastDate(hourly, idPoints[nPoint]).date())
                            {
                                break;
                            }
                        }
                        QDate myFirstHourly = meteoPointsDbHandler->getFirstDate(hourly, idPoints[nPoint]).date();
                        i = myFirstHourly.daysTo(myDate);
                        for (int j = 0; j<meteoPoints.size(); j++)
                        {
                            if (meteoPoints[j].id == idPoints[nPoint])
                            {
                                indexMp = j;
                                break;
                            }
                        }
                    }
                    float myHourlyValue = meteoPoints[indexMp].getMeteoPointValueH(getCrit3DDate(myDate), myHour, 0, myVar);
                    if (i<0 || i>meteoPoints[indexMp].nrObsDataDaysH)
                    {
                        check = quality::missing_data;
                    }
                    else
                    {
                        check = quality->checkFastValueHourly_SingleValue(myVar, climateParameters, myHourlyValue, myDate.month(), meteoPoints[indexMp].point.z);
                    }
                    if (check == quality::accepted)
                    {
                        if (myVar == precipitation)
                        {
                            if (myHourlyValue < meteoSettings->getRainfallThreshold())
                            {
                                myHourlyValue = 0;
                            }
                        }
                        series.push_back(myHourlyValue);
                        if (isFirstData)
                        {
                            myMinValue = myHourlyValue;
                            myMaxValue = myHourlyValue;
                            isFirstData = false;
                        }
                        else if (myHourlyValue < myMinValue)
                        {
                            myMinValue = myHourlyValue;
                        }
                        else if (myHourlyValue > myMaxValue)
                        {
                            myMaxValue = myHourlyValue;
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

        valMaxValue = valMax.text().toInt(&ok);
        if (!ok || valMax.text().isEmpty() || valMaxValue == NODATA)
        {
            valMaxValue = maxValueInt;
            valMax.setText(QString::number(valMaxValue));
        }
        valMinValue = valMin.text().toInt(&ok);
        if (!ok || valMin.text().isEmpty() || valMinValue == NODATA)
        {
            valMinValue = minValueInt;
            valMin.setText(QString::number(valMinValue));
        }
        valMax.blockSignals(false);
        valMin.blockSignals(false);

        // init
        std::vector<float> bucket;
        for (int i = 0; i<= (valMaxValue - valMinValue)/classWidthValue; i++)
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
        double beta;
        double alpha;
        double pzero;

        int visualizedNrValues = 0;
        if (myVar == precipitation)
        {
            for (int i = 0; i < nrValues; i++)
            {
                if (series[i] > 0)
                {
                    int index = (series[i] - valMinValue)/classWidthValue;
                    if( index >= 0 && index<bucket.size())
                    {
                        bucket[index] = bucket[index] + 1;
                        visualizedNrValues = visualizedNrValues + 1;
                    }
                }
            }
            if (!generalizedGammaFitting(series, nrValues, &beta, &alpha,  &pzero))
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
                    int index = (series[i] - valMinValue)/classWidthValue;
                    if( index >= 0 && index<bucket.size())
                    {
                        bucket[index] = bucket[index] + 1;
                        visualizedNrValues = visualizedNrValues + 1;
                    }
                }
            }
            avg = statistics::mean(series, nrValues);
            dev_std = statistics::standardDeviation(series, nrValues);
            millile3dev = sorting::percentile(sortedSeries, nrValues, 99.73, true);
            millile_3Dev = sorting::percentile(sortedSeries, nrValues, 0.27, false);
        }
        availability.setText(QString::number((float)nrValues/(float)totDays * 100, 'f', 3));
        average.setText(QString::number(avg, 'f', 1));

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
            float myMode = minValueInt + (modeVal*classWidthValue) + (classWidthValue/2.0); // use minValueInt not the displayed minValue
            mode.setText(QString::number(myMode, 'f', 1));
        }
        if (dev_std != NODATA)
        {
            sigma.setText(QString::number(dev_std, 'f', 1));
        }
        median.setText(QString::number(sorting::percentile(sortedSeries, nrValues, 50, false), 'f', 1));

        QList<QPointF> lineValues;
        for (int i = 0; i<bucket.size(); i++)
        {
            float x = valMinValue + (i*classWidthValue) + (classWidthValue/2.0);
            if (x < float(valMaxValue))
            {
                if (myVar == precipitation)
                {
                    if (x > 0)
                    {
                        if (alpha > 0 && beta > 0)
                        {
                            float probGamma = probabilityGamma(x,alpha,beta);
                            lineValues.append(QPointF(x,probGamma));
                        }
                        else
                        {
                            QMessageBox::information(nullptr, "Error", "Error in gamma distribution");
                            return;
                        }
                    }
                }
                else if (myVar != airRelHumidity && myVar != windVectorDirection)
                {
                    float gauss = gaussianFunction(x, avg, dev_std);
                    lineValues.append(QPointF(x,gauss));
                }
            }
        }
        for (int i = 0; i<bucket.size(); i++)
        {
            bucket[i] = bucket[i]/visualizedNrValues;
        }
        chartView->drawDistribution(bucket, lineValues, valMinValue, valMaxValue, classWidthValue);
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
        meteoPointTemp.latitude = meteoPoints[0].latitude;
        meteoPointTemp.elaboration = meteoPoints[0].elaboration;
        bool dataAlreadyLoaded;
        if (idPoints.size() == 1)
        {
            // meteoPointTemp should be init
            meteoPointTemp.nrObsDataDaysH = 0;
            meteoPointTemp.nrObsDataDaysD = 0;
            dataAlreadyLoaded = false;
        }
        else
        {
            QDate endDate(clima.yearEnd(), clima.genericPeriodDateEnd().month(), clima.genericPeriodDateEnd().day());
            int numberOfDays = meteoPoints[0].obsDataD[0].date.daysTo(getCrit3DDate(endDate))+1;
            meteoPointTemp.initializeObsDataD(numberOfDays, meteoPoints[0].obsDataD[0].date);
            meteoPointTemp.initializeObsDataH(1, numberOfDays, meteoPoints[0].getMeteoPointHourlyValuesDate(0));
            meteoPointTemp.initializeObsDataDFromMp(meteoPoints[0].nrObsDataDaysD, meteoPoints[0].obsDataD[0].date, meteoPoints[0]);
            meteoPointTemp.initializeObsDataHFromMp(1,meteoPoints[0].nrObsDataDaysH, meteoPoints[0].getMeteoPointHourlyValuesDate(0), meteoPoints[0]);
            QDate lastDateCopyed = meteoPointsDbHandler->getLastDate(daily, meteoPoints[0].id).date();
            for (int i = 1; i<idPoints.size(); i++)
            {
                QDate lastDateNew = meteoPointsDbHandler->getLastDate(daily, idPoints[i]).date();
                if (lastDateNew > lastDateCopyed)
                {
                    int indexMp;
                    for (int j = 0; j<meteoPoints.size(); j++)
                    {
                        if (meteoPoints[j].id == idPoints[i])
                        {
                            indexMp = j;
                            break;
                        }
                    }
                    for (QDate myDate=lastDateCopyed.addDays(1); myDate<=lastDateNew; myDate=myDate.addDays(1))
                    {
                        setMpValues(meteoPoints[indexMp], &meteoPointTemp, myDate);
                    }
                }
                lastDateCopyed = lastDateNew;
            }
            dataAlreadyLoaded = true;
        }
        std::vector<int> vectorYears;
        int validYears = computeAnnualSeriesOnPointFromDaily(&myError, meteoPointsDbHandler, meteoGridDbHandler,
                                                 &meteoPointTemp, &clima, isGrid, isAnomaly, meteoSettings, outputValues, vectorYears, dataAlreadyLoaded);
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
        int yearsLength = lastYear - firstYear;
        int nYearsToAdd = 0;
        if (yearsLength > 20)
        {
            for (int inc = 0; inc<=3; inc++)
            {
                if ( (yearsLength+inc) % 2 == 0 &&  (yearsLength+inc)/2 <= 20)
                {
                    nYearsToAdd = inc;
                    break;
                }
                if ( (yearsLength+inc) % 3 == 0 &&  (yearsLength+inc)/3 <= 20)
                {
                    nYearsToAdd = inc;
                    break;
                }
                if ( (yearsLength+inc) % 4 == 0 &&  (yearsLength+inc)/4 <= 20)
                {
                    nYearsToAdd = inc;
                    break;
                }
            }
            for (int i = nYearsToAdd; i> 0; i--)
            {
                years.push_back(firstYear-i);
                outputValues.insert(outputValues.begin(),NODATA);
            }

        }
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

void Crit3DPointStatisticsWidget::updatePlotByVal()
{
    // check valMax and valMin
    if (valMin.text().toInt() == valMinValue && valMax.text().toInt() == valMaxValue)
    {
        return; //nothing changed
    }
    if (valMin.text().toInt() >= valMax.text().toInt())
    {
        valMax.blockSignals(true);
        valMin.blockSignals(true);

        valMin.setText(QString::number(valMinValue));
        valMax.setText(QString::number(valMaxValue));

        valMax.blockSignals(false);
        valMin.blockSignals(false);
        QMessageBox::information(nullptr, "Error", "Min value >= Max vaue");
        return;
    }
    plot();
}

void Crit3DPointStatisticsWidget::updatePlot()
{
    // does not compute valMax and valMin
    plot();
}

void Crit3DPointStatisticsWidget::computePlot()
{
    // compute valMax and valMin
    valMax.clear();
    valMin.clear();
    plot();
}

void Crit3DPointStatisticsWidget::on_actionChangeLeftAxis()
{
    DialogChangeAxis changeAxisDialog(true);
    if (changeAxisDialog.result() == QDialog::Accepted)
    {
        chartView->setYmax(changeAxisDialog.getMaxVal());
        chartView->setYmin(changeAxisDialog.getMinVal());
    }
}


void Crit3DPointStatisticsWidget::on_actionExportGraph()
{

    QString fileName = QFileDialog::getSaveFileName(this, tr("Save current graph"), "", tr("png files (*.png)"));

    if (fileName != "")
    {
        const auto dpr = chartView->devicePixelRatioF();
        QPixmap buffer(chartView->width() * dpr, chartView->height() * dpr);
        buffer.setDevicePixelRatio(dpr);
        buffer.fill(Qt::transparent);

        QPainter *paint = new QPainter(&buffer);
        paint->setPen(*(new QColor(255,34,255,255)));
        chartView->render(paint);

        QFile file(fileName);
        file.open(QIODevice::WriteOnly);
        buffer.save(&file, "PNG");
    }
}

void Crit3DPointStatisticsWidget::on_actionExportData()
{
    QString csvFileName = QFileDialog::getSaveFileName(this, tr("Save current data"), "", tr("csv files (*.csv)"));

    if (csvFileName != "")
    {
        QFile myFile(csvFileName);
        if (!myFile.open(QIODevice::WriteOnly | QFile::Truncate))
        {
            QMessageBox::information(nullptr, "Error", "Open CSV failed: " + csvFileName + "\n ");
            return;
        }

        QTextStream myStream (&myFile);
        myStream.setRealNumberNotation(QTextStream::FixedNotation);
        myStream.setRealNumberPrecision(5);
        if (graphType.currentText() == "Trend" || graphType.currentText() == "Anomaly trend")
        {
            QString header = "x,y";
            myStream << header << "\n";
            QList<QPointF> dataPoins = chartView->exportTrend();
            for (int i = 0; i < dataPoins.size(); i++)
            {
                myStream << dataPoins[i].toPoint().x() << "," << dataPoins[i].y() << "\n";
            }
        }
        else if (graphType.currentText() == "Climate")
        {
            myStream << "Daily" << "\n";
            QString header = "x,y";
            myStream << header << "\n";
            QList<QPointF> dataPoins = chartView->exportClimaDaily();
            for (int i = 0; i < dataPoins.size(); i++)
            {
                myStream << dataPoins[i].x() << "," << dataPoins[i].y() << "\n";
            }
            dataPoins.clear();
            myStream << "Decadal" << "\n";
            header = "x,y";
            myStream << header << "\n";
            dataPoins = chartView->exportClimaDecadal();
            for (int i = 0; i < dataPoins.size(); i++)
            {
                myStream << dataPoins[i].x() << "," << dataPoins[i].y() << "\n";
            }
            dataPoins.clear();
            myStream << "Monthly" << "\n";
            header = "x,y";
            myStream << header << "\n";
            dataPoins = chartView->exportClimaMonthly();
            for (int i = 0; i < dataPoins.size(); i++)
            {
                myStream << dataPoins[i].x() << "," << dataPoins[i].y() << "\n";
            }
        }
        else if (graphType.currentText() == "Distribution")
        {
            QString header = "x1,x2,frequency";
            myStream << header << "\n";
            QList< QList<float> > bar = chartView->exportDistribution();
            for (int i = 0; i < bar.size(); i++)
            {
                int x1 = bar[i].at(0);
                int x2 = bar[i].at(1);
                myStream << x1 << "," << x2 << "," << bar[i].at(2) << "\n";
            }
        }

        myFile.close();

        return;
    }
}

void Crit3DPointStatisticsWidget::addStationClicked()
{
    if (jointStationsList.currentText().isEmpty())
    {
        return;
    }
    std::string newId;
    if (jointStationsSelected.findItems(jointStationsList.currentText(), Qt::MatchExactly).isEmpty())
    {
        jointStationsSelected.addItem(jointStationsList.currentText());
        deleteStation.setEnabled(true);
        saveToDb.setEnabled(true);
        newId = jointStationsList.currentText().section(" ",0,0).toStdString();
        idPoints << newId;

        updateYears();
        int indexMp;
        for (int j = 0; j<meteoPoints.size(); j++)
        {
            if (meteoPoints[j].id == newId)
            {
                indexMp = j;
                break;
            }
        }
        // load all Data
        QDate firstDaily = meteoPointsDbHandler->getFirstDate(daily, newId).date();
        QDate lastDaily = meteoPointsDbHandler->getLastDate(daily, newId).date();

        QDateTime firstHourly = meteoPointsDbHandler->getFirstDate(hourly, newId);
        QDateTime lastHourly = meteoPointsDbHandler->getLastDate(hourly, newId);
        meteoPointsDbHandler->loadDailyData(getCrit3DDate(firstDaily), getCrit3DDate(lastDaily), &meteoPoints[indexMp]);
        meteoPointsDbHandler->loadHourlyData(getCrit3DDate(firstHourly.date()), getCrit3DDate(lastHourly.date()), &meteoPoints[indexMp]);
    }

}

void Crit3DPointStatisticsWidget::deleteStationClicked()
{
    QList<QListWidgetItem*> items = jointStationsSelected.selectedItems();
    foreach(QListWidgetItem * item, items)
    {
        idPoints.removeOne(item->text().section(" ",0,0).toStdString());
        delete jointStationsSelected.takeItem(jointStationsSelected.row(item));
    }
    updateYears();
}

void Crit3DPointStatisticsWidget::saveToDbClicked()
{
    QList<QString> stationsList;
    for (int row = 0; row < jointStationsSelected.count(); row++)
    {
        QString textSelected = jointStationsSelected.item(row)->text();
        stationsList.append(textSelected.section(" ",0,0));
    }
    if (!meteoPointsDbHandler->setJointStations(QString::fromStdString(meteoPoints[0].id), stationsList))
    {
        QMessageBox::critical(nullptr, "Error", meteoPointsDbHandler->getErrorString());
    }
}

void Crit3DPointStatisticsWidget::updateYears()
{

    lastDaily = meteoPointsDbHandler->getLastDate(daily, meteoPoints[0].id).date();
    lastHourly = meteoPointsDbHandler->getLastDate(hourly, meteoPoints[0].id);

    for (int i = 1; i<idPoints.size(); i++)
    {

        QDate lastDailyJointStation = meteoPointsDbHandler->getLastDate(daily, idPoints[i]).date();
        if (lastDailyJointStation.isValid() && lastDailyJointStation > lastDaily )
        {
            lastDaily = lastDailyJointStation;
        }

        QDateTime lastHourlyJointStation = meteoPointsDbHandler->getLastDate(hourly, idPoints[i]);
        if (lastHourlyJointStation.isValid() && lastHourlyJointStation > lastHourly )
        {
            lastHourly = lastHourlyJointStation;
        }
    }
    // save current yearFrom
    QString currentYearFrom = yearFrom.currentText();
    QString currentAnalysisYearFrom = analysisYearFrom.currentText();
    yearFrom.clear();
    yearTo.clear();
    if (currentFrequency == daily)
    {
        analysisYearFrom.clear();
        analysisYearTo.clear();
        for(int i = 0; i <= lastDaily.year()-firstDaily.year(); i++)
        {
            yearFrom.addItem(QString::number(firstDaily.year()+i));
            yearTo.addItem(QString::number(firstDaily.year()+i));
            analysisYearFrom.addItem(QString::number(firstDaily.year()+i));
            analysisYearTo.addItem(QString::number(firstDaily.year()+i));
        }
        yearTo.setCurrentText(QString::number(lastDaily.year()));
        analysisYearTo.setCurrentText(QString::number(lastDaily.year()));
        yearFrom.setCurrentText(currentYearFrom);
        analysisYearTo.setCurrentText(currentAnalysisYearFrom);
    }
    else if (currentFrequency == hourly)
    {
        for(int i = 0; i <= lastHourly.date().year() - firstHourly.date().year(); i++)
        {
            yearFrom.addItem(QString::number(firstHourly.date().year()+i));
            yearTo.addItem(QString::number(firstHourly.date().year()+i));
        }
        yearFrom.setCurrentText(currentYearFrom);
        yearTo.setCurrentText(QString::number(lastHourly.date().year()));
    }
}

void Crit3DPointStatisticsWidget::setMpValues(Crit3DMeteoPoint meteoPointGet, Crit3DMeteoPoint* meteoPointSet, QDate myDate)
{

    bool automaticETP = meteoSettings->getAutomaticET0HS();
    bool automaticTmed = meteoSettings->getAutomaticTavg();

    switch(myVar)
    {

        case dailyLeafWetness:
        {
            QDateTime myDateTime(myDate,QTime(1,0,0));
            QDateTime endDateTime(myDate.addDays(1),QTime(0,0,0));
            while(myDateTime<=endDateTime)
            {
                float value = meteoPointGet.getMeteoPointValueH(getCrit3DDate(myDateTime.date()), myDateTime.time().hour(), 0, leafWetness);
                meteoPointSet->setMeteoPointValueH(getCrit3DDate(myDateTime.date()), myDateTime.time().hour(), 0, leafWetness, value);
                myDateTime = myDateTime.addSecs(3600);
            }
            break;
        }

        case dailyThomDaytime:
        {
            float value = meteoPointGet.getMeteoPointValueD(getCrit3DDate(myDate), dailyAirRelHumidityMin, meteoSettings);
            meteoPointSet->setMeteoPointValueD(getCrit3DDate(myDate), dailyAirRelHumidityMin, value);
            value = meteoPointGet.getMeteoPointValueD(getCrit3DDate(myDate), dailyAirTemperatureMax, meteoSettings);
            meteoPointSet->setMeteoPointValueD(getCrit3DDate(myDate), dailyAirTemperatureMax, value);
            break;
        }

        case dailyThomNighttime:
        {
            float value = meteoPointGet.getMeteoPointValueD(getCrit3DDate(myDate), dailyAirRelHumidityMax, meteoSettings);
            meteoPointSet->setMeteoPointValueD(getCrit3DDate(myDate), dailyAirRelHumidityMax, value);
            value = meteoPointGet.getMeteoPointValueD(getCrit3DDate(myDate), dailyAirTemperatureMin, meteoSettings);
            meteoPointSet->setMeteoPointValueD(getCrit3DDate(myDate), dailyAirTemperatureMin, value);
            break;
        }
        case dailyThomAvg: case dailyThomMax: case dailyThomHoursAbove:
        {
            QDateTime myDateTime(myDate,QTime(1,0,0));
            QDateTime endDateTime(myDate.addDays(1),QTime(0,0,0));
            while(myDateTime<=endDateTime)
            {
                float value = meteoPointGet.getMeteoPointValueH(getCrit3DDate(myDateTime.date()), myDateTime.time().hour(), 0, airTemperature);
                meteoPointSet->setMeteoPointValueH(getCrit3DDate(myDateTime.date()), myDateTime.time().hour(), 0, airTemperature, value);
                value = meteoPointGet.getMeteoPointValueH(getCrit3DDate(myDateTime.date()), myDateTime.time().hour(), 0, airRelHumidity);
                meteoPointSet->setMeteoPointValueH(getCrit3DDate(myDateTime.date()), myDateTime.time().hour(), 0, airRelHumidity, value);
                myDateTime = myDateTime.addSecs(3600);
            }
            break;
        }
        case dailyBIC:
        {
            float value = meteoPointGet.getMeteoPointValueD(getCrit3DDate(myDate), dailyReferenceEvapotranspirationHS, meteoSettings);
            meteoPointSet->setMeteoPointValueD(getCrit3DDate(myDate), dailyReferenceEvapotranspirationHS, value);
            if (automaticETP)
            {
                float value = meteoPointGet.getMeteoPointValueD(getCrit3DDate(myDate), dailyAirTemperatureMin, meteoSettings);
                meteoPointSet->setMeteoPointValueD(getCrit3DDate(myDate), dailyAirTemperatureMin, value);
                value = meteoPointGet.getMeteoPointValueD(getCrit3DDate(myDate), dailyAirTemperatureMax, meteoSettings);
                meteoPointSet->setMeteoPointValueD(getCrit3DDate(myDate), dailyAirTemperatureMax, value);
            }
            value = meteoPointGet.getMeteoPointValueD(getCrit3DDate(myDate), dailyPrecipitation, meteoSettings);
            meteoPointSet->setMeteoPointValueD(getCrit3DDate(myDate), dailyPrecipitation, value);
            break;
        }

        case dailyAirTemperatureRange:
        {
            float value = meteoPointGet.getMeteoPointValueD(getCrit3DDate(myDate), dailyAirTemperatureMin, meteoSettings);
            meteoPointSet->setMeteoPointValueD(getCrit3DDate(myDate), dailyAirTemperatureMin, value);
            value = meteoPointGet.getMeteoPointValueD(getCrit3DDate(myDate), dailyAirTemperatureMax, meteoSettings);
            meteoPointSet->setMeteoPointValueD(getCrit3DDate(myDate), dailyAirTemperatureMax, value);
            break;
        }

        case dailyAirTemperatureAvg:
        {
            if (automaticTmed)
            {
                float value = meteoPointGet.getMeteoPointValueD(getCrit3DDate(myDate), dailyAirTemperatureMin, meteoSettings);
                meteoPointSet->setMeteoPointValueD(getCrit3DDate(myDate), dailyAirTemperatureMin, value);
                value = meteoPointGet.getMeteoPointValueD(getCrit3DDate(myDate), dailyAirTemperatureMax, meteoSettings);
                meteoPointSet->setMeteoPointValueD(getCrit3DDate(myDate), dailyAirTemperatureMax, value);
            }
            break;
        }

        case dailyReferenceEvapotranspirationHS:
        {
            if (automaticETP)
            {
                float value = meteoPointGet.getMeteoPointValueD(getCrit3DDate(myDate), dailyAirTemperatureMin, meteoSettings);
                meteoPointSet->setMeteoPointValueD(getCrit3DDate(myDate), dailyAirTemperatureMin, value);
                value = meteoPointGet.getMeteoPointValueD(getCrit3DDate(myDate), dailyAirTemperatureMax, meteoSettings);
                meteoPointSet->setMeteoPointValueD(getCrit3DDate(myDate), dailyAirTemperatureMax, value);
            }
            break;
        }
        case dailyHeatingDegreeDays: case dailyCoolingDegreeDays:
        {
            float value = meteoPointGet.getMeteoPointValueD(getCrit3DDate(myDate), dailyAirTemperatureAvg, meteoSettings);
            meteoPointSet->setMeteoPointValueD(getCrit3DDate(myDate), dailyAirTemperatureAvg, value);
            value = meteoPointGet.getMeteoPointValueD(getCrit3DDate(myDate), dailyAirTemperatureMin, meteoSettings);
            meteoPointSet->setMeteoPointValueD(getCrit3DDate(myDate), dailyAirTemperatureMin, value);
            value = meteoPointGet.getMeteoPointValueD(getCrit3DDate(myDate), dailyAirTemperatureMax, meteoSettings);
            meteoPointSet->setMeteoPointValueD(getCrit3DDate(myDate), dailyAirTemperatureMax, value);
            break;
        }

        default:
        {
            float value = meteoPointGet.getMeteoPointValueD(getCrit3DDate(myDate), myVar, meteoSettings);
            meteoPointSet->setMeteoPointValueD(getCrit3DDate(myDate), myVar, value);
            break;
        }
    }

}


