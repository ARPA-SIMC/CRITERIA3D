/*!
    \copyright 2020 Fausto Tomei, Gabriele Antolini,
    Alberto Pistocchi, Marco Bittelli, Antonio Volta, Laura Costantini

    This file is part of AGROLIB.
    AGROLIB has been developed under contract issued by ARPAE Emilia-Romagna

    AGROLIB is free software: you can redistribute it and/or modify
    it under the terms of the GNU Lesser General Public License as published by
    the Free Software Foundation, either version 3 of the License, or
    (at your option) any later version.

    AGROLIB is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
    GNU Lesser General Public License for more details.

    You should have received a copy of the GNU Lesser General Public License
    along with AGROLIB.  If not, see <http://www.gnu.org/licenses/>.

    contacts:
    ftomei@arpae.it
    gantolini@arpae.it
*/

#include "meteo.h"
#include "homogeneityWidget.h"
#include "utilities.h"
#include "interpolation.h"
#include "spatialControl.h"
#include "commonConstants.h"
#include "basicMath.h"
#include "climate.h"
#include "dialogChangeAxis.h"
#include "gammaFunction.h"
#include "furtherMathFunctions.h"
#include "formInfo.h"

#include <QLayout>
#include <QDate>

Crit3DHomogeneityWidget::Crit3DHomogeneityWidget(Crit3DMeteoPointsDbHandler* meteoPointsDbHandler, QList<Crit3DMeteoPoint> meteoPoints,
                                                         QDate firstDaily, QDate lastDaily, Crit3DMeteoSettings *meteoSettings, QSettings *settings, Crit3DClimateParameters *climateParameters, Crit3DQuality *quality)
:meteoPointsDbHandler(meteoPointsDbHandler), meteoPoints(meteoPoints), firstDaily(firstDaily),
  lastDaily(lastDaily), meteoSettings(meteoSettings), settings(settings), climateParameters(climateParameters), quality(quality)
{
    this->setWindowTitle("Homogeneity Test Id:"+QString::fromStdString(meteoPoints[0].id)+" "+QString::fromStdString(meteoPoints[0].name));
    this->resize(1000, 700);
    this->setSizePolicy(QSizePolicy::Expanding, QSizePolicy::Expanding);
    this->setAttribute(Qt::WA_DeleteOnClose);

    idPoints << meteoPoints[0].id;

    // layout
    QHBoxLayout *mainLayout = new QHBoxLayout();
    QVBoxLayout *leftLayout = new QVBoxLayout();
    QVBoxLayout *plotLayout = new QVBoxLayout();

    QHBoxLayout *firstLayout = new QHBoxLayout();
    QVBoxLayout *methodLayout = new QVBoxLayout;
    QGroupBox *methodGroupBox = new QGroupBox();
    QVBoxLayout *variableLayout = new QVBoxLayout;
    QGroupBox *variableGroupBox = new QGroupBox();

    QGroupBox *jointStationsGroupBox = new QGroupBox();
    QHBoxLayout *jointStationsLayout = new QHBoxLayout;
    QVBoxLayout *jointStationsSelectLayout = new QVBoxLayout;

    QHBoxLayout *secondLayout = new QHBoxLayout();
    QGroupBox *parametersGroupBox = new QGroupBox();
    parametersGroupBox->setTitle("Parameter");
    QGridLayout *gridParamtersLayout = new QGridLayout;

    QHBoxLayout *findStationsLayout = new QHBoxLayout();
    QVBoxLayout *selectStationsLayout = new QVBoxLayout();
    QHBoxLayout *headerLayout = new QHBoxLayout;
    QHBoxLayout *stationsLayout = new QHBoxLayout;
    QVBoxLayout *arrowLayout = new QVBoxLayout();

    QHBoxLayout *resultLayout = new QHBoxLayout();
    QGroupBox *resultGroupBox = new QGroupBox();
    resultGroupBox->setTitle("Homogeneity results");


    QLabel *methodLabel = new QLabel(tr("Method: "));
    method.setMaximumWidth(120);
    method.addItem("SNHT");
    method.addItem("CRADDOCK");
    method.setSizeAdjustPolicy(QComboBox::AdjustToContents);
    methodLayout->addWidget(methodLabel);
    methodLayout->addWidget(&method);
    methodGroupBox->setLayout(methodLayout);

    QLabel *variableLabel = new QLabel(tr("Variable: "));
    std::map<meteoVariable, std::string>::const_iterator it;
    for(it = MapDailyMeteoVarToString.begin(); it != MapDailyMeteoVarToString.end(); ++it)
    {
        variable.addItem(QString::fromStdString(it->second));
    }
    myVar = getKeyMeteoVarMeteoMap(MapDailyMeteoVarToString, variable.currentText().toStdString());
    variable.setSizeAdjustPolicy(QComboBox::AdjustToContents);
    variable.setMaximumWidth(150);
    variableLayout->addWidget(variableLabel);
    variableLayout->addWidget(&variable);
    variableGroupBox->setLayout(variableLayout);

    QLabel *minNumStationsLabel = new QLabel(tr("Minimum number of stations: "));
    gridParamtersLayout->addWidget(minNumStationsLabel,0,0,1,1);
    minNumStations.setMaximumWidth(50);
    minNumStations.setMaximumHeight(24);
    minNumStations.setText("1");
    minNumStations.setValidator(new QIntValidator(1.0, 20.0));
    gridParamtersLayout->addWidget(&minNumStations,3,0,1,-1);

    parametersGroupBox->setMaximumHeight(this->height()/8);
    parametersGroupBox->setLayout(gridParamtersLayout);

    QLabel *jointStationsLabel = new QLabel(tr("Stations:"));
    jointStationsSelectLayout->addWidget(jointStationsLabel);
    jointStationsSelectLayout->addWidget(&jointStationsList);
    for (int i = 1; i<meteoPoints.size(); i++)
    {
        jointStationsList.addItem(QString::fromStdString(meteoPoints[i].id)+" "+QString::fromStdString(meteoPoints[i].name));
    }
    if (jointStationsList.count() != 0)
    {
        addJointStation.setEnabled(true);
    }
    else
    {
        addJointStation.setEnabled(false);
    }
    QHBoxLayout *addDeleteStationLayout = new QHBoxLayout;
    addDeleteStationLayout->addWidget(&addJointStation);
    addJointStation.setText("Add");
    addJointStation.setMaximumWidth(120);
    deleteJointStation.setText("Delete");
    deleteJointStation.setMaximumWidth(120);
    saveToDb.setText("Save to DB");
    saveToDb.setMaximumWidth(120);
    deleteJointStation.setEnabled(false);
    saveToDb.setEnabled(false);
    addDeleteStationLayout->addWidget(&deleteJointStation);
    jointStationsSelectLayout->addLayout(addDeleteStationLayout);
    jointStationsSelectLayout->addWidget(&saveToDb);
    jointStationsLayout->addLayout(jointStationsSelectLayout);
    jointStationsLayout->addWidget(&jointStationsSelected);
    jointStationsGroupBox->setTitle("Joint stations");
    jointStationsGroupBox->setLayout(jointStationsLayout);

    QLabel *yearFromLabel = new QLabel(tr("From"));
    findStationsLayout->addWidget(yearFromLabel);
    yearFrom.setMaximumWidth(100);
    findStationsLayout->addWidget(&yearFrom);
    findStationsLayout->addStretch(120);
    QLabel *yearToLabel = new QLabel(tr("To"));
    findStationsLayout->addWidget(yearToLabel);
    yearTo.setMaximumWidth(100);
    findStationsLayout->addWidget(&yearTo);
    findStationsLayout->addStretch(500);
    find.setText("Find stations");
    find.setMaximumWidth(120);
    findStationsLayout->addWidget(&find);
    for(int i = 0; i <= lastDaily.year()-firstDaily.year(); i++)
    {
        yearFrom.addItem(QString::number(firstDaily.year()+i));
        yearTo.addItem(QString::number(firstDaily.year()+i));
    }
    yearTo.setCurrentText(QString::number(lastDaily.year()));

    QLabel *allHeader = new QLabel("Stations found");
    QLabel *selectedHeader = new QLabel("Stations selected");
    addButton.setText("➡");
    deleteButton.setText("⬅");
    addButton.setEnabled(false);
    deleteButton.setEnabled(false);
    arrowLayout->addWidget(&addButton);
    arrowLayout->addWidget(&deleteButton);
    stationsLayout->addWidget(&listFoundStations);
    stationsLayout->addLayout(arrowLayout);
    stationsLayout->addWidget(&listSelectedStations);

    headerLayout->addWidget(allHeader);
    headerLayout->addSpacing(listFoundStations.width());
    headerLayout->addWidget(selectedHeader);
    selectStationsLayout->addLayout(headerLayout);
    selectStationsLayout->addLayout(stationsLayout);

    stationsTable.setColumnCount(4);
    QList<QString> tableHeader;
    tableHeader <<"Name"<<"R^2"<<"getDistance [km]"<<"Delta Z [m]";
    stationsTable.setHorizontalHeaderLabels(tableHeader);
    stationsTable.adjustSize();
    selectStationsLayout->addWidget(&stationsTable);

    execute.setText("Execute");
    execute.setMaximumWidth(100);
    if (listSelectedStations.count() == 0)
    {
        execute.setEnabled(false);
    }

    resultLayout->addWidget(&execute);
    resultLayout->addWidget(&resultLabel);
    resultGroupBox->setLayout(resultLayout);

    annualSeriesChartView = new AnnualSeriesChartView();
    annualSeriesChartView->setMinimumWidth(this->width()*2/3);
    annualSeriesChartView->setYTitle(QString::fromStdString(getUnitFromVariable(myVar)));
    plotLayout->addWidget(annualSeriesChartView);

    homogeneityChartView = new HomogeneityChartView();
    homogeneityChartView->setMinimumWidth(this->width()*2/3);
    plotLayout->addWidget(homogeneityChartView);

    firstLayout->addWidget(methodGroupBox);
    firstLayout->addWidget(variableGroupBox);

    secondLayout->addWidget(parametersGroupBox);

    leftLayout->addLayout(firstLayout);
    leftLayout->addWidget(jointStationsGroupBox);
    leftLayout->addLayout(secondLayout);
    leftLayout->addLayout(findStationsLayout);
    leftLayout->addLayout(selectStationsLayout);
    leftLayout->addWidget(resultGroupBox);

    // menu
    QMenuBar* menuBar = new QMenuBar();
    QMenu *editMenu = new QMenu("Edit");

    menuBar->addMenu(editMenu);
    mainLayout->setMenuBar(menuBar);

    QAction* changeHomogeneityLeftAxis = new QAction(tr("&Change homogenity chart axis left"), this);
    QAction* exportHomogeneityGraph = new QAction(tr("&Export homogenity graph"), this);
    QAction* exportHomogeneityData = new QAction(tr("&Export homogenity data"), this);

    editMenu->addAction(changeHomogeneityLeftAxis);
    editMenu->addAction(exportHomogeneityGraph);
    editMenu->addAction(exportHomogeneityData);

    mainLayout->addLayout(leftLayout);
    mainLayout->addLayout(plotLayout);
    setLayout(mainLayout);

    connect(&variable, &QComboBox::currentTextChanged, [=](const QString &newVariable){ this->changeVar(newVariable); });
    connect(&yearFrom, &QComboBox::currentTextChanged, [=](){ this->changeYears(); });
    connect(&yearTo, &QComboBox::currentTextChanged, [=](){ this->changeYears(); });
    connect(&method, &QComboBox::currentTextChanged, [=](const QString &newMethod){ this->changeMethod(newMethod); });
    connect(&addJointStation, &QPushButton::clicked, [=](){ addStationClicked(); });
    connect(&deleteJointStation, &QPushButton::clicked, [=](){ deleteStationClicked(); });
    connect(&saveToDb, &QPushButton::clicked, [=](){ saveToDbClicked(); });
    connect(changeHomogeneityLeftAxis, &QAction::triggered, this, &Crit3DHomogeneityWidget::on_actionChangeLeftAxis);
    connect(exportHomogeneityGraph, &QAction::triggered, this, &Crit3DHomogeneityWidget::on_actionExportGraph);
    connect(exportHomogeneityData, &QAction::triggered, this, &Crit3DHomogeneityWidget::on_actionExportData);

    plotAnnualSeries();

    show();
}


Crit3DHomogeneityWidget::~Crit3DHomogeneityWidget()
{

}

void Crit3DHomogeneityWidget::closeEvent(QCloseEvent *event)
{
    event->accept();
}

void Crit3DHomogeneityWidget::plotAnnualSeries()
{
    int firstYear = yearFrom.currentText().toInt();
    int lastYear = yearTo.currentText().toInt();

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
    clima.setGenericPeriodDateStart(QDate(firstYear, 1, 1));
    clima.setGenericPeriodDateEnd(QDate(lastYear, 12, 31));
    clima.setNYears(0);

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
        QDate endDate(QDate(lastYear, 12, 31));
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

    annualSeriesChartView->clearSeries();
    int validYears = computeAnnualSeriesOnPointFromDaily(&myError, meteoPointsDbHandler, nullptr,
                                             &meteoPointTemp, &clima, false, isAnomaly, meteoSettings, outputValues, dataAlreadyLoaded);
    formInfo.close();

    if (validYears > 0)
    {
        double sum = 0;
        int count = 0;
        int validData = 0;
        for (int i = firstYear; i<=lastYear; i++)
        {
            years.push_back(i);
            if (outputValues[count] != NODATA)
            {
                sum += double(outputValues[unsigned(count)]);
                validData = validData + 1;
            }
            count = count + 1;
        }
        averageValue = sum / validYears;
        // draw
        annualSeriesChartView->draw(years, outputValues);
    }
}

void Crit3DHomogeneityWidget::setMpValues(Crit3DMeteoPoint meteoPointGet, Crit3DMeteoPoint* meteoPointSet, QDate myDate)
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
        case dailyAirDewTemperatureMax:
        {
            float value = meteoPointGet.getMeteoPointValueD(getCrit3DDate(myDate), dailyAirTemperatureMax, meteoSettings);
            meteoPointSet->setMeteoPointValueD(getCrit3DDate(myDate), dailyAirTemperatureMax, value);
            value = meteoPointGet.getMeteoPointValueD(getCrit3DDate(myDate), dailyAirRelHumidityMin, meteoSettings);
            meteoPointSet->setMeteoPointValueD(getCrit3DDate(myDate), dailyAirRelHumidityMin, value);
            break;
        }

        case dailyAirDewTemperatureMin:
        {
            float value = meteoPointGet.getMeteoPointValueD(getCrit3DDate(myDate), dailyAirTemperatureMin, meteoSettings);
            meteoPointSet->setMeteoPointValueD(getCrit3DDate(myDate), dailyAirTemperatureMin, value);
            value = meteoPointGet.getMeteoPointValueD(getCrit3DDate(myDate), dailyAirRelHumidityMax, meteoSettings);
            meteoPointSet->setMeteoPointValueD(getCrit3DDate(myDate), dailyAirRelHumidityMax, value);
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

void Crit3DHomogeneityWidget::changeVar(const QString varName)
{
    myVar = getKeyMeteoVarMeteoMap(MapDailyMeteoVarToString, varName.toStdString());
    listFoundStations.clear();
    listSelectedStations.clear();
    stationsTable.clear();
    resultLabel.clear();
    annualSeriesChartView->setYTitle(QString::fromStdString(getUnitFromVariable(myVar)));
    plotAnnualSeries();
}

void Crit3DHomogeneityWidget::changeYears()
{
    listFoundStations.clear();
    listSelectedStations.clear();
    stationsTable.clear();
    resultLabel.clear();
    plotAnnualSeries();
}

void Crit3DHomogeneityWidget::changeMethod(const QString methodName)
{
    if (methodName == "SNHT")
    {

    }
    else if (methodName == "CRADDOCK")
    {

    }
    //plot();
}

void Crit3DHomogeneityWidget::addStationClicked()
{
    if (jointStationsList.currentText().isEmpty())
    {
        return;
    }
    std::string newId;
    if (jointStationsSelected.findItems(jointStationsList.currentText(), Qt::MatchExactly).isEmpty())
    {
        jointStationsSelected.addItem(jointStationsList.currentText());
        deleteJointStation.setEnabled(true);
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

        meteoPointsDbHandler->loadDailyData(getCrit3DDate(firstDaily), getCrit3DDate(lastDaily), &meteoPoints[indexMp]);
    }

}

void Crit3DHomogeneityWidget::deleteStationClicked()
{
    QList<QListWidgetItem*> items = jointStationsSelected.selectedItems();
    foreach(QListWidgetItem * item, items)
    {
        idPoints.removeOne(item->text().section(" ",0,0).toStdString());
        delete jointStationsSelected.takeItem(jointStationsSelected.row(item));
    }
    updateYears();
}

void Crit3DHomogeneityWidget::saveToDbClicked()
{
    QList<QString> stationsList;
    for (int row = 0; row < jointStationsSelected.count(); row++)
    {
        QString textSelected = jointStationsSelected.item(row)->text();
        stationsList.append(textSelected.section(" ",0,0));
    }
    if (!meteoPointsDbHandler->setJointStations(QString::fromStdString(meteoPoints[0].id), stationsList))
    {
        QMessageBox::critical(nullptr, "Error", meteoPointsDbHandler->error);
    }
}

void Crit3DHomogeneityWidget::updateYears()
{

    yearFrom.blockSignals(true);
    yearTo.blockSignals(true);

    lastDaily = meteoPointsDbHandler->getLastDate(daily, meteoPoints[0].id).date();
    for (int i = 1; i<idPoints.size(); i++)
    {

        QDate lastDailyJointStation = meteoPointsDbHandler->getLastDate(daily, idPoints[i]).date();
        if (lastDailyJointStation.isValid() && lastDailyJointStation > lastDaily )
        {
            lastDaily = lastDailyJointStation;
        }
    }
    // save current yearFrom
    QString currentYearFrom = yearFrom.currentText();
    yearFrom.clear();
    yearTo.clear();

    for(int i = 0; i <= lastDaily.year()-firstDaily.year(); i++)
    {
        yearFrom.addItem(QString::number(firstDaily.year()+i));
        yearTo.addItem(QString::number(firstDaily.year()+i));
    }
    yearTo.setCurrentText(QString::number(lastDaily.year()));
    yearFrom.setCurrentText(currentYearFrom);

    yearFrom.blockSignals(true);
    yearTo.blockSignals(true);
    plotAnnualSeries();
}

void Crit3DHomogeneityWidget::on_actionChangeLeftAxis()
{
    DialogChangeAxis changeAxisDialog(true);
    if (changeAxisDialog.result() == QDialog::Accepted)
    {
        homogeneityChartView->setYmax(changeAxisDialog.getMaxVal());
        homogeneityChartView->setYmin(changeAxisDialog.getMinVal());
    }
}


void Crit3DHomogeneityWidget::on_actionExportGraph()
{

    QString fileName = QFileDialog::getSaveFileName(this, tr("Save current graph"), "", tr("png files (*.png)"));

    if (fileName != "")
    {
        const auto dpr = homogeneityChartView->devicePixelRatioF();
        QPixmap buffer(homogeneityChartView->width() * dpr, homogeneityChartView->height() * dpr);
        buffer.setDevicePixelRatio(dpr);
        buffer.fill(Qt::transparent);

        QPainter *paint = new QPainter(&buffer);
        paint->setPen(*(new QColor(255,34,255,255)));
        homogeneityChartView->render(paint);

        QFile file(fileName);
        file.open(QIODevice::WriteOnly);
        buffer.save(&file, "PNG");
    }
}

void Crit3DHomogeneityWidget::on_actionExportData()
{
    /*
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
        myStream.setRealNumberPrecision(3);
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
    */
}

