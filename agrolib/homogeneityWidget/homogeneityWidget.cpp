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

Crit3DHomogeneityWidget::Crit3DHomogeneityWidget(Crit3DMeteoPointsDbHandler* meteoPointsDbHandler, QList<Crit3DMeteoPoint> meteoPointsNearDistanceList, QList<std::string> sortedId, std::vector<float> distanceId,
                                                         QList<QString> jointStationsMyMp, QDate firstDaily, QDate lastDaily, Crit3DMeteoSettings *meteoSettings, QSettings *settings, Crit3DClimateParameters *climateParameters, Crit3DQuality *quality)
:meteoPointsDbHandler(meteoPointsDbHandler), meteoPointsNearDistanceList(meteoPointsNearDistanceList), sortedId(sortedId), distanceId(distanceId), firstDaily(firstDaily),
  lastDaily(lastDaily), meteoSettings(meteoSettings), settings(settings), climateParameters(climateParameters), quality(quality)
{
    this->setWindowTitle("Homogeneity Test Id:"+QString::fromStdString(this->meteoPointsNearDistanceList[0].id)+" "+QString::fromStdString(this->meteoPointsNearDistanceList[0].name));
    this->resize(1240, 700);
    this->setSizePolicy(QSizePolicy::Expanding, QSizePolicy::Expanding);
    this->setAttribute(Qt::WA_DeleteOnClose);

    idPointsJointed << this->meteoPointsNearDistanceList[0].id;
    for (int i = 0; i<jointStationsMyMp.size(); i++)
    {
        idPointsJointed << jointStationsMyMp[i].toStdString();
    }

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
    QHBoxLayout *paramtersLayout = new QHBoxLayout;

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
    variable.addItem("DAILY_TAVG");
    variable.addItem("DAILY_PREC");
    variable.addItem("DAILY_RHAVG");
    variable.addItem("DAILY_RAD");
    variable.addItem("DAILY_W_VEC_INT_AVG");
    variable.addItem("DAILY_W_SCAL_INT_AVG");

    myVar = getKeyMeteoVarMeteoMap(MapDailyMeteoVarToString, variable.currentText().toStdString());
    variable.setSizeAdjustPolicy(QComboBox::AdjustToContents);
    variable.setMaximumWidth(150);
    variableLayout->addWidget(variableLabel);
    variableLayout->addWidget(&variable);
    variableGroupBox->setLayout(variableLayout);

    QLabel *minNumStationsLabel = new QLabel(tr("Minimum number of stations: "));
    paramtersLayout->addWidget(minNumStationsLabel);
    minNumStations.setMaximumWidth(50);
    minNumStations.setMaximumHeight(24);
    minNumStations.setText("1");
    minNumStations.setValidator(new QIntValidator(1.0, 20.0));
    paramtersLayout->addWidget(&minNumStations);

    QLabel *jointStationsLabel = new QLabel(tr("Stations:"));
    jointStationsSelectLayout->addWidget(jointStationsLabel);
    jointStationsSelectLayout->addWidget(&jointStationsList);

    for (int i = 0; i<jointStationsMyMp.size(); i++)
    {
        int indexMp;
        for (int j = 0; j<this->meteoPointsNearDistanceList.size(); j++)
        {
            if (this->meteoPointsNearDistanceList[j].id == jointStationsMyMp[i].toStdString())
            {
                indexMp = j;
                break;
            }
        }
        jointStationsSelected.addItem(QString::fromStdString(this->meteoPointsNearDistanceList[indexMp].id)+" "+QString::fromStdString(this->meteoPointsNearDistanceList[indexMp].name));
        // load all Data
        QDate firstDaily = meteoPointsDbHandler->getFirstDate(daily, jointStationsMyMp[i].toStdString()).date();
        QDate lastDaily = meteoPointsDbHandler->getLastDate(daily, jointStationsMyMp[i].toStdString()).date();
        meteoPointsDbHandler->loadDailyData(getCrit3DDate(firstDaily), getCrit3DDate(lastDaily), &this->meteoPointsNearDistanceList[indexMp]);
    }

    for (int i = 1; i<this->meteoPointsNearDistanceList.size(); i++)
    {
        jointStationsList.addItem(QString::fromStdString(this->meteoPointsNearDistanceList[i].id)+" "+QString::fromStdString(this->meteoPointsNearDistanceList[i].name));
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
    saveToDb.setEnabled(false);

    if (jointStationsSelected.count() == 0)
    {
        deleteJointStation.setEnabled(false);
    }
    else
    {
        deleteJointStation.setEnabled(true);
    }
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
    addStationFoundButton.setText("➡");
    deleteStationFoundButton.setText("⬅");
    addStationFoundButton.setEnabled(false);
    deleteStationFoundButton.setEnabled(false);
    arrowLayout->addWidget(&addStationFoundButton);
    arrowLayout->addWidget(&deleteStationFoundButton);
    listFoundStations.setSelectionMode(QAbstractItemView::ExtendedSelection);
    listSelectedStations.setSelectionMode(QAbstractItemView::ExtendedSelection);
    stationsLayout->addWidget(&listFoundStations);
    stationsLayout->addLayout(arrowLayout);
    stationsLayout->addWidget(&listSelectedStations);

    headerLayout->addWidget(allHeader);
    headerLayout->addStretch(listFoundStations.width());
    headerLayout->addWidget(selectedHeader);
    selectStationsLayout->addLayout(headerLayout);
    selectStationsLayout->addLayout(stationsLayout);

    stationsTable.setColumnCount(4);
    QList<QString> tableHeader;
    tableHeader <<"Name"<<"R^2"<<"getDistance [km]"<<"Delta Z [m]";
    stationsTable.setHorizontalHeaderLabels(tableHeader);
    stationsTable.adjustSize();
    stationsTable.horizontalHeader()->setSectionResizeMode(QHeaderView::Stretch);
    stationsTable.resizeColumnsToContents();
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

    leftLayout->addLayout(firstLayout);
    leftLayout->addWidget(jointStationsGroupBox);
    leftLayout->addLayout(paramtersLayout);
    leftLayout->addLayout(findStationsLayout);
    leftLayout->addLayout(selectStationsLayout);
    leftLayout->addWidget(resultGroupBox);

    // menu
    QMenuBar* menuBar = new QMenuBar();
    QMenu *editMenu = new QMenu("Edit");

    menuBar->addMenu(editMenu);
    mainLayout->setMenuBar(menuBar);

    QAction* changeHomogeneityLeftAxis = new QAction(tr("&Change homogenity chart axis left"), this);
    QAction* exportAnnualGraph = new QAction(tr("&Export annual series graph"), this);
    QAction* exportHomogeneityGraph = new QAction(tr("&Export homogenity graph"), this);
    QAction* exportAnnualData = new QAction(tr("&Export annual series data"), this);
    QAction* exportHomogeneityData = new QAction(tr("&Export homogenity data"), this);

    editMenu->addAction(changeHomogeneityLeftAxis);
    editMenu->addAction(exportAnnualGraph);
    editMenu->addAction(exportHomogeneityGraph);
    editMenu->addAction(exportAnnualData);
    editMenu->addAction(exportHomogeneityData);

    mainLayout->addLayout(leftLayout);
    mainLayout->addLayout(plotLayout);
    setLayout(mainLayout);

    connect(&variable, &QComboBox::currentTextChanged, [=](const QString &newVariable){ this->changeVar(newVariable); });
    connect(&yearFrom, &QComboBox::currentTextChanged, [=](){ this->changeYears(); });
    connect(&yearTo, &QComboBox::currentTextChanged, [=](){ this->changeYears(); });
    connect(&method, &QComboBox::currentTextChanged, [=](const QString &newMethod){ this->changeMethod(newMethod); });
    connect(&addJointStation, &QPushButton::clicked, [=](){ addJointStationClicked(); });
    connect(&deleteJointStation, &QPushButton::clicked, [=](){ deleteJointStationClicked(); });
    connect(&saveToDb, &QPushButton::clicked, [=](){ saveToDbClicked(); });
    connect(&find, &QPushButton::clicked, [=](){ findReferenceStations(); });
    connect(&addStationFoundButton, &QPushButton::clicked, [=](){ addFoundStationClicked(); });
    connect(&deleteStationFoundButton, &QPushButton::clicked, [=](){ deleteFoundStationClicked(); });
    connect(&execute, &QPushButton::clicked, [=](){ executeClicked(); });
    connect(changeHomogeneityLeftAxis, &QAction::triggered, this, &Crit3DHomogeneityWidget::on_actionChangeLeftAxis);
    connect(exportAnnualGraph, &QAction::triggered, this, &Crit3DHomogeneityWidget::on_actionExportAnnualGraph);
    connect(exportHomogeneityGraph, &QAction::triggered, this, &Crit3DHomogeneityWidget::on_actionExportHomogeneityGraph);
    connect(exportAnnualData, &QAction::triggered, this, &Crit3DHomogeneityWidget::on_actionExportAnnualData);
    connect(exportHomogeneityData, &QAction::triggered, this, &Crit3DHomogeneityWidget::on_actionExportHomogeneityData);

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
    annualSeriesChartView->clearSeries();
    myAnnualSeries.clear();
    int firstYear = yearFrom.currentText().toInt();
    int lastYear = yearTo.currentText().toInt();
    QDate firstDate(firstYear, 1, 1);
    QDate lastDate(lastYear, 12, 31);

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
    clima.setGenericPeriodDateStart(firstDate);
    clima.setGenericPeriodDateEnd(lastDate);
    clima.setNYears(0);

    std::vector<int> years;

    bool isAnomaly = false;

    FormInfo formInfo;
    formInfo.showInfo("compute annual series...");
    // copy data to MPTemp
    Crit3DMeteoPoint meteoPointTemp;
    meteoPointTemp.id = meteoPointsNearDistanceList[0].id;
    meteoPointTemp.latitude = meteoPointsNearDistanceList[0].latitude;
    meteoPointTemp.elaboration = meteoPointsNearDistanceList[0].elaboration;
    QDate lastDateCopyed = meteoPointsDbHandler->getLastDate(daily, meteoPointsNearDistanceList[0].id).date();
    bool dataAlreadyLoaded;
    if (idPointsJointed.size() == 1 || lastDateCopyed >= lastDate)
    {
        // no joint station inside date interval
        // meteoPointTemp should be init
        meteoPointTemp.nrObsDataDaysD = 0;
        dataAlreadyLoaded = false;
    }
    else
    {
        QDate endDate(QDate(lastYear, 12, 31));
        int numberOfDays = meteoPointsNearDistanceList[0].obsDataD[0].date.daysTo(getCrit3DDate(endDate))+1;
        meteoPointTemp.initializeObsDataD(numberOfDays, meteoPointsNearDistanceList[0].obsDataD[0].date);
        meteoPointTemp.initializeObsDataDFromMp(meteoPointsNearDistanceList[0].nrObsDataDaysD, meteoPointsNearDistanceList[0].obsDataD[0].date, meteoPointsNearDistanceList[0]);
        if (lastDateCopyed < lastDate)
        {
            for (int i = 1; i<idPointsJointed.size(); i++)
            {
                QDate lastDateNew = meteoPointsDbHandler->getLastDate(daily, idPointsJointed[i]).date();
                if (lastDateNew > lastDateCopyed)
                {
                    int indexMp;
                    for (int j = 0; j<meteoPointsNearDistanceList.size(); j++)
                    {
                        if (meteoPointsNearDistanceList[j].id == idPointsJointed[i])
                        {
                            indexMp = j;
                            break;
                        }
                    }
                    for (QDate myDate=lastDateCopyed.addDays(1); myDate<=lastDateNew; myDate=myDate.addDays(1))
                    {
                        setMpValues(meteoPointsNearDistanceList[indexMp], &meteoPointTemp, myDate, myVar, meteoSettings);
                    }
                }
                lastDateCopyed = lastDateNew;
            }
        }
        dataAlreadyLoaded = true;
    }

    int validYears = computeAnnualSeriesOnPointFromDaily(&myError, meteoPointsDbHandler, nullptr,
                                             &meteoPointTemp, &clima, false, isAnomaly, meteoSettings, myAnnualSeries, dataAlreadyLoaded);
    formInfo.close();

    if (validYears > 0)
    {
        double sum = 0;
        int count = 0;
        int validData = 0;
        int yearsLength = lastYear - firstYear;
        int nYearsToAdd;
        std::vector<float> seriesToView = myAnnualSeries;
        if (yearsLength > 15)
        {
            for (int inc = 0; inc<=3; inc++)
            {
                if ( (yearsLength+inc) % 2 == 0 &&  (yearsLength+inc)/2 <= 15)
                {
                    nYearsToAdd = inc;
                    break;
                }
                if ( (yearsLength+inc) % 3 == 0 &&  (yearsLength+inc)/3 <= 15)
                {
                    nYearsToAdd = inc;
                    break;
                }
                if ( (yearsLength+inc) % 4 == 0 &&  (yearsLength+inc)/4 <= 15)
                {
                    nYearsToAdd = inc;
                    break;
                }
            }
            for (int i = nYearsToAdd; i> 0; i--)
            {
                years.push_back(firstYear-i);
                seriesToView.insert(seriesToView.begin(),NODATA);
            }
        }
        for (int i = firstYear; i<=lastYear; i++)
        {
            years.push_back(i);
            if (myAnnualSeries[count] != NODATA)
            {
                sum += double(myAnnualSeries[unsigned(count)]);
                validData = validData + 1;
            }
            count = count + 1;
        }
        averageValue = sum / validYears;
        // draw
        annualSeriesChartView->draw(years, seriesToView);
    }
    else
    {
        myAnnualSeries.clear();
        return;
    }
}



void Crit3DHomogeneityWidget::changeVar(const QString varName)
{
    myVar = getKeyMeteoVarMeteoMap(MapDailyMeteoVarToString, varName.toStdString());
    listFoundStations.clear();
    listAllFound.clear();
    listSelectedStations.clear();
    stationsTable.clearContents();
    resultLabel.clear();
    annualSeriesChartView->setYTitle(QString::fromStdString(getUnitFromVariable(myVar)));
    execute.setEnabled(false);
    homogeneityChartView->clearSNHTSeries();
    homogeneityChartView->clearCraddockSeries();
    plotAnnualSeries();
}

void Crit3DHomogeneityWidget::changeYears()
{
    listFoundStations.clear();
    listAllFound.clear();
    listSelectedStations.clear();
    stationsTable.clearContents();
    stationsTable.setRowCount(0);
    resultLabel.clear();
    execute.setEnabled(false);
    homogeneityChartView->clearSNHTSeries();
    homogeneityChartView->clearCraddockSeries();
    plotAnnualSeries();
}

void Crit3DHomogeneityWidget::changeMethod(const QString methodName)
{
    homogeneityChartView->clearSNHTSeries();
    homogeneityChartView->clearCraddockSeries();
    resultLabel.clear();
    if (execute.isEnabled())
    {
        executeClicked();
    }
}

void Crit3DHomogeneityWidget::addJointStationClicked()
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
        idPointsJointed << newId;

        int indexMp;
        for (int j = 0; j<meteoPointsNearDistanceList.size(); j++)
        {
            if (meteoPointsNearDistanceList[j].id == newId)
            {
                indexMp = j;
                break;
            }
        }
        // load all Data
        QDate firstDailyNewId = meteoPointsDbHandler->getFirstDate(daily, newId).date();
        QDate lastDailyNewId = meteoPointsDbHandler->getLastDate(daily, newId).date();

        meteoPointsDbHandler->loadDailyData(getCrit3DDate(firstDailyNewId), getCrit3DDate(lastDailyNewId), &meteoPointsNearDistanceList[indexMp]);
        updateYears();
    }

}

void Crit3DHomogeneityWidget::deleteJointStationClicked()
{
    QList<QListWidgetItem*> items = jointStationsSelected.selectedItems();
    if (items.isEmpty())
    {
        return;
    }
    foreach(QListWidgetItem * item, items)
    {
        idPointsJointed.removeOne(item->text().section(" ",0,0).toStdString());
        delete jointStationsSelected.takeItem(jointStationsSelected.row(item));
    }
    saveToDb.setEnabled(true);
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
    if (!meteoPointsDbHandler->setJointStations(QString::fromStdString(meteoPointsNearDistanceList[0].id), stationsList))
    {
        QMessageBox::critical(nullptr, "Error", meteoPointsDbHandler->error);
    }
    saveToDb.setEnabled(false);
}

void Crit3DHomogeneityWidget::updateYears()
{

    yearFrom.blockSignals(true);
    yearTo.blockSignals(true);

    for (int i = 1; i<idPointsJointed.size(); i++)
    {

        QDate lastDailyJointStation = meteoPointsDbHandler->getLastDate(daily, idPointsJointed[i]).date();
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

    yearFrom.blockSignals(false);
    yearTo.blockSignals(false);
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

void Crit3DHomogeneityWidget::on_actionExportHomogeneityGraph()
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

void Crit3DHomogeneityWidget::on_actionExportAnnualGraph()
{

    QString fileName = QFileDialog::getSaveFileName(this, tr("Save current graph"), "", tr("png files (*.png)"));

    if (fileName != "")
    {
        const auto dpr = annualSeriesChartView->devicePixelRatioF();
        QPixmap buffer(annualSeriesChartView->width() * dpr, annualSeriesChartView->height() * dpr);
        buffer.setDevicePixelRatio(dpr);
        buffer.fill(Qt::transparent);

        QPainter *paint = new QPainter(&buffer);
        paint->setPen(*(new QColor(255,34,255,255)));
        annualSeriesChartView->render(paint);

        QFile file(fileName);
        file.open(QIODevice::WriteOnly);
        buffer.save(&file, "PNG");
    }
}

void Crit3DHomogeneityWidget::on_actionExportHomogeneityData()
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
        myStream.setRealNumberPrecision(3);
        if (method.currentText() == "SNHT")
        {
            QString header = "year,value";
            myStream << header << "\n";
            QList<QPointF> dataPoints = homogeneityChartView->exportSNHTValues();
            for (int i = 0; i < dataPoints.size(); i++)
            {
                myStream << dataPoints[i].toPoint().x() << "," << dataPoints[i].y() << "\n";
            }
        }
        else if (method.currentText() == "CRADDOCK")
        {
            QList<QString> refNames;
            QList<QList<QPointF>> pointsAllSeries = homogeneityChartView->exportCraddockValues(refNames);
            for (int point = 0; point<refNames.size(); point++)
            {
                QString name = refNames[point];
                myStream << name << "\n";
                QString header = "year,value";
                myStream << header << "\n";
                QList<QPointF> dataPoints = pointsAllSeries[point];
                for (int i = 0; i < dataPoints.size(); i++)
                {
                    myStream << dataPoints[i].toPoint().x() << "," << dataPoints[i].y() << "\n";
                }
            }
        }
        myFile.close();

        return;
    }
}

void Crit3DHomogeneityWidget::on_actionExportAnnualData()
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
        myStream.setRealNumberPrecision(3);
        QString header = "year,value";
        myStream << header << "\n";
        QList<QPointF> dataPoins = annualSeriesChartView->exportAnnualValues();
        for (int i = 0; i < dataPoins.size(); i++)
        {
            myStream << dataPoins[i].toPoint().x() << "," << dataPoins[i].y() << "\n";
        }
        myFile.close();

        return;
    }
}

void Crit3DHomogeneityWidget::findReferenceStations()
{
    stationsTable.clearContents();
    listFoundStations.clear();
    listAllFound.clear();
    listSelectedStations.clear();

    QList<std::vector<float>> myAnnualSeriesFound;
    QList<std::string> sortedIdFound;
    QList<float> distanceIdFound;
    int firstYear = yearFrom.currentText().toInt();
    int lastYear = yearTo.currentText().toInt();
    QDate firstDate(firstYear, 1, 1);
    QDate lastDate(lastYear, 12, 31);

    if (myAnnualSeries.size() == 0)
    {
        QMessageBox::critical(nullptr, "Error", "Data unavailable for candidate station");
        return;
    }
    int myNrStations = 0;
    QProgressDialog progress("Finding stations...", "Abort", 0, minNumStations.text().toInt(), this);
    progress.setWindowModality(Qt::WindowModal);
    progress.show();
    int i = 0;
    for (i = 0; i<sortedId.size(); i++)
    {
        progress.setValue(myNrStations);
        if (idPointsJointed.contains(sortedId[i]))
        {
            continue;
        }
        clima.setYearStart(firstYear);
        clima.setYearEnd(lastYear);
        clima.setGenericPeriodDateStart(QDate(firstYear, 1, 1));
        clima.setGenericPeriodDateEnd(QDate(lastYear, 12, 31));
        clima.setNYears(0);

        Crit3DMeteoPoint mpToBeComputed;
        mpToBeComputed.id = sortedId[i];
        QString name = meteoPointsDbHandler->getNameGivenId(QString::fromStdString(sortedId[i]));
        QList<QString> jointStationsListMpToBeComputed = meteoPointsDbHandler->getJointStations(QString::fromStdString(mpToBeComputed.id));
        meteoPointsDbHandler->loadDailyData(getCrit3DDate(firstDate), getCrit3DDate(lastDate), &mpToBeComputed);

        // copy data to MPTemp
        Crit3DMeteoPoint meteoPointTemp;
        meteoPointTemp.id = mpToBeComputed.id;
        meteoPointTemp.latitude = mpToBeComputed.latitude;
        meteoPointTemp.elaboration = mpToBeComputed.elaboration;
        bool dataAlreadyLoaded;
        if (jointStationsListMpToBeComputed.size() == 0)
        {
            // meteoPointTemp should be init
            meteoPointTemp.nrObsDataDaysH = 0;
            meteoPointTemp.nrObsDataDaysD = 0;
            dataAlreadyLoaded = false;
        }
        else
        {
            QDate endDate(QDate(yearTo.currentText().toInt(), 12, 31));
            int numberOfDays = mpToBeComputed.obsDataD[0].date.daysTo(getCrit3DDate(endDate))+1;
            meteoPointTemp.initializeObsDataD(numberOfDays, mpToBeComputed.obsDataD[0].date);
            meteoPointTemp.initializeObsDataDFromMp(mpToBeComputed.nrObsDataDaysD, mpToBeComputed.obsDataD[0].date, mpToBeComputed);
            QDate lastDateCopyed = meteoPointsDbHandler->getLastDate(daily, mpToBeComputed.id).date();
            for (int j = 0; j<jointStationsListMpToBeComputed.size(); j++)
            {
                QDate lastDateNew = meteoPointsDbHandler->getLastDate(daily, jointStationsListMpToBeComputed[j].toStdString()).date();
                if (lastDateNew > lastDateCopyed)
                {
                    Crit3DMeteoPoint mpGet;
                    mpGet.id = jointStationsListMpToBeComputed[j].toStdString();
                    meteoPointsDbHandler->loadDailyData(getCrit3DDate(firstDate), getCrit3DDate(lastDate), &mpGet);

                    for (QDate myDate=lastDateCopyed.addDays(1); myDate<=lastDateNew; myDate=myDate.addDays(1))
                    {
                        setMpValues(mpGet, &meteoPointTemp, myDate, myVar, meteoSettings);
                    }
                    QString nameJoint = meteoPointsDbHandler->getNameGivenId(jointStationsListMpToBeComputed[j]);
                    name = name+"_J_"+nameJoint;
                }
                lastDateCopyed = lastDateNew;
            }
            dataAlreadyLoaded = true;
        }
        std::vector<float> mpToBeComputedAnnualSeries;
        int validYears = computeAnnualSeriesOnPointFromDaily(&myError, meteoPointsDbHandler, nullptr,
                                                 &meteoPointTemp, &clima, false, false, meteoSettings, mpToBeComputedAnnualSeries, dataAlreadyLoaded);
        if (validYears != 0)
        {
            if ( (float)validYears / (float)(lastYear - firstYear + 1) > meteoSettings->getMinimumPercentage() / 100.f)
            {
                myNrStations = myNrStations + 1;
                sortedIdFound.append(sortedId[i]);   
                mapNameId.insert(name, sortedId[i]);
                mapNameAnnualSeries.insert(name,mpToBeComputedAnnualSeries);
                distanceIdFound.append(distanceId[i]);
                myAnnualSeriesFound.append(mpToBeComputedAnnualSeries);
            }
        }
        if (myNrStations == minNumStations.text().toInt())
        {
            break;
        }
    }
    progress.setValue(myNrStations);
    progress.close();

    if (myNrStations == 0)
    {
        QMessageBox::critical(nullptr, "Error", "No reference stations found");
        return;
    }
    stationsTable.setRowCount(myNrStations);
    for (int z = 0; z<myNrStations; z++)
    {
        float r2, y_intercept, trend;
        QString name;
        QMapIterator<QString,std::string> iterator(mapNameId);
        while (iterator.hasNext()) {
            iterator.next();
            if (iterator.value() == sortedIdFound[z])
            {
                name = iterator.key();
                break;
            }
        }
        statistics::linearRegression(myAnnualSeries, myAnnualSeriesFound[z], int(myAnnualSeries.size()), false, &y_intercept, &trend, &r2);
        double altitude = meteoPointsDbHandler->getAltitudeGivenId(QString::fromStdString(sortedIdFound[z]));
        double delta =  meteoPointsNearDistanceList[0].point.z - altitude;
        stationsTable.setItem(z,0,new QTableWidgetItem(name));
        stationsTable.setItem(z,1,new QTableWidgetItem(QString::number(r2, 'f', 3)));
        stationsTable.setItem(z,2,new QTableWidgetItem(QString::number(distanceIdFound[z]/1000, 'f', 1)));
        stationsTable.setItem(z,3,new QTableWidgetItem(QString::number(delta)));
        if (listFoundStations.findItems(name, Qt::MatchExactly).isEmpty())
        {
            listAllFound.append(name);
            listFoundStations.addItem(name);
            addStationFoundButton.setEnabled(true);
        }
    }
}

void Crit3DHomogeneityWidget::addFoundStationClicked()
{
    QList<QListWidgetItem *> items = listFoundStations.selectedItems();
    for (int i = 0; i<items.size(); i++)
    {
        listFoundStations.takeItem(listFoundStations.row(items[i]));
        listSelectedStations.addItem(items[i]);
    }

    if(listFoundStations.count() == 0)
    {
        addStationFoundButton.setEnabled(false);
    }
    else
    {
        addStationFoundButton.setEnabled(true);
    }

    if(listSelectedStations.count() == 0)
    {
        deleteStationFoundButton.setEnabled(false);
        execute.setEnabled(false);
    }
    else
    {
        deleteStationFoundButton.setEnabled(true);
        execute.setEnabled(true);
    }
}

void Crit3DHomogeneityWidget::deleteFoundStationClicked()
{
    QList<QListWidgetItem *> items = listSelectedStations.selectedItems();
    for (int i = 0; i<items.size(); i++)
    {
        listSelectedStations.takeItem(listSelectedStations.row(items[i]));
    }
    // add station, keep order
    listFoundStations.clear();
    for (int i = 0; i<listAllFound.size(); i++)
    {
        if (listSelectedStations.findItems(listAllFound[i], Qt::MatchExactly).isEmpty())
        {
            listFoundStations.addItem(listAllFound[i]);
        }
    }

    if(listFoundStations.count() == 0)
    {
        addStationFoundButton.setEnabled(false);
    }
    else
    {
        addStationFoundButton.setEnabled(true);
    }

    if(listSelectedStations.count() == 0)
    {
        deleteStationFoundButton.setEnabled(false);
        execute.setEnabled(false);
    }
    else
    {
        deleteStationFoundButton.setEnabled(true);
        execute.setEnabled(true);
    }
}

void Crit3DHomogeneityWidget::executeClicked()
{
    bool isHomogeneous = false;
    std::vector<double> myTValues;
    double myYearTmax = NODATA;
    double myTmax = NODATA;
    resultLabel.clear();

    int myFirstYear = yearFrom.currentText().toInt();
    int myLastYear = yearTo.currentText().toInt();
    if (myAnnualSeries.empty())
    {
        QMessageBox::critical(nullptr, "Error", "Data unavailable for candidate station");
        return;
    }
    if (mapNameAnnualSeries.isEmpty())
    {
        QMessageBox::critical(nullptr, "Error", "No reference stations found");
        return;
    }

    FormInfo formInfo;
    formInfo.showInfo("compute homogeneity test...");
    int nrReference = listSelectedStations.count();

    if (method.currentText() == "SNHT")
    {
        std::vector<double> myValidValues;
        for (int i = 0; i<myAnnualSeries.size(); i++)
        {
            if (myAnnualSeries[i] != NODATA)
            {
                myValidValues.push_back((double)myAnnualSeries[i]);
            }
        }
        double myAverage = statistics::mean(myValidValues, int(myValidValues.size()));
        std::vector<double> myRefAverage;
        std::vector<float> r2;
        std::vector<std::vector<float>> refSeriesVector;
        float r2Value, y_intercept, trend;

        for (int row = 0; row < nrReference; row++)
        {
            std::vector<double> myRefValidValues;
            QString name = listSelectedStations.item(row)->text();
            std::vector<float> refSeries = mapNameAnnualSeries.value(name);
            refSeriesVector.push_back(refSeries);
            for (int i = 0; i<refSeries.size(); i++)
            {
                if (refSeries[i] != NODATA)
                {
                    myRefValidValues.push_back((double)refSeries[i]);
                }
            }
            myRefAverage.push_back(statistics::mean(myRefValidValues, int(myRefValidValues.size())));
            statistics::linearRegression(myAnnualSeries, refSeries, long(myAnnualSeries.size()), false, &y_intercept, &trend, &r2Value);
            r2.push_back(r2Value);
        }
        double tmp, sumV;
        std::vector<double> myQ;
        if (myVar == dailyPrecipitation)
        {
            for (int i = 0; i<myAnnualSeries.size(); i++)
            {
                tmp = 0;
                sumV = 0;
                for (int j = 0; j<nrReference; j++)
                {
                    if (refSeriesVector[j][i] != NODATA)
                    {
                        tmp = tmp + (r2[j] * refSeriesVector[j][i] * myAverage / myRefAverage[j]);
                        sumV = r2[j] + sumV;
                    }
                }
                if (myAnnualSeries[i] != NODATA && tmp!= 0 && sumV!= 0)
                {
                    myQ.push_back((double)myAnnualSeries[i]/(tmp/sumV));
                }
                else
                {
                    myQ.push_back(NODATA);
                }
            }
        }
        else
        {
            for (int i = 0; i<myAnnualSeries.size(); i++)
            {
                tmp = 0;
                sumV = 0;
                for (int j = 0; j<nrReference; j++)
                {
                    if (refSeriesVector[j][i] != NODATA)
                    {
                        tmp = tmp + (r2[j] * (refSeriesVector[j][i] - myRefAverage[j] + myAverage));
                        sumV = r2[j] + sumV;
                    }
                }
                if (myAnnualSeries[i] != NODATA)
                {
                    if (sumV > 0)
                    {
                         myQ.push_back((double)myAnnualSeries[i]-(tmp/sumV));
                    }
                    else
                    {
                        myQ.push_back(NODATA);
                    }
                }
                else
                {
                    myQ.push_back(NODATA);
                }
            }
        }
        myValidValues.clear();
        for (int i = 0; i<myQ.size(); i++)
        {
            if (myQ[i] != NODATA)
            {
                myValidValues.push_back(myQ[i]);
            }
        }
        double myQAverage = statistics::mean(myValidValues, int(myValidValues.size()));
        double myQDevStd = statistics::standardDeviation(myValidValues, int(myValidValues.size()));
        std::vector<double> myZ;
        for (int i = 0; i<myQ.size(); i++)
        {
            if (myQ[i] != NODATA)
            {
                myZ.push_back((myQ[i] - myQAverage) / myQDevStd);
            }
            else
            {
                myZ.push_back(NODATA);
            }
        }
        myValidValues.clear();
        for (int i = 0; i<myZ.size(); i++)
        {
            if (myZ[i] != NODATA)
            {
                myValidValues.push_back(myZ[i]);
            }
        }

        double myZAverage = statistics::mean(myValidValues, int(myValidValues.size()));

        isHomogeneous = (qAbs(myZAverage) <= TOLERANCE);
        std::vector<double> z1;
        std::vector<double> z2;

        for (int i = 0; i< myZ.size()-1; i++)
        {
            myTValues.push_back(NODATA);
        }

        for (int a = 0; a < myZ.size()-1; a++)
        {
            z1.resize(a+1);
            z2.resize(myZ.size()-a-1);
            for (int i = 0; i< myZ.size(); i++)
            {
                if (i<=a)
                {
                    z1[i] = myZ[i];
                }
                else
                {
                    z2[i-a-1] = myZ[i];
                }
            }
            myValidValues.clear();
            for (int i = 0; i<z1.size(); i++)
            {
                if (z1[i] != NODATA)
                {
                    myValidValues.push_back(z1[i]);
                }
            }
            double myZ1Average = statistics::mean(myValidValues, int(myValidValues.size()));
            myValidValues.clear();
            for (int i = 0; i<z2.size(); i++)
            {
                if (z2[i] != NODATA)
                {
                    myValidValues.push_back(z2[i]);
                }
            }
            double myZ2Average = statistics::mean(myValidValues, int(myValidValues.size()));
            if (myZ1Average != NODATA && myZ2Average != NODATA)
            {
                myTValues[a] = ( (a+1) * pow(myZ1Average,2)) + ((myZ.size() - (a+1)) * pow(myZ2Average,2));
                if (myTmax == NODATA)
                {
                    myTmax = myTValues[a];
                    myYearTmax = myFirstYear + a;
                }
                else if (myTValues[a] > myTmax)
                {
                    myTmax = myTValues[a];
                    myYearTmax = myFirstYear + a;
                }
            }
        }
        std::vector<int> years;
        std::vector<double> outputValues;
        QList<QPointF> t95Points;
        double myValue;
        double myMaxValue = NODATA;
        double myT95;

        int myNrYears = yearTo.currentText().toInt() - myFirstYear + 1;
        for (int i = 0; i < myTValues.size(); i++)
        {
            years.push_back(myFirstYear+i);
            myValue = myTValues[i];
            if (myValue != NODATA)
            {
                if ((myMaxValue == NODATA) || (myValue > myMaxValue))
                {
                    myMaxValue = myValue;
                }
            }
            outputValues.push_back(myValue);
        }
        int myT95Index = round(myNrYears / 10);
        if (myT95Index > 0 && myT95Index <= 10)
        {
            int index = round(myNrYears / 10);
            myT95 = SNHT_T95_VALUES[index-1];
            if (myT95 != NODATA)
            {
                t95Points.append(QPointF(myFirstYear,myT95));
                t95Points.append(QPointF(myFirstYear+myTValues.size()-1,myT95));
            }
        }
        else
        {
            QMessageBox::critical(nullptr, "Info", "T95 value available only for number of years < 100");
        }

        int nYearsToAdd;
        if (years.size()-1 > 15)
        {
            for (int inc = 0; inc<=3; inc++)
            {
                if ( (years.size()-1+inc) % 2 == 0 &&  (years.size()-1+inc)/2 <= 15)
                {
                    nYearsToAdd = inc;
                    break;
                }
                if ( (years.size()-1+inc) % 3 == 0 &&  (years.size()-1+inc)/3 <= 15)
                {
                    nYearsToAdd = inc;
                    break;
                }
                if ( (years.size()-1+inc) % 4 == 0 &&  (years.size()-1+inc)/4 <= 15)
                {
                    nYearsToAdd = inc;
                    break;
                }
            }
            int pos = 0;
            for (int i = nYearsToAdd; i> 0; i--)
            {
                years.insert(years.begin()+pos,myFirstYear-i);
                outputValues.insert(outputValues.begin(),NODATA);
                pos = pos + 1;
            }
        }
        homogeneityChartView->drawSNHT(years,outputValues,t95Points);
        if (myTmax >= myT95 && myYearTmax != NODATA)
        {
            QString text = "Series is not homogeneous\n";
            text = text + "Year of discontinuity: " + QString::number(myYearTmax);
            resultLabel.setText(text);
            resultLabel.setWordWrap(true);
        }
        else
        {
            resultLabel.setText("Series is homogeneous");
        }

    }
    else if (method.currentText() == "CRADDOCK")
    {
        float myLastSum;
        float myReferenceSum;
        float myAverage;
        int myValidYears;
        std::vector<float> myRefAverage;
        std::vector<float> myC;
        std::vector<std::vector<float>> mySValues;
        std::vector<std::vector<float>> myD;
        // init myD
        for (int row = 0; row < nrReference; row++)
        {
            std::vector<float> vectD;
            for (int myYear = 0; myYear<myAnnualSeries.size(); myYear++)
            {
                vectD.push_back(NODATA);
            }
            myD.push_back(vectD);
            mySValues.push_back(vectD);
        }
        std::vector<QString> refNames;
        for (int row = 0; row < nrReference; row++)
        {
            // compute mean only for common years
            myLastSum = 0;
            myReferenceSum = 0;
            myValidYears = 0;
            QString name = listSelectedStations.item(row)->text();
            refNames.push_back(name);
            std::vector<float> refSeries = mapNameAnnualSeries.value(name);
            for (int myYear = 0; myYear<myAnnualSeries.size(); myYear++)
            {
                if (myAnnualSeries[myYear] != NODATA && refSeries[myYear] != NODATA)
                {
                    myLastSum = myLastSum + myAnnualSeries[myYear];
                    myReferenceSum = myReferenceSum + refSeries[myYear];
                    myValidYears = myValidYears + 1;
                }
            }
            myAverage = myLastSum / myValidYears;
            myRefAverage.push_back(myReferenceSum / myValidYears);
            if (myVar == dailyPrecipitation)
            {
                if (myRefAverage[row] == 0)
                {
                    QMessageBox::critical(nullptr, "Error", "Can not divide by zero");
                    return;
                }
                myC.push_back(myRefAverage[row] / myAverage);
            }
            else
            {
                 myC.push_back(myRefAverage[row] - myAverage);
            }
            for (int myYear = 0; myYear<myAnnualSeries.size(); myYear++)
            {
                if (myAnnualSeries[myYear] != NODATA && refSeries[myYear] != NODATA)
                {
                    if (myVar == dailyPrecipitation)
                    {
                         myD[row][myYear] = myC[row]*myAnnualSeries[myYear] - refSeries[myYear];
                    }
                    else
                    {
                        myD[row][myYear] = myC[row]+myAnnualSeries[myYear] - refSeries[myYear];
                    }
                }
            }
        }
        // sum
        for (int row = 0; row < nrReference; row++)
        {
            myLastSum = 0;
            for (int myYear = 0; myYear<myAnnualSeries.size(); myYear++)
            {
                if (myD[row][myYear] != NODATA)
                {
                    mySValues[row][myYear] = myLastSum + myD[row][myYear];
                    myLastSum = mySValues[row][myYear];
                }
            }
        }
        // draw
        homogeneityChartView->drawCraddock(myFirstYear, myLastYear, mySValues, refNames, myVar, averageValue);

    }
    formInfo.close();

}

