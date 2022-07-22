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
#include "synchronicityWidget.h"
#include "synchronicityChartView.h"
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

Crit3DSynchronicityWidget::Crit3DSynchronicityWidget(Crit3DMeteoPointsDbHandler* meteoPointsDbHandler, QList<Crit3DMeteoPoint> meteoPointsNearDistanceList, QList<std::string> sortedId, std::vector<float> distanceId,
                                                         QDate firstDaily, QDate lastDaily, Crit3DMeteoSettings *meteoSettings, QSettings *settings, Crit3DClimateParameters *climateParameters, Crit3DQuality *quality)
:meteoPointsDbHandler(meteoPointsDbHandler), meteoPointsNearDistanceList(meteoPointsNearDistanceList), sortedId(sortedId), distanceId(distanceId), firstDaily(firstDaily),
  lastDaily(lastDaily), meteoSettings(meteoSettings), settings(settings), climateParameters(climateParameters), quality(quality)
{
    this->setWindowTitle("Synchronicity analysis Id:"+QString::fromStdString(meteoPointsNearDistanceList[0].id)+" "+QString::fromStdString(meteoPointsNearDistanceList[0].name));
    this->resize(1240, 700);
    this->setSizePolicy(QSizePolicy::Expanding, QSizePolicy::Expanding);
    this->setAttribute(Qt::WA_DeleteOnClose);

    idPointsJointed << meteoPointsNearDistanceList[0].id;

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
    variable.addItem("DAILY_TMIN");
    variable.addItem("DAILY_TMAX");
    variable.addItem("DAILY_PREC");

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
    for (int i = 1; i<meteoPointsNearDistanceList.size(); i++)
    {
        jointStationsList.addItem(QString::fromStdString(meteoPointsNearDistanceList[i].id)+" "+QString::fromStdString(meteoPointsNearDistanceList[i].name));
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
    addStationFoundButton.setText("➡");
    deleteStationFoundButton.setText("⬅");
    addStationFoundButton.setEnabled(false);
    deleteStationFoundButton.setEnabled(false);
    arrowLayout->addWidget(&addStationFoundButton);
    arrowLayout->addWidget(&deleteStationFoundButton);
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

    synchronicityChartView = new SynchronicityChartView();
    synchronicityChartView->setMinimumWidth(this->width()*2/3);
    plotLayout->addWidget(synchronicityChartView);

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
    connect(changeHomogeneityLeftAxis, &QAction::triggered, this, &Crit3DSynchronicityWidget::on_actionChangeLeftAxis);
    connect(exportAnnualGraph, &QAction::triggered, this, &Crit3DSynchronicityWidget::on_actionExportAnnualGraph);
    connect(exportHomogeneityGraph, &QAction::triggered, this, &Crit3DSynchronicityWidget::on_actionExportHomogeneityGraph);
    connect(exportAnnualData, &QAction::triggered, this, &Crit3DSynchronicityWidget::on_actionExportAnnualData);
    connect(exportHomogeneityData, &QAction::triggered, this, &Crit3DSynchronicityWidget::on_actionExportHomogeneityData);

    plotAnnualSeries();

    show();
}


Crit3DSynchronicityWidget::~Crit3DSynchronicityWidget()
{

}

void Crit3DSynchronicityWidget::closeEvent(QCloseEvent *event)
{
    event->accept();
}

/*
void Crit3DSynchronicityWidget::changeVar(const QString varName)
{
    myVar = getKeyMeteoVarMeteoMap(MapDailyMeteoVarToString, varName.toStdString());
    listFoundStations.clear();
    listSelectedStations.clear();
    stationsTable.clearContents();
    resultLabel.clear();
    annualSeriesChartView->setYTitle(QString::fromStdString(getUnitFromVariable(myVar)));
    execute.setEnabled(false);
    synchronicityChartView->clearSNHTSeries();
    synchronicityChartView->clearCraddockSeries();
    plotAnnualSeries();
}

void Crit3DSynchronicityWidget::changeYears()
{
    listFoundStations.clear();
    listSelectedStations.clear();
    stationsTable.clearContents();
    resultLabel.clear();
    execute.setEnabled(false);
    synchronicityChartView->clearSNHTSeries();
    synchronicityChartView->clearCraddockSeries();
    plotAnnualSeries();
}

void Crit3DSynchronicityWidget::changeMethod(const QString methodName)
{
    synchronicityChartView->clearSNHTSeries();
    synchronicityChartView->clearCraddockSeries();
    resultLabel.clear();
    if (execute.isEnabled())
    {
        executeClicked();
    }
}

void Crit3DSynchronicityWidget::updateYears()
{

    yearFrom.blockSignals(true);
    yearTo.blockSignals(true);

    lastDaily = meteoPointsDbHandler->getLastDate(daily, meteoPointsNearDistanceList[0].id).date();
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

    yearFrom.blockSignals(true);
    yearTo.blockSignals(true);
    plotAnnualSeries();
}

void Crit3DSynchronicityWidget::on_actionChangeLeftAxis()
{
    DialogChangeAxis changeAxisDialog(true);
    if (changeAxisDialog.result() == QDialog::Accepted)
    {
        synchronicityChartView->setYmax(changeAxisDialog.getMaxVal());
        synchronicityChartView->setYmin(changeAxisDialog.getMinVal());
    }
}


void Crit3DSynchronicityWidget::on_actionExportHomogeneityGraph()
{

    QString fileName = QFileDialog::getSaveFileName(this, tr("Save current graph"), "", tr("png files (*.png)"));

    if (fileName != "")
    {
        const auto dpr = synchronicityChartView->devicePixelRatioF();
        QPixmap buffer(synchronicityChartView->width() * dpr, synchronicityChartView->height() * dpr);
        buffer.setDevicePixelRatio(dpr);
        buffer.fill(Qt::transparent);

        QPainter *paint = new QPainter(&buffer);
        paint->setPen(*(new QColor(255,34,255,255)));
        synchronicityChartView->render(paint);

        QFile file(fileName);
        file.open(QIODevice::WriteOnly);
        buffer.save(&file, "PNG");
    }
}

void Crit3DSynchronicityWidget::on_actionExportAnnualGraph()
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

void Crit3DSynchronicityWidget::on_actionExportHomogeneityData()
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
            QList<QPointF> dataPoints = synchronicityChartView->exportSNHTValues();
            for (int i = 0; i < dataPoints.size(); i++)
            {
                myStream << dataPoints[i].toPoint().x() << "," << dataPoints[i].y() << "\n";
            }
        }
        else if (method.currentText() == "CRADDOCK")
        {
            QList<QString> refNames;
            QList<QList<QPointF>> pointsAllSeries = synchronicityChartView->exportCraddockValues(refNames);
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

void Crit3DSynchronicityWidget::on_actionExportAnnualData()
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
*/

