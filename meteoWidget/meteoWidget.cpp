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


#include "meteoWidget.h"
#include "dialogSelectVar.h"
#include "dialogRemoveStation.h"
#include "dialogMeteoTable.h"
#include "dialogChangeAxis.h"
#include "dialogVariableToSum.h"
#include "utilities.h"
#include "commonConstants.h"
#include "formInfo.h"

#include <QLayout>
#include <QDate>


Crit3DMeteoWidget::Crit3DMeteoWidget(bool isGrid_, QString projectPath, Crit3DMeteoSettings* meteoSettings_)
{
    isInitialized = false;
    meteoSettings = meteoSettings_;
    isGrid = isGrid_;
    isEnsemble = false;
    nrMembers = NODATA;
    maxEnsembleBar = -1;
    maxEnsembleLine = NODATA;
    minEnsembleLine = -NODATA;

    if (isGrid)
    {
        this->setWindowTitle("Grid");
    }
    else
    {
        this->setWindowTitle("Point");
    }

    this->resize(1240, 700);
    this->setAttribute(Qt::WA_DeleteOnClose);
    currentFreq = noFrequency;

    QDate noDate = QDate(1800,1,1);
    currentDate = noDate;
    firstDailyDate = noDate;
    firstHourlyDate = noDate;
    lastDailyDate = noDate;
    lastHourlyDate = noDate;

    isLine = false;
    isBar = false;
    QVector<QLineSeries*> vectorLine;
    QVector<QBarSet*> vectorBarSet;

    QString csvPath, defaultPath, stylesPath;
    if (! projectPath.isEmpty())
    {
        defaultPath = projectPath + "SETTINGS/Crit3DPlotDefault.csv";
        stylesPath = projectPath + "SETTINGS/Crit3DPlotStyles.csv";
        if (QFileInfo::exists(defaultPath) == false)
        {
            defaultPath = "";
        }
        if (QFileInfo::exists(stylesPath) == false)
        {
            stylesPath = "";
        }
    }
    if (defaultPath.isEmpty() || stylesPath.isEmpty())
    {
        if (searchDataPath(&csvPath))
        {
            if (defaultPath.isEmpty())
            {
                defaultPath = csvPath + "SETTINGS/Crit3DPlotDefault.csv";
            }
            if (stylesPath.isEmpty())
            {
                stylesPath = csvPath + "SETTINGS/Crit3DPlotStyles.csv";
            }
        }
    }

    // read Crit3DPlotDefault and fill MapCSVDefault
    int CSVRequiredInfo = 3;
    QFile fileDefaultGraph(defaultPath);
    if ( !fileDefaultGraph.open(QFile::ReadOnly | QFile::Text) )
    {
        qDebug() << "File not exists";
        QMessageBox::information(nullptr, "Warning", "Missing Crit3DPlotDefault.csv");
        return;
    }

    currentVariables.clear();
    QTextStream in(&fileDefaultGraph);
    in.readLine(); //skip first line
    while (!in.atEnd())
    {
        QString line = in.readLine();
        QList<QString> items = line.split(",");
        if (items.size() < CSVRequiredInfo)
        {
            qDebug() << "invalid format CSV, missing data";
            break;
        }
        QString key = items[0];
        items.removeFirst();
        if (key.isEmpty() || items[0].isEmpty())
        {
            qDebug() << "invalid format CSV, missing data";
            break;
        }
        if (key.contains("DAILY"))
        {
            currentFreq = daily;
        }
        else
        {
            currentFreq = hourly;
        }
        MapCSVDefault.insert(key,items);
        if (items[0] == "line")
        {
            auto search = MapDailyMeteoVar.find(key.toStdString());
            auto searchHourly = MapHourlyMeteoVar.find(key.toStdString());
            if (search != MapDailyMeteoVar.end() || searchHourly != MapHourlyMeteoVar.end())
            {
                isLine = true;
                QLineSeries* line = new QLineSeries();
                line->setName(key);
                line->setColor(QColor(items[1]));
                vectorLine.append(line);
                currentVariables.append(key);
                nameLines.append(key);
                colorLines.append(QColor(items[1]));
            }
        }
        else if (items[0] == "bar")
        {
            auto search = MapDailyMeteoVar.find(key.toStdString());
            auto searchHourly = MapHourlyMeteoVar.find(key.toStdString());
            if (search != MapDailyMeteoVar.end() || searchHourly != MapHourlyMeteoVar.end())
            {
                isBar = true;
                QBarSet* set = new QBarSet(key);
                set->setColor(QColor(items[1]));
                set->setBorderColor(QColor(items[1]));
                vectorBarSet.append(set);
                currentVariables.append(key);
                nameBar.append(key);
                colorBar.append(QColor(items[1]));
            }
        }
        else
        {
            qDebug() << "invalid format CSV, missing line or bar";
            break;
        }
    }

    // check valid data
    int dailyVar = 0;
    int hourlyVar = 0;
    for (int i = 0; i < currentVariables.size(); i++)
    {
        if (currentVariables[i].contains("DAILY"))
        {
            dailyVar = dailyVar+1;
        }
        else
        {
            hourlyVar = hourlyVar+1;
        }
    }
    if (currentVariables.isEmpty() || (dailyVar != 0 && hourlyVar != 0))
    {
        QMessageBox::information(nullptr, "Warning", "Wrong variables in Crit3DPlotDefault.csv");
        currentFreq = noFrequency;
        currentVariables.clear();
        nameLines.clear();
        nameBar.clear();
        MapCSVDefault.clear();
        isLine = false;
        isBar = false;
        return;
    }

    if (isLine)
    {
        lineSeries.append(vectorLine);
    }
    if (isBar)
    {
        setVector.append(vectorBarSet);
    }

    // read Crit3DPlotStyles and fill MapCSVStyles
    QFile fileStylesGraph(stylesPath);
    if ( !fileStylesGraph.open(QFile::ReadOnly | QFile::Text) ) {
        QMessageBox::information(nullptr, "Error", "Missing Crit3DPlotStyles.csv");
        qDebug() << "File not exists";
        return;
    }
    else
    {
        QTextStream in(&fileStylesGraph);
        in.readLine(); //skip first line
        while (!in.atEnd())
        {
            QString line = in.readLine();
            QList<QString> items = line.split(",");
            if (items.size() < CSVRequiredInfo)
            {
                QMessageBox::information(nullptr, "Error", "invalid format Crit3DPlotStyles.csv");
                qDebug() << "invalid format CSV, missing data";
            }
            QString key = items[0];
            items.removeFirst();
            if (key.isEmpty() || items[0].isEmpty())
            {
                QMessageBox::information(nullptr, "Error", "invalid format Crit3DPlotStyles.csv");
                qDebug() << "invalid format CSV, missing data";
            }
            MapCSVStyles.insert(key,items);
        }
    }

    // layout
    QVBoxLayout *mainLayout = new QVBoxLayout();
    QGroupBox *horizontalGroupBox = new QGroupBox();
    QHBoxLayout *buttonLayout = new QHBoxLayout;
    QVBoxLayout *plotLayout = new QVBoxLayout;

    dailyButton = new QPushButton(tr("daily"));
    hourlyButton = new QPushButton(tr("hourly"));
    monthlyButton = new QPushButton(tr("monthly"));
    addVarButton = new QPushButton(tr("+/- var"));
    tableButton = new QPushButton(tr("view table"));
    redrawButton = new QPushButton(tr("redraw"));
    shiftPreviousButton = new QPushButton(tr("<"));
    shiftFollowingButton = new QPushButton(tr(">"));
    QLabel *labelFirstDate = new QLabel(tr("Start Date: "));
    QLabel *labelEndDate = new QLabel(tr("End Date: "));
    firstDate = new QDateTimeEdit(currentDate);
    lastDate = new QDateTimeEdit(currentDate);
    dailyButton->setMaximumWidth(100);
    hourlyButton->setMaximumWidth(100);
    monthlyButton->setMaximumWidth(100);
    addVarButton->setMaximumWidth(100);
    tableButton->setMaximumWidth(100);
    redrawButton->setMaximumWidth(100);
    shiftPreviousButton->setMaximumWidth(30);
    shiftFollowingButton->setMaximumWidth(30);

    firstDate->setDisplayFormat("yyyy-MM-dd");
    firstDate->setCalendarPopup(true);
    lastDate->setDisplayFormat("yyyy-MM-dd");
    lastDate->setCalendarPopup(true);
    firstDate->setMaximumWidth(120);
    lastDate->setMaximumWidth(120);
    firstDate->setMinimumWidth(firstDate->width()-firstDate->width()*0.3);
    lastDate->setMinimumWidth(lastDate->width()-lastDate->width()*0.3);

    if (currentFreq == daily || currentFreq == noFrequency)
    {
        dailyButton->setEnabled(false);
        hourlyButton->setEnabled(true);
        monthlyButton->setEnabled(true);
    }
    else if (currentFreq == hourly)
    {
        hourlyButton->setEnabled(false);
        dailyButton->setEnabled(true);
        monthlyButton->setEnabled(true);
    }
    else if (currentFreq == monthly)
    {
        monthlyButton->setEnabled(false);
        dailyButton->setEnabled(true);
        hourlyButton->setEnabled(true);
    }

    buttonLayout->addWidget(monthlyButton);
    buttonLayout->addWidget(dailyButton);
    buttonLayout->addWidget(hourlyButton);
    buttonLayout->addWidget(addVarButton);
    buttonLayout->addWidget(labelFirstDate);
    buttonLayout->addWidget(firstDate);
    buttonLayout->addWidget(labelEndDate);
    buttonLayout->addWidget(lastDate);
    buttonLayout->addWidget(shiftPreviousButton);
    buttonLayout->addWidget(shiftFollowingButton);
    buttonLayout->addWidget(redrawButton);
    buttonLayout->addWidget(tableButton);
    buttonLayout->setAlignment(Qt::AlignLeft);
    chart = new QChart();
    chartView = new QChartView(chart);
    chartView->setChart(chart);

    axisX = new QBarCategoryAxis();
    axisXvirtual = new QBarCategoryAxis();

    axisY = new QValueAxis();
    axisYdx = new QValueAxis();

    axisX->setTitleText("Date");
    axisXvirtual->setTitleText("Date");
    axisXvirtual->setGridLineVisible(false);

    axisY->setRange(0,30);
    axisY->setGridLineVisible(false);

    axisYdx->setRange(0,8);
    axisYdx->setGridLineVisible(false);

    chart->addAxis(axisX, Qt::AlignBottom);
    chart->addAxis(axisXvirtual, Qt::AlignBottom);
    chart->addAxis(axisY, Qt::AlignLeft);
    chart->addAxis(axisYdx, Qt::AlignRight);

    chart->legend()->setVisible(true);
    chart->legend()->setAlignment(Qt::AlignBottom);
    chartView->setRenderHint(QPainter::Antialiasing);
    axisX->hide();

    m_tooltip = new Callout(chart);
    m_tooltip->hide();

    // menu
    QMenuBar* menuBar = new QMenuBar();
    QMenu *editMenu = new QMenu("Edit");
    QMenu *viewMenu = new QMenu("View");

    menuBar->addMenu(editMenu);
    menuBar->addMenu(viewMenu);
    mainLayout->setMenuBar(menuBar);

    QAction* changeLeftAxis = new QAction(tr("&Change axis left"), this);
    QAction* changeRightAxis = new QAction(tr("&Change axis right"), this);
    QAction* exportGraph = new QAction(tr("&Export graph"), this);
    QAction* removeStation = new QAction(tr("&Remove stations"), this);

    editMenu->addAction(changeLeftAxis);
    editMenu->addAction(changeRightAxis);
    editMenu->addAction(exportGraph);
    editMenu->addAction(removeStation);

    QAction* infoPoint = new QAction(tr("&Info meteo point"), this);
    QAction* dataAvailability = new QAction(tr("&Data availability"), this);
    QAction* dataSum = new QAction(tr("&Sum"), this);

    viewMenu->addAction(infoPoint);
    viewMenu->addAction(dataAvailability);
    viewMenu->addAction(dataSum);

    connect(addVarButton, &QPushButton::clicked, [=](){ showVar(); });
    connect(monthlyButton, &QPushButton::clicked, [=](){ showMonthlyGraph(); });
    connect(dailyButton, &QPushButton::clicked, [=](){ showDailyGraph(); });
    connect(hourlyButton, &QPushButton::clicked, [=](){ showHourlyGraph(); });
    connect(tableButton, &QPushButton::clicked, [=](){ showTable(); });
    connect(redrawButton, &QPushButton::clicked, [=](){ redraw(); });
    connect(shiftPreviousButton, &QPushButton::clicked, [=](){ shiftPrevious(); });
    connect(shiftFollowingButton, &QPushButton::clicked, [=](){ shiftFollowing(); });
    connect(changeLeftAxis, &QAction::triggered, this, &Crit3DMeteoWidget::on_actionChangeLeftAxis);
    connect(changeRightAxis, &QAction::triggered, this, &Crit3DMeteoWidget::on_actionChangeRightAxis);
    connect(exportGraph, &QAction::triggered, this, &Crit3DMeteoWidget::on_actionExportGraph);
    connect(removeStation, &QAction::triggered, this, &Crit3DMeteoWidget::on_actionRemoveStation);
    connect(infoPoint, &QAction::triggered, this, &Crit3DMeteoWidget::on_actionInfoPoint);
    connect(dataAvailability, &QAction::triggered, this, &Crit3DMeteoWidget::on_actionDataAvailability);
    connect(dataSum, &QAction::triggered, this, &Crit3DMeteoWidget::on_actionDataSum);

    plotLayout->addWidget(chartView);
    horizontalGroupBox->setLayout(buttonLayout);
    mainLayout->addWidget(horizontalGroupBox);
    mainLayout->addLayout(plotLayout);

    QStatusBar* statusBar = new QStatusBar();
    mainLayout->addWidget(statusBar);
    setLayout(mainLayout);

    isInitialized = true;
}


Crit3DMeteoWidget::~Crit3DMeteoWidget()
{

}


void Crit3DMeteoWidget::setDateIntervalDaily(QDate firstDate, QDate lastDate)
{
    firstDailyDate = firstDate;
    lastDailyDate = lastDate;
}


void Crit3DMeteoWidget::setDateIntervalHourly(QDate firstDate, QDate lastDate)
{
    firstHourlyDate = firstDate;
    lastHourlyDate = lastDate;
}


// search bigger data interval to show between meteoPoints
void Crit3DMeteoWidget::updateTimeRange()
{
    for (int i = 0; i < meteoPoints.size(); i++)
    {
        QDate myDailyDateFirst;
        myDailyDateFirst.setDate(meteoPoints[i].obsDataD[0].date.year,
                                 meteoPoints[i].obsDataD[0].date.month,
                                 meteoPoints[i].obsDataD[0].date.day);
        QDate myDailyDateLast = myDailyDateFirst.addDays(meteoPoints[i].nrObsDataDaysD-1);

        // updates daily range
        if (myDailyDateFirst.isValid() &&
            (! firstDailyDate.isValid() || firstDailyDate.year() == 1800 || myDailyDateFirst < firstDailyDate))
        {
            firstDailyDate = myDailyDateFirst;
        }
        if (myDailyDateLast.isValid() &&
            (! lastDailyDate.isValid() || lastDailyDate.year() == 1800 || myDailyDateLast > lastDailyDate))
        {
            lastDailyDate = myDailyDateLast;
        }

        QDate myHourlyDateFirst;
        myHourlyDateFirst.setDate(meteoPoints[i].getMeteoPointHourlyValuesDate(0).year,
                                  meteoPoints[i].getMeteoPointHourlyValuesDate(0).month,
                                  meteoPoints[i].getMeteoPointHourlyValuesDate(0).day);
        QDate myHourlyDateLast = myHourlyDateFirst.addDays(meteoPoints[i].nrObsDataDaysH-1);

        // updates hourly range
        if (myHourlyDateFirst.isValid() &&
            (! firstHourlyDate.isValid() || firstHourlyDate.year() == 1800 || myHourlyDateFirst < firstHourlyDate))
        {
            firstHourlyDate = myHourlyDateFirst;
        }
        if (myHourlyDateLast.isValid() &&
            (! lastHourlyDate.isValid() || lastHourlyDate.year() == 1800 || myHourlyDateLast > lastHourlyDate))
        {
            lastHourlyDate = myHourlyDateLast;
        }
    }
}


void Crit3DMeteoWidget::drawMeteoPoint(Crit3DMeteoPoint mp, bool isAppend)
{
    if (! isInitialized) return;

    meteoPoints.append(mp);
    updateTimeRange();

    if (! isAppend)
    {
        // set dates
        firstDate->blockSignals(true);
        lastDate->blockSignals(true);

        lastDate->setDate(currentDate);

        // draw period (31 days for daily, 3 days for hourly)
        if (currentFreq == daily)
        {
            firstDate->setDate(currentDate.addDays(-30));
        }
        else if (currentFreq == hourly)
        {
            firstDate->setDate(currentDate.addDays(-2));
        }

        firstDate->blockSignals(false);
        lastDate->blockSignals(false);
    }

    redraw();
    show();
}

void Crit3DMeteoWidget::addMeteoPointsEnsemble(Crit3DMeteoPoint mp)
{
    meteoPointsEnsemble.append(mp);
}


QString getFormattedLabel(QString pointName, QString varName)
{
    QString label;

    QList<QString> words = pointName.split(' ');

    if (words.size() == 1)
    {
        int size = std::min(int(words[0].size()), 12);
        pointName = words[0].left(size);
    }
    else
    {
        int last = words.size() - 1;
        int size1 = std::min(int(words[0].size()), 7);
        int size2 = std::min(int(words[last].size()), 12-size1);
        pointName = words[0].left(size1) + '_' + words[last].left(size2);
    }

    if (varName.contains("DAILY_"))
    {
        varName = varName.remove("DAILY_");
    }

    label = pointName + '_' + varName;

    return label;
}


void Crit3DMeteoWidget::drawEnsemble()
{
    if (! isInitialized || meteoPointsEnsemble.isEmpty() || meteoPointsEnsemble.size() != nrMembers)
        return;

    firstDate->blockSignals(true);
    lastDate->blockSignals(true);

    // set date
    QDate myDailyDateFirst;
    QDate myDailyDateLast;
    QDate myHourlyDateFirst;
    QDate myHourlyDateLast;
    myDailyDateFirst.setDate(meteoPointsEnsemble[0].obsDataD[0].date.year, meteoPointsEnsemble[0].obsDataD[0].date.month, meteoPointsEnsemble[0].obsDataD[0].date.day);
    myDailyDateLast = myDailyDateFirst.addDays(meteoPointsEnsemble[0].nrObsDataDaysD-1);
    if (myDailyDateFirst.isValid() && myDailyDateFirst < firstDailyDate)
    {
        firstDailyDate = myDailyDateFirst;
    }
    if (myDailyDateLast.isValid() && myDailyDateLast > lastDailyDate)
    {
        lastDailyDate = myDailyDateLast;
    }
    myHourlyDateFirst.setDate(meteoPointsEnsemble[0].getMeteoPointHourlyValuesDate(0).year, meteoPointsEnsemble[0].getMeteoPointHourlyValuesDate(0).month,
                              meteoPointsEnsemble[0].getMeteoPointHourlyValuesDate(0).day);
    myHourlyDateLast = myHourlyDateFirst.addDays(meteoPointsEnsemble[0].nrObsDataDaysH-1);
    if (myHourlyDateFirst.isValid() && myHourlyDateFirst < firstHourlyDate)
    {
        firstHourlyDate = myHourlyDateFirst;
    }
    if (myHourlyDateLast.isValid() && myHourlyDateLast > lastHourlyDate)
    {
        lastHourlyDate = myHourlyDateLast;
    }

    lastDate->setDate(currentDate);

    // draw period (31 days for daily, 3 days for hourly)
    if (currentFreq == daily)
        firstDate->setDate(currentDate.addDays(-30));
    else if (currentFreq == hourly)
        firstDate->setDate(currentDate.addDays(-2));

    redraw();

    firstDate->blockSignals(false);
    lastDate->blockSignals(false);

    show();
}


void Crit3DMeteoWidget::resetValues()
{
    int nMeteoPoints = meteoPoints.size();
    // clear prev series values
    if (!lineSeries.isEmpty())
    {
        for (int mp = 0; mp < lineSeries.size(); mp++)
        {
            for (int i = 0; i < lineSeries[mp].size(); i++)
            {
                lineSeries[mp][i]->clear();
                if (chart->series().contains(lineSeries[mp][i]))
                {
                    chart->removeSeries(lineSeries[mp][i]);
                }
            }
            lineSeries[mp].clear();
        }
        lineSeries.clear();
    }
    if (! barSeries.isEmpty())
    {
        for (int mp = 0; mp < barSeries.size(); mp++)
        {
            setVector[mp].clear();
            barSeries[mp]->clear();
            if (chart->series().contains(barSeries[mp]))
            {
                chart->removeSeries(barSeries[mp]);
            }
        }
        barSeries.clear();
        setVector.clear();
    }

    delete m_tooltip;

    if (isLine)
    {
        QVector<QLineSeries*> vectorLine;
        // add lineSeries elements for each mp, clone lineSeries[0]
        for (int mp = 0; mp < nMeteoPoints; mp++)
        {
            QString pointName = QString::fromStdString(meteoPoints[mp].name);
            vectorLine.clear();

            for (int i = 0; i < nameLines.size(); i++)
            {
                QLineSeries* line = new QLineSeries();
                line->setName(getFormattedLabel(pointName, nameLines[i]));
                QColor lineColor = colorLines[i];
                if (nMeteoPoints == 1)
                {
                    lineColor.setAlpha(255);
                }
                else
                {
                    lineColor.setAlpha( 255-(mp*(150/(nMeteoPoints-1))));
                }
                line->setColor(lineColor);
                vectorLine.append(line);
            }
            if (vectorLine.size() != 0)
            {
                lineSeries.append(vectorLine);
            }
        }
    }

    // possono essere veri entrambi
    if (isBar)
    {
        QVector<QBarSet*> vectorBarSet;
        // add vectorBarSet elements for each mp
        for (int mp = 0; mp < nMeteoPoints; mp++)
        {
            QString pointName = QString::fromStdString(meteoPoints[mp].name);
            vectorBarSet.clear();

            for (int i = 0; i < nameBar.size(); i++)
            {
                QBarSet* bar = new QBarSet(getFormattedLabel(pointName, nameBar[i]));
                bar->setLabel(getFormattedLabel(pointName, nameBar[i]));

                QColor barColor = colorBar[i];
                if (meteoPointsEnsemble.size() == 0)
                {
                    if (nMeteoPoints == 1)
                    {
                        barColor.setAlpha(255);
                    }
                    else
                    {
                        barColor.setAlpha( 255-(mp*(150/(nMeteoPoints-1))) );
                    }
                    bar->setColor(barColor);
                }
                else
                {
                    bar->setColor(Qt::transparent);
                }
                bar->setBorderColor(barColor);
                vectorBarSet.append(bar);
            }
            if (vectorBarSet.size() != 0)
            {
                setVector.append(vectorBarSet);
            }
        }
    }
}

void Crit3DMeteoWidget::resetEnsembleValues()
{
    // clear prev series values
    ensembleSet.clear();
    for (int i = 0; i < ensembleSeries.size(); i++)
    {
        if (chart->series().contains(ensembleSeries[i]))
        {
            chart->removeSeries(ensembleSeries[i]);
        }
    }
    ensembleSeries.clear();
    categories.clear();
    categoriesVirtual.clear();
    maxEnsembleBar = 0;
    maxEnsembleLine = NODATA;
    minEnsembleLine = -NODATA;
}


void Crit3DMeteoWidget::drawEnsembleDailyVar()
{
    if (! isInitialized) return;

    FormInfo formInfo;
    formInfo.showInfo("Draw daily data...");

    firstDate->blockSignals(true);
    lastDate->blockSignals(true);

    Crit3DDate myDate;
    int nDays = 0;
    maxEnsembleBar = -1;
    maxEnsembleLine = NODATA;
    minEnsembleLine = -NODATA;

    Crit3DDate firstCrit3DDate = getCrit3DDate(firstDate->date());
    Crit3DDate lastCrit3DDate = getCrit3DDate(lastDate->date());
    nDays = firstCrit3DDate.daysTo(lastCrit3DDate)+1;

    categories.clear();
    categoriesVirtual.clear();

    // virtual x axis
    int nrIntervals;
    if (nDays <= 12)
    {
        nrIntervals = nDays;
    }
    else if (nDays <= 45)
    {
        nrIntervals = nDays/3;
    }
    else
    {
        nrIntervals = 12;
    }
    double step = double(nDays) / double(nrIntervals);
    double nextIndex = step / 2 - 0.5;
    for (int day = 0; day < nDays; day++)
    {
        myDate = firstCrit3DDate.addDays(day);
        if (day == round(nextIndex))
        {
            categoriesVirtual.append(getQDate(myDate).toString("MMM dd <br> yyyy"));
            nextIndex += step;
        }
    }

    for (int day = 0; day < nDays; day++)
    {
        categories.append(QString::number(day));
    }

    QList<double> sortedList;
    QList<QBoxSet*> listBoxSet;

    if (isLine)
    {
        for (int i = 0; i < nameLines.size(); i++)
        {
            listBoxSet.clear();
            QBoxPlotSeries *series = new QBoxPlotSeries();
            series->setName(QString::fromStdString(meteoPointsEnsemble[0].id)+"_"+ QString::fromStdString(meteoPointsEnsemble[0].name)+"_"+nameLines[i]+"_Ensemble");
            series->setBrush(colorLines[i]);
            for (int day = 0; day < nDays; day++)
            {
                sortedList.clear();
                myDate = firstCrit3DDate.addDays(day);
                meteoVariable meteoVar = getMeteoVar(nameLines[i].toStdString());
                if (meteoVar == noMeteoVar)
                {
                    continue;
                }

                for (int mp=0; mp<nrMembers;mp++)
                {
                    double value = meteoPointsEnsemble[mp].getMeteoPointValueD(myDate, meteoVar, meteoSettings);
                    if (value != NODATA)
                    {
                        sortedList.append(value);
                        if (value > maxEnsembleLine)
                        {
                            maxEnsembleLine = value;
                        }
                        if (value < minEnsembleLine)
                        {
                            minEnsembleLine = value;
                        }
                    }
                }
                QBoxSet *box = new QBoxSet();
                if (!sortedList.isEmpty())
                {
                    std::sort(sortedList.begin(), sortedList.end());
                    int count = sortedList.count();
                    box->setValue(QBoxSet::LowerExtreme, sortedList.first());
                    box->setValue(QBoxSet::UpperExtreme, sortedList.last());
                    box->setValue(QBoxSet::Median, findMedian(sortedList, 0, count));
                    box->setValue(QBoxSet::LowerQuartile, findMedian(sortedList, 0, count / 2));
                    box->setValue(QBoxSet::UpperQuartile, findMedian(sortedList, count / 2 + (count % 2), count));
                }
                else
                {
                    box->setValue(QBoxSet::LowerExtreme, 0);
                    box->setValue(QBoxSet::UpperExtreme, 0);
                    box->setValue(QBoxSet::Median, 0);
                    box->setValue(QBoxSet::LowerQuartile, 0);
                    box->setValue(QBoxSet::UpperQuartile, 0);
                }
                box->setBrush(colorLines[i]);
                listBoxSet.append(box);
                ensembleSet.append(listBoxSet);
            }
            if(!ensembleSet.isEmpty())
            {
                series->append(ensembleSet.last());
                ensembleSeries.append(series);
                chart->addSeries(series);
                series->attachAxis(axisX);
                series->attachAxis(axisY);
            }
        }
    }

    if (isBar)
    {
        for (int i = 0; i < nameBar.size(); i++)
        {
            listBoxSet.clear();
            QBoxPlotSeries *series = new QBoxPlotSeries();
            series->setName(QString::fromStdString(meteoPointsEnsemble[0].id)+"_"+ QString::fromStdString(meteoPointsEnsemble[0].name)+"_"+nameBar[i]+"_Ensemble");
            series->setBrush(colorBar[i]);
            for (int day = 0; day < nDays; day++)
            {
                myDate = firstCrit3DDate.addDays(day);
                meteoVariable meteoVar = getMeteoVar(nameBar[i].toStdString());
                if (meteoVar == noMeteoVar)
                {
                    continue;
                }
                sortedList.clear();
                for (int mp=0; mp<nrMembers;mp++)
                {
                    double value = meteoPointsEnsemble[mp].getMeteoPointValueD(myDate, meteoVar, meteoSettings);
                    if (value != NODATA)
                    {
                        sortedList.append(value);
                        if (value > maxEnsembleBar)
                        {
                            maxEnsembleBar = value;
                        }
                    }
                }
                QBoxSet *box = new QBoxSet();
                if (!sortedList.isEmpty())
                {
                    std::sort(sortedList.begin(), sortedList.end());
                    int count = sortedList.count();
                    box->setValue(QBoxSet::LowerExtreme, sortedList.first());
                    box->setValue(QBoxSet::UpperExtreme, sortedList.last());
                    box->setValue(QBoxSet::Median, findMedian(sortedList, 0, count));
                    box->setValue(QBoxSet::LowerQuartile, findMedian(sortedList, 0, count / 2));
                    box->setValue(QBoxSet::UpperQuartile, findMedian(sortedList, count / 2 + (count % 2), count));
                }
                else
                {
                    box->setValue(QBoxSet::LowerExtreme, 0);
                    box->setValue(QBoxSet::UpperExtreme, 0);
                    box->setValue(QBoxSet::Median, 0);
                    box->setValue(QBoxSet::LowerQuartile, 0);
                    box->setValue(QBoxSet::UpperQuartile, 0);
                }
                box->setBrush(colorBar[i]);
                listBoxSet.append(box);
                ensembleSet.append(listBoxSet);
            }
            if(!ensembleSet.isEmpty())
            {
                series->append(ensembleSet.last());
                ensembleSeries.append(series);
                chart->addSeries(series);
                series->attachAxis(axisX);
                series->attachAxis(axisYdx);
            }
        }
    }

    if(isLine)
    {
        if (maxEnsembleLine == NODATA && minEnsembleLine == -NODATA)
        {
            axisY->setVisible(false);
        }
        else
        {
            axisY->setVisible(true);
            axisY->setMax(maxEnsembleLine);
            axisY->setMin(minEnsembleLine);
            if (axisY->max() == axisY->min())
            {
                axisY->setRange(axisY->min()-axisY->min()/100, axisY->max()+axisY->max()/100);
            }
        }
    }
    else
    {
        axisY->setVisible(false);
    }

    if (isBar)
    {
        if (maxEnsembleBar == -1)
        {
            axisYdx->setVisible(false);
        }
        else
        {
            axisYdx->setVisible(true);
            if (maxEnsembleBar != 0)
            {
                axisYdx->setRange(0,maxEnsembleBar);
            }
            else
            {
                axisYdx->setRange(0,1);
            }
        }
    }
    else
    {
        axisYdx->setVisible(false);
    }

    axisX->setCategories(categories);
    axisXvirtual->setCategories(categoriesVirtual);
    axisXvirtual->setGridLineVisible(false);

    firstDate->blockSignals(false);
    lastDate->blockSignals(false);

    formInfo.close();
}


void Crit3DMeteoWidget::drawDailyVar()
{
    if (! isInitialized) return;

    FormInfo formInfo;
    formInfo.showInfo("Draw daily data...");

    firstDate->blockSignals(true);
    lastDate->blockSignals(true);

    Crit3DDate myDate;
    int nDays = 0;
    double maxBar = -1;
    double maxLine = NODATA;
    double minLine = -NODATA;

    Crit3DDate firstCrit3DDate = getCrit3DDate(firstDate->date());
    Crit3DDate lastCrit3DDate = getCrit3DDate(lastDate->date());
    nDays = firstCrit3DDate.daysTo(lastCrit3DDate)+1;

    categories.clear();
    categoriesVirtual.clear();
    m_tooltip = new Callout(chart);
    m_tooltip->hide();

    // virtual x axis
    int nrIntervals;
    if (nDays <= 12)
    {
        nrIntervals = nDays;
    }
    else if (nDays <= 45)
    {
        nrIntervals = nDays/3;
    }
    else
    {
        nrIntervals = 12;
    }
    double step = double(nDays) / double(nrIntervals);
    double nextIndex = step / 2 - 0.5;
    for (int day = 0; day < nDays; day++)
    {
        myDate = firstCrit3DDate.addDays(day);
        if (day == round(nextIndex))
        {
            categoriesVirtual.append(getQDate(myDate).toString("MMM dd <br> yyyy"));
            nextIndex += step;
        }
    }

    int nMeteoPoints = meteoPoints.size();
    for (int day = 0; day < nDays; day++)
    {
        myDate = firstCrit3DDate.addDays(day);
        categories.append(QString::number(day));

        for (int mp=0; mp<nMeteoPoints;mp++)
        {
            if (isLine)
            {
                for (int i = 0; i < nameLines.size(); i++)
                {
                    meteoVariable meteoVar = getMeteoVar(nameLines[i].toStdString());
                    if (meteoVar == noMeteoVar)
                    {
                        continue;
                    }
                    double value = meteoPoints[mp].getMeteoPointValueD(myDate, meteoVar, meteoSettings);
                    if (value != NODATA)
                    {
                        lineSeries[mp][i]->append(day, value);
                        if (value > maxLine)
                        {
                            maxLine = value;
                        }
                        if (value < minLine)
                        {
                            minLine = value;
                        }
                    }
                    else
                    {
                        if (meteoPoints[mp].isDateLoadedD(myDate))
                        {
                            lineSeries[mp][i]->append(day, value); // nodata days are not drawed if they are the first of the last day of the serie
                        }
                    }
                }
            }
            if (isBar)
            {
                for (int j = 0; j < nameBar.size(); j++)
                {
                    meteoVariable meteoVar = getMeteoVar(nameBar[j].toStdString());
                    if (meteoVar == noMeteoVar)
                    {
                        continue;
                    }
                    double value = meteoPoints[mp].getMeteoPointValueD(myDate, meteoVar, meteoSettings);
                    if (value != NODATA)
                    {
                        *setVector[mp][j] << value;
                        if (value > maxBar)
                        {
                            maxBar = value;
                        }
                    }
                    else
                    {
                        *setVector[mp][j] << 0;
                    }
                }
            }
        }
    }

    if (isBar)
    {
        for (int mp = 0; mp < nMeteoPoints; mp++)
        {
            QBarSeries* barMpSeries = new QBarSeries();
            for (int i = 0; i < nameBar.size(); i++)
            {
                barMpSeries->append(setVector[mp][i]);
            }
            barSeries.append(barMpSeries);
        }

        for (int mp = 0; mp < nMeteoPoints; mp++)
        {
            connect(barSeries[mp], &QBarSeries::hovered, this, &Crit3DMeteoWidget::tooltipBar);
            if (nameBar.size() != 0)
            {
                chart->addSeries(barSeries[mp]);
                barSeries[mp]->attachAxis(axisX);
                barSeries[mp]->attachAxis(axisYdx);
            }
        }
        if (maxEnsembleBar == -1 && maxBar == -1)
        {
            axisYdx->setVisible(false);
        }
        else
        {
            axisYdx->setVisible(true);
            if (maxEnsembleBar > maxBar)
            {
                axisYdx->setRange(0,maxEnsembleBar);
            }
            else
            {
                axisYdx->setRange(0,maxBar);
            }
            if (axisYdx->max() == axisYdx->min())
            {
                axisYdx->setRange(0,1);
            }
        }
    }
    else
    {
        axisYdx->setVisible(false);
    }

    if (isLine)
    {
        for (int mp=0; mp<nMeteoPoints;mp++)
        {
            if (isLine)
            {
                for (int i = 0; i < nameLines.size(); i++)
                {
                    chart->addSeries(lineSeries[mp][i]);
                    lineSeries[mp][i]->attachAxis(axisX);
                    lineSeries[mp][i]->attachAxis(axisY);
                    connect(lineSeries[mp][i], &QLineSeries::hovered, this, &Crit3DMeteoWidget::tooltipLineSeries);
                }
            }
        }
        if (maxLine == NODATA && minLine == -NODATA && maxEnsembleLine == NODATA && minEnsembleLine == -NODATA)
        {
            axisY->setVisible(false);
        }
        else
        {
            axisY->setVisible(true);
            if (maxEnsembleLine > maxLine)
            {
                axisY->setMax(maxEnsembleLine);
            }
            else
            {
                axisY->setMax(maxLine);
            }

            if (minEnsembleLine < minLine)
            {
                axisY->setMin(minEnsembleLine);
            }
            else
            {
                axisY->setMin(minLine);
            }
            if (axisY->max() == axisY->min())
            {
                axisY->setRange(axisY->min()-axisY->min()/100, axisY->max()+axisY->max()/100);
            }
        }
    }
    else
    {
        axisY->setVisible(false);
    }

    // add minimimum values required
    if (nDays==1)
    {
        categories.append(QString::number(1));
        categoriesVirtual.append(firstDate->date().addDays(1).toString("MMM dd <br> yyyy"));
        for (int mp=0; mp<nMeteoPoints;mp++)
        {
            if (isLine)
            {
                for (int i = 0; i < nameLines.size(); i++)
                {
                    lineSeries[mp][0]->append(1, NODATA);
                }
            }

            if (isBar)
            {
                for (int j = 0; j < nameBar.size(); j++)
                {
                    *setVector[mp][j] << 0;
                }
            }
        }
    }

    for (int mp=0; mp<nMeteoPoints;mp++)
    {
        for (int j = 0; j < nameBar.size(); j++)
        {
            if (nDays < 5)
            {
                setVector[mp][j]->setColor(QColor("transparent"));
            }
            else
            {
                QColor barColor = colorBar[j];
                if (meteoPointsEnsemble.size() == 0)
                {
                    if (nMeteoPoints == 1)
                    {
                        barColor.setAlpha(255);
                    }
                    else
                    {
                        barColor.setAlpha( 255-(mp*(150/(nMeteoPoints-1))) );
                    }
                    setVector[mp][j]->setColor(barColor);
                }
                else
                {
                    setVector[mp][j]->setColor(Qt::transparent);
                }
                setVector[mp][j]->setBorderColor(barColor);
            }
        }
    }

    axisX->setCategories(categories);
    axisXvirtual->setCategories(categoriesVirtual);
    axisXvirtual->setGridLineVisible(false);

    firstDate->blockSignals(false);
    lastDate->blockSignals(false);

    foreach(QLegendMarker* marker, chart->legend()->markers())
    {
        marker->setVisible(true);
        marker->series()->setVisible(true);
        QObject::connect(marker, &QLegendMarker::clicked, this, &Crit3DMeteoWidget::handleMarkerClicked);
    }

    formInfo.close();
}


void Crit3DMeteoWidget::drawHourlyVar()
{
    if (! isInitialized) return;

    categories.clear();
    categoriesVirtual.clear();
    m_tooltip = new Callout(chart);
    m_tooltip->hide();

    if (! firstHourlyDate.isValid() || ! lastHourlyDate.isValid()
        || firstHourlyDate.year() == 1800 || lastHourlyDate.year() == 1800)
    {
        // nothing to draw
        return;
    }

    FormInfo formInfo;
    formInfo.showInfo("Draw hourly data...");

    firstDate->blockSignals(true);
    lastDate->blockSignals(true);

    double maxBar = -1;
    double maxLine = NODATA;
    double minLine = -NODATA;

    int nrDays = firstDate->date().daysTo(lastDate->date())+1;
    int nrValues = nrDays*24;

    // virtual x axis
    int nrIntervals;
    if (nrValues <= 36)
    {
        nrIntervals = nrValues/3;
    }
    else
    {
        nrIntervals = 12;
    }
    double step = double(nrValues) / double(nrIntervals);
    double nextIndex = step / 2 - 0.5;
    long index;

    int nMeteoPoints = meteoPoints.size();
    QDate myDate = firstDate->date();
    Crit3DDate myCrit3DDate;
    QDateTime myDateTime;

    for (int d = 0; d < nrDays; d++)
    {
        myCrit3DDate = getCrit3DDate(myDate);

        for (int h = 0; h < 24; h++)
        {
            index = d*24+h;
            // set categories
            categories.append(QString::number(index));
            if (index == round(nextIndex))
            {
                myDateTime.setDate(myDate);
                myDateTime.setTime(QTime(h, 0, 0));
                myDateTime.setTimeSpec(Qt::UTC);
                categoriesVirtual.append(myDateTime.toString("MMM dd <br> yyyy <br> hh:mm"));
                nextIndex += step;
            }

            for (int mp=0; mp < nMeteoPoints; mp++)
            {
                if (isLine)
                {
                    for (int i = 0; i < nameLines.size(); i++)
                    {
                        meteoVariable meteoVar = MapHourlyMeteoVar.at(nameLines[i].toStdString());
                        double value = meteoPoints[mp].getMeteoPointValueH(myCrit3DDate, h, 0, meteoVar);
                        if (value != NODATA)
                        {
                            lineSeries[mp][i]->append(index, value);
                            if (value > maxLine)
                            {
                                maxLine = value;
                            }
                            if (value < minLine)
                            {
                                minLine = value;
                            }
                        }
                        else
                        {
                            if (meteoPoints[mp].isDateTimeLoadedH(Crit3DTime(myCrit3DDate,h)))
                            {
                                lineSeries[mp][i]->append(index, value); // nodata hours are not drawed if they are the first of the last hour of the serie
                            }
                        }
                    }
                }

                if (isBar)
                {
                    for (int j = 0; j < nameBar.size(); j++)
                    {
                        meteoVariable meteoVar = MapHourlyMeteoVar.at(nameBar[j].toStdString());
                        double value = meteoPoints[mp].getMeteoPointValueH(myCrit3DDate, h, 0, meteoVar);
                        if (value != NODATA)
                        {
                            *setVector[mp][j] << value;
                            if (value > maxBar)
                            {
                                maxBar = value;
                            }
                        }
                        else
                        {
                            *setVector[mp][j] << 0;
                        }
                    }
                }
            }
        }

        myDate=myDate.addDays(1);
    }

    if (isBar)
    {
        for (int mp=0; mp < nMeteoPoints; mp++)
        {
            QBarSeries* barMpSeries = new QBarSeries();
            for (int i = 0; i < nameBar.size(); i++)
            {
                QColor barColor = colorBar[i];
                if (meteoPointsEnsemble.size() == 0)
                {
                    if (nMeteoPoints == 1)
                    {
                        barColor.setAlpha(255);
                    }
                    else
                    {
                        barColor.setAlpha( 255-(mp*(150/(nMeteoPoints-1))) );
                    }
                    setVector[mp][i]->setColor(barColor);
                }
                else
                {
                    setVector[mp][i]->setColor(Qt::transparent);
                }
                setVector[mp][i]->setBorderColor(barColor);
                barMpSeries->append(setVector[mp][i]);
            }
            barSeries.append(barMpSeries);
        }

        for (int mp=0; mp<nMeteoPoints;mp++)
        {
            connect(barSeries[mp], &QBarSeries::hovered, this, &Crit3DMeteoWidget::tooltipBar);
            if (nameBar.size() != 0)
            {
                chart->addSeries(barSeries[mp]);
                barSeries[mp]->attachAxis(axisX);
                barSeries[mp]->attachAxis(axisYdx);
            }
        }
        if (maxBar == -1)
        {
            axisYdx->setVisible(false);
        }
        else
        {
            axisYdx->setVisible(true);
            if (maxBar != 0)
            {
                axisYdx->setRange(0,maxBar);
            }
            else
            {
                axisYdx->setRange(0,1);
            }
        }
    }
    else
    {
        axisYdx->setVisible(false);
    }

    if (isLine)
    {
        for (int mp=0; mp < nMeteoPoints; mp++)
        {
            for (int i = 0; i < nameLines.size(); i++)
            {
                chart->addSeries(lineSeries[mp][i]);
                lineSeries[mp][i]->attachAxis(axisX);
                lineSeries[mp][i]->attachAxis(axisY);
                connect(lineSeries[mp][i], &QLineSeries::hovered, this, &Crit3DMeteoWidget::tooltipLineSeries);
            }
        }
        if (maxLine == NODATA && minLine == -NODATA)
        {
            axisY->setVisible(false);
        }
        else
        {
            axisY->setVisible(true);
            axisY->setMax(maxLine);
            axisY->setMin(minLine);

            if (axisY->max() == axisY->min())
            {
                axisY->setRange(axisY->min()-axisY->min()/100, axisY->max()+axisY->max()/100);
            }
        }
    }
    else
    {
        axisY->setVisible(false);
    }

    axisX->setCategories(categories);
    axisXvirtual->setCategories(categoriesVirtual);
    axisX->setGridLineVisible(false);

    firstDate->blockSignals(false);
    lastDate->blockSignals(false);

    foreach(QLegendMarker* marker, chart->legend()->markers())
    {
        marker->setVisible(true);
        marker->series()->setVisible(true);
        QObject::connect(marker, &QLegendMarker::clicked, this, &Crit3DMeteoWidget::handleMarkerClicked);
    }

    formInfo.close();

}


void Crit3DMeteoWidget::showVar()
{
    if (! isInitialized) return;

    if (currentFreq == noFrequency)
    {
        if (dailyButton->isChecked()) // dailyButton is pressed
        {
            currentFreq = daily;
        }
        else if (hourlyButton->isChecked())
        {
            currentFreq = hourly;
        }
        else if (monthlyButton->isChecked())
        {
            currentFreq = monthly;
        }
    }
    QList<QString> allKeys = MapCSVStyles.keys();
    QList<QString> selectedVar = currentVariables;
    QList<QString> allVar;
    for (int i = 0; i<allKeys.size(); i++)
    {
        if (currentFreq == daily)
        {
            if (allKeys[i].contains("DAILY") && !selectedVar.contains(allKeys[i]))
            {
                allVar.append(allKeys[i]);
            }
        }
        else if (currentFreq == hourly)
        {
            if (!allKeys[i].contains("DAILY") && !selectedVar.contains(allKeys[i]))
            {
                allVar.append(allKeys[i]);
            }
        }
        else if (currentFreq == monthly)
        {
            if (!allKeys[i].contains("MONTHLY") && !selectedVar.contains(allKeys[i]))
            {
                allVar.append(allKeys[i]);
            }
        }
    }
    DialogSelectVar selectDialog(allVar, selectedVar);
    if (selectDialog.result() == QDialog::Accepted)
    {
        currentVariables.clear();
        currentVariables = selectDialog.getSelectedVariables();

        updateSeries();
        redraw();
    }
}

void Crit3DMeteoWidget::showMonthlyGraph()
{
    if (! isInitialized) return;

    currentFreq = monthly;

    dailyButton->setEnabled(true);
    hourlyButton->setEnabled(true);
    monthlyButton->setEnabled(false);

     // TO DO
/*
    QList<QString> currentMonthlyVar = currentVariables;
    currentVariables.clear();

    for (int i = 0; i<currentMonthlyVar.size(); i++)
    {
        QString name = currentMonthlyVar[i];
        auto searchMonthly = MapMonthlyMeteoVar.find(name.toStdString());
        if (searchMonthly != MapMonthlyMeteoVar.end())
        {
            meteoVariable monthlyVar = MapMonthlyMeteoVar.at(name.toStdString());
            meteoVariable dailyVar = updateMeteoVariable(monthlyVar, daily);
            if (dailyVar != noMeteoVar)
            {
                QString varString = QString::fromStdString(MapDailyMeteoVarToString.at(dailyVar));
                if (!currentVariables.contains(varString))
                {
                    currentVariables.append(varString);
                }
            }
        }
    }

    updateSeries();
    redraw();
*/
}


void Crit3DMeteoWidget::showDailyGraph()
{
    if (! isInitialized) return;
    frequencyType prevFreq = currentFreq;
    currentFreq = daily;

    dailyButton->setEnabled(false);
    hourlyButton->setEnabled(true);
    monthlyButton->setEnabled(true);

    if (prevFreq == hourly)
    {
        QList<QString> currentHourlyVar = currentVariables;
        currentVariables.clear();
        for (int i = 0; i<currentHourlyVar.size(); i++)
        {
            QString name = currentHourlyVar[i];
            auto searchHourly = MapHourlyMeteoVar.find(name.toStdString());
            if (searchHourly != MapHourlyMeteoVar.end())
            {
                meteoVariable hourlyVar = MapHourlyMeteoVar.at(name.toStdString());
                meteoVariable dailyVar = updateMeteoVariable(hourlyVar, daily);
                if (dailyVar != noMeteoVar)
                {
                    QString varString = QString::fromStdString(MapDailyMeteoVarToString.at(dailyVar));
                    if (!currentVariables.contains(varString))
                    {
                        currentVariables.append(varString);
                    }
                }
            }
        }
    }
    else if (prevFreq == monthly)
    {
        QList<QString> currentMonthlyVar = currentVariables;
        currentVariables.clear();
        for (int i = 0; i<currentMonthlyVar.size(); i++)
        {
            QString name = currentMonthlyVar[i];
            auto searchMonthly = MapMonthlyMeteoVar.find(name.toStdString());
            if (searchMonthly != MapMonthlyMeteoVar.end())
            {
                meteoVariable monthlyVar = MapMonthlyMeteoVar.at(name.toStdString());
                meteoVariable dailyVar = updateMeteoVariable(monthlyVar, daily);
                if (dailyVar != noMeteoVar)
                {
                    QString varString = QString::fromStdString(MapDailyMeteoVarToString.at(dailyVar));
                    if (!currentVariables.contains(varString))
                    {
                        currentVariables.append(varString);
                    }
                }
            }
        }
    }

    updateSeries();
    redraw();
}


void Crit3DMeteoWidget::showHourlyGraph()
{
    if (! isInitialized) return;
    frequencyType prevFreq = currentFreq;
    currentFreq = hourly;

    hourlyButton->setEnabled(false);
    dailyButton->setEnabled(true);
    monthlyButton->setEnabled(true);

    if (prevFreq == daily)
    {
        QList<QString> currentDailyVar = currentVariables;
        currentVariables.clear();
        for (int i = 0; i < currentDailyVar.size(); i++)
        {
            QString name = currentDailyVar[i];
            auto searchDaily = MapDailyMeteoVar.find(name.toStdString());
            if (searchDaily != MapDailyMeteoVar.end())
            {
                meteoVariable dailyVar = getMeteoVar(name.toStdString());
                meteoVariable hourlyVar= updateMeteoVariable(dailyVar, hourly);
                if (hourlyVar != noMeteoVar)
                {
                    QString varString = QString::fromStdString(MapHourlyMeteoVarToString.at(hourlyVar));
                    if (!currentVariables.contains(varString))
                    {
                        currentVariables.append(varString);
                    }
                }
            }
        }
    }
    else if (prevFreq == monthly)
    {
        QList<QString> currentMonthlyVar = currentVariables;
        currentVariables.clear();
        for (int i = 0; i<currentMonthlyVar.size(); i++)
        {
            QString name = currentMonthlyVar[i];
            auto searchMonthly = MapMonthlyMeteoVar.find(name.toStdString());
            if (searchMonthly != MapMonthlyMeteoVar.end())
            {
                meteoVariable monthlyVar = MapMonthlyMeteoVar.at(name.toStdString());
                meteoVariable hourlyVar= updateMeteoVariable(monthlyVar, hourly);
                if (hourlyVar != noMeteoVar)
                {
                    QString varString = QString::fromStdString(MapHourlyMeteoVarToString.at(hourlyVar));
                    if (!currentVariables.contains(varString))
                    {
                        currentVariables.append(varString);
                    }
                }
            }
        }
    }

    updateSeries();
    redraw();
}


void Crit3DMeteoWidget::updateSeries()
{
    if (! isInitialized) return;

    nameLines.clear();
    colorLines.clear();
    nameBar.clear();
    colorBar.clear();
    isLine = false;
    isBar = false;

    QMapIterator<QString, QList<QString>> i(MapCSVStyles);
    while (i.hasNext())
    {
        i.next();
        for (int j=0; j<currentVariables.size(); j++)
        {
            if (i.key() == currentVariables[j])
            {
                QList<QString> items = i.value();
                if (items[0] == "line")
                {
                    isLine = true;
                    nameLines.append(i.key());
                    colorLines.append(QColor(items[1]));
                }
                if (items[0] == "bar")
                {
                    isBar = true;
                    nameBar.append(i.key());
                    colorBar.append(QColor(items[1]));
                }
            }
        }
    }
}


void Crit3DMeteoWidget::redraw()
{
    if (! isInitialized) return;

    if (lastDate->dateTime() < firstDate->dateTime())
    {
        QMessageBox::information(nullptr, "Error", "Invalid data range");
        return;
    }

    if (isEnsemble || meteoPointsEnsemble.size() != 0)
    {
        resetEnsembleValues();
    }
    if(! isEnsemble)
    {
        resetValues();
    }

    if (currentFreq == daily)
    {
        if (isEnsemble || meteoPointsEnsemble.size() != 0)
        {
            drawEnsembleDailyVar();
        }
        if(! isEnsemble)
        {
            drawDailyVar();
        }
    }
    else if (currentFreq == hourly)
    {
        if (isEnsemble || meteoPointsEnsemble.size() != 0)
        {
            // TO DO
        }
        if(!isEnsemble)
        {
            drawHourlyVar();
        }
    }

}


void Crit3DMeteoWidget::shiftPrevious()
{
    int nDays = firstDate->date().daysTo(lastDate->date());

    QDate firstValidDate;
    if (currentFreq == daily)
    {
        if (! firstDailyDate.isValid() || firstDailyDate.year() == 1800)
            return;

        firstValidDate = firstDailyDate;
    }
    else
    {
        if (! firstHourlyDate.isValid() || firstHourlyDate.year() == 1800)
            return;

        firstValidDate = firstHourlyDate;
    }

    if (firstValidDate < firstDate->date().addDays(-nDays-1))
    {
        firstDate->setDate(firstDate->date().addDays(-nDays-1));
    }
    else
    {
        firstDate->setDate(firstValidDate);
    }

    lastDate->setDate(firstDate->date().addDays(nDays));

    redraw();
}


void Crit3DMeteoWidget::shiftFollowing()
{
    int nDays = firstDate->date().daysTo(lastDate->date());

    QDate lastValidDate;
    if (currentFreq == daily)
    {
        if (! lastDailyDate.isValid() || lastDailyDate.year() == 1800)
            return;

        lastValidDate = lastDailyDate;
    }
    else
    {
        if (! lastHourlyDate.isValid() || lastHourlyDate.year() == 1800)
            return;

        lastValidDate = lastHourlyDate;
    }

    if (lastDate->date().addDays(nDays+1) < lastValidDate)
    {
        lastDate->setDate(lastDate->date().addDays(nDays+1));
    }
    else
    {
        lastDate->setDate(lastValidDate);
    }

    firstDate->setDate(lastDate->date().addDays(-nDays));

    redraw();
}


void Crit3DMeteoWidget::showTable()
{
    DialogMeteoTable meteoTable(meteoSettings, meteoPoints, firstDate->date(), lastDate->date(), currentFreq, currentVariables);
}

void Crit3DMeteoWidget::tooltipLineSeries(QPointF point, bool state)
{
    QLineSeries *series = qobject_cast<QLineSeries *>(sender());
    computeTooltipLineSeries(series, point, state);
}

bool Crit3DMeteoWidget::computeTooltipLineSeries(QLineSeries *series, QPointF point, bool state)
{
    if (state)
    {
        int doy = point.x();
        int doyRelative = point.x();
        bool valueExist = false;

        if (categories.size() != series->count())
        {
            for(int i = 0; i < series->count(); i++)
            {
                if (series->at(i).x() == doy)
                {
                    doyRelative = i;
                    valueExist = true;
                    break;
                }
            }

            if (!valueExist)
            {
                // missing data
                if (currentFreq == daily)
                {
                    QDate xDate = firstDate->date().addDays(doy);
                    m_tooltip->setText(QString("%1 \n%2 nan ").arg(series->name()).arg(xDate.toString("MMM dd yyyy")));
                }
                if (currentFreq == hourly)
                {
                    QDateTime xDate(firstDate->date(), QTime(0,0,0), Qt::UTC);
                    xDate = xDate.addSecs(3600*doy);
                    m_tooltip->setText(QString("%1 \n%2 nan ").arg(series->name()).arg(xDate.toString("MMM dd yyyy hh:mm")));
                }
                m_tooltip->setSeries(series);
                m_tooltip->setAnchor(point);
                m_tooltip->setZValue(11);
                m_tooltip->updateGeometry();
                m_tooltip->show();
                return true;
            }
        }

        QPoint CursorPoint = QCursor::pos();
        QPoint mapPoint = chartView->mapFromGlobal(CursorPoint);
        QPoint pointDoY = series->at(doyRelative).toPoint();

        if (doyRelative == 0)
        {
            QPoint pointNext = series->at(doyRelative+1).toPoint();
            int distStep = qAbs(chart->mapToPosition(pointDoY).x()-chart->mapToPosition(pointNext).x());
            int distDoY = qAbs(mapPoint.x()-chart->mapToPosition(pointDoY).x());
            int distNext = qAbs(mapPoint.x()-chart->mapToPosition(pointNext).x());
            int minDist = qMin(distDoY, distNext);
            if (minDist !=  distDoY)
            {
                if (distStep > 0 && distNext > distStep/10)
                {
                    return false;
                }
                else
                {
                    doy = doy + 1;
                }
            }
            else
            {
                if (distDoY > distStep/10)
                {
                    return false;
                }
            }

        }
        else if (doyRelative > 0 && doyRelative < series->count())
        {
            QPoint pointBefore = series->at(doyRelative-1).toPoint();
            QPoint pointNext = series->at(doyRelative+1).toPoint();

            int distStep = qAbs(chart->mapToPosition(pointDoY).x()-chart->mapToPosition(pointNext).x());
            int distDoY = qAbs(mapPoint.x()-chart->mapToPosition(pointDoY).x());
            int distNext = qAbs(mapPoint.x()-chart->mapToPosition(pointNext).x());
            int distBefore = qAbs(mapPoint.x()-chart->mapToPosition(pointBefore).x());

            int minDist = qMin(qMin(distBefore,distDoY), distNext);
            if ( minDist != distDoY)
            {
                if (minDist == distBefore)
                {
                    if (distStep > 0 && distBefore > distStep/10)
                    {
                        return false;
                    }
                    else
                    {
                        doy = doy - 1;
                    }
                }
                else if (minDist == distNext)
                {
                    if (distStep > 0 && distNext > distStep/10)
                    {
                        return false;
                    }
                    else
                    {
                        doy = doy + 1;
                    }
                }
            }
            else
            {
                if (distStep > 0 && distDoY > distStep/10)
                {
                    return false;
                }
            }

        }
        else if (doyRelative == series->count())
        {
            QPoint pointBefore = series->at(doyRelative-1).toPoint();
            QPoint pointDoY = series->at(doyRelative).toPoint();
            int distStep = qAbs(chart->mapToPosition(pointDoY).x()-chart->mapToPosition(pointBefore).x());

            int distBefore = qAbs(mapPoint.x()-chart->mapToPosition(pointBefore).x());
            int distDoY = qAbs(mapPoint.x()-chart->mapToPosition(pointDoY).x());
            int minDist = qMin(distDoY, distBefore);
            if (minDist != distDoY)
            {
                if (distStep > 0 && distBefore > distStep/10)
                {
                    return false;
                }
                else
                {
                    doy = doy - 1;
                }
            }
            else
            {
                if (distStep > 0 && distDoY > distStep/10)
                {
                    return false;
                }
            }

        }

        if (currentFreq == daily)
        {
            QDate xDate = firstDate->date().addDays(doy);
            for(int i = 0; i < series->count(); i++)
            {
                if (series->at(i).x() == doy)
                {
                    doyRelative = i;
                    break;
                }
            }
            double value = series->at(doyRelative).y();
            m_tooltip->setText(QString("%1 \n%2 %3 ").arg(series->name()).arg(xDate.toString("MMM dd yyyy")).arg(value, 0, 'f', 1));
        }
        if (currentFreq == hourly)
        {
            QDateTime xDate(firstDate->date(), QTime(0,0,0), Qt::UTC);
            xDate = xDate.addSecs(3600*doy);
            for(int i = 0; i < series->count(); i++)
            {
                if (series->at(i).x() == doy)
                {
                    doyRelative = i;
                    break;
                }
            }
            double value = series->at(doyRelative).y();
            m_tooltip->setText(QString("%1 \n%2 %3 ").arg(series->name()).arg(xDate.toString("MMM dd yyyy hh:mm")).arg(value, 0, 'f', 1));
        }
        m_tooltip->setSeries(series);
        m_tooltip->setAnchor(point);
        m_tooltip->setZValue(11);
        m_tooltip->updateGeometry();
        m_tooltip->show();
        return true;
    }
    else
    {
        m_tooltip->hide();
        return false;
    }
}

void Crit3DMeteoWidget::tooltipBar(bool state, int index, QBarSet *barset)
{
    QBarSeries *series = qobject_cast<QBarSeries *>(sender());

    if (state && barset!=nullptr && index < barset->count())
    {

        QPoint CursorPoint = QCursor::pos();
        QPoint mapPoint = chartView->mapFromGlobal(CursorPoint);
        QPointF pointF = chart->mapToValue(mapPoint,series);

        // check if bar is hiding QlineSeries
        if(isLine)
        {
            for (int mp=0; mp<meteoPoints.size();mp++)
            {
                for (int i = 0; i < nameLines.size(); i++)
                {
                    int index = pointF.toPoint().x();
                    if (index < lineSeries[mp][i]->count())
                    {
                        double lineSeriesY = lineSeries[mp][i]->at(index).y();
                        if (static_cast<int>( lineSeriesY) == pointF.toPoint().y())
                        {
                            if (computeTooltipLineSeries(lineSeries[mp][i], pointF, true))
                                return;
                        }
                    }
                }
            }
        }

        QString valueStr;
        if (currentFreq == daily)
        {
            QDate xDate = firstDate->date().addDays(index);
            valueStr = QString("%1 \n%2 %3 ").arg(xDate.toString("MMM dd yyyy")).arg(barset->label()).arg(barset->at(index), 0, 'f', 1);
        }
        if (currentFreq == hourly)
        {

            QDateTime xDate(firstDate->date(), QTime(0,0,0), Qt::UTC);
            xDate = xDate.addSecs(3600*index);
            valueStr = QString("%1 \n%2 %3 ").arg(xDate.toString("MMM dd yyyy hh:mm")).arg(barset->label()).arg(barset->at(index), 0, 'f', 1);
        }

        m_tooltip->setSeries(series);
        m_tooltip->setText(valueStr);
        m_tooltip->setAnchor(pointF);
        m_tooltip->setZValue(11);
        m_tooltip->updateGeometry();
        m_tooltip->show();

    }
    else
    {
        m_tooltip->hide();
    }

}

void Crit3DMeteoWidget::handleMarkerClicked()
{

    QLegendMarker* marker = qobject_cast<QLegendMarker*> (sender());

    // Toggle visibility of series
    bool isVisible = marker->series()->isVisible();
    marker->series()->setVisible(!isVisible);

    // Turn legend marker back to visible, since otherwise hiding series also hides the marker
    marker->setVisible(true);

    // change marker alpha, if series is not visible
    qreal alpha;
    if (isVisible)
    {
        alpha = 0.5;
    }
    else
    {
        alpha = 1.0;
    }

    QColor color;
    QBrush brush = marker->labelBrush();
    color = brush.color();
    color.setAlphaF(alpha);
    brush.setColor(color);
    marker->setLabelBrush(brush);

    brush = marker->brush();
    color = brush.color();
    color.setAlphaF(alpha);
    brush.setColor(color);
    marker->setBrush(brush);

    QPen pen = marker->pen();
    color = pen.color();
    color.setAlphaF(alpha);
    pen.setColor(color);
    marker->setPen(pen);

}

void Crit3DMeteoWidget::closeEvent(QCloseEvent *event)
{

    if(isGrid)
    {
        emit closeWidgetGrid(meteoWidgetID);
    }
    else
    {
        emit closeWidgetPoint(meteoWidgetID);
    }
    delete m_tooltip;
    event->accept();
}

void Crit3DMeteoWidget::setIsEnsemble(bool value)
{
    isEnsemble = value;
    tableButton->setEnabled(!value);
}

bool Crit3DMeteoWidget::getIsEnsemble()
{
    return isEnsemble;
}

void Crit3DMeteoWidget::setNrMembers(int value)
{
    nrMembers = value;
}

int Crit3DMeteoWidget::getMeteoWidgetID() const
{
    return meteoWidgetID;
}

void Crit3DMeteoWidget::setMeteoWidgetID(int value)
{
    meteoWidgetID = value;
}

void Crit3DMeteoWidget::setCurrentDate(QDate myDate)
{
    currentDate = myDate;
}

void Crit3DMeteoWidget::on_actionChangeLeftAxis()
{
    DialogChangeAxis changeAxisDialog(true);
    if (changeAxisDialog.result() == QDialog::Accepted)
    {
        axisY->setMax(changeAxisDialog.getMaxVal());
        axisY->setMin(changeAxisDialog.getMinVal());
    }
}


void Crit3DMeteoWidget::on_actionChangeRightAxis()
{
    DialogChangeAxis changeAxisDialog(false);
    if (changeAxisDialog.result() == QDialog::Accepted)
    {
        axisYdx->setMax(changeAxisDialog.getMaxVal());
        axisYdx->setMin(changeAxisDialog.getMinVal());
    }
}


void Crit3DMeteoWidget::on_actionExportGraph()
{
    QString fileName = QFileDialog::getSaveFileName(this, tr("Save current graph"), "", tr("png files (*.png)"));

    if (fileName != "")
    {
        /*const auto dpr = chartView->devicePixelRatioF();
        QPixmap buffer(chartView->width() * dpr, chartView->height() * dpr);
        buffer.setDevicePixelRatio(dpr);*/

        QPixmap buffer(chartView->width() * 2, chartView->height() * 2);
        buffer.fill(Qt::transparent);

        QPainter *paint = new QPainter(&buffer);
        paint->setPen(*(new QColor(255,34,255,255)));
        chartView->render(paint);

        QFile file(fileName);
        file.open(QIODevice::WriteOnly);
        buffer.save(&file, "PNG");
    }
}


void Crit3DMeteoWidget::on_actionRemoveStation()
{
    QList<QString> allStations;
    for (int mp=0; mp<meteoPoints.size();mp++)
    {
        QString stationId = QString::fromStdString(meteoPoints[mp].id);
        QString stationsName = QString::fromStdString(meteoPoints[mp].name);
        QString station = stationId+"_"+stationsName;
        allStations << station;
    }
    DialogRemoveStation selectStation(allStations);
    if (selectStation.result() == QDialog::Accepted)
    {
        QList<QString> stationsToRemoveList = selectStation.getSelectedStations();
        for (int n=0; n<stationsToRemoveList.size();n++)
        {
            QString id = stationsToRemoveList[n].split("_")[0];
            for (int indexMp=0; indexMp<meteoPoints.size();indexMp++)
            {
                if (meteoPoints[indexMp].id == id.toStdString())
                {
                    meteoPoints.removeAt(indexMp);
                    indexMp = indexMp - 1;
                }
            }
        }
        updateSeries();
        redraw();
    }
}

void Crit3DMeteoWidget::on_actionInfoPoint()
{
    QDialog infoWindow;
    infoWindow.setWindowTitle("Info meteo points");
    QVBoxLayout* layout = new QVBoxLayout();
    for (int mp=0; mp<meteoPoints.size();mp++)
    {
        QString stationId = QString::fromStdString(meteoPoints[mp].id);
        QString stationsName = QString::fromStdString(meteoPoints[mp].name);
        QString station = stationId+"_"+stationsName;
        QString dataset = QString::fromStdString(meteoPoints[mp].dataset);
        QString altitude = QString::number(meteoPoints[mp].point.z);
        QString lapseRateName = QString::fromStdString(getLapseRateCodeName(meteoPoints[mp].lapseRateCode));
        QLabel* label = new QLabel(station);
        layout->addWidget(label);
        QString infoStr;
        if (isGrid)
        {
            infoStr = QString("Cell: <b> %1 </b> <br/> ID: %2 <br/> altitude: %3 m ")
                          .arg(stationsName, stationId, altitude);
        }
        else
        {
            infoStr = QString("Point: <b> %1 </b> <br/> ID: %2 <br/> dataset: %3 <br/> altitude: %4 m <br/> lapse rate code: %5")
                                  .arg(stationsName, stationId, dataset, altitude, lapseRateName);
        }
        QTextEdit* plainTextEdit = new QTextEdit(infoStr);
        plainTextEdit->setReadOnly(true);
        layout->addWidget(plainTextEdit);
    }
    infoWindow.setLayout(layout);
    infoWindow.exec();
}


void Crit3DMeteoWidget::on_actionDataAvailability()
{
    QDialog infoWindow;
    infoWindow.setWindowTitle("DataAvailability");
    QVBoxLayout* layout = new QVBoxLayout();

    if (isGrid)
    {
        QGroupBox *groupBox = new QGroupBox("MeteoGrid");
        QVBoxLayout *vbox = new QVBoxLayout;

        if (!firstDailyDate.isNull() && firstDailyDate.year() != 1800)
        {
            QLabel* labelDaily = new QLabel("Daily Data:");
            vbox->addWidget(labelDaily);

            QString dailyInfo = QString("%1 - %2").arg(firstDailyDate.toString("yyyy/MM/dd"), lastDailyDate.toString("yyyy/MM/dd"));
            QTextEdit* dailyTextEdit = new QTextEdit(dailyInfo);

            dailyTextEdit->setMaximumHeight(QFontMetrics(dailyTextEdit->font()).height() + 10);
            dailyTextEdit->setReadOnly(true);

            vbox->addWidget(dailyTextEdit);
        }

        if (!firstHourlyDate.isNull() && firstHourlyDate.year() != 1800)
        {
            QLabel* labelHourly = new QLabel("Hourly Data:");
            vbox->addWidget(labelHourly);

            QString hourlyInfo = QString("%1 - %2").arg(firstHourlyDate.toString("yyyy/MM/dd"), lastHourlyDate.toString("yyyy/MM/dd"));
            QTextEdit* hourlyTextEdit = new QTextEdit(hourlyInfo);

            hourlyTextEdit->setMaximumHeight(QFontMetrics(hourlyTextEdit->font()).height() + 10);
            hourlyTextEdit->setReadOnly(true);

            vbox->addWidget(hourlyTextEdit);
        }

        groupBox->setLayout(vbox);
        layout->addWidget(groupBox);
    }
    else
    {
        QDate myFirstDailyDate, myLastDailyDate;
        QDate myFirstHourlyDate, myLastHourlyDate;

        for (int mp=0; mp<meteoPoints.size();mp++)
        {
            QString stationId = QString::fromStdString(meteoPoints[mp].id);
            QString stationsName = QString::fromStdString(meteoPoints[mp].name);
            QString stationString = stationId + "_" + stationsName;

            QGroupBox *groupBox = new QGroupBox(stationString);
            QVBoxLayout *vbox = new QVBoxLayout;

            QString infoDaily = QString("Daily Data:");
            QLabel* labelDaily = new QLabel(infoDaily);
            vbox->addWidget(labelDaily);
            myFirstDailyDate.setDate(meteoPoints[mp].obsDataD[0].date.year, meteoPoints[mp].obsDataD[0].date.month, meteoPoints[mp].obsDataD[0].date.day);
            myLastDailyDate = myFirstDailyDate.addDays(meteoPoints[mp].nrObsDataDaysD-1);
            QString dailyInfo = QString("%1 - %2")
                                    .arg(myFirstDailyDate.toString("yyyy/MM/dd"), myLastDailyDate.toString("yyyy/MM/dd"));
            QTextEdit* dailyTextEdit = new QTextEdit(dailyInfo);
            QFont font = dailyTextEdit->font();
            QFontMetrics fontMetrics = QFontMetrics(font);
            dailyTextEdit->setMaximumHeight(fontMetrics.height()+10);
            dailyTextEdit->setReadOnly(true);
            vbox->addWidget(dailyTextEdit);

            QString infoHourly = QString("Hourly Data:");
            QLabel* labelHourly = new QLabel(infoHourly);
            vbox->addWidget(labelHourly);
            myFirstHourlyDate.setDate(meteoPoints[mp].getMeteoPointHourlyValuesDate(0).year, meteoPoints[mp].getMeteoPointHourlyValuesDate(0).month,
                                      meteoPoints[mp].getMeteoPointHourlyValuesDate(0).day);
            myLastHourlyDate = myFirstHourlyDate.addDays(meteoPoints[mp].nrObsDataDaysH-1);

            QString hourlyInfo = QString("%1 - %2")
                                     .arg(myFirstHourlyDate.toString("yyyy/MM/dd"), myLastHourlyDate.toString("yyyy/MM/dd"));
            QTextEdit* hourlyTextEdit = new QTextEdit(hourlyInfo);
            hourlyTextEdit->setMaximumHeight(fontMetrics.height()+10);
            hourlyTextEdit->setReadOnly(true);
            vbox->addWidget(hourlyTextEdit);
            groupBox->setLayout(vbox);
            layout->addWidget(groupBox);
        }
    }

    infoWindow.setLayout(layout);
    infoWindow.exec();
}

void Crit3DMeteoWidget::on_actionDataSum()
{
    DialogVariableToSum varToSum(currentVariables);
    QList<QString> varToSumList;
    if (varToSum.result() == QDialog::Accepted)
    {
        varToSumList = varToSum.getSelectedVariable();
    }
    else
    {
        return;
    }
    if (!varToSumList.isEmpty())
    {
        drawSum(varToSumList);
    }
}

void Crit3DMeteoWidget::drawSum(QList<QString> varToSumList)
{
    int nMeteoPoints = meteoPoints.size();
    for (int i = 0; i < varToSumList.size(); i++)
    {
        if (!lineSeries.isEmpty())
        {
            for (int j = 0; j < nameLines.size(); j++)
            {
                qreal max = NODATA;
                if (nameLines[j] == varToSumList[i])
                {
                    QVector<QPointF> points;
                    QVector<QPointF> cumulativePoints;   
                    for (int mp=0; mp<nMeteoPoints;mp++)
                    {
                        for (int n = 0; n<lineSeries[mp][j]->points().size(); n++)
                        {
                            points.append(QPointF(lineSeries[mp][j]->points()[n].x(),lineSeries[mp][j]->points()[n].y()));
                        }
                        cumulativePoints.append(points[0]);
                        for (int n = 1; n<points.size(); n++)
                        {
                            cumulativePoints.append(QPointF(points[n].rx(), points[n].ry()+cumulativePoints[n-1].ry()));
                        }
                        lineSeries[mp][j]->replace(cumulativePoints);
                        if (max < cumulativePoints.last().ry())
                        {
                            max = cumulativePoints.last().ry();
                        }
                        points.clear();
                        cumulativePoints.clear();
                    }
                }
                axisY->setRange(axisY->min(),max);
            }
        }
        if (! barSeries.isEmpty())
        {
            for (int j = 0; j < nameBar.size(); j++)
            {
                double max = NODATA;
                if (nameBar[j] == varToSumList[i])
                {
                    QVector<double> values;
                    QVector<double> cumulativeValues;
                    for (int mp=0; mp<nMeteoPoints;mp++)
                    {
                        for (int n = 0; n<setVector[mp][j]->count(); n++)
                        {
                            values << setVector[mp][j]->at(n);
                        }
                        cumulativeValues.append(values[0]);
                        for (int n = 1; n<values.size(); n++)
                        {
                            cumulativeValues.append(values[n]+cumulativeValues[n-1]);
                            setVector[mp][j]->replace(n,cumulativeValues[n]);
                        }
                        if (max < cumulativeValues.last())
                        {
                            max = cumulativeValues.last();
                        }
                        values.clear();
                        cumulativeValues.clear();
                    }
                }
                axisYdx->setRange(axisYdx->min(),max);
            }
        }
    }
}


qreal findMedian(QList<double> sortedList, int begin, int end)
{
    int count = end - begin;
    if (count % 2) {
        return sortedList.at(count / 2 + begin);
    } else {
        qreal right = sortedList.at(count / 2 + begin);
        qreal left = sortedList.at(count / 2 - 1 + begin);
        return (right + left) / 2.0;
    }
}



