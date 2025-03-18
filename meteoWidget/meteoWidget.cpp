/*!
    \file meteoWidget.cpp

    \abstract this widget displays hourly or daily weather data sets

    \copyright
    This file is part of CRITERIA3D.
    CRITERIA3D has been developed by ARPAE Emilia-Romagna.

    CRITERIA3D is free software: you can redistribute it and/or modify
    it under the terms of the GNU Lesser General Public License as published by
    the Free Software Foundation, either version 3 of the License, or
    (at your option) any later version.

    CRITERIA3D is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
    GNU Lesser General Public License for more details.

    You should have received a copy of the GNU Lesser General Public License
    along with CRITERIA3D.  If not, see <http://www.gnu.org/licenses/>.

    \authors
    Laura Costantini
    Fausto Tomei ftomei@arpae.it
    Gabriele Antolini gantolini@arpae.it
*/


#include "meteoWidget.h"
#include "dialogSelectVar.h"
#include "dialogRemoveStation.h"
#include "dialogAddStation.h"
#include "dialogMeteoTable.h"
#include "dialogChangeAxis.h"
#include "dialogVariableToSum.h"
#include "utilities.h"
#include "commonConstants.h"
#include "formInfo.h"

#include <QLayout>
#include <QDate>
#include <QColorDialog>


Crit3DMeteoWidget::Crit3DMeteoWidget(bool isGrid, QString projectPath, Crit3DMeteoSettings* meteoSettings)
: _isGrid(isGrid), _meteoSettings(meteoSettings)
{
    _isInitialized = false;
    _isEnsemble = false;
    _nrMembers = NODATA;
    _allMeteoPointsPointer = nullptr;

    maxEnsembleBar = -1;
    maxEnsembleLine = NODATA;
    minEnsembleLine = -NODATA;

    if (_isGrid)
        this->setWindowTitle("Grid");
    else
        this->setWindowTitle("Point");

    this->resize(1240, 700);
    this->setAttribute(Qt::WA_DeleteOnClose);
    _currentFrequency = noFrequency;

    QDate noDate = QDate(1800,1,1);
    _currentDate = noDate;
    firstDailyDate = noDate;
    firstHourlyDate = noDate;
    firstMonthlyDate = noDate;
    lastDailyDate = noDate;
    lastHourlyDate = noDate;
    lastMonthlyDate = noDate;

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
    while (! in.atEnd())
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
            _currentFrequency = daily;
        }
        else
        {
            _currentFrequency = hourly;
        }

        MapCSVDefault.insert(key,items);
        zeroLine = new QLineSeries();
        zeroLine->setColor(Qt::gray);
        zeroLine->setName("zero");
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
            dailyVar++;
        }
        else
        {
            hourlyVar++;
        }
    }

    if (currentVariables.isEmpty() || (dailyVar != 0 && hourlyVar != 0))
    {
        QMessageBox::information(nullptr, "Warning", "Wrong variables in Crit3DPlotDefault.csv");
        _currentFrequency = noFrequency;
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
    firstDate = new QDateTimeEdit(_currentDate);
    lastDate = new QDateTimeEdit(_currentDate);
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

    if (_currentFrequency == daily || _currentFrequency == noFrequency)
    {
        dailyButton->setEnabled(false);
        hourlyButton->setEnabled(true);
        monthlyButton->setEnabled(true);
    }
    else if (_currentFrequency == hourly)
    {
        hourlyButton->setEnabled(false);
        dailyButton->setEnabled(true);
        monthlyButton->setEnabled(true);
    }
    else if (_currentFrequency == monthly)
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

    axisY_sx = new QValueAxis();
    axisY_dx = new QValueAxis();

    axisX->setTitleText("Date");
    axisXvirtual->setTitleText("Date");
    axisXvirtual->setGridLineVisible(false);

    axisY_sx->setRange(0,30);
    axisY_sx->setGridLineVisible(false);

    axisY_dx->setRange(0,8);
    axisY_dx->setGridLineVisible(false);

    chart->addAxis(axisX, Qt::AlignBottom);
    chart->addAxis(axisXvirtual, Qt::AlignBottom);
    chart->addAxis(axisY_sx, Qt::AlignLeft);
    chart->addAxis(axisY_dx, Qt::AlignRight);

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
    QAction* addStation = new QAction(tr("Add stations"), this);

    editMenu->addAction(changeLeftAxis);
    editMenu->addAction(changeRightAxis);
    editMenu->addAction(exportGraph);
    editMenu->addAction(removeStation);
    editMenu->addAction(addStation);

    QAction* infoPoint = new QAction(tr("&Info meteo point"), this);
    QAction* dataAvailability = new QAction(tr("&Data availability"), this);
    dataSum = new QAction(tr("&Sum"), this);
    dataSum->setCheckable(true);

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
    connect(addStation, &QAction::triggered, this, &Crit3DMeteoWidget::on_actionAddStation);
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

    _isInitialized = true;
}


Crit3DMeteoWidget::~Crit3DMeteoWidget()
{

}


void Crit3DMeteoWidget::setFrequency(frequencyType frequency)
{
    _currentFrequency = frequency;

    // update gui
    if (_currentFrequency == daily || _currentFrequency == noFrequency)
    {
        dailyButton->setEnabled(false);
        hourlyButton->setEnabled(true);
        monthlyButton->setEnabled(true);
    }
    else if (_currentFrequency == hourly)
    {
        hourlyButton->setEnabled(false);
        dailyButton->setEnabled(true);
        monthlyButton->setEnabled(true);
    }
    else if (_currentFrequency == monthly)
    {
        monthlyButton->setEnabled(false);
        dailyButton->setEnabled(true);
        hourlyButton->setEnabled(true);
    }
}


void Crit3DMeteoWidget::setDailyRange(QDate firstDate, QDate lastDate)
{
    firstDailyDate = firstDate;
    lastDailyDate = lastDate;
}


void Crit3DMeteoWidget::setHourlyRange(QDate firstDate, QDate lastDate)
{
    firstHourlyDate = firstDate;
    lastHourlyDate = lastDate;
}

void Crit3DMeteoWidget::setMonthlyRange(QDate firstDate, QDate lastDate)
{
    firstMonthlyDate = firstDate;
    lastMonthlyDate = lastDate;
}


// search bigger data interval to show between my meteoPoints
void Crit3DMeteoWidget::updateTimeRange()
{
    for (int i = 0; i < _meteoPoints.size(); i++)
    {
        QDate myDailyDateFirst;
        QDate myDailyDateLast;
        if (_meteoPoints[i].obsDataD.size() != 0)
        {
            myDailyDateFirst.setDate(_meteoPoints[i].obsDataD[0].date.year,
                                     _meteoPoints[i].obsDataD[0].date.month,
                                     _meteoPoints[i].obsDataD[0].date.day);
            myDailyDateLast = myDailyDateFirst.addDays(_meteoPoints[i].nrObsDataDaysD-1);
        }
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
        myHourlyDateFirst.setDate(_meteoPoints[i].getMeteoPointHourlyValuesDate(0).year,
                                  _meteoPoints[i].getMeteoPointHourlyValuesDate(0).month,
                                  _meteoPoints[i].getMeteoPointHourlyValuesDate(0).day);
        QDate myHourlyDateLast = myHourlyDateFirst.addDays(_meteoPoints[i].nrObsDataDaysH-1);
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

        QDate myMonthlyDateFirst;
        QDate myMonthlyDateLast;
        if (_meteoPoints[i].obsDataM.size() != 0)
        {
            myMonthlyDateFirst.setDate(_meteoPoints[i].obsDataM[0]._year,
                                       _meteoPoints[i].obsDataM[0]._month,
                                       1);
            myMonthlyDateLast = myMonthlyDateFirst.addMonths(_meteoPoints[i].nrObsDataDaysM-1);
        }
        // updates monthly range
        if (myMonthlyDateFirst.isValid() &&
            (! firstMonthlyDate.isValid() || firstMonthlyDate.year() == 1800 || myMonthlyDateFirst < firstMonthlyDate))
        {
            firstMonthlyDate = myMonthlyDateFirst;
        }
        if (myMonthlyDateLast.isValid() &&
            (! lastMonthlyDate.isValid() || lastMonthlyDate.year() == 1800 || myMonthlyDateLast > lastMonthlyDate))
        {
            lastMonthlyDate = myMonthlyDateLast;
        }
    }

    // enable/disable monthly button if monthly data are available
    if ( (!firstMonthlyDate.isValid() || firstMonthlyDate.year() == 1800) && (!lastMonthlyDate.isValid() || lastMonthlyDate.year() == 1800) )
    {
        monthlyButton->setVisible(false);
    }
    else
    {
        monthlyButton->setVisible(true);
    }
}


void Crit3DMeteoWidget::drawMeteoPoint(Crit3DMeteoPoint mp, bool isAppend)
{
    if (! _isInitialized) return;

    _meteoPoints.append(mp);
    updateTimeRange();

    if (! isAppend)
    {
        // set dates
        firstDate->blockSignals(true);
        lastDate->blockSignals(true);

        lastDate->setDate(_currentDate);

        // draw period (31 days for daily, 3 days for hourly)
        if (_currentFrequency == daily)
        {
            firstDate->setDate(_currentDate.addDays(-30));
        }
        else if (_currentFrequency == hourly)
        {
            firstDate->setDate(_currentDate.addDays(-2));
        }

        firstDate->blockSignals(false);
        lastDate->blockSignals(false);
    }

    redraw();
    show();
}

void Crit3DMeteoWidget::addMeteoPointsEnsemble(Crit3DMeteoPoint mp)
{
    _meteoPointsEnsemble.append(mp);
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
    else if (varName.contains("MONTHLY_"))
    {
        varName = varName.remove("MONTHLY_");
    }

    label = pointName + '_' + varName;

    return label;
}


void Crit3DMeteoWidget::drawEnsemble()
{
    if (! _isInitialized || _meteoPointsEnsemble.isEmpty() || _meteoPointsEnsemble.size() != _nrMembers)
        return;

    firstDate->blockSignals(true);
    lastDate->blockSignals(true);

    // set date
    QDate myDailyDateFirst;
    QDate myDailyDateLast;
    QDate myHourlyDateFirst;
    QDate myHourlyDateLast;
    myDailyDateFirst.setDate(_meteoPointsEnsemble[0].obsDataD[0].date.year, _meteoPointsEnsemble[0].obsDataD[0].date.month, _meteoPointsEnsemble[0].obsDataD[0].date.day);
    myDailyDateLast = myDailyDateFirst.addDays(_meteoPointsEnsemble[0].nrObsDataDaysD-1);
    if (myDailyDateFirst.isValid() && myDailyDateFirst < firstDailyDate)
    {
        firstDailyDate = myDailyDateFirst;
    }
    if (myDailyDateLast.isValid() && myDailyDateLast > lastDailyDate)
    {
        lastDailyDate = myDailyDateLast;
    }

    myHourlyDateFirst.setDate(_meteoPointsEnsemble[0].getMeteoPointHourlyValuesDate(0).year, _meteoPointsEnsemble[0].getMeteoPointHourlyValuesDate(0).month,
                              _meteoPointsEnsemble[0].getMeteoPointHourlyValuesDate(0).day);
    myHourlyDateLast = myHourlyDateFirst.addDays(_meteoPointsEnsemble[0].nrObsDataDaysH-1);

    if (myHourlyDateFirst.isValid() && myHourlyDateFirst < firstHourlyDate)
    {
        firstHourlyDate = myHourlyDateFirst;
    }

    if (myHourlyDateLast.isValid() && myHourlyDateLast > lastHourlyDate)
    {
        lastHourlyDate = myHourlyDateLast;
    }

    lastDate->setDate(_currentDate);

    // draw period (31 days for daily, 3 days for hourly)
    if (_currentFrequency == daily)
        firstDate->setDate(_currentDate.addDays(-30));
    else if (_currentFrequency == hourly)
        firstDate->setDate(_currentDate.addDays(-2));

    redraw();

    firstDate->blockSignals(false);
    lastDate->blockSignals(false);

    show();
}


void Crit3DMeteoWidget::resetValues()
{
    if (chart->series().contains(zeroLine))
    {
        chart->removeSeries(zeroLine);
    }
    int nMeteoPoints = _meteoPoints.size();
    // clear prev series values
    if (!lineSeries.isEmpty())
    {
        for (int mp = 0; mp < lineSeries.size(); mp++)
        {
            for (int i = 0; i < lineSeries[mp].size(); i++)
            {
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
            QString pointName = QString::fromStdString(_meteoPoints[mp].name);
            vectorLine.clear();

            for (int i = 0; i < nameLines.size(); i++)
            {
                QLineSeries* line = new QLineSeries();
                line->setName(getFormattedLabel(pointName, nameLines[i]));
                QColor lineColor = colorLines[i];
                if (mp == 0)
                {
                    line->setColor(lineColor);
                }
                else
                {
                    QColor newColor = lineColor.toHsl();
                    int h = newColor.hslHue();
                    if (h >= 180)
                    {
                        newColor.setHsl(h-(mp*27), newColor.hslSaturation(), newColor.lightness());
                    }
                    else
                    {
                        newColor.setHsl(h+(mp*27), newColor.hslSaturation(), newColor.lightness());
                    }
                    if (colorLinesMpAppended.isEmpty() || !colorLinesMpAppended.contains(lineColor.name()))
                    {
                        line->setColor(newColor);
                        QList<QColor> myList;
                        myList.append(newColor);
                        colorLinesMpAppended.insert(lineColor.name(), myList);
                    }
                    else
                    {
                        QMapIterator<QString, QList<QColor>> iterator(colorLinesMpAppended);
                        while (iterator.hasNext())
                        {
                            iterator.next();
                            if (iterator.key() == lineColor.name())
                            {
                                QList<QColor> myList = colorLinesMpAppended[lineColor.name()];
                                if (myList.size()>=mp)
                                {
                                    line->setColor(myList[mp-1]);
                                }
                                else
                                {
                                    myList.append(newColor);
                                    colorLinesMpAppended[lineColor.name()] = myList;
                                    line->setColor(newColor);
                                }
                                break;
                            }
                        }
                    }
                }
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
            QString pointName = QString::fromStdString(_meteoPoints[mp].name);
            vectorBarSet.clear();

            for (int i = 0; i < nameBar.size(); i++)
            {
                QBarSet* bar = new QBarSet(getFormattedLabel(pointName, nameBar[i]));
                bar->setLabel(getFormattedLabel(pointName, nameBar[i]));

                QColor barColor = colorBar[i];
                if (_meteoPointsEnsemble.size() == 0)
                {
                    if (mp == 0)
                    {
                        bar->setColor(barColor);
                        bar->setBorderColor(barColor);
                    }
                    else
                    {
                        QColor newColor = barColor.toHsl();
                        int h = newColor.hslHue();
                        if (h >= 180)
                        {
                            newColor.setHsl(h-(mp*27), newColor.hslSaturation(), newColor.lightness());
                        }
                        else
                        {
                            newColor.setHsl(h+(mp*27), newColor.hslSaturation(), newColor.lightness());
                        }
                        if (colorBarMpAppended.isEmpty() || !colorBarMpAppended.contains(barColor.name()))
                        {
                            bar->setColor(newColor);
                            bar->setBorderColor(newColor);
                            QList<QColor> myList;
                            myList.append(newColor);
                            colorBarMpAppended.insert(barColor.name(), myList);
                        }
                        else
                        {
                            QMapIterator<QString, QList<QColor>> iterator(colorBarMpAppended);
                            while (iterator.hasNext())
                            {
                                iterator.next();

                                if (iterator.key() == barColor.name())
                                {
                                    QList<QColor> myList = colorBarMpAppended[barColor.name()];
                                    if (myList.size() >= mp)
                                    {
                                        bar->setColor(myList[mp-1]);
                                        bar->setBorderColor(myList[mp-1]);
                                    }
                                    else
                                    {
                                        myList.append(newColor);
                                        colorBarMpAppended[barColor.name()] = myList;
                                        bar->setColor(newColor);
                                        bar->setBorderColor(newColor);
                                    }
                                    break;
                                }
                            }
                        }
                    }
                }
                else
                {
                    bar->setColor(Qt::transparent);
                    bar->setBorderColor(barColor);
                }
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
    if (! _isInitialized) return;

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

    std::vector<double> sortedList;
    QList<QBoxSet*> listBoxSet;

    if (isLine)
    {
        for (int i = 0; i < nameLines.size(); i++)
        {
            listBoxSet.clear();
            QBoxPlotSeries *series = new QBoxPlotSeries();
            series->setName(QString::fromStdString(_meteoPointsEnsemble[0].id)+"_"+ QString::fromStdString(_meteoPointsEnsemble[0].name)+"_"+nameLines[i]+"_Ensemble");
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

                for (int mp=0; mp<_nrMembers;mp++)
                {
                    double value = _meteoPointsEnsemble[mp].getMeteoPointValueD(myDate, meteoVar, _meteoSettings);
                    if (value != NODATA)
                    {
                        sortedList.push_back(value);
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
                if (! sortedList.empty())
                {
                    std::sort(sortedList.begin(), sortedList.end());
                    int lastIndex = int(sortedList.size())-1;
                    box->setValue(QBoxSet::LowerExtreme, sortedList.front());
                    box->setValue(QBoxSet::UpperExtreme, sortedList.back());
                    box->setValue(QBoxSet::Median, findMedian(sortedList, 0, lastIndex));
                    box->setValue(QBoxSet::LowerQuartile, findMedian(sortedList, 0, lastIndex / 2));
                    box->setValue(QBoxSet::UpperQuartile, findMedian(sortedList, lastIndex / 2 + (lastIndex % 2), lastIndex));
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
                series->attachAxis(axisY_sx);
            }
        }
    }

    if (isBar)
    {
        for (int i = 0; i < nameBar.size(); i++)
        {
            listBoxSet.clear();
            QBoxPlotSeries *series = new QBoxPlotSeries();
            series->setName(QString::fromStdString(_meteoPointsEnsemble[0].id)+"_"+ QString::fromStdString(_meteoPointsEnsemble[0].name)+"_"+nameBar[i]+"_Ensemble");
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
                for (int mp=0; mp<_nrMembers;mp++)
                {
                    double value = _meteoPointsEnsemble[mp].getMeteoPointValueD(myDate, meteoVar, _meteoSettings);
                    if (value != NODATA)
                    {
                        sortedList.push_back(value);
                        if (value > maxEnsembleBar)
                        {
                            maxEnsembleBar = value;
                        }
                    }
                }
                QBoxSet *box = new QBoxSet();
                if (! sortedList.empty())
                {
                    std::sort(sortedList.begin(), sortedList.end());
                    int lastIndex = int(sortedList.size())-1;
                    box->setValue(QBoxSet::LowerExtreme, sortedList.front());
                    box->setValue(QBoxSet::UpperExtreme, sortedList.back());
                    box->setValue(QBoxSet::Median, findMedian(sortedList, 0, lastIndex));
                    box->setValue(QBoxSet::LowerQuartile, findMedian(sortedList, 0, lastIndex / 2));
                    box->setValue(QBoxSet::UpperQuartile, findMedian(sortedList, lastIndex / 2 + (lastIndex % 2), lastIndex));
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
                series->attachAxis(axisY_dx);
            }
        }
    }

    if(isLine)
    {
        if (maxEnsembleLine == NODATA && minEnsembleLine == -NODATA)
        {
            axisY_sx->setVisible(false);
        }
        else
        {
            axisY_sx->setVisible(true);
            axisY_sx->setMax(maxEnsembleLine);
            axisY_sx->setMin(minEnsembleLine);
            if (axisY_sx->max() == axisY_sx->min())
            {
                axisY_sx->setRange(axisY_sx->min()-axisY_sx->min()/100, axisY_sx->max()+axisY_sx->max()/100);
            }
        }
    }
    else
    {
        axisY_sx->setVisible(false);
    }

    if (isBar)
    {
        if (maxEnsembleBar == -1)
        {
            axisY_dx->setVisible(false);
        }
        else
        {
            axisY_dx->setVisible(true);
            if (maxEnsembleBar != 0)
            {
                axisY_dx->setRange(0,maxEnsembleBar);
            }
            else
            {
                axisY_dx->setRange(0,1);
            }
        }
    }
    else
    {
        axisY_dx->setVisible(false);
    }

    axisX->setCategories(categories);
    axisXvirtual->setCategories(categoriesVirtual);
    axisXvirtual->setGridLineVisible(false);

    drawAxisTitle();
    formInfo.close();

    firstDate->blockSignals(false);
    lastDate->blockSignals(false);
}


void Crit3DMeteoWidget::drawAxisTitle()
{
    if (! _isInitialized)
        return;

    QList<QString> unitList;
    QString axisTitle = "";

    // axis sx (lines)
    for (int i = 0; i < nameLines.size(); i++)
    {
        meteoVariable meteoVar = getMeteoVar(nameLines[i].toStdString());
        QString unitStr = QString::fromStdString(getUnitFromVariable(meteoVar));
        if (! unitList.contains(unitStr))
        {
            unitList.append(unitStr);
        }
    }

    if (! unitList.empty())
    {
        for (int i = 0; i < unitList.size(); i++)
        {
            if (i > 0)
                axisTitle += "    ,    ";

            axisTitle += unitList[i];
        }
    }
    axisY_sx->setTitleText(axisTitle);

    // axis dx (bar)
    unitList.clear();
    for (int i = 0; i < nameBar.size(); i++)
    {
        meteoVariable meteoVar = getMeteoVar(nameBar[i].toStdString());
        QString unitStr = QString::fromStdString(getUnitFromVariable(meteoVar));
        if (! unitList.contains(unitStr))
        {
            unitList.append(unitStr);
        }
    }

    axisTitle.clear();
    if (! unitList.empty())
    {
        for (int i = 0; i < unitList.size(); i++)
        {
            if (i > 0)
                axisTitle += "    ,    ";

            axisTitle += unitList[i];
        }
    }

    axisY_dx->setTitleText(axisTitle);
}


void Crit3DMeteoWidget::drawDailyVar()
{
    if (! _isInitialized)
        return;

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

    int nMeteoPoints = _meteoPoints.size();
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
                    double value = _meteoPoints[mp].getMeteoPointValueD(myDate, meteoVar, _meteoSettings);
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
                        if (_meteoPoints[mp].isDateLoadedD(myDate))
                        {
                            // nodata days are not drawed if they are the first of the last day of the series
                            lineSeries[mp][i]->append(day, value);
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
                    double value = _meteoPoints[mp].getMeteoPointValueD(myDate, meteoVar, _meteoSettings);
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
            connect(barSeries[mp], &QBarSeries::clicked, this, &Crit3DMeteoWidget::editBar);
            if (nameBar.size() != 0)
            {
                chart->addSeries(barSeries[mp]);
                barSeries[mp]->attachAxis(axisX);
                barSeries[mp]->attachAxis(axisY_dx);
            }
        }
        if (maxEnsembleBar == -1 && maxBar == -1)
        {
            axisY_dx->setVisible(false);
        }
        else
        {
            axisY_dx->setVisible(true);
            if (maxEnsembleBar > maxBar)
            {
                axisY_dx->setRange(0,maxEnsembleBar);
            }
            else
            {
                axisY_dx->setRange(0,maxBar);
            }
            if (axisY_dx->max() == axisY_dx->min())
            {
                axisY_dx->setRange(0,1);
            }
        }
    }
    else
    {
        axisY_dx->setVisible(false);
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
                    lineSeries[mp][i]->attachAxis(axisY_sx);
                    connect(lineSeries[mp][i], &QLineSeries::hovered, this, &Crit3DMeteoWidget::tooltipLineSeries);
                    connect(lineSeries[mp][i], &QLineSeries::clicked, this, &Crit3DMeteoWidget::editLineSeries);
                }
            }
        }
        if (maxLine == NODATA && minLine == -NODATA && maxEnsembleLine == NODATA && minEnsembleLine == -NODATA)
        {
            axisY_sx->setVisible(false);
        }
        else
        {
            axisY_sx->setVisible(true);
            if (maxEnsembleLine > maxLine)
            {
                axisY_sx->setMax(maxEnsembleLine);
            }
            else
            {
                axisY_sx->setMax(maxLine);
            }

            if (minEnsembleLine < minLine)
            {
                axisY_sx->setMin(minEnsembleLine);
            }
            else
            {
                axisY_sx->setMin(minLine);
            }
            if (axisY_sx->max() == axisY_sx->min())
            {
                axisY_sx->setRange(axisY_sx->min()-axisY_sx->min()/100, axisY_sx->max()+axisY_sx->max()/100);
            }
        }
    }
    else
    {
        axisY_sx->setVisible(false);
    }

    // add zeroLine
    if (axisY_sx->min() <= 0 && axisY_sx->max() >= 0)
    {
        zeroLine->clear();
        for (int day = 0; day < nDays; day++)
        {
            zeroLine->append(day, 0);
        }
        chart->addSeries(zeroLine);
        zeroLine->attachAxis(axisX);
        zeroLine->attachAxis(axisY_sx);
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
            if (nDays < 5 || _meteoPointsEnsemble.size() != 0)
            {
                setVector[mp][j]->setColor(QColor("transparent"));
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
        if (marker->series()->name() != "zero")
        {
            marker->setVisible(true);
            marker->series()->setVisible(true);
            QObject::connect(marker, &QLegendMarker::clicked, this, &Crit3DMeteoWidget::handleMarkerClicked);
        }
        else
        {
            marker->setVisible(false);
            marker->series()->setVisible(true);
        }
    }

    drawAxisTitle();
    formInfo.close();
}


void Crit3DMeteoWidget::drawHourlyVar()
{
    if (! _isInitialized) return;

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

    int nDays = firstDate->date().daysTo(lastDate->date())+1;
    int nrValues = nDays*24;

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

    int nMeteoPoints = _meteoPoints.size();
    QDate myDate = firstDate->date();
    Crit3DDate myCrit3DDate;
    QDateTime myDateTime;

    for (int d = 0; d < nDays; d++)
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
                        meteoVariable meteoVar = getMeteoVar(nameLines[i].toStdString());
                        if (meteoVar == noMeteoVar)
                        {
                            continue;
                        }

                        double value = _meteoPoints[mp].getMeteoPointValueH(myCrit3DDate, h, 0, meteoVar);
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
                            if (_meteoPoints[mp].isDateTimeLoadedH(Crit3DTime(myCrit3DDate,h)))
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
                        meteoVariable meteoVar = getMeteoVar(nameBar[j].toStdString());
                        if (meteoVar == noMeteoVar)
                        {
                            continue;
                        }

                        double value = _meteoPoints[mp].getMeteoPointValueH(myCrit3DDate, h, 0, meteoVar);
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
        for (int mp = 0; mp < nMeteoPoints; mp++)
        {
            QBarSeries* barMpSeries = new QBarSeries();
            for (int i = 0; i < nameBar.size(); i++)
            {
                barMpSeries->append(setVector[mp][i]);
                if (_meteoPointsEnsemble.size() != 0)
                {
                    setVector[mp][i]->setColor(Qt::transparent);
                }
            }
            barSeries.append(barMpSeries);
        }

        for (int mp=0; mp<nMeteoPoints;mp++)
        {
            connect(barSeries[mp], &QBarSeries::hovered, this, &Crit3DMeteoWidget::tooltipBar);
            connect(barSeries[mp], &QBarSeries::clicked, this, &Crit3DMeteoWidget::editBar);
            if (nameBar.size() != 0)
            {
                chart->addSeries(barSeries[mp]);
                barSeries[mp]->attachAxis(axisX);
                barSeries[mp]->attachAxis(axisY_dx);
            }
        }
        if (maxBar == -1)
        {
            axisY_dx->setVisible(false);
        }
        else
        {
            axisY_dx->setVisible(true);
            if (maxBar != 0)
            {
                axisY_dx->setRange(0,maxBar);
            }
            else
            {
                axisY_dx->setRange(0,1);
            }
        }
    }
    else
    {
        axisY_dx->setVisible(false);
    }

    if (isLine)
    {
        for (int mp=0; mp < nMeteoPoints; mp++)
        {
            for (int i = 0; i < nameLines.size(); i++)
            {
                chart->addSeries(lineSeries[mp][i]);
                lineSeries[mp][i]->attachAxis(axisX);
                lineSeries[mp][i]->attachAxis(axisY_sx);
                connect(lineSeries[mp][i], &QLineSeries::hovered, this, &Crit3DMeteoWidget::tooltipLineSeries);
                connect(lineSeries[mp][i], &QLineSeries::clicked, this, &Crit3DMeteoWidget::editLineSeries);
            }
        }
        if (maxLine == NODATA && minLine == -NODATA)
        {
            axisY_sx->setVisible(false);
        }
        else
        {
            axisY_sx->setVisible(true);
            axisY_sx->setMax(maxLine);
            axisY_sx->setMin(minLine);

            if (axisY_sx->max() == axisY_sx->min())
            {
                axisY_sx->setRange(axisY_sx->min()-axisY_sx->min()/100, axisY_sx->max()+axisY_sx->max()/100);
            }
        }
    }
    else
    {
        axisY_sx->setVisible(false);
    }

    // add zeroLine
    if (axisY_sx->min() <= 0 && axisY_sx->max() >= 0)
    {
        zeroLine->clear();
        for (int d = 0; d < nDays; d++)
        {
            for (int h = 0; h < 24; h++)
            {
                int index = d*24+h;
                zeroLine->append(index, 0);
            }
        }
        chart->addSeries(zeroLine);
        zeroLine->attachAxis(axisX);
        zeroLine->attachAxis(axisY_sx);
    }

    axisX->setCategories(categories);
    axisXvirtual->setCategories(categoriesVirtual);
    axisX->setGridLineVisible(false);

    firstDate->blockSignals(false);
    lastDate->blockSignals(false);

    foreach(QLegendMarker* marker, chart->legend()->markers())
    {
        if (marker->series()->name() != "zero")
        {
            marker->setVisible(true);
            marker->series()->setVisible(true);
            QObject::connect(marker, &QLegendMarker::clicked, this, &Crit3DMeteoWidget::handleMarkerClicked);
        }
        else
        {
            marker->setVisible(false);
            marker->series()->setVisible(true);
        }
    }

    drawAxisTitle();
    formInfo.close();
}


void Crit3DMeteoWidget::drawMonthlyVar()
{
    if (! _isInitialized) return;

    FormInfo formInfo;
    formInfo.showInfo("Draw monthly data...");

    firstDate->blockSignals(true);
    lastDate->blockSignals(true);

    Crit3DDate myDate;
    double maxBar = -1;
    double maxLine = NODATA;
    double minLine = -NODATA;

    int numberOfMonths = (lastDate->date().year()- firstDate->date().year())*12 + lastDate->date().month() - (firstDate->date().month()-1);

    categories.clear();
    categoriesVirtual.clear();
    m_tooltip = new Callout(chart);
    m_tooltip->hide();

    // virtual x axis
    int nrIntervals;
    if (numberOfMonths <= 12)
    {
        nrIntervals = numberOfMonths;
    }
    else if (numberOfMonths <= 45)
    {
        nrIntervals = numberOfMonths/3;
    }
    else
    {
        nrIntervals = 12;
    }
    double step = double(numberOfMonths) / double(nrIntervals);
    double nextIndex = step / 2 - 0.5;
    for (int month = 0; month < numberOfMonths; month++)
    {
        myDate = getCrit3DDate(firstDate->date().addMonths(month));
        if (month == round(nextIndex))
        {
            categoriesVirtual.append(getQDate(myDate).toString("MMM <br> yyyy"));
            nextIndex += step;
        }
    }

    int nMeteoPoints = _meteoPoints.size();
    for (int month = 0; month < numberOfMonths; month++)
    {
        myDate = getCrit3DDate(firstDate->date().addMonths(month));
        categories.append(QString::number(month));

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
                    double value = _meteoPoints[mp].getMeteoPointValueM(myDate, meteoVar);
                    if (value != NODATA)
                    {
                        lineSeries[mp][i]->append(month, value);
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
                        if (_meteoPoints[mp].isDateLoadedM(myDate))
                        {
                            lineSeries[mp][i]->append(month, value); // nodata days are not drawed if they are the first of the last day of the serie
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
                    double value = _meteoPoints[mp].getMeteoPointValueM(myDate, meteoVar);
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
            connect(barSeries[mp], &QBarSeries::clicked, this, &Crit3DMeteoWidget::editBar);
            if (nameBar.size() != 0)
            {
                chart->addSeries(barSeries[mp]);
                barSeries[mp]->attachAxis(axisX);
                barSeries[mp]->attachAxis(axisY_dx);
            }
        }
        if (maxEnsembleBar == -1 && maxBar == -1)
        {
            axisY_dx->setVisible(false);
        }
        else
        {
            axisY_dx->setVisible(true);
            if (maxEnsembleBar > maxBar)
            {
                axisY_dx->setRange(0,maxEnsembleBar);
            }
            else
            {
                axisY_dx->setRange(0,maxBar);
            }
            if (axisY_dx->max() == axisY_dx->min())
            {
                axisY_dx->setRange(0,1);
            }
        }
    }
    else
    {
        axisY_dx->setVisible(false);
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
                    lineSeries[mp][i]->attachAxis(axisY_sx);
                    connect(lineSeries[mp][i], &QLineSeries::hovered, this, &Crit3DMeteoWidget::tooltipLineSeries);
                    connect(lineSeries[mp][i], &QLineSeries::clicked, this, &Crit3DMeteoWidget::editLineSeries);
                }
            }
        }
        if (maxLine == NODATA && minLine == -NODATA && maxEnsembleLine == NODATA && minEnsembleLine == -NODATA)
        {
            axisY_sx->setVisible(false);
        }
        else
        {
            axisY_sx->setVisible(true);
            if (maxEnsembleLine > maxLine)
            {
                axisY_sx->setMax(maxEnsembleLine);
            }
            else
            {
                axisY_sx->setMax(maxLine);
            }

            if (minEnsembleLine < minLine)
            {
                axisY_sx->setMin(minEnsembleLine);
            }
            else
            {
                axisY_sx->setMin(minLine);
            }
            if (axisY_sx->max() == axisY_sx->min())
            {
                axisY_sx->setRange(axisY_sx->min()-axisY_sx->min()/100, axisY_sx->max()+axisY_sx->max()/100);
            }
        }
    }
    else
    {
        axisY_sx->setVisible(false);
    }

    // add zeroLine
    if (axisY_sx->min() <= 0 && axisY_sx->max() >= 0)
    {
        zeroLine->clear();
        for (int month = 0; month < numberOfMonths; month++)
        {
            zeroLine->append(month, 0);
        }
        chart->addSeries(zeroLine);
        zeroLine->attachAxis(axisX);
        zeroLine->attachAxis(axisY_sx);
    }

    // add minimimum values required
    if (numberOfMonths==1)
    {
        categories.append(QString::number(1));
        categoriesVirtual.append(firstDate->date().addDays(1).toString("MMM <br> yyyy"));
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
            if (numberOfMonths < 5)
            {
                setVector[mp][j]->setColor(QColor("transparent"));
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
        if (marker->series()->name() != "zero")
        {
            marker->setVisible(true);
            marker->series()->setVisible(true);
            QObject::connect(marker, &QLegendMarker::clicked, this, &Crit3DMeteoWidget::handleMarkerClicked);
        }
        else
        {
            marker->setVisible(false);
            marker->series()->setVisible(true);
        }
    }

    drawAxisTitle();
    formInfo.close();
}


void Crit3DMeteoWidget::showVar()
{
    if (! _isInitialized) return;

    if (_currentFrequency == noFrequency)
    {
        if (dailyButton->isChecked()) // dailyButton is pressed
        {
            _currentFrequency = daily;
        }
        else if (hourlyButton->isChecked())
        {
            _currentFrequency = hourly;
        }
        else if (monthlyButton->isChecked())
        {
            _currentFrequency = monthly;
        }
    }
    QList<QString> allKeys = MapCSVStyles.keys();
    QList<QString> selectedVar = currentVariables;
    QList<QString> allVar;
    for (int i = 0; i<allKeys.size(); i++)
    {
        if (_currentFrequency == daily)
        {
            if (allKeys[i].contains("DAILY") && !selectedVar.contains(allKeys[i]))
            {
                allVar.append(allKeys[i]);
            }
        }
        else if (_currentFrequency == hourly)
        {
            if (!allKeys[i].contains("DAILY") && !allKeys[i].contains("MONTHLY") && !selectedVar.contains(allKeys[i]))
            {
                allVar.append(allKeys[i]);
            }
        }
        else if (_currentFrequency == monthly)
        {
            if (allKeys[i].contains("MONTHLY") && !selectedVar.contains(allKeys[i]))
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
    if (! _isInitialized) return;
    frequencyType prevFreq = _currentFrequency;
    _currentFrequency = monthly;

    // change freq, reset sum data settings
    varToSumList.clear();
    dataSum->setChecked(false);

    monthlyButton->setEnabled(false);
    dailyButton->setEnabled(true);
    hourlyButton->setEnabled(true);

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
                meteoVariable monthlyVar= updateMeteoVariable(dailyVar, monthly);
                if (monthlyVar != noMeteoVar)
                {
                    QString varString = QString::fromStdString(getMeteoVarName(monthlyVar));
                    if (!varString.isEmpty() && !currentVariables.contains(varString))
                    {
                        currentVariables.append(varString);
                    }
                }
            }
        }
    }
    else if (prevFreq == hourly)
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
                meteoVariable monthlyVar= updateMeteoVariable(hourlyVar, monthly);
                if (monthlyVar != noMeteoVar)
                {
                    QString varString = QString::fromStdString(getMeteoVarName(monthlyVar));
                    if (!varString.isEmpty() && !currentVariables.contains(varString))
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


void Crit3DMeteoWidget::showDailyGraph()
{
    if (! _isInitialized) return;
    frequencyType prevFreq = _currentFrequency;
    _currentFrequency = daily;

    // change freq, reset sum data settings
    varToSumList.clear();
    dataSum->setChecked(false);

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
                    QString varString = QString::fromStdString(getMeteoVarName(dailyVar));
                    if (!varString.isEmpty() && !currentVariables.contains(varString))
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
                    QString varString = QString::fromStdString(getMeteoVarName(dailyVar));
                    if (!varString.isEmpty() && !currentVariables.contains(varString))
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
    if (! _isInitialized) return;
    frequencyType prevFreq = _currentFrequency;
    _currentFrequency = hourly;

    // change freq, reset sum data settings
    varToSumList.clear();
    dataSum->setChecked(false);

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
                    QString varString = QString::fromStdString(getMeteoVarName(hourlyVar));
                    if (!varString.isEmpty() && !currentVariables.contains(varString))
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
                    QString varString = QString::fromStdString(getMeteoVarName(hourlyVar));
                    if (!varString.isEmpty() && !currentVariables.contains(varString))
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
    if (! _isInitialized) return;

    nameLines.clear();
    colorLines.clear();
    colorLinesMpAppended.clear();
    nameBar.clear();
    colorBarMpAppended.clear();
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
                    QColor myColor = QColor(items[1]);
                    colorLines.append(myColor);
                    QList<QColor> appendedList;
                    for (int colorAppended = 2; colorAppended < items.size(); colorAppended++)
                    {
                        appendedList.append(QColor(items[colorAppended]));
                    }
                    if (!appendedList.isEmpty())
                    {
                        colorLinesMpAppended.insert(myColor.name(), appendedList);
                    }
                }
                if (items[0] == "bar")
                {
                    isBar = true;
                    nameBar.append(i.key());
                    QColor myColor = QColor(items[1]);
                    colorBar.append(myColor);
                    QList<QColor> appendedList;
                    for (int colorAppended = 2; colorAppended < items.size(); colorAppended++)
                    {
                        appendedList.append(QColor(items[colorAppended]));
                    }
                    if (!appendedList.isEmpty())
                    {
                        colorBarMpAppended.insert(myColor.name(), appendedList);
                    }
                }
            }
        }
    }
}


void Crit3DMeteoWidget::redraw()
{
    if (! _isInitialized) return;

    if (lastDate->dateTime() < firstDate->dateTime())
    {
        QMessageBox::information(nullptr, "Error", "Invalid data range");
        return;
    }

    if (_isEnsemble || _meteoPointsEnsemble.size() != 0)
    {
        resetEnsembleValues();
    }
    if(! _isEnsemble)
    {
        resetValues();
    }

    if (_currentFrequency == daily)
    {
        if (_isEnsemble || _meteoPointsEnsemble.size() != 0)
        {
            drawEnsembleDailyVar();
        }
        if(! _isEnsemble)
        {
            drawDailyVar();
            if (varToSumList.size() != 0)
            {
                drawSum();
            }
        }
    }
    else if (_currentFrequency == hourly)
    {
        if (_isEnsemble || _meteoPointsEnsemble.size() != 0)
        {
            // TO DO
        }
        if(!_isEnsemble)
        {
            drawHourlyVar();
            if (varToSumList.size() != 0)
            {
                drawSum();
            }
        }
    }
    else if (_currentFrequency == monthly)
    {
        if (_isEnsemble || _meteoPointsEnsemble.size() != 0)
        {
            // TO DO
        }
        if(!_isEnsemble)
        {
            drawMonthlyVar();
            if (varToSumList.size() != 0)
            {
                drawSum();
            }
        }
    }
}


void Crit3DMeteoWidget::shiftPrevious()
{
    int nDays = firstDate->date().daysTo(lastDate->date());

    QDate firstValidDate;
    if (_currentFrequency == daily)
    {
        if (! firstDailyDate.isValid() || firstDailyDate.year() == 1800)
            return;

        firstValidDate = firstDailyDate;
    }
    else if (_currentFrequency == hourly)
    {
        if (! firstHourlyDate.isValid() || firstHourlyDate.year() == 1800)
            return;

        firstValidDate = firstHourlyDate;
    }
    else if (_currentFrequency == monthly)
    {
        if (! firstMonthlyDate.isValid() || firstMonthlyDate.year() == 1800)
            return;

        firstValidDate = firstMonthlyDate;
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
    if (_currentFrequency == daily)
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
    DialogMeteoTable meteoTable(_meteoSettings, _meteoPoints, firstDate->date(), lastDate->date(), _currentFrequency, currentVariables);
}

void Crit3DMeteoWidget::tooltipLineSeries(QPointF point, bool state)
{
    QLineSeries *series = qobject_cast<QLineSeries *>(sender());
    computeTooltipLineSeries(series, point, state);
}

void Crit3DMeteoWidget::editLineSeries()
{
    QLineSeries *series = qobject_cast<QLineSeries *>(sender());
    QMenu menu("Edit");
    QAction* editColor = menu.addAction(QString("Set color"));
    QAction *selection =  menu.exec(QCursor::pos());

    if (selection != nullptr)
    {
        if (selection == editColor)
        {
            QColorDialog colorSelection;
            QColor newColor = colorSelection.getColor(series->color(), this );
            if( newColor.isValid() )
            {
                for (int i = 0; i<nameLines.size(); i++)
                {
                    QString myName = nameLines[i];
                    if (nameLines[i].contains("DAILY"))
                    {
                        myName.remove("DAILY_");
                    }
                    if (nameLines[i].contains("MONTHLY"))
                    {
                        myName.remove("MONTHLY_");
                    }
                    if (series->name().contains(myName))
                    {
                        QMapIterator<QString, QList<QString>> iterator(MapCSVStyles);
                        while (iterator.hasNext())
                        {
                            iterator.next();

                            if (iterator.key() == nameLines[i])
                            {
                                QList<QString> newItems = iterator.value();
                                // check which mp is
                                for (int mp = 0; mp < _meteoPoints.size(); mp++)
                                {
                                    QString pointName = QString::fromStdString(_meteoPoints[mp].name);
                                    QString label = getFormattedLabel(pointName, nameLines[i]);
                                    if (series->name() == label)
                                    {
                                        if (newItems.size() > mp+1)
                                        {
                                            newItems[mp+1] = newColor.name();
                                        }
                                        else
                                        {
                                            newItems.append(newColor.name());
                                        }
                                    }
                                }
                                MapCSVStyles[nameLines[i]] = newItems;
                                break;
                            }
                        }
                    }
                }
                updateSeries();
                redraw();
            }
        }
    }

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
                if (_currentFrequency == daily)
                {
                    QDate xDate = firstDate->date().addDays(doy);
                    m_tooltip->setText(QString("%1 \n%2 nan ").arg(series->name()).arg(xDate.toString("MMM dd yyyy")));
                }
                else if (_currentFrequency == hourly)
                {
                    QDateTime xDate(firstDate->date(), QTime(0,0,0), Qt::UTC);
                    xDate = xDate.addSecs(3600*doy);
                    m_tooltip->setText(QString("%1 \n%2 nan ").arg(series->name()).arg(xDate.toString("MMM dd yyyy hh:mm")));
                }
                else if (_currentFrequency == monthly)
                {
                    QDate xDate = firstDate->date().addMonths(doy);
                    m_tooltip->setText(QString("%1 \n%2 nan ").arg(series->name()).arg(xDate.toString("MMM yyyy")));
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

        if (_currentFrequency == daily)
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
        else if (_currentFrequency == hourly)
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
        else if (_currentFrequency == monthly)
        {
            QDate xDate = firstDate->date().addMonths(doy);
            for(int i = 0; i < series->count(); i++)
            {
                if (series->at(i).x() == doy)
                {
                    doyRelative = i;
                    break;
                }
            }
            double value = series->at(doyRelative).y();
            m_tooltip->setText(QString("%1 \n%2 %3 ").arg(series->name()).arg(xDate.toString("MMM yyyy")).arg(value, 0, 'f', 1));
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
            for (int mp=0; mp<_meteoPoints.size();mp++)
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
        if (_currentFrequency == daily)
        {
            QDate xDate = firstDate->date().addDays(index);
            valueStr = QString("%1 \n%2 %3 ").arg(xDate.toString("MMM dd yyyy")).arg(barset->label()).arg(barset->at(index), 0, 'f', 1);
        }
        else if (_currentFrequency == hourly)
        {

            QDateTime xDate(firstDate->date(), QTime(0,0,0), Qt::UTC);
            xDate = xDate.addSecs(3600*index);
            valueStr = QString("%1 \n%2 %3 ").arg(xDate.toString("MMM dd yyyy hh:mm")).arg(barset->label()).arg(barset->at(index), 0, 'f', 1);
        }
        else if (_currentFrequency == monthly)
        {
            QDate xDate = firstDate->date().addMonths(index);
            valueStr = QString("%1 \n%2 %3 ").arg(xDate.toString("MMM yyyy")).arg(barset->label()).arg(barset->at(index), 0, 'f', 1);
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

void Crit3DMeteoWidget::editBar()
{
    QBarSeries *series = qobject_cast<QBarSeries *>(sender());
    QBarSet *barset = series->barSets()[0];
    QMenu menu("Edit");
    QAction* editColor = menu.addAction(QString("Set color"));
    QAction *selection =  menu.exec(QCursor::pos());

    if (selection != nullptr)
    {
        if (selection == editColor)
        {
            QColorDialog colorSelection;
            QColor newColor = colorSelection.getColor(barset->color(), this );
            if( newColor.isValid() )
            {
                for (int i = 0; i<nameBar.size(); i++)
                {
                    QString myName = nameBar[i];
                    if (nameBar[i].contains("DAILY"))
                    {
                        myName.remove("DAILY_");
                    }
                    if (nameBar[i].contains("MONTHLY"))
                    {
                        myName.remove("MONTHLY_");
                    }
                    if (barset->label().contains(myName))
                    {
                        QMapIterator<QString, QList<QString>> iterator(MapCSVStyles);
                        while (iterator.hasNext())
                        {
                            iterator.next();

                            if (iterator.key() == nameBar[i])
                            {
                                QList<QString> newItems = iterator.value();
                                // check which mp is
                                for (int mp = 0; mp < _meteoPoints.size(); mp++)
                                {
                                    QString pointName = QString::fromStdString(_meteoPoints[mp].name);
                                    QString label = getFormattedLabel(pointName, nameBar[i]);
                                    if (barset->label() == label)
                                    {
                                        if (newItems.size() > mp+1)
                                        {
                                            newItems[mp+1] = newColor.name();
                                        }
                                        else
                                        {
                                            newItems.append(newColor.name());
                                        }
                                        break;
                                    }
                                }
                                MapCSVStyles[nameBar[i]] = newItems;
                                break;
                            }
                        }
                    }
                }
                updateSeries();
                redraw();
            }
        }
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
    if(_isGrid)
        emit closeWidgetGrid(_meteoWidgetID);
    else
        emit closeWidgetPoint(_meteoWidgetID);

    delete m_tooltip;
    event->accept();
}

void Crit3DMeteoWidget::setIsEnsemble(bool value)
{
    _isEnsemble = value;
    tableButton->setEnabled(!value);
}


void Crit3DMeteoWidget::on_actionChangeLeftAxis()
{
    DialogChangeAxis changeAxisDialog(1, false);
    if (changeAxisDialog.result() == QDialog::Accepted)
    {
        axisY_sx->setMax(changeAxisDialog.getMaxVal());
        axisY_sx->setMin(changeAxisDialog.getMinVal());
    }
}


void Crit3DMeteoWidget::on_actionChangeRightAxis()
{
    DialogChangeAxis changeAxisDialog(2, false);
    if (changeAxisDialog.result() == QDialog::Accepted)
    {
        axisY_dx->setMax(changeAxisDialog.getMaxVal());
        axisY_dx->setMin(changeAxisDialog.getMinVal());
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
    for (int mp=0; mp<_meteoPoints.size();mp++)
    {
        QString stationId = QString::fromStdString(_meteoPoints[mp].id);
        QString stationsName = QString::fromStdString(_meteoPoints[mp].name);
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
            for (int indexMp=0; indexMp<_meteoPoints.size();indexMp++)
            {
                if (_meteoPoints[indexMp].id == id.toStdString())
                {
                    _meteoPoints.removeAt(indexMp);
                    indexMp = indexMp - 1;
                }
            }
        }
        updateSeries();
        redraw();
    }
}

void Crit3DMeteoWidget::on_actionAddStation()
{
    if (_isGrid)
    {
        QMessageBox::information(nullptr, "Warning", "This option can only be used on meteo points. Meteo grid is not compatible.");
        return;
    }

    QList<QString> allStations;
    for (int mp=0; mp<_meteoPoints.size();mp++)
    {
        QString stationId = QString::fromStdString(_meteoPoints[mp].id);
        QString stationsName = QString::fromStdString(_meteoPoints[mp].name);
        QString station = stationId+"_"+stationsName;
        allStations << station;
    }
    DialogAddStation selectStation(allStations);

}

void Crit3DMeteoWidget::on_actionInfoPoint()
{
    QDialog infoWindow;
    infoWindow.setWindowTitle("Info meteo points");
    QVBoxLayout* layout = new QVBoxLayout();
    for (int mp=0; mp<_meteoPoints.size();mp++)
    {
        QString stationId = QString::fromStdString(_meteoPoints[mp].id);
        QString stationsName = QString::fromStdString(_meteoPoints[mp].name);
        QString station = stationId+"_"+stationsName;
        QString dataset = QString::fromStdString(_meteoPoints[mp].dataset);
        QString altitude = QString::number(_meteoPoints[mp].point.z);
        QString lapseRateName = QString::fromStdString(getLapseRateCodeName(_meteoPoints[mp].lapseRateCode));
        QLabel* label = new QLabel(station);
        layout->addWidget(label);
        QString infoStr;
        if (_isGrid)
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

    if (_isGrid)
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

        if (!firstMonthlyDate.isNull() && firstMonthlyDate.year() != 1800)
        {
            QLabel* labelMonthly = new QLabel("Monthly Data:");
            vbox->addWidget(labelMonthly);

            QString monthlyInfo = QString("%1 - %2").arg(firstMonthlyDate.toString("yyyy/MM"), lastMonthlyDate.toString("yyyy/MM"));
            QTextEdit* monthlyTextEdit = new QTextEdit(monthlyInfo);

            monthlyTextEdit->setMaximumHeight(QFontMetrics(monthlyTextEdit->font()).height() + 10);
            monthlyTextEdit->setReadOnly(true);

            vbox->addWidget(monthlyTextEdit);
        }

        groupBox->setLayout(vbox);
        layout->addWidget(groupBox);
    }
    else
    {
        QDate myFirstDailyDate, myLastDailyDate;
        QDate myFirstHourlyDate, myLastHourlyDate;

        for (int mp=0; mp<_meteoPoints.size();mp++)
        {
            QString stationId = QString::fromStdString(_meteoPoints[mp].id);
            QString stationsName = QString::fromStdString(_meteoPoints[mp].name);
            QString stationString = stationId + "_" + stationsName;

            QGroupBox *groupBox = new QGroupBox(stationString);
            QVBoxLayout *vbox = new QVBoxLayout;

            QString infoDaily = QString("Daily Data:");
            QLabel* labelDaily = new QLabel(infoDaily);
            vbox->addWidget(labelDaily);
            myFirstDailyDate.setDate(_meteoPoints[mp].obsDataD[0].date.year, _meteoPoints[mp].obsDataD[0].date.month, _meteoPoints[mp].obsDataD[0].date.day);
            myLastDailyDate = myFirstDailyDate.addDays(_meteoPoints[mp].nrObsDataDaysD-1);
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
            myFirstHourlyDate.setDate(_meteoPoints[mp].getMeteoPointHourlyValuesDate(0).year, _meteoPoints[mp].getMeteoPointHourlyValuesDate(0).month,
                                      _meteoPoints[mp].getMeteoPointHourlyValuesDate(0).day);
            myLastHourlyDate = myFirstHourlyDate.addDays(_meteoPoints[mp].nrObsDataDaysH-1);

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
    DialogVariableToSum varToSum(currentVariables, varToSumList);
    QList<QString> varToSumListPrev = varToSumList;
    if (varToSum.result() == QDialog::Accepted)
    {
        varToSumList = varToSum.getSelectedVariable();
    }
    else
    {
        if (varToSumList.isEmpty())
        {
            dataSum->setChecked(false);
        }
        else
        {
            dataSum->setChecked(true);
        }
        return;
    }
    if (!varToSumList.isEmpty())
    {
        dataSum->setChecked(true);
    }
    else
    {
        dataSum->setChecked(false);
    }
    if (varToSumListPrev != varToSumList) //something is changed
    {
        redraw();
    }
}

void Crit3DMeteoWidget::drawSum()
{
    int nMeteoPoints = _meteoPoints.size();
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
                axisY_sx->setRange(axisY_sx->min(),max);
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
                axisY_dx->setRange(axisY_dx->min(),max);
            }
        }
    }
}


qreal findMedian(std::vector<double> sortedList, int begin, int end)
{
    int count = end - begin;
    if (count % 2)
    {
        return sortedList.at(count / 2 + begin);
    }
    else
    {
        qreal right = sortedList.at(count / 2 + begin);
        qreal left = sortedList.at(count / 2 - 1 + begin);
        return (right + left) / 2.0;
    }
}
