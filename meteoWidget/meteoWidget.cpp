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
    You should have received a copy of the GNU Lesser General Public License
    along with CRITERIA3D.  If not, see <http://www.gnu.org/licenses/>.
    contacts:
    fausto.tomei@gmail.com
    ftomei@arpae.it
*/


#include "meteoWidget.h"
#include "dialogSelectVar.h"
#include "dialogMeteoTable.h"
#include "utilities.h"
#include "commonConstants.h"
#include "formInfo.h"

#include <QMessageBox>
#include <QLayout>
#include <QMenu>
#include <QMenuBar>
#include <QPushButton>
#include <QDate>
#include <QtGlobal>
#include <QDebug>


Crit3DMeteoWidget::Crit3DMeteoWidget(bool isGrid, QString projectPath)
{
    this->isGrid = isGrid;

    if (this->isGrid)
    {
        this->setWindowTitle("Grid");
    }
    else
    {
        this->setWindowTitle("Point");
    }

    this->resize(1240, 700);
    currentFreq = noFrequency;
    firstDailyDate = QDate::currentDate();
    firstHourlyDate = QDate::currentDate();
    lastDailyDate = QDate(1800,1,1);
    lastHourlyDate = QDate(1800,1,1);

    isLine = false;
    isBar = false;
    QVector<QLineSeries*> vectorLine;
    QVector<QBarSet*> vectorBarSet;

    QString csvPath, defaultPath, stylesPath;
    if (!projectPath.isEmpty())
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
    if ( !fileDefaultGraph.open(QFile::ReadOnly | QFile::Text) ) {
        qDebug() << "File not exists";
        QMessageBox::information(nullptr, "Warning", "Missing Crit3DPlotDefault.csv");
    }
    else
    {
        QTextStream in(&fileDefaultGraph);
        in.readLine(); //skip first line
        while (!in.atEnd())
        {
            QString line = in.readLine();
            QStringList items = line.split(",");
            if (items.size() < CSVRequiredInfo)
            {
                qDebug() << "invalid format CSV, missing data";
                currentVariables.clear();
                break;
            }
            QString key = items[0];
            items.removeFirst();
            if (key.isEmpty() || items[0].isEmpty())
            {
                qDebug() << "invalid format CSV, missing data";
                currentVariables.clear();
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
                currentVariables.clear();
                break;
            }
        }
    }
    // check valid data
    int dailyVar = 0;
    int hourlyVar = 0;
    for (int i = 0; i<currentVariables.size(); i++)
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
        qDebug() << "invalid format CSV";
        currentFreq = noFrequency;
        currentVariables.clear();
        nameLines.clear();
        nameBar.clear();
        MapCSVDefault.clear();
        isLine = false;
        isBar = false;
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
    }
    else
    {
        QTextStream in(&fileStylesGraph);
        in.readLine(); //skip first line
        while (!in.atEnd())
        {
            QString line = in.readLine();
            QStringList items = line.split(",");
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
    addVarButton = new QPushButton(tr("+/- var"));
    tableButton = new QPushButton(tr("view table"));
    redrawButton = new QPushButton(tr("redraw"));
    QLabel *labelFirstDate = new QLabel(tr("Start Date: "));
    QLabel *labelEndDate = new QLabel(tr("End Date: "));
    firstDate = new QDateTimeEdit(QDate::currentDate());
    lastDate = new QDateTimeEdit(QDate::currentDate());
    dailyButton->setMaximumWidth(100);
    hourlyButton->setMaximumWidth(100);
    addVarButton->setMaximumWidth(100);
    tableButton->setMaximumWidth(100);
    redrawButton->setMaximumWidth(100);

    if (currentFreq == daily || currentFreq == noFrequency)
    {
        dailyButton->setEnabled(false);
        hourlyButton->setEnabled(true);
        firstDate->setDisplayFormat("dd/MM/yyyy");
        lastDate->setDisplayFormat("dd/MM/yyyy");
        firstDate->setMaximumWidth(100);
        lastDate->setMaximumWidth(100);
    }
    else
    {
        hourlyButton->setEnabled(false);
        dailyButton->setEnabled(true);
        firstDate->setDisplayFormat("dd/MM/yyyy hh:mm");
        lastDate->setDisplayFormat("dd/MM/yyyy hh:mm");
        firstDate->setMaximumWidth(140);
        lastDate->setMaximumWidth(140);
    }

    buttonLayout->addWidget(dailyButton);
    buttonLayout->addWidget(hourlyButton);
    buttonLayout->addWidget(addVarButton);
    buttonLayout->addWidget(labelFirstDate);
    buttonLayout->addWidget(firstDate);
    buttonLayout->addWidget(labelEndDate);
    buttonLayout->addWidget(lastDate);
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

    QDate first(QDate::currentDate().year(), 1, 1);
    QDate last(QDate::currentDate().year(), 12, 31);
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

    connect(addVarButton, &QPushButton::clicked, [=](){ showVar(); });
    connect(dailyButton, &QPushButton::clicked, [=](){ showDailyGraph(); });
    connect(hourlyButton, &QPushButton::clicked, [=](){ showHourlyGraph(); });
    connect(tableButton, &QPushButton::clicked, [=](){ showTable(); });
    connect(redrawButton, &QPushButton::clicked, [=](){ updateDate(); });

    plotLayout->addWidget(chartView);
    horizontalGroupBox->setLayout(buttonLayout);
    mainLayout->addWidget(horizontalGroupBox);
    mainLayout->addLayout(plotLayout);
    setLayout(mainLayout);

}

void Crit3DMeteoWidget::draw(Crit3DMeteoPoint mp)
{
    firstDate->blockSignals(true);
    lastDate->blockSignals(true);

    meteoPoints.append(mp);
    QDate myDailyDateFirst;
    QDate myDailyDateLast;
    QDate myHourlyDateFirst;
    QDate myHourlyDateLast;
    // search bigger data interval to show between meteoPoints
    for (int i = 0; i < meteoPoints.size(); i++)
    {
        myDailyDateFirst.setDate(meteoPoints[i].obsDataD[0].date.year, meteoPoints[i].obsDataD[0].date.month, meteoPoints[i].obsDataD[0].date.day);
        myDailyDateLast = myDailyDateFirst.addDays(meteoPoints[i].nrObsDataDaysD-1);
        if (myDailyDateFirst.isValid() && myDailyDateFirst < firstDailyDate)
        {
            firstDailyDate = myDailyDateFirst;
        }
        if (myDailyDateLast.isValid() && myDailyDateLast > lastDailyDate)
        {
            lastDailyDate = myDailyDateLast;
        }
        myHourlyDateFirst.setDate(meteoPoints[i].getMeteoPointHourlyValuesDate(0).year, meteoPoints[i].getMeteoPointHourlyValuesDate(0).month,
                                  meteoPoints[i].getMeteoPointHourlyValuesDate(0).day);
        myHourlyDateLast = myHourlyDateFirst.addDays(meteoPoints[i].nrObsDataDaysH-1);
        if (myHourlyDateFirst.isValid() && myHourlyDateFirst < firstHourlyDate)
        {
            firstHourlyDate = myHourlyDateFirst;
        }
        if (myHourlyDateLast.isValid() && myHourlyDateLast > lastHourlyDate)
        {
            lastHourlyDate = myHourlyDateLast;
        }
    }
    resetValues();
    if (currentFreq == daily)
    {
        if (firstDailyDate == QDate::currentDate() && lastDailyDate == QDate(1800,1,1))
        {
            firstDate->setDate(QDate::currentDate());
            lastDate->setDate(QDate::currentDate());
        }
        else
        {
            firstDate->setMinimumDate(firstDailyDate);
            firstDate->setDate(firstDailyDate);
            lastDate->setDate(lastDailyDate);
            lastDate->setMaximumDate(lastDailyDate);
        }
        drawDailyVar();
    }
    else if (currentFreq == hourly)
    {
        if (firstHourlyDate == QDate::currentDate() && lastHourlyDate == QDate(1800,1,1))
        {
            firstDate->setDate(QDate::currentDate());
            lastDate->setDate(QDate::currentDate());
        }
        else
        {
            firstDate->setMinimumDate(firstHourlyDate);
            firstDate->setDate(firstHourlyDate);
            lastDate->setDate(lastHourlyDate);
            lastDate->setMaximumDate(lastHourlyDate);
        }
        drawHourlyVar();
    }
    firstDate->blockSignals(false);
    lastDate->blockSignals(false);

    show();

}

void Crit3DMeteoWidget::resetValues()
{

    int nMeteoPoints = meteoPoints.size();
    QVector<QColor> colorLine;
    for (int i = 0; i< nameLines.size(); i++)
    {
        colorLine.append(lineSeries[0][i]->color());
    }

    for (int i = 0; i < nameBar.size(); i++)
    {
        colorBar.append(setVector[0][i]->color());
    }

    // clear prev series values
    if (isLine)
    {
        for (int mp = 0; mp<lineSeries.size(); mp++)
        {
            for (int i = 0; i < lineSeries[mp].size(); i++)
            {
                lineSeries[mp][i]->clear();
            }
            lineSeries[mp].clear();
        }
        lineSeries.clear();
    }
    if (isBar)
    {
        for (int mp = 0; mp<lineSeries.size(); mp++)
        {
            setVector[mp].clear();
            barSeries[mp]->clear();
        }
        barSeries.clear();
        setVector.clear();
    }

    chart->removeAllSeries();
    delete m_tooltip;

    if (isLine)
    {
        QVector<QLineSeries*> vectorLine;
        // add lineSeries elements for each mp, clone lineSeries[0]
        for (int mp=0; mp<nMeteoPoints;mp++)
        {
            vectorLine.clear();
            for (int i = 0; i<nameLines.size(); i++)
            {
                QLineSeries* line = new QLineSeries();
                QString pointName = QString::fromStdString(meteoPoints[mp].name);
                QStringList elementsName = pointName.split(' ');
                if (elementsName.size() == 1)
                {
                    pointName = elementsName[0].left(8);
                }
                else
                {
                    pointName = elementsName[0].left(4)+elementsName[elementsName.size()-1].left(4);
                }
                line->setName(QString::fromStdString(meteoPoints[mp].id)+"_"+pointName+"_"+nameLines[i]);
                QColor lineColor = colorLine[i];
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

    if (isBar)
    {
        QVector<QBarSet*> vectorBarSet;
        QString name;
        // add vectorBarSet elements for each mp
        for (int mp=0; mp<nMeteoPoints;mp++)
        {
            vectorBarSet.clear();
            for (int i = 0; i < nameBar.size(); i++)
            {
                QString pointName = QString::fromStdString(meteoPoints[mp].name);
                QStringList elementsName = pointName.split(' ');
                if (elementsName.size() == 1)
                {
                    pointName = elementsName[0].left(8);
                }
                else
                {
                    pointName = elementsName[0].left(4)+elementsName[elementsName.size()-1].left(4);
                }
                name = QString::fromStdString(meteoPoints[mp].id)+"_"+pointName+"_"+nameBar[i];
                QBarSet* set = new QBarSet(name);
                QColor barColor = colorBar[i];
                if (nMeteoPoints == 1)
                {
                    barColor.setAlpha(255);
                }
                else
                {
                    barColor.setAlpha( 255-(mp*(150/(nMeteoPoints-1))) );
                }
                set->setColor(barColor);
                set->setBorderColor(barColor);
                vectorBarSet.append(set);
            }
            if (vectorBarSet.size() != 0)
            {
                setVector.append(vectorBarSet);
            }
        }
    }
}

void Crit3DMeteoWidget::drawDailyVar()
{

    FormInfo formInfo;
    formInfo.showInfo("Draw daily data...");

    firstDate->blockSignals(true);
    lastDate->blockSignals(true);

    Crit3DDate myDate;
    int nDays = 0;
    double maxBar = 0;
    double maxLine = NODATA;
    double minLine = -NODATA;

    Crit3DDate firstCrit3DDate = getCrit3DDate(firstDate->date());
    Crit3DDate lastfirstCrit3DDate = getCrit3DDate(lastDate->date());
    nDays = firstCrit3DDate.daysTo(lastfirstCrit3DDate)+1;

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
                    meteoVariable meteoVar = MapDailyMeteoVar.at(nameLines[i].toStdString());
                    double value = meteoPoints[mp].getMeteoPointValueD(myDate, meteoVar);
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
                }
            }
            if (isBar)
            {

                for (int j = 0; j < nameBar.size(); j++)
                {
                    meteoVariable meteoVar = MapDailyMeteoVar.at(nameBar[j].toStdString());
                    double value = meteoPoints[mp].getMeteoPointValueD(myDate, meteoVar);
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
        for (int mp=0; mp<nMeteoPoints;mp++)
        {
            QBarSeries* barMpSeries = new QBarSeries();
            for (int i = 0; i < nameBar.size(); i++)
            {
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
        axisYdx->setVisible(true);
        axisYdx->setRange(0,maxBar);
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

        axisY->setVisible(true);
        axisY->setMax(maxLine);
        axisY->setMin(minLine);
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

    FormInfo formInfo;
    formInfo.showInfo("Draw hourly data...");

    firstDate->blockSignals(true);
    lastDate->blockSignals(true);

    Crit3DTime myDate;
    int nDays = 0;
    double maxBar = 0;
    double maxLine = NODATA;
    double minLine = -NODATA;

    Crit3DTime firstCrit3DDate(getCrit3DDate(firstDate->date()),0);
    Crit3DTime lastCrit3DDate(getCrit3DDate(lastDate->date()),0);
    nDays = firstCrit3DDate.date.daysTo(lastCrit3DDate.date)+1;
    int nValues = nDays*24;

    categories.clear();
    categoriesVirtual.clear();
    m_tooltip = new Callout(chart);
    m_tooltip->hide();

    // virtual x axis
    int nrIntervals;

    if (nValues <= 45)
    {
        nrIntervals = nValues/3;
    }
    else
    {
        nrIntervals = 12;
    }
    double step = double(nValues) / double(nrIntervals);
    double nextIndex = step / 2 - 0.5;

    for (int value = 0; value < nValues; value++)
    {
        myDate = firstCrit3DDate.addSeconds(value*3600);
        if (value == round(nextIndex))
        {
            categoriesVirtual.append(getQDateTime(myDate).toString("MMM dd <br> yyyy <br> hh:mm"));
            nextIndex += step;
        }
    }

    int nMeteoPoints = meteoPoints.size();
    for (int cont = 0; cont< nValues; cont++)
    {
        myDate = firstCrit3DDate.addSeconds(cont*3600);
        categories.append(QString::number(cont));

        for (int mp=0; mp<nMeteoPoints;mp++)
        {
            if (isLine)
            {
                for (int i = 0; i < nameLines.size(); i++)
                {
                    meteoVariable meteoVar = MapHourlyMeteoVar.at(nameLines[i].toStdString());
                    double value = meteoPoints[mp].getMeteoPointValueH(myDate.date, myDate.getHour(), 0, meteoVar);
                    if (value != NODATA)
                    {
                        lineSeries[mp][i]->append(cont, value);
                        if (value > maxLine)
                        {
                            maxLine = value;
                        }
                        if (value < minLine)
                        {
                            minLine = value;
                        }
                    }
                }
            }
            if (isBar)
            {
                for (int j = 0; j < nameBar.size(); j++)
                {
                    meteoVariable meteoVar = MapHourlyMeteoVar.at(nameBar[j].toStdString());
                    double value = meteoPoints[mp].getMeteoPointValueH(myDate.date, myDate.getHour(), 0, meteoVar);
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
        for (int mp=0; mp<nMeteoPoints;mp++)
        {
            QBarSeries* barMpSeries = new QBarSeries();
            for (int i = 0; i < nameBar.size(); i++)
            {
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
        axisYdx->setVisible(true);
        axisYdx->setRange(0,maxBar);
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

        axisY->setVisible(true);
        axisY->setMax(maxLine);
        axisY->setMin(minLine);
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
    if (currentFreq == noFrequency)
    {
        if (!dailyButton->isEnabled()) // dailyButton is pressed
        {
            currentFreq = daily;
        }
        else
        {
            currentFreq = hourly;
        }
    }
    QStringList allKeys = MapCSVStyles.keys();
    QStringList selectedVar = currentVariables;
    QStringList allVar;
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
    }
    DialogSelectVar selectDialog(allVar, selectedVar);
    if (selectDialog.result() == QDialog::Accepted)
    {
        currentVariables.clear();
        currentVariables = selectDialog.getSelectedVariables();
        updateSeries();
        if (currentFreq == daily)
        {
            drawDailyVar();
        }
        else if (currentFreq == hourly)
        {
            drawHourlyVar();
        }
    }
}

void Crit3DMeteoWidget::showDailyGraph()
{
    currentFreq = daily;
    if (firstDailyDate == QDate::currentDate() && lastDailyDate == QDate(1800,1,1))
    {
        firstDate->setMinimumDate(QDate::currentDate());
        firstDate->setDate(QDate::currentDate());
        lastDate->setMaximumDate(QDate::currentDate());
        lastDate->setDate(QDate::currentDate());
    }
    else
    {
        firstDate->setMinimumDate(firstDailyDate);
        firstDate->setDate(firstDailyDate);
        lastDate->setDate(lastDailyDate);
        lastDate->setMaximumDate(lastDailyDate);
    }
    firstDate->setDisplayFormat("dd/MM/yyyy");
    lastDate->setDisplayFormat("dd/MM/yyyy");
    firstDate->setMaximumWidth(100);
    lastDate->setMaximumWidth(100);

    firstDate->setMinimumWidth(firstDate->width()-firstDate->width()*0.3);
    lastDate->setMinimumWidth(lastDate->width()-lastDate->width()*0.3);

    dailyButton->setEnabled(false);
    hourlyButton->setEnabled(true);

    QStringList currentHourlyVar = currentVariables;
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
                currentVariables.append(QString::fromStdString(MapDailyMeteoVarToString.at(dailyVar)));
            }
        }
    }

    updateSeries();
    drawDailyVar();

}

void Crit3DMeteoWidget::showHourlyGraph()
{
    currentFreq = hourly;
    if (firstHourlyDate == QDate::currentDate() && lastHourlyDate == QDate(1800,1,1))
    {
        firstDate->setMinimumDate(QDate::currentDate());
        firstDate->setDate(QDate::currentDate());
        lastDate->setMaximumDate(QDate::currentDate());
        lastDate->setDate(QDate::currentDate());
    }
    else
    {
        firstDate->setMinimumDate(firstHourlyDate);
        firstDate->setDate(firstHourlyDate);
        lastDate->setDate(lastHourlyDate);
        lastDate->setMaximumDate(lastHourlyDate);
    }
    firstDate->setDisplayFormat("dd/MM/yyyy hh:mm");
    lastDate->setDisplayFormat("dd/MM/yyyy hh:mm");
    firstDate->setMaximumWidth(140);
    lastDate->setMaximumWidth(140);

    firstDate->setMinimumWidth(firstDate->width()+firstDate->width()*0.3);
    lastDate->setMinimumWidth(lastDate->width()+lastDate->width()*0.3);

    hourlyButton->setEnabled(false);
    dailyButton->setEnabled(true);

    QStringList currentDailyVar = currentVariables;
    currentVariables.clear();

    for (int i = 0; i<currentDailyVar.size(); i++)
    {
        QString name = currentDailyVar[i];
        auto searchDaily = MapDailyMeteoVar.find(name.toStdString());
        if (searchDaily != MapDailyMeteoVar.end())
        {
            meteoVariable dailyVar = MapDailyMeteoVar.at(name.toStdString());
            meteoVariable hourlyVar= updateMeteoVariable(dailyVar, hourly);
            if (hourlyVar != noMeteoVar)
            {
                currentVariables.append(QString::fromStdString(MapHourlyMeteoVarToString.at(hourlyVar)));
            }
        }
    }
    updateSeries();
    drawHourlyVar();
}

void Crit3DMeteoWidget::updateSeries()
{
    barSeries.clear();
    setVector.clear();
    lineSeries.clear();
    chart->removeAllSeries();
    nameLines.clear();
    nameBar.clear();
    isLine = false;
    isBar = false;

    QVector<QLineSeries*> vectorLine;
    QVector<QBarSet*> vectorBarSet;
    QMapIterator<QString, QStringList> i(MapCSVStyles);

    while (i.hasNext())
    {
        i.next();
        for (int j=0; j<currentVariables.size(); j++)
        {
            if (i.key() == currentVariables[j])
            {
                QStringList items = i.value();
                if (items[0] == "line")
                {
                    isLine = true;
                    nameLines.append(i.key());
                    QLineSeries* line = new QLineSeries();
                    line->setName(i.key());
                    line->setColor(QColor(items[1]));
                    vectorLine.append(line);
                }
                if (items[0] == "bar")
                {
                    isBar = true;
                    nameBar.append(i.key());
                    colorBar.append(QColor(items[1]));
                    QBarSet* set = new QBarSet(i.key());
                    set->setColor(QColor(items[1]));
                    set->setBorderColor(QColor(items[1]));
                    vectorBarSet.append(set);
                }
            }
        }
    }

    if (isLine)
    {
        lineSeries.append(vectorLine);
    }
    if (isBar)
    {
        setVector.append(vectorBarSet);
        QBarSeries* barFirstSeries = new QBarSeries();
        for (int i = 0; i < setVector[0].size(); i++)
        {
            barFirstSeries->append(setVector[0][i]);
        }
        barSeries.append(barFirstSeries);
    }

    resetValues();

}

void Crit3DMeteoWidget::updateDate()
{

    resetValues();
    if (currentFreq == daily)
    {
        drawDailyVar();
    }
    else if (currentFreq == hourly)
    {
        drawHourlyVar();
    }

}

void Crit3DMeteoWidget::showTable()
{
    DialogMeteoTable meteoTable(meteoPoints, firstDate->date(), lastDate->date(), currentFreq, currentVariables);
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
        int doy = point.x(); //start from 0
        QPoint CursorPoint = QCursor::pos();
        QPoint mapPoint = chartView->mapFromGlobal(CursorPoint);
        QPoint pointDoY = series->at(doy).toPoint();

        if (doy == 0)
        {
            QPoint pointNext = series->at(doy+1).toPoint();
            int distStep = qAbs(chart->mapToPosition(pointDoY).x()-chart->mapToPosition(pointNext).x());
            int distDoY = qAbs(mapPoint.x()-chart->mapToPosition(pointDoY).x());
            int distNext = qAbs(mapPoint.x()-chart->mapToPosition(pointNext).x());

            if (qMin(distDoY, distNext) == distNext)
            {
                if (distNext > distStep/10)
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
        else if (doy > 0 && doy < series->count())
        {
            QPoint pointBefore = series->at(doy-1).toPoint();
            QPoint pointNext = series->at(doy+1).toPoint();

            int distStep = qAbs(chart->mapToPosition(pointDoY).x()-chart->mapToPosition(pointNext).x());
            int distDoY = qAbs(mapPoint.x()-chart->mapToPosition(pointDoY).x());
            int distNext = qAbs(mapPoint.x()-chart->mapToPosition(pointNext).x());
            int distBefore = qAbs(mapPoint.x()-chart->mapToPosition(pointBefore).x());

            if (qMin(qMin(distBefore,distDoY), distNext) == distBefore)
            {
                if (distBefore > distStep/10)
                {
                    return false;
                }
                else
                {
                    doy = doy - 1;
                }
            }
            else if (qMin(qMin(distBefore,distDoY), distNext) == distNext)
            {
                if (distNext > distStep/10)
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
        else if (doy == series->count())
        {
            QPoint pointBefore = series->at(doy-1).toPoint();
            QPoint pointDoY = series->at(doy).toPoint();
            int distStep = qAbs(chart->mapToPosition(pointDoY).x()-chart->mapToPosition(pointBefore).x());

            int distBefore = qAbs(mapPoint.x()-chart->mapToPosition(pointBefore).x());
            int distDoY = qAbs(mapPoint.x()-chart->mapToPosition(pointDoY).x());

            if (qMin(distDoY, distBefore) == distBefore)
            {
                if (distBefore > distStep/10)
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
                if (distDoY > distStep/10)
                {
                    return false;
                }
            }

        }

        if (currentFreq == daily)
        {
            QDate xDate = firstDate->date().addDays(doy);
            double value = series->at(doy).y();
            m_tooltip->setText(QString("%1 \n%2 %3 ").arg(series->name()).arg(xDate.toString("MMM dd yyyy")).arg(value, 0, 'f', 1));
        }
        if (currentFreq == hourly)
        {
            QDateTime xDate(firstDate->date(),QTime(0,0,0));
            xDate = xDate.addSecs(3600*doy);
            double value = series->at(doy).y();
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
                    double lineSeriesY = lineSeries[mp][i]->at(pointF.toPoint().x()).y();
                    if (  static_cast<int>( lineSeriesY) == pointF.toPoint().y())
                    {
                        if (computeTooltipLineSeries(lineSeries[mp][i], pointF, true))
                        {
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

            QDateTime xDate(firstDate->date(),QTime(0,0,0));
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
    if (marker->type() == QLegendMarker::LegendMarkerTypeXY)
    {
        // Toggle visibility of series
        marker->series()->setVisible(!marker->series()->isVisible());

        // Turn legend marker back to visible, since otherwise hiding series also hides the marker
        marker->setVisible(true);

        // change marker alpha, if series is not visible
        qreal alpha = 1.0;

        if (!marker->series()->isVisible()) {
            alpha = 0.5;
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
    else if (marker->type() == QLegendMarker::LegendMarkerTypeBar)
    {
        // Toggle visibility of series
        marker->series()->setVisible(!marker->series()->isVisible());

        // change marker alpha, if series is not visible
        qreal alpha = 1.0;

        // Turn legend marker back to visible, since otherwise hiding series also hides the marker
        foreach(QLegendMarker* marker, chart->legend()->markers())
        {
            if (marker->type() == QLegendMarker::LegendMarkerTypeBar)
            {
                marker->setVisible(true);
            }
            if (!marker->series()->isVisible()) {
                alpha = 0.5;
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

    }

}

void Crit3DMeteoWidget::closeEvent(QCloseEvent *event)
{
    event->accept();
    emit closeWidget(meteoWidgetID);
}

int Crit3DMeteoWidget::getMeteoWidgetID() const
{
    return meteoWidgetID;
}

void Crit3DMeteoWidget::setMeteoWidgetID(int value)
{
    meteoWidgetID = value;
}



