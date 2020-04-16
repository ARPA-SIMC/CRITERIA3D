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


Crit3DMeteoWidget::Crit3DMeteoWidget()
{
    this->setWindowTitle(QStringLiteral("Graph"));
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
    if (searchDataPath(&csvPath))
    {
        defaultPath = csvPath + "SETTINGS/Crit3DPlotDefault.csv";
        stylesPath = csvPath + "SETTINGS/Crit3DPlotStyles.csv";
    }

    // read Crit3DPlotDefault and fill MapCSVDefault
    int CSVRequiredInfo = 3;
    QFile fileDefaultGraph(defaultPath);
    if ( !fileDefaultGraph.open(QFile::ReadOnly | QFile::Text) ) {
        qDebug() << "File not exists";
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
                qDebug() << "invalid format CSV, missing data";
            }
            QString key = items[0];
            items.removeFirst();
            if (key.isEmpty() || items[0].isEmpty())
            {
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
    QLabel *labelFirstDate = new QLabel(tr("Start Date: "));
    QLabel *labelEndDate = new QLabel(tr("End Date: "));
    firstDate = new QDateTimeEdit(QDate::currentDate());
    lastDate = new QDateTimeEdit(QDate::currentDate());
    dailyButton->setMaximumWidth(this->width()/8);
    hourlyButton->setMaximumWidth(this->width()/8);
    addVarButton->setMaximumWidth(this->width()/8);

    if (currentFreq == daily || currentFreq == noFrequency)
    {
        dailyButton->setEnabled(false);
        hourlyButton->setEnabled(true);
        firstDate->setDisplayFormat("dd/MM/yyyy");
        lastDate->setDisplayFormat("dd/MM/yyyy");
    }
    else
    {
        hourlyButton->setEnabled(false);
        dailyButton->setEnabled(true);
        firstDate->setDisplayFormat("dd/MM/yyyy hh:mm");
        lastDate->setDisplayFormat("dd/MM/yyyy hh:mm");
    }

    buttonLayout->addWidget(dailyButton);
    buttonLayout->addWidget(hourlyButton);
    buttonLayout->addWidget(addVarButton);
    buttonLayout->addWidget(labelFirstDate);
    buttonLayout->addWidget(firstDate);
    buttonLayout->addWidget(labelEndDate);
    buttonLayout->addWidget(lastDate);
    buttonLayout->setAlignment(Qt::AlignLeft);
    chart = new QChart();
    chartView = new QChartView(chart);
    chartView->setChart(chart);

    axisX = new QBarCategoryAxis();
    axisXvirtual = new QDateTimeAxis();
    axisY = new QValueAxis();
    axisYdx = new QValueAxis();

    QDate first(QDate::currentDate().year(), 1, 1);
    QDate last(QDate::currentDate().year(), 12, 31);
    axisX->setTitleText("Date");
    axisX->setGridLineVisible(false);
    axisXvirtual->setTitleText("Date");
    axisXvirtual->setFormat("MMM dd <br> yyyy");
    axisXvirtual->setMin(QDateTime(first, QTime(0,0,0)));
    axisXvirtual->setMax(QDateTime(last, QTime(0,0,0)));
    axisXvirtual->setTickCount(13);

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

    connect(addVarButton, &QPushButton::clicked, [=](){ showVar(); });
    connect(dailyButton, &QPushButton::clicked, [=](){ showDailyGraph(); });
    connect(hourlyButton, &QPushButton::clicked, [=](){ showHourlyGraph(); });
    connect(firstDate, &QDateTimeEdit::editingFinished, [=](){ updateDate(); });
    connect(lastDate, &QDateTimeEdit::editingFinished, [=](){ updateDate(); });

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
}

void Crit3DMeteoWidget::resetValues()
{

    QVector<QColor> colorLine;
    for (int i = 0; i< nameLines.size(); i++)
    {
        colorLine.append(lineSeries[0][i]->color());
    }

    QVector<QColor> colorBar;
    for (int i = 0; i < nameBar.size(); i++)
    {
        colorBar.append(setVector[0][i]->color());
    }

    // clear prev series values
    for (int mp=0; mp<meteoPoints.size()-1;mp++)
    {
        if (isLine)
        {
            for (int i = 0; i < lineSeries[mp].size(); i++)
            {
                lineSeries[mp][i]->clear();
            }
            lineSeries[mp].clear();
        }
        if (isBar)
        {
            setVector[mp].clear();
            barSeries[mp]->clear();
        }
    }

    if (isLine)
    {
        lineSeries.clear();
    }
    if (isBar)
    {
        barSeries.clear();
        setVector.clear();
    }

    chart->removeAllSeries();

    if (isLine)
    {
        QVector<QLineSeries*> vectorLine;
        // add lineSeries elements for each mp, clone lineSeries[0]
        for (int mp=0; mp<meteoPoints.size();mp++)
        {
            vectorLine.clear();
            for (int i = 0; i<nameLines.size(); i++)
            {
                QLineSeries* line = new QLineSeries();
                line->setName("ID"+QString::fromStdString(meteoPoints[mp].id)+"_"+nameLines[i]);
                line->setColor(colorLine[i]);
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
        for (int mp=0; mp<meteoPoints.size();mp++)
        {
            vectorBarSet.clear();
            for (int i = 0; i < nameBar.size(); i++)
            {
                name = "ID"+QString::fromStdString(meteoPoints[mp].id)+"_"+nameBar[i];
                QBarSet* set = new QBarSet(name);
                set->setColor(colorBar[i]);
                set->setBorderColor(colorBar[i]);
                vectorBarSet.append(set);
            }
            if (vectorBarSet.size() != 0)
            {
                setVector.append(vectorBarSet);
            }
        }
    }


    for (int mp=0; mp<meteoPoints.size();mp++)
    {
        if (isLine)
        {
            for (int i = 0; i < nameLines.size(); i++)
            {
                chart->addSeries(lineSeries[mp][i]);
                lineSeries[mp][i]->attachAxis(axisX);
                lineSeries[mp][i]->attachAxis(axisY);
            }
        }
    }
}

void Crit3DMeteoWidget::drawDailyVar()
{

    FormInfo formInfo;
    firstDate->blockSignals(true);
    lastDate->blockSignals(true);

    Crit3DDate myDate;
    int nDays = 0;
    double maxBar = 0;
    double maxLine = NODATA;

    Crit3DDate firstCrit3DDate = getCrit3DDate(firstDate->date());
    Crit3DDate lastfirstCrit3DDate = getCrit3DDate(lastDate->date());
    nDays = firstCrit3DDate.daysTo(lastfirstCrit3DDate);

    int step = formInfo.start("Compute model...", nDays);
    int cont = 0;

    categories.clear();
    for (int day = 0; day<=nDays; day++)
    {
        myDate = firstCrit3DDate.addDays(day);
        categories.append(QString::number(day));
        for (int mp=0; mp<meteoPoints.size();mp++)
        {
            if ( (cont % step) == 0) formInfo.setValue(cont);
            if (isLine)
            {
                for (int i = 0; i < nameLines.size(); i++)
                {
                    meteoVariable meteoVar = MapDailyMeteoVar.at(nameLines[i].toStdString());
                    double value = meteoPoints[mp].getMeteoPointValueD(myDate, meteoVar);
                    lineSeries[mp][i]->append(day, value);
                    if (value > maxLine)
                    {
                        maxLine = value;
                    }
                }
            }
            if (isBar)
            {

                for (int j = 0; j < nameBar.size(); j++)
                {
                    meteoVariable meteoVar = MapDailyMeteoVar.at(nameBar[j].toStdString());
                    double value = meteoPoints[mp].getMeteoPointValueD(myDate, meteoVar);
                    *setVector[mp][j] << value;
                    if (value > maxBar)
                    {
                        maxBar = value;
                    }
                }
            }
            cont++; // formInfo update
        }
    }
    formInfo.close();

    if (isBar)
    {
        for (int mp=0; mp<meteoPoints.size();mp++)
        {
            QBarSeries* barMpSeries = new QBarSeries();
            for (int i = 0; i < nameBar.size(); i++)
            {
                barMpSeries->append(setVector[mp][i]);
            }
            barSeries.append(barMpSeries);
        }

        for (int mp=0; mp<meteoPoints.size();mp++)
        {
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
        axisY->setVisible(true);
        axisY->setMax(maxLine);
    }
    else
    {
        axisY->setVisible(false);
    }

    int tCount = 0;
    if (nDays<=2)
    {
        // add minimimum values required
        if (nDays==0)
        {
            categories.append(QString::number(1));
            for (int mp=0; mp<meteoPoints.size();mp++)
            {
                if (isLine)
                {
                    for (int i = 0; i < nameLines.size(); i++)
                    {
                        lineSeries[0][0]->append(1, NODATA);
                    }
                }
            }
        }
        tCount = 2;
    }
    else
    {
        tCount = qMin(nDays+1,13);
    }

    axisX->setCategories(categories);
    axisX->setGridLineVisible(false);
    // update virtual x axis
    axisXvirtual->setFormat("MMM dd <br> yyyy");

    axisXvirtual->setTickCount(tCount);
    axisXvirtual->setMin(QDateTime(firstDate->date(), QTime(0,0,0)));
    if (firstDate->date() == lastDate->date())
    {
        axisXvirtual->setMax(QDateTime(lastDate->date().addDays(1), QTime(0,0,0)));
    }
    else
    {
        axisXvirtual->setMax(QDateTime(lastDate->date(), QTime(0,0,0)));
    }
    firstDate->blockSignals(false);
    lastDate->blockSignals(false);
}

void Crit3DMeteoWidget::drawHourlyVar()
{

    FormInfo formInfo;
    firstDate->blockSignals(true);
    lastDate->blockSignals(true);

    Crit3DDate myDate;
    long nDays = 0;
    int nValues = 0;
    double maxBar = 0;
    double maxLine = NODATA;

    Crit3DDate firstCrit3DDate = getCrit3DDate(firstDate->date());
    Crit3DDate lastCrit3DDate = getCrit3DDate(lastDate->date());
    nDays = firstCrit3DDate.daysTo(lastCrit3DDate);

    int step = formInfo.start("Compute model...", (nDays+1)*24);

    categories.clear();
    for (int day = 0; day<=nDays; day++)
    {
        myDate = firstCrit3DDate.addDays(day);
        for (int hour = 1; hour < 25; hour++)
        {
            if ( (nValues % step) == 0) formInfo.setValue(nValues);
            categories.append(QString::number(nValues));
            for (int mp=0; mp<meteoPoints.size();mp++)
            {
                if (isLine)
                {
                    for (int i = 0; i < nameLines.size(); i++)
                    {
                        meteoVariable meteoVar = MapHourlyMeteoVar.at(nameLines[i].toStdString());
                        double value = meteoPoints[mp].getMeteoPointValueH(myDate, hour, 0, meteoVar);
                        lineSeries[mp][i]->append(nValues, value);
                        if (value > maxLine)
                        {
                            maxLine = value;
                        }
                    }
                }
                if (isBar)
                {
                    for (int j = 0; j < nameBar.size(); j++)
                    {
                        meteoVariable meteoVar = MapHourlyMeteoVar.at(nameBar[j].toStdString());
                        double value = meteoPoints[mp].getMeteoPointValueH(myDate, hour, 0, meteoVar);
                        *setVector[mp][j] << value;
                        if (value > maxBar)
                        {
                            maxBar = value;
                        }
                    }
                }
            }
            nValues = nValues + 1;
        }
    }
    formInfo.close();

    if (isBar)
    {

        for (int mp=0; mp<meteoPoints.size();mp++)
        {
            QBarSeries* barMpSeries = new QBarSeries();
            for (int i = 0; i < nameBar.size(); i++)
            {
                barMpSeries->append(setVector[mp][i]);
            }
            barSeries.append(barMpSeries);
        }

        for (int mp=0; mp<meteoPoints.size();mp++)
        {
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
        axisY->setVisible(true);
        axisY->setMax(maxLine);
    }
    else
    {
        axisY->setVisible(false);
    }


    axisX->setCategories(categories);
    axisX->setGridLineVisible(false);
    // update virtual x axis
    axisXvirtual->setFormat("MMM dd <br> yyyy <br> hh:mm");
    axisXvirtual->setTickCount(20); // TO DO how many?
    axisXvirtual->setMin(QDateTime(this->firstDate->date(), QTime(0,0,0)));
    if (firstDate->date() == lastDate->date())
    {
        axisXvirtual->setMax(QDateTime(lastDate->date().addDays(1), QTime(0,0,0)));
    }
    else
    {
        axisXvirtual->setMax(QDateTime(this->lastDate->date(), QTime(0,0,0)));
    }

    firstDate->blockSignals(false);
    lastDate->blockSignals(false);

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
    firstDate->setMinimumWidth(firstDate->width()-firstDate->width()*0.3);
    lastDate->setMinimumWidth(lastDate->width()-lastDate->width()*0.3);

    dailyButton->setEnabled(false);
    hourlyButton->setEnabled(true);

    for (int i = 0; i<currentVariables.size(); i++)
    {
        QString name = currentVariables[i];
        name = "DAILY_"+name;
        currentVariables[i] = name;
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
    firstDate->setMinimumWidth(firstDate->width()+firstDate->width()*0.3);
    lastDate->setMinimumWidth(lastDate->width()+lastDate->width()*0.3);

    hourlyButton->setEnabled(false);
    dailyButton->setEnabled(true);

    for (int i = 0; i<currentVariables.size(); i++)
    {
        QString name = currentVariables[i];
        name = name.replace("DAILY_","");
        currentVariables[i] = name;
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


