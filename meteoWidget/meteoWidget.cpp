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

#include <QMessageBox>
#include <QLayout>
#include <QMenu>
#include <QMenuBar>
#include <QPushButton>
#include <QDate>

#include <QDebug>


Crit3DMeteoWidget::Crit3DMeteoWidget()
{
    this->setWindowTitle(QStringLiteral("Graph"));
    this->resize(1240, 700);
    currentFreq = daily; //default
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
            }
            QString key = items[0];
            items.removeFirst();
            if (key.isEmpty() || items[0].isEmpty())
            {
                qDebug() << "invalid format CSV, missing data";
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
                QLineSeries* line = new QLineSeries();
                line->setName(key);
                line->setColor(QColor(items[1]));
                vectorLine.append(line);
            }
            if (items[0] == "bar")
            {
                QBarSet* set = new QBarSet(key);
                set->setColor(QColor(items[1]));
                vectorBarSet.append(set);
            }
        }
    }
    lineSeries.append(vectorLine);
    setVector.append(vectorBarSet);

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
    dailyButton->setMaximumWidth(this->width()/8);
    hourlyButton->setMaximumWidth(this->width()/8);
    addVarButton->setMaximumWidth(this->width()/8);

    if (currentFreq == daily)
    {
        dailyButton->setEnabled(false);
        hourlyButton->setEnabled(true);
    }
    else if(currentFreq == hourly)
    {
        hourlyButton->setEnabled(false);
        dailyButton->setEnabled(true);
    }

    buttonLayout->addWidget(dailyButton);
    buttonLayout->addWidget(hourlyButton);
    buttonLayout->addWidget(addVarButton);
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

    QBarSeries* barFirstSeries = new QBarSeries();
    for (int i = 0; i < setVector[0].size(); i++)
    {
        barFirstSeries->append(setVector[0][i]);
    }
    barSeries.append(barFirstSeries);
    if (setVector[0].size() != 0)
    {
        chart->addSeries(barSeries[0]);
        barSeries[0]->attachAxis(axisX);
        barSeries[0]->attachAxis(axisYdx);
    }

    for (int i = 0; i < lineSeries[0].size(); i++)
    {
        chart->addSeries(lineSeries[0][i]);
        lineSeries[0][i]->attachAxis(axisX);
        lineSeries[0][i]->attachAxis(axisY);
    }

    chart->legend()->setVisible(true);
    chart->legend()->setAlignment(Qt::AlignBottom);
    chartView->setRenderHint(QPainter::Antialiasing);
    axisX->hide();

    connect(addVarButton, &QPushButton::clicked, [=](){ showVar(); });
    connect(dailyButton, &QPushButton::clicked, [=](){ showDailyGraph(); });
    connect(hourlyButton, &QPushButton::clicked, [=](){ showHourlyGraph(); });

    plotLayout->addWidget(chartView);
    horizontalGroupBox->setLayout(buttonLayout);
    mainLayout->addWidget(horizontalGroupBox);
    mainLayout->addLayout(plotLayout);
    setLayout(mainLayout);
}

void Crit3DMeteoWidget::draw(Crit3DMeteoPoint mpVector)
{
    meteoPoints.append(mpVector);
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

void Crit3DMeteoWidget::resetValues()
{
    // clear lineSeries values
    for (int i = 0; i < lineSeries[0].size(); i++)
    {
        lineSeries[0][i]->clear();
    }
    // add lineSeries elements for each mp
    for (int np=0; np<meteoPoints.size();np++)
    {
        QVector<QLineSeries*> vectorLine = lineSeries[0];
        lineSeries.append(vectorLine);
    }

    QStringList nameBar;
    QVector<QColor> colorBar;
    int sizeBarSet = setVector[0].size();

    for (int i = 0; i < sizeBarSet; i++)
    {
        nameBar.append(setVector[0][i]->label());
        colorBar.append(setVector[0][i]->color());
    }
    barSeries.clear();
    setVector.clear();
    QVector<QBarSet*> vectorBarSet;
    for (int i = 0; i < sizeBarSet; i++)
    {
        QBarSet* set = new QBarSet(nameBar[i]);
        set->setColor(colorBar[i]);
        set->setBorderColor(colorBar[i]);
        vectorBarSet.append(set);
    }
    setVector.append(vectorBarSet);
    // add vectorBarSet elements for each mp
    for (int np=0; np<meteoPoints.size();np++)
    {
        QVector<QBarSet*> vectorBarSet = setVector[0];
        setVector.append(vectorBarSet);
    }

    categories.clear();
}

void Crit3DMeteoWidget::drawDailyVar()
{

    // TO DO general case replace meteoPoints[0] with all mp used
    Crit3DDate firstDate;
    Crit3DDate myDate;
    long nDays = 0;
    double maxBar = 0;
    double maxLine = NODATA;

    nDays = meteoPoints[0].nrObsDataDaysD;
    firstDate = meteoPoints[0].obsDataD[0].date;

    for (int day = 0; day<nDays; day++)
    {
        myDate = firstDate.addDays(day);
        categories.append(QString::number(day+1));
        for (int i = 0; i < lineSeries[0].size(); i++)
        {
            meteoVariable meteoVar = MapDailyMeteoVar.at(lineSeries[0][i]->name().toStdString());
            double value = meteoPoints[0].getMeteoPointValueD(myDate, meteoVar);
            lineSeries[0][i]->append(day+1, value);
            if (value > maxLine)
            {
                maxLine = value;
            }
        }
        for (int j = 0; j < setVector[0].size(); j++)
        {
            meteoVariable meteoVar = MapDailyMeteoVar.at(setVector[0][j]->label().toStdString());
            double value = meteoPoints[0].getMeteoPointValueD(myDate, meteoVar);
            *setVector[0][j] << value;
            if (value > maxBar)
            {
                maxBar = value;
            }
        }
    }
    QBarSeries* barFirstSeries = new QBarSeries();
    for (int i = 0; i < setVector[0].size(); i++)
    {
        barFirstSeries->append(setVector[0][i]);
    }
    barSeries.append(barFirstSeries);
    if (setVector[0].size() != 0)
    {
        chart->addSeries(barSeries[0]);
        barSeries[0]->attachAxis(axisX);
        barSeries[0]->attachAxis(axisYdx);
    }
    axisX->append(categories);
    axisX->setGridLineVisible(false);
    // update virtual x axis
    QDate first(firstDate.year, firstDate.month, firstDate.day);
    QDate last = first.addDays(nDays);
    axisXvirtual->setFormat("MMM dd <br> yyyy");
    axisXvirtual->setTickCount(13);
    axisXvirtual->setMin(QDateTime(first, QTime(0,0,0)));
    axisXvirtual->setMax(QDateTime(last, QTime(0,0,0)));
    axisYdx->setRange(0,maxBar);
    axisY->setMax(maxLine);
}

void Crit3DMeteoWidget::drawHourlyVar()
{
    // TO DO general case replace meteoPoints[0] with all mp used
    Crit3DDate firstDate;
    Crit3DDate myDate;
    long nDays = 0;
    int nValues = 0;
    double maxBar = 0;
    double maxLine = NODATA;

    nDays = meteoPoints[0].nrObsDataDaysH;
    firstDate = meteoPoints[0].getMeteoPointHourlyValuesDate(0);

    for (int day = 0; day<nDays; day++)
    {
        myDate = firstDate.addDays(day);
        for (int hour = 0; hour < 24; hour++)
        {
            categories.append(QString::number(nValues));
            for (int i = 0; i < lineSeries[0].size(); i++)
            {
                meteoVariable meteoVar = MapHourlyMeteoVar.at(lineSeries[0][i]->name().toStdString());
                double value = meteoPoints[0].getMeteoPointValueH(myDate, hour, 0, meteoVar);
                lineSeries[0][i]->append(nValues, value);
                if (value > maxLine)
                {
                    maxLine = value;
                }
            }
            for (int j = 0; j < setVector[0].size(); j++)
            {
                meteoVariable meteoVar = MapHourlyMeteoVar.at(setVector[0][j]->label().toStdString());
                double value = meteoPoints[0].getMeteoPointValueH(myDate, hour, 0, meteoVar);
                *setVector[0][j] << value;
                if (value > maxBar)
                {
                    maxBar = value;
                }
            }
            nValues = nValues + 1;
        }
    }
    QBarSeries* barFirstSeries = new QBarSeries();
    for (int i = 0; i < setVector[0].size(); i++)
    {
        barFirstSeries->append(setVector[0][i]);
    }
    barSeries.append(barFirstSeries);
    if (setVector[0].size() != 0)
    {
        chart->addSeries(barSeries[0]);
        barSeries[0]->attachAxis(axisX);
        barSeries[0]->attachAxis(axisYdx);
    }
    axisX->append(categories);
    axisX->setGridLineVisible(false);
    // update virtual x axis
    QDate first(firstDate.year, firstDate.month, firstDate.day);
    QDate last = first.addDays(nDays);
    axisXvirtual->setFormat("MMM dd <br> yyyy <br> hh:mm");
    axisXvirtual->setTickCount(20);
    axisXvirtual->setMin(QDateTime(first, QTime(0,0,0)));
    axisXvirtual->setMax(QDateTime(last, QTime(0,0,0)));
    axisYdx->setRange(0,maxBar);
    axisY->setMax(maxLine);
}

void Crit3DMeteoWidget::showVar()
{
    QStringList allKeys = MapCSVStyles.keys();
    QStringList selectedVar = MapCSVDefault.keys();
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
}

void Crit3DMeteoWidget::showDailyGraph()
{
    currentFreq = daily;

    dailyButton->setEnabled(false);
    hourlyButton->setEnabled(true);

    // TO DO

}

void Crit3DMeteoWidget::showHourlyGraph()
{
    currentFreq = hourly;

    hourlyButton->setEnabled(false);
    dailyButton->setEnabled(true);

    // save currently daily var
    QStringList nameVar;
    for (int i = 0; i < lineSeries[0].size(); i++)
    {
        QString name = lineSeries[0][i]->name();
        nameVar.append(name.replace("DAILY_",""));
    }
    lineSeries.clear();
    for (int i = 0; i < setVector[0].size(); i++)
    {
        QString name = setVector[0][i]->label();
        nameVar.append(name.replace("DAILY_",""));
    }
    barSeries.clear();
    setVector.clear();

    QVector<QLineSeries*> vectorLine;
    QVector<QBarSet*> vectorBarSet;
    QMapIterator<QString, QStringList> i(MapCSVStyles);

    while (i.hasNext())
    {
        i.next();
        for (int j=0; j<nameVar.size(); j++)
        {
            if (i.key() == nameVar[j])
            {
                QStringList items = i.value();
                if (items[0] == "line")
                {
                    QLineSeries* line = new QLineSeries();
                    line->setName(i.key());
                    line->setColor(QColor(items[1]));
                    vectorLine.append(line);
                }
                if (items[0] == "bar")
                {
                    QBarSet* set = new QBarSet(i.key());
                    set->setColor(QColor(items[1]));
                    vectorBarSet.append(set);
                }
            }
        }
    }
    lineSeries.append(vectorLine);
    setVector.append(vectorBarSet);
    chart->removeAllSeries();

    QBarSeries* barFirstSeries = new QBarSeries();
    for (int i = 0; i < setVector[0].size(); i++)
    {
        barFirstSeries->append(setVector[0][i]);
    }
    barSeries.append(barFirstSeries);
    /*
    if (setVector[0].size() != 0)
    {
        chart->addSeries(barSeries[0]);
        barSeries[0]->attachAxis(axisX);
        barSeries[0]->attachAxis(axisYdx);
    }
    */

    for (int i = 0; i < lineSeries[0].size(); i++)
    {
        chart->addSeries(lineSeries[0][i]);
        lineSeries[0][i]->attachAxis(axisX);
        lineSeries[0][i]->attachAxis(axisY);
    }


    drawHourlyVar();
}



