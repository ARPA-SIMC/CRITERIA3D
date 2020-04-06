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
#include "utilities.h"

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
            MapCSVDefault.insert(key,items);
            if (items[0] == "line")
            {
                QLineSeries* line = new QLineSeries();
                line->setName(key);
                line->setColor(QColor(items[1]));
                lineSeries.append(line);
            }
            if (items[0] == "bar")
            {
                QBarSet* set = new QBarSet(key);
                set->setColor(QColor(items[1]));
                setVector.append(set);
            }
        }
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
    QVBoxLayout *plotLayout = new QVBoxLayout;
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

    for (int i = 0; i < lineSeries.size(); i++)
    {
        chart->addSeries(lineSeries[i]);
        lineSeries[i]->attachAxis(axisX);
        lineSeries[i]->attachAxis(axisY);
    }

    barSeries = new QBarSeries();
    for (int i = 0; i < setVector.size(); i++)
    {
        barSeries->append(setVector[i]);
    }
    if (setVector.size() != 0)
    {
        chart->addSeries(barSeries);
        barSeries->attachAxis(axisX);
        barSeries->attachAxis(axisYdx);
    }

    chart->legend()->setVisible(true);
    chart->legend()->setAlignment(Qt::AlignBottom);
    chartView->setRenderHint(QPainter::Antialiasing);
    axisX->hide();

    plotLayout->addWidget(chartView);
    mainLayout->addLayout(plotLayout);
    setLayout(mainLayout);
}

void Crit3DMeteoWidget::draw(QVector<Crit3DMeteoPoint> mpVector, frequencyType freq)
{

    // clear all
    for (int i = 0; i < lineSeries.size(); i++)
    {
        lineSeries[0]->clear();
    }
    QStringList nameBar;
    QVector<QColor> colorBar;
    int sizeBarSet = setVector.size();

    for (int i = 0; i < sizeBarSet; i++)
    {
        nameBar.append(setVector[i]->label());
        colorBar.append(setVector[i]->color());
        barSeries->remove(setVector[i]);
        setVector.clear();

    }
    for (int i = 0; i < sizeBarSet; i++)
    {
        QBarSet* set = new QBarSet(nameBar[i]);
        set->setColor(colorBar[i]);
        setVector.append(set);
    }

    categories.clear();
    Crit3DDate firstDate;
    Crit3DDate myDate;
    long nDays = 0;
    double maxBar = 0;
    if (freq == daily)
    {
        nDays = mpVector[0].nrObsDataDaysD;
        firstDate = mpVector[0].obsDataD[0].date;
        for (int day = 0; day<nDays; day++)
        {
            myDate = firstDate.addDays(day);
            categories.append(QString::number(day+1));
            for (int i = 0; i < lineSeries.size(); i++)
            {
                meteoVariable meteoVar = MapDailyMeteoVar.at(lineSeries[i]->name().toStdString());
                double value = mpVector[0].getMeteoPointValueD(myDate, meteoVar);
                lineSeries[i]->append(day+1, value);
            }
            for (int j = 0; j < setVector.size(); j++)
            {
                meteoVariable meteoVar = MapDailyMeteoVar.at(setVector[j]->label().toStdString());
                double value = mpVector[0].getMeteoPointValueD(myDate, meteoVar);
                *setVector[j] << value;
                if (value > maxBar)
                {
                    maxBar = value;
                }
            }
        }
        for (int i = 0; i < setVector.size(); i++)
        {
            barSeries->append(setVector[i]);
        }
        axisX->append(categories);
        axisX->setGridLineVisible(false);
        // update virtual x axis
        QDate first(firstDate.year, firstDate.month, firstDate.day);
        QDate last = first.addDays(nDays);
        axisXvirtual->setMin(QDateTime(first, QTime(0,0,0)));
        axisXvirtual->setMax(QDateTime(last, QTime(0,0,0)));
        axisYdx->setRange(0,maxBar);
    }
    else if (freq == hourly)
    {
        nDays = mpVector[0].nrObsDataDaysH;
        firstDate = mpVector[0].getMeteoPointHourlyValuesDate(0);
    }

}



