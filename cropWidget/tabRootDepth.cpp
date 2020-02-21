#include "tabRootDepth.h"
#include "commonConstants.h"
#include <QMessageBox>


TabRootDepth::TabRootDepth()
{
    QHBoxLayout *mainLayout = new QHBoxLayout;
    QVBoxLayout *plotLayout = new QVBoxLayout;
    chart = new QChart();   
    chartView = new Crit3DChartView(chart);
    chart->setTitle("Root Depth");
    chartView->setChart(chart);
    series = new QLineSeries();
    axisX = new QDateTimeAxis();
    axisY = new QValueAxis();

    chart->addSeries(series);
    QDate first(QDate::currentDate().year(), 1, 1);
    QDate last(QDate::currentDate().year(), 12, 31);
    axisX->setTitleText("Date");
    axisX->setFormat("MMM dd");
    axisX->setMin(QDateTime(first, QTime(0,0,0)));
    axisX->setMax(QDateTime(last, QTime(0,0,0)));
    axisX->setTickCount(13);
    chart->addAxis(axisX, Qt::AlignBottom);
    series->attachAxis(axisX);

    axisY->setTitleText("Depth  [m]");
    axisY->setRange(0,6);
    axisY->setTickCount(7);
    chart->addAxis(axisY, Qt::AlignLeft);
    series->attachAxis(axisY);

    plotLayout->addWidget(chartView);
    mainLayout->addLayout(plotLayout);
    setLayout(mainLayout);
}



