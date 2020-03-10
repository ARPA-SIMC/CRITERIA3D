#include "tabIrrigation.h"

TabIrrigation::TabIrrigation()
{
    QHBoxLayout *mainLayout = new QHBoxLayout;
    QVBoxLayout *plotLayout = new QVBoxLayout;
    chart = new QChart();
    chartView = new QChartView(chart);
    chartView->setChart(chart);


    axisX = new QDateTimeAxis();
    axisY = new QValueAxis();

    QDate first(QDate::currentDate().year(), 1, 1);
    QDate last(QDate::currentDate().year(), 12, 31);
    axisX->setTitleText("Date");
    axisX->setFormat("MMM dd");
    axisX->setMin(QDateTime(first, QTime(0,0,0)));
    axisX->setMax(QDateTime(last, QTime(0,0,0)));
    axisX->setTickCount(13);
    chart->addAxis(axisX, Qt::AlignBottom);
    chart->addAxis(axisY, Qt::AlignLeft);


    plotLayout->addWidget(chartView);
    mainLayout->addLayout(plotLayout);
    setLayout(mainLayout);
}
