#include "tabLAI.h"
#include <QtCharts>


TabLAI::TabLAI()
{
    QHBoxLayout *mainLayout = new QHBoxLayout;
    QVBoxLayout *plotLayout = new QVBoxLayout;
    // Create a graph view
    QChartView *chartView = new QChartView();
    QLineSeries *series = new QLineSeries();

    QChart *chart = new QChart();
    chart->setTitle("LAI development");
    chart->addSeries(series);
    QValueAxis *axisX = new QValueAxis();
    axisX->setTitleText("days");
    axisX->setRange(1,365);
    axisX->setLabelFormat("%i");
    // axisX->setFormat("MM-dd-yy");
    axisX->setTickCount(5);
    chart->addAxis(axisX, Qt::AlignBottom);
    series->attachAxis(axisX);

    QValueAxis *axisY = new QValueAxis();
    axisY->setTitleText("LAI");
    axisY->setTickCount(5);
    chart->addAxis(axisY, Qt::AlignLeft);
    series->attachAxis(axisY);

    chartView->setChart(chart);
    plotLayout->addWidget(chartView);
    mainLayout->addLayout(plotLayout);
    setLayout(mainLayout);
}
