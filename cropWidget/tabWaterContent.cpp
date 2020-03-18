#include "tabWaterContent.h"
#include "commonConstants.h"

TabWaterContent::TabWaterContent()
{
    QVBoxLayout *mainLayout = new QVBoxLayout;
    QVBoxLayout *plotLayout = new QVBoxLayout;

    chart = new QChart();
    chartView = new QChartView(chart);
    chartView->setChart(chart);

    seriesWaterContent = new QHorizontalBarSeries();
    seriesWaterContent->setName("Water Content");
    set = new QBarSet("");
    seriesWaterContent->append(set);
    chart->addSeries(seriesWaterContent);

    axisX = new QDateTimeAxis();
    axisY = new QBarCategoryAxis();

    QDate first(QDate::currentDate().year(), 1, 1);
    QDate last(QDate::currentDate().year(), 12, 31);
    axisX->setTitleText("Date");
    axisX->setFormat("MMM dd");
    axisX->setMin(QDateTime(first, QTime(0,0,0)));
    axisX->setMax(QDateTime(last, QTime(0,0,0)));
    axisX->setTickCount(13);
    chart->addAxis(axisX, Qt::AlignBottom);
    seriesWaterContent->attachAxis(axisX);

    axisY->setTitleText("Depth [m]");

    double i = 1.95;
    while (i > 0)
    {
        categories.append(QString::number(i, 'f', 2));
        i = i-0.1;
    }
    axisY->append(categories);
    chart->addAxis(axisY, Qt::AlignLeft);
    seriesWaterContent->attachAxis(axisY);

    chart->legend()->setVisible(false);

    m_tooltip = new Callout(chart);
    m_tooltip->hide();

    plotLayout->addWidget(chartView);
    mainLayout->addLayout(plotLayout);
    setLayout(mainLayout);
}

void TabWaterContent::computeWaterContent(Crit1DCase myCase, int currentYear, bool isVolumetricWaterContent)
{

}

