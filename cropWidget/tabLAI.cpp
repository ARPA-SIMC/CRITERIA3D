#include "tabLAI.h"
#include "commonConstants.h"
#include <QMessageBox>


TabLAI::TabLAI()
{
    QHBoxLayout *mainLayout = new QHBoxLayout;
    QVBoxLayout *plotLayout = new QVBoxLayout;
    // Create empty graph view
    chartView = new QChartView();
    chart = new QChart();
    chart->setTitle("LAI development");
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
    axisX->setTickCount(15);
    chart->addAxis(axisX, Qt::AlignBottom);
    series->attachAxis(axisX);

    axisY->setTitleText("LAI");
    axisY->setRange(0,6);
    axisY->setTickCount(5);
    chart->addAxis(axisY, Qt::AlignLeft);
    series->attachAxis(axisY);

    plotLayout->addWidget(chartView);
    mainLayout->addLayout(plotLayout);
    setLayout(mainLayout);
}

void TabLAI::computeLAI(Crit3DCrop* myCrop, Crit3DMeteoPoint *meteoPoint, int year, int nrLayers, double totalSoilDepth, int currentDoy)
{
    this->year = year;
    myCrop->initialize(meteoPoint->latitude, nrLayers, totalSoilDepth, currentDoy);

    double waterTableDepth = NODATA;
    std::string error;

    Crit3DDate firstDate = Crit3DDate(1,1,year);
    Crit3DDate lastDate = Crit3DDate(31,12,year);
    double tmin;
    double tmax;
    QDateTime x;

    series->clear();
    for (Crit3DDate myDate = firstDate; myDate <= lastDate; ++myDate)
    {
        tmin = meteoPoint->getMeteoPointValueD(myDate, dailyAirTemperatureMin);
        tmax = meteoPoint->getMeteoPointValueD(myDate, dailyAirTemperatureMax);
        if (!myCrop->dailyUpdate(myDate, meteoPoint->latitude, nrLayers, totalSoilDepth, tmin, tmax, waterTableDepth, &error))
        {
            QMessageBox::critical(nullptr, "Error!", QString::fromStdString(error));
            return;
        }
        x.setDate(QDate(myDate.year, myDate.month, myDate.day));
        series->append(x.toMSecsSinceEpoch(), myCrop->LAI);
    }

    // update x axis
    QDate first(year, 1, 1);
    QDate last(year, 12, 31);
    axisX->setMin(QDateTime(first, QTime(0,0,0)));
    axisX->setMax(QDateTime(last, QTime(0,0,0)));

}


