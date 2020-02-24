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
    axisY->setRange(0,2);
    axisY->setTickCount(5);
    chart->addAxis(axisY, Qt::AlignLeft);
    series->attachAxis(axisY);

    plotLayout->addWidget(chartView);
    mainLayout->addLayout(plotLayout);
    setLayout(mainLayout);
}

void TabRootDepth::computeRootDepth(Crit3DCrop* myCrop, Crit3DMeteoPoint *meteoPoint, int currentYear, const std::vector<soil::Crit3DLayer> &soilLayers)
{
/*
    unsigned int nrLayers = unsigned(soilLayers.size());
    double totalSoilDepth = 0;
    if (nrLayers > 0) totalSoilDepth = soilLayers[nrLayers-1].depth + soilLayers[nrLayers-1].thickness / 2;

    year = currentYear;
    int prevYear = currentYear - 1;

    double waterTableDepth = NODATA;
    std::string error;

    Crit3DDate firstDate = Crit3DDate(1, 1, prevYear);
    Crit3DDate lastDate = Crit3DDate(31, 12, year);
    double tmin;
    double tmax;
    QDateTime x;

    series->clear();

    int currentDoy = 1;
    myCrop->initialize(meteoPoint->latitude, nrLayers, totalSoilDepth, currentDoy);

    for (Crit3DDate myDate = firstDate; myDate <= lastDate; ++myDate)
    {
        tmin = meteoPoint->getMeteoPointValueD(myDate, dailyAirTemperatureMin);
        tmax = meteoPoint->getMeteoPointValueD(myDate, dailyAirTemperatureMax);

        if (!myCrop->dailyUpdate(myDate, meteoPoint->latitude, soilLayers, tmin, tmax, waterTableDepth, &error))
        {
            QMessageBox::critical(nullptr, "Error!", QString::fromStdString(error));
            return;
        }

        // display only current year
        if (myDate.year == year)
        {
            x.setDate(QDate(myDate.year, myDate.month, myDate.day));
            series->append(x.toMSecsSinceEpoch(), myCrop->LAI);
        }
    }

    // update x axis
    QDate first(year, 1, 1);
    QDate last(year, 12, 31);
    axisX->setMin(QDateTime(first, QTime(0,0,0)));
    axisX->setMax(QDateTime(last, QTime(0,0,0)));
*/
}



