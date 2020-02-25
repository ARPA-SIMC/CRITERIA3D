#include "tabRootDepth.h"
#include "commonConstants.h"
#include "utilities.h"
#include <QMessageBox>


TabRootDepth::TabRootDepth()
{
    QHBoxLayout *mainLayout = new QHBoxLayout;
    QVBoxLayout *plotLayout = new QVBoxLayout;
    chart = new QChart();
    chartView = new Crit3DChartView(chart);
    chart->setTitle("Root Depth");
    chartView->setChart(chart);
    series = new QStackedBarSeries();
    rootDepth = new QBarSet("");
    rootDepthMin = new QBarSet("");
    chart->addSeries(series);

    series->append(rootDepth);
    series->append(rootDepthMin);
    QDate first(QDate::currentDate().year(), 1, 1);
    QDate last(QDate::currentDate().year(), 12, 31);
    QDate myDate;
    for (myDate = first; myDate <= last; myDate=myDate.addDays(1))
    {
        categories << myDate.toString("MMM dd");
    }

    axisX = new QBarCategoryAxis();
    axisX->append(categories);
    chart->addAxis(axisX, Qt::AlignBottom);
    QValueAxis *axisY = new QValueAxis();
    axisY->setReverse(true);
    axisY->setRange(0,2);
    chart->addAxis(axisY, Qt::AlignLeft);
    series->attachAxis(axisY);
    chart->legend()->setVisible(false);
    chartView->setRenderHint(QPainter::Antialiasing);
    plotLayout->addWidget(chartView);
    mainLayout->addLayout(plotLayout);
    setLayout(mainLayout);
}

void TabRootDepth::computeRootDepth(Crit3DCrop* myCrop, Crit3DMeteoPoint *meteoPoint, int currentYear, const std::vector<soil::Crit3DLayer> &soilLayers)
{

    categories.clear();
    for (int i = 0; i<rootDepth->count(); i++)
    {
        rootDepth->remove(i);
        rootDepthMin->remove(i);
    }
    chart->removeSeries(series);

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
            if (myCrop->roots.rootDepthMin != NODATA && myCrop->roots.rootDepth != NODATA)
            {
                categories << (getQDate(myDate)).toString("MMM dd");
                qDebug() << "date " << (getQDate(myDate)).toString("MMM dd");
                qDebug() << "myCrop->roots.rootDepthMin" << myCrop->roots.rootDepthMin;
                qDebug() << "myCrop->roots.rootDepth" << myCrop->roots.rootDepth;
                *rootDepthMin << myCrop->roots.rootDepthMin;
                *rootDepth << myCrop->roots.rootDepth;

            }
        }
    }

    chart->addSeries(series);
}



