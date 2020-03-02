#include "tabRootDensity.h"
#include "commonConstants.h"

TabRootDensity::TabRootDensity()
{
    QHBoxLayout *mainLayout = new QHBoxLayout;
    QVBoxLayout *plotLayout = new QVBoxLayout;
    chart = new QChart();
    chartView = new QChartView(chart);
    chart->setTitle("Root Density");
    chartView->setChart(chart);

    seriesRootDensity = new QHorizontalPercentBarSeries();
    seriesRootDensity->setName("rooth density");
    set = new QBarSet("");
    seriesRootDensity->append(set);

    axisX = new QValueAxis();
    axisY = new QBarCategoryAxis();

    chart->addSeries(seriesRootDensity);
    axisX->setTitleText("Rooth density [%]");
    axisX->setRange(0,0.022);
    axisX->setTickCount(12);
    chart->addAxis(axisX, Qt::AlignBottom);
    seriesRootDensity->attachAxis(axisX);

    axisY->setTitleText("Depth  [m]");
    axisY->setReverse(true);

    categories << "2.00" << "1.80" << "1.60" << "1.40" << "1.20" << "1.00" << "0.80" << "0.60" << "0.40" << "0.20" << "0.00";
    axisY->append(categories);
    chart->addAxis(axisY, Qt::AlignLeft);
    seriesRootDensity->attachAxis(axisY);

    chart->legend()->setVisible(true);
    chart->legend()->setAlignment(Qt::AlignBottom);

    chart->setAcceptHoverEvents(true);
    //m_tooltip = new Callout(chart);
    //connect(seriesRootDensity, &QLineSeries::hovered, this, &TabRootDensity::tooltip);

    plotLayout->addWidget(chartView);
    mainLayout->addLayout(plotLayout);
    setLayout(mainLayout);
}

void TabRootDensity::computeRootDensity(Crit3DCrop* myCrop, Crit3DMeteoPoint *meteoPoint, int currentYear, const std::vector<soil::Crit3DLayer> &soilLayers)
{

    unsigned int nrLayers = unsigned(soilLayers.size());
    double totalSoilDepth = 0;
    if (nrLayers > 0) totalSoilDepth = soilLayers[nrLayers-1].depth + soilLayers[nrLayers-1].thickness / 2;

    categories.clear();
    for (int i = 0; i<nrLayers; i++)
    {
        categories << QString::number(soilLayers[i].depth);
    }

    year = currentYear;
    int prevYear = currentYear - 1;

    double waterTableDepth = NODATA;
    std::string error;

    Crit3DDate firstDate = Crit3DDate(1, 1, prevYear);
    Crit3DDate lastDate = Crit3DDate(31, 12, year);
    double tmin;
    double tmax;
    QDateTime x;

    seriesRootDensity->clear();

    int currentDoy = 1;
    myCrop->initialize(meteoPoint->latitude, nrLayers, totalSoilDepth, currentDoy);

    for (Crit3DDate myDate = firstDate; myDate <= lastDate; ++myDate)
    {
        tmin = meteoPoint->getMeteoPointValueD(myDate, dailyAirTemperatureMin);
        tmax = meteoPoint->getMeteoPointValueD(myDate, dailyAirTemperatureMax);

        if (!myCrop->dailyUpdate(myDate, meteoPoint->latitude, soilLayers, tmin, tmax, waterTableDepth, error))
        {
            QMessageBox::critical(nullptr, "Error!", QString::fromStdString(error));
            return;
        }

        // display only current year
        if (myDate.year == year)
        {
            for (int i = 0; i<nrLayers; i++)
            {
                qDebug() << "myCrop->roots.rootDensity[i] " << myCrop->roots.rootDensity[i];
                //*set << myCrop->roots.rootDensity[i];
            }

        }
    }

}

void TabRootDensity::tooltip(QPointF point, bool state)
{
    /*
    if (m_tooltip == nullptr)
        m_tooltip = new Callout(chart);

    if (state)
    {
        QDateTime xDate;
        xDate.setMSecsSinceEpoch(point.x());
        m_tooltip->setText(QString("%1 \nroot ini %2 ").arg(xDate.date().toString("MMM dd")).arg(point.y()));
        m_tooltip->setAnchor(point);
        m_tooltip->setZValue(11);
        m_tooltip->updateGeometry();
        m_tooltip->show();
    } else {
        m_tooltip->hide();
    }
    */
}
