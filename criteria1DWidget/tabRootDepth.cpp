#include <QMessageBox>

#include "tabRootDepth.h"
#include "commonConstants.h"
#include "utilities.h"
#include "meteoPoint.h"
#include "formInfo.h"


TabRootDepth::TabRootDepth()
{
    QHBoxLayout *mainLayout = new QHBoxLayout;
    QVBoxLayout *plotLayout = new QVBoxLayout;
    chart = new QChart();
    chartView = new QChartView(chart);
    chart->setTitle("Root Depth");
    chartView->setChart(chart);
    seriesRootDepth = new QLineSeries();
    seriesRootDepth->setName("rooth depth [m]");
    seriesRootDepth->setColor(QColor(Qt::red));
    seriesRootDepthMin = new QLineSeries();
    seriesRootDepthMin->setName("root depht zero [m]");
    seriesRootDepthMin->setColor(QColor(Qt::green));
    axisX = new QDateTimeAxis();
    axisY = new QValueAxis();

    chart->addSeries(seriesRootDepth);
    chart->addSeries(seriesRootDepthMin);
    QDate first(QDate::currentDate().year(), 1, 1);
    QDate last(QDate::currentDate().year(), 12, 31);
    axisX->setTitleText("Date");
    axisX->setFormat("MMM dd <br> yyyy");
    axisX->setMin(QDateTime(first, QTime(0,0,0)));
    axisX->setMax(QDateTime(last, QTime(0,0,0)));
    axisX->setTickCount(13);
    chart->addAxis(axisX, Qt::AlignBottom);
    seriesRootDepth->attachAxis(axisX);
    seriesRootDepthMin->attachAxis(axisX);

    axisY->setTitleText("Depth  [m]");
    axisY->setReverse(true);
    axisY->setRange(0,2);
    axisY->setTickCount(5);
    chart->addAxis(axisY, Qt::AlignLeft);
    seriesRootDepth->attachAxis(axisY);
    seriesRootDepthMin->attachAxis(axisY);

    chart->legend()->setVisible(true);
    chart->legend()->setAlignment(Qt::AlignBottom);
    QFont legendFont = chart->legend()->font();
    legendFont.setPointSize(8);
    legendFont.setBold(true);
    chart->legend()->setFont(legendFont);

    chart->setAcceptHoverEvents(true);
    m_tooltip = new Callout(chart);
    m_tooltip->hide();

    connect(seriesRootDepthMin, &QLineSeries::hovered, this, &TabRootDepth::tooltipRDM);
    connect(seriesRootDepth, &QLineSeries::hovered, this, &TabRootDepth::tooltipRD);

    plotLayout->addWidget(chartView);
    mainLayout->addLayout(plotLayout);
    setLayout(mainLayout);
}

void TabRootDepth::computeRootDepth(Crit3DCrop* myCrop, Crit3DMeteoPoint *meteoPoint, int firstYear, int lastYear, QDate lastDBMeteoDate, const std::vector<soil::Crit1DLayer> &soilLayers)
{
    unsigned int nrLayers = unsigned(soilLayers.size());
    double totalSoilDepth = 0;
    if (nrLayers > 0) totalSoilDepth = soilLayers[nrLayers-1].depth + soilLayers[nrLayers-1].thickness / 2;

    int prevYear = firstYear - 1;
    std::string error;

    Crit3DDate firstDate = Crit3DDate(1, 1, prevYear);
    Crit3DDate lastDate;
    if (lastYear != lastDBMeteoDate.year())
    {
        lastDate = Crit3DDate(31, 12, lastYear);
    }
    else
    {
        lastDate = Crit3DDate(lastDBMeteoDate.day(), lastDBMeteoDate.month(), lastYear);
    }
    double tmin, tmax, waterTableDepth;
    QDateTime x;

    chart->removeSeries(seriesRootDepth);
    chart->removeSeries(seriesRootDepthMin);
    seriesRootDepth->clear();
    seriesRootDepthMin->clear();

    int currentDoy = 1;
    myCrop->initialize(meteoPoint->latitude, nrLayers, totalSoilDepth, currentDoy);

    for (Crit3DDate myDate = firstDate; myDate <= lastDate; ++myDate)
    {
        tmin = meteoPoint->getMeteoPointValueD(myDate, dailyAirTemperatureMin);
        tmax = meteoPoint->getMeteoPointValueD(myDate, dailyAirTemperatureMax);
        waterTableDepth = meteoPoint->getMeteoPointValueD(myDate, dailyWaterTableDepth);

        if (!myCrop->dailyUpdate(myDate, meteoPoint->latitude, soilLayers, tmin, tmax, waterTableDepth, error))
        {
            QMessageBox::critical(nullptr, "Error!", QString::fromStdString(error));
            return;
        }

        // display only from firstYear
        if (myDate.year >= firstYear)
        {
            x.setDate(QDate(myDate.year, myDate.month, myDate.day));
            seriesRootDepthMin->append(x.toMSecsSinceEpoch(), myCrop->roots.rootDepthMin);
            if (myCrop->roots.rootDepth!= NODATA)
            {
                seriesRootDepth->append(x.toMSecsSinceEpoch(), myCrop->roots.rootDepth);
            }
            else
            {
                seriesRootDepth->append(x.toMSecsSinceEpoch(), myCrop->roots.rootDepthMin);
            }
        }
    }

    // update x axis
    QDate first(firstYear, 1, 1);
    QDate last(lastDate.year, lastDate.month, lastDate.day);
    axisX->setMin(QDateTime(first, QTime(0,0,0)));
    axisX->setMax(QDateTime(last, QTime(0,0,0)));

    chart->addSeries(seriesRootDepth);
    chart->addSeries(seriesRootDepthMin);
    seriesRootDepth->attachAxis(axisY);
    seriesRootDepthMin->attachAxis(axisY);

}

void TabRootDepth::tooltipRDM(QPointF point, bool state)
{
    if (state)
    {
        QDateTime xDate;
        xDate.setMSecsSinceEpoch(point.x());
        m_tooltip->setText(QString("%1 \nroot ini %2 ").arg(xDate.date().toString("yyyy-MM-dd")).arg(point.y()));
        m_tooltip->setAnchor(point);
        m_tooltip->setZValue(11);
        m_tooltip->updateGeometry();
        m_tooltip->show();
    } else {
        m_tooltip->hide();
    }
}

void TabRootDepth::tooltipRD(QPointF point, bool state)
{
    if (state)
    {
        QDateTime xDate;
        xDate.setMSecsSinceEpoch(point.x());
        m_tooltip->setText(QString("%1 \nroot depth %2 ").arg(xDate.date().toString("yyyy-MM-dd")).arg(point.y()));
        m_tooltip->setAnchor(point);
        m_tooltip->setZValue(11);
        m_tooltip->updateGeometry();
        m_tooltip->show();
    } else {
        m_tooltip->hide();
    }
}


