#include "synchronicityChartView.h"
#include "commonConstants.h"
#include <QtCharts/QLegendMarker>
#include <QtGui/QImage>
#include <QtGui/QPainter>
#include <QtCore/QtMath>
#include <qdebug.h>

SynchronicityChartView::SynchronicityChartView(QWidget *parent) :
    QChartView(new QChart(), parent)
{
    axisX = new QValueAxis();
    axisY = new QValueAxis();

    chart()->addAxis(axisX, Qt::AlignBottom);
    chart()->addAxis(axisY, Qt::AlignLeft);
    chart()->setDropShadowEnabled(false);

    chart()->legend()->setVisible(true);
    chart()->legend()->setAlignment(Qt::AlignBottom);
    m_tooltip = new Callout(chart());
    m_tooltip->hide();
}

void SynchronicityChartView::setYmax(float value)
{
    axisY->setMax(value);
}

void SynchronicityChartView::setYmin(float value)
{
    axisY->setMin(value);
}


void SynchronicityChartView::drawGraphStation(int firstYear, std::vector<float> outputValues)
{
    chart()->legend()->setVisible(true);

    float maxValue = NODATA;
    float minValue = -NODATA;
    QLineSeries* graphSeries = new QLineSeries();
    unsigned int nYears;
    for (nYears = 0; nYears < outputValues.size(); nYears++)
    {
        if (outputValues[nYears] != NODATA)
        {
            if (outputValues[nYears] > maxValue)
            {
                maxValue = outputValues[nYears];
            }
            if (outputValues[nYears] < minValue)
            {
                minValue = outputValues[nYears];
            }
            graphSeries->append(firstYear+nYears,outputValues[nYears]);
        }
    }
    if (maxValue != minValue)
    {
        axisY->setMax(maxValue);
        axisY->setMin(minValue);
    }
    else
    {
        axisY->setMax(maxValue+3);
        axisY->setMin(minValue-3);
    }
    axisX->setRange(firstYear, firstYear+nYears);
    if ( nYears+1 <= 15)
    {
        axisX->setTickCount(nYears+1);
    }
    else
    {
        axisX->setTickCount(15);
    }
    axisX->setLabelFormat("%d");
    axisY->setLabelFormat("%.1f");
    axisX->setTitleText("years");
    axisY->setTitleText("r2");
    chart()->addSeries(graphSeries);
    graphSeries->attachAxis(axisX);
    graphSeries->attachAxis(axisY);
    stationGraphSeries.push_back(graphSeries);
    connect(graphSeries, &QLineSeries::hovered, this, &SynchronicityChartView::tooltipGraphStationSeries);
}

void SynchronicityChartView::clearStationGraphSeries()
{
    if (chart()->series().size() > 0)
    {
        for(int i = 0; i<stationGraphSeries.size(); i++)
        {
            if (chart()->series().contains(stationGraphSeries[i]))
            {
                chart()->removeSeries(stationGraphSeries[i]);
                stationGraphSeries[i]->clear();
            }
        }
    }
    stationGraphSeries.clear();
}

void SynchronicityChartView::tooltipGraphStationSeries(QPointF point, bool state)
{

    auto serie = qobject_cast<QLineSeries *>(sender());
    if (state)
    {
        int xValue = point.x();
        double yValue = point.y();

        m_tooltip->setText(QString("year %1: %2").arg(xValue).arg(yValue, 0, 'f', 3));
        m_tooltip->setSeries(serie);
        m_tooltip->setAnchor(point);
        m_tooltip->setZValue(11);
        m_tooltip->updateGeometry();
        m_tooltip->show();
    }
    else
    {
        m_tooltip->hide();
    }
}

