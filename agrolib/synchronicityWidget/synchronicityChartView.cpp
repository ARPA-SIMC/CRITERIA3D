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
    maxValue = NODATA;
    minValue = -NODATA;
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


void SynchronicityChartView::drawGraphStation(QList<QPointF> pointList, QString var, int lag)
{
    chart()->legend()->setVisible(true);
    QString name = var+" lag="+QString::number(lag);
    for(int i = 0; i<stationGraphSeries.size(); i++)
    {
        if (stationGraphSeries[i]->name() == name)
        {
            return;
        }
    }
    QLineSeries* graphSeries = new QLineSeries();
    graphSeries->setName(name);
    for (unsigned int nYears = 0; nYears < pointList.size(); nYears++)
    {
        if (pointList[nYears].y() != NODATA)
        {
            if (pointList[nYears].y() > maxValue)
            {
                maxValue = pointList[nYears].y();
            }
            if (pointList[nYears].y() < minValue)
            {
                minValue = pointList[nYears].y();
            }
            graphSeries->append(pointList[nYears]);
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
    axisX->setRange(pointList[0].x(), pointList[pointList.size()-1].x());
    if ( pointList[pointList.size()-1].x()-pointList[0].x()+1 <= 15)
    {
        axisX->setTickCount(pointList[pointList.size()-1].x()-pointList[0].x()+1);
    }
    else
    {
        axisX->setTickCount(15);
    }
    axisX->setLabelFormat("%d");
    axisY->setLabelFormat("%.3f");
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
    maxValue = NODATA;
    minValue = -NODATA;
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

