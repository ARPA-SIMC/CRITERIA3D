#include "chartView.h"
#include <QtCharts/QLegendMarker>
#include <QtGui/QImage>
#include <QtGui/QPainter>
#include <QtCore/QtMath>

ChartView::ChartView(QWidget *parent) :
    QChartView(new QChart(), parent)
{
    series1 = new QScatterSeries();
    series1->setName("Primary");
    series1->setMarkerShape(QScatterSeries::MarkerShapeCircle);
    series1->setColor(Qt::red);
    series1->setMarkerSize(15.0);

    series2 = new QScatterSeries();
    series2->setName("Secondary");
    series2->setMarkerShape(QScatterSeries::MarkerShapeCircle);
    series2->setColor(Qt::black);
    series2->setMarkerSize(15.0);

    setRenderHint(QPainter::Antialiasing);
    chart()->addSeries(series1);
    chart()->addSeries(series2);

    chart()->createDefaultAxes();
    chart()->setDropShadowEnabled(false);


    chart()->legend()->setMarkerShape(QLegend::MarkerShapeFromSeries);
}

void ChartView::appendPointSeries1(QList<QPointF> pointList)
{
    chart()->removeSeries(series1);
    series1->clear();
    for (int i = 0; i < pointList.size(); i++)
    {
        series1->append(pointList[i]);
    }
    chart()->addSeries(series1);
    chart()->createDefaultAxes();
}

void ChartView::appendPointSeries2(QList<QPointF> pointList)
{
    chart()->removeSeries(series2);
    series2->clear();
    for (int i = 0; i < pointList.size(); i++)
    {
        series2->append(pointList[i]);
    }
    chart()->addSeries(series2);
    chart()->createDefaultAxes();
}
