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
    series1->setMarkerSize(10.0);

    series2 = new QScatterSeries();
    series2->setName("Secondary");
    series2->setMarkerShape(QScatterSeries::MarkerShapeCircle);
    series2->setColor(Qt::black);
    series2->setMarkerSize(10.0);

    series3 = new QScatterSeries();
    series3->setName("Supplemental");
    series3->setMarkerShape(QScatterSeries::MarkerShapeCircle);
    QPen pen;
    pen.setColor(Qt::black);
    series3->setPen(pen);
    series3->setColor(Qt::white);
    series3->setMarkerSize(10.0);

    climLapseRatelineSeries = new QLineSeries();
    climLapseRatelineSeries->setName("Climatological Lapse Rate");

    setRenderHint(QPainter::Antialiasing);
    chart()->addSeries(series1);
    chart()->addSeries(series2);
    chart()->addSeries(series3);

    chart()->createDefaultAxes();
    chart()->setDropShadowEnabled(false);


    chart()->legend()->setMarkerShape(QLegend::MarkerShapeFromSeries);
}

void ChartView::drawPointSeriesPrimary(QList<QPointF> pointList)
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

void ChartView::drawPointSeriesSecondary(QList<QPointF> pointList)
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

void ChartView::drawPointSeriesSupplemental(QList<QPointF> pointList)
{
    chart()->removeSeries(series3);
    series3->clear();
    for (int i = 0; i < pointList.size(); i++)
    {
        series3->append(pointList[i]);
    }
    chart()->addSeries(series3);
    chart()->createDefaultAxes();
}

void ChartView::cleanClimLapseRate()
{
    chart()->removeSeries(climLapseRatelineSeries);
    climLapseRatelineSeries->clear();
}

void ChartView::drawClimLapseRate()
{
    // TO DO
    chart()->addSeries(climLapseRatelineSeries);
}

