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

    axisX = new QValueAxis();
    axisY = new QValueAxis();

    chart()->addAxis(axisX, Qt::AlignBottom);
    chart()->addAxis(axisY, Qt::AlignLeft);
    chart()->setDropShadowEnabled(false);


    chart()->legend()->setMarkerShape(QLegend::MarkerShapeFromSeries);
}

void ChartView::drawScatterSeries(QList<QPointF> pointListSeries1, QList<QPointF> pointListSeries2, QList<QPointF> pointListSeries3)
{
    chart()->removeSeries(series1);
    series1->clear();
    for (int i = 0; i < pointListSeries1.size(); i++)
    {
        series1->append(pointListSeries1[i]);
    }

    chart()->removeSeries(series2);
    series2->clear();
    for (int i = 0; i < pointListSeries2.size(); i++)
    {
        series2->append(pointListSeries2[i]);
    }

    chart()->removeSeries(series3);
    series3->clear();
    for (int i = 0; i < pointListSeries3.size(); i++)
    {
        series3->append(pointListSeries3[i]);
    }

    pointListSeries1.append(pointListSeries2);
    pointListSeries1.append(pointListSeries3);
    double xMin = std::numeric_limits<int>::max();
    double xMax = std::numeric_limits<int>::min();
    double yMin = std::numeric_limits<int>::max();
    double yMax = std::numeric_limits<int>::min();
    foreach (QPointF p, pointListSeries1) {
        xMin = qMin(xMin, p.x());
        xMax = qMax(xMax, p.x());
        yMin = qMin(yMin, p.y());
        yMax = qMax(yMax, p.y());
    }

    double xRange = xMax - abs(xMin);
    double yRange = yMax - abs(yMin);
    double deltaX = xRange/100;
    double deltaY = yRange/100;
    axisX->setMax(xMax+deltaX);
    axisX->setMin(xMin-deltaX);
    axisY->setMax(yMax+deltaY);
    axisY->setMin(yMin-deltaY);

    chart()->addSeries(series1);
    chart()->addSeries(series2);
    chart()->addSeries(series3);

    series1->attachAxis(axisX);
    series1->attachAxis(axisY);

    series2->attachAxis(axisX);
    series2->attachAxis(axisY);

    series3->attachAxis(axisX);
    series3->attachAxis(axisY);
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

