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
    QPen pen;
    pen.setColor(Qt::black);
    series1->setPen(pen);
    series1->setColor(Qt::white);
    series1->setMarkerSize(10.0);

    series2 = new QScatterSeries();
    series2->setName("Secondary");
    series2->setMarkerShape(QScatterSeries::MarkerShapeCircle);
    series2->setPen(pen);
    series2->setColor(Qt::black);
    series2->setMarkerSize(10.0);

    series3 = new QScatterSeries();
    series3->setName("Supplemental");
    series3->setMarkerShape(QScatterSeries::MarkerShapeCircle);
    series3->setPen(pen);
    series3->setColor(Qt::gray);
    series3->setMarkerSize(8.0);

    climLapseRatelineSeries = new QLineSeries();
    climLapseRatelineSeries->setName("Climatological Lapse Rate");
    climLapseRatelineSeries->setColor(Qt::blue);

    modelLapseRatelineSeries = new QLineSeries();
    modelLapseRatelineSeries->setName("Model lapse rate");
    modelLapseRatelineSeries->setColor(Qt::red);

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
    m_tooltip = new Callout(chart());
    m_tooltip->hide();
}

void ChartView::cleanScatterSeries()
{
    if (chart()->series().contains(series1))
    {
        chart()->removeSeries(series1);
        series1->clear();
    }
    if (chart()->series().contains(series2))
    {
        chart()->removeSeries(series2);
        series2->clear();
    }
    if (chart()->series().contains(series3))
    {
        chart()->removeSeries(series3);
        series3->clear();
    }
}

void ChartView::drawScatterSeries(QList<QPointF> pointListSeries1, QList<QPointF> pointListSeries2, QList<QPointF> pointListSeries3)
{
    for (int i = 0; i < pointListSeries1.size(); i++)
    {
        series1->append(pointListSeries1[i]);
    }

    for (int i = 0; i < pointListSeries2.size(); i++)
    {
        series2->append(pointListSeries2[i]);
    }

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

    double xRange = xMax - xMin;
    double yRange = yMax - yMin;
    double deltaX = xRange/100;
    double deltaY = yRange/100;
    axisX->setMax(xMax+3*deltaX);
    axisX->setMin(xMin-3*deltaX);
    axisY->setMax(yMax+3*deltaY);
    axisY->setMin(yMin-3*deltaY);

    chart()->addSeries(series1);
    chart()->addSeries(series2);
    chart()->addSeries(series3);

    series1->attachAxis(axisX);
    series1->attachAxis(axisY);

    series2->attachAxis(axisX);
    series2->attachAxis(axisY);

    series3->attachAxis(axisX);
    series3->attachAxis(axisY);

    connect(series1, &QScatterSeries::hovered, this, &ChartView::tooltipScatterSeries);
    connect(series2, &QScatterSeries::hovered, this, &ChartView::tooltipScatterSeries);
    connect(series3, &QScatterSeries::hovered, this, &ChartView::tooltipScatterSeries);
}

void ChartView::cleanClimLapseRate()
{
    if (chart()->series().contains(climLapseRatelineSeries))
    {
        chart()->removeSeries(climLapseRatelineSeries);
        climLapseRatelineSeries->clear();
    }
}

void ChartView::drawClimLapseRate(QPointF firstPoint, QPointF lastPoint)
{
    climLapseRatelineSeries->append(firstPoint);
    climLapseRatelineSeries->append(lastPoint);
    chart()->addSeries(climLapseRatelineSeries);
    climLapseRatelineSeries->attachAxis(axisX);
    climLapseRatelineSeries->attachAxis(axisY);
}

void ChartView::cleanModelLapseRate()
{
    if (chart()->series().contains(modelLapseRatelineSeries))
    {
        chart()->removeSeries(modelLapseRatelineSeries);
        modelLapseRatelineSeries->clear();
    }
}

void ChartView::drawModelLapseRate(QList<QPointF> pointList)
{
    for (int i = 0; i < pointList.size(); i++)
    {
        modelLapseRatelineSeries->append(pointList[i]);
    }
    chart()->addSeries(modelLapseRatelineSeries);
    modelLapseRatelineSeries->attachAxis(axisX);
    modelLapseRatelineSeries->attachAxis(axisY);
}

void ChartView::setIdPointMap(const QMap<QString, QPointF> &valuePrimary, const QMap<QString, QPointF> &valueSecondary, const QMap<QString, QPointF> &valueSupplemental)
{
    idPointMap.clear();
    idPointMap2.clear();
    idPointMap3.clear();
    idPointMap = valuePrimary;
    idPointMap2 = valueSecondary;
    idPointMap3 = valueSupplemental;
}

void ChartView::tooltipScatterSeries(QPointF point, bool state)
{

    auto serie = qobject_cast<QScatterSeries *>(sender());
    if (state)
    {
        double xValue = point.x();
        double yValue = point.y();

        QString key;

        if (serie->name() == "Primary")
        {
            QMapIterator<QString, QPointF> i(idPointMap);
            while (i.hasNext()) {
                i.next();
                if (i.value() == point)
                {
                    key = i.key();
                }
            }
        }
        else if (serie->name() == "Secondary")
        {
            QMapIterator<QString, QPointF> i(idPointMap2);
            while (i.hasNext()) {
                i.next();
                if (i.value() == point)
                {
                    key = i.key();
                }
            }
        }
        else if (serie->name() == "Supplemental")
        {
            QMapIterator<QString, QPointF> i(idPointMap3);
            while (i.hasNext()) {
                i.next();
                if (i.value() == point)
                {
                    key = i.key();
                }
            }
        }

        m_tooltip->setText(QString("%1\n%2 %3 ").arg(key).arg(xValue, 0, 'f', 1).arg(yValue, 0, 'f', 3));
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

