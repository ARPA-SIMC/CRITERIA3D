#include "pointStatisticsChartView.h"
#include "commonConstants.h"
#include <QtCharts/QLegendMarker>
#include <QtGui/QImage>
#include <QtGui/QPainter>
#include <QtCore/QtMath>

PointStatisticsChartView::PointStatisticsChartView(QWidget *parent) :
    QChartView(new QChart(), parent)
{
    trend = new QScatterSeries();
    trend->setName("Trend");
    trend->setColor(Qt::red);
    trend->setMarkerSize(10.0);
    setRenderHint(QPainter::Antialiasing);

    axisXvalue = new QValueAxis();
    //axisX = new QBarCategoryAxis();
    axisY = new QValueAxis();

    //chart()->addAxis(axisX, Qt::AlignBottom);
    chart()->addAxis(axisXvalue, Qt::AlignBottom);
    //axisX->setVisible(false);
    chart()->addAxis(axisY, Qt::AlignLeft);
    chart()->setDropShadowEnabled(false);

    chart()->legend()->setVisible(true);
    chart()->legend()->setAlignment(Qt::AlignBottom);
    m_tooltip = new PointStatisticsCallout(chart());
    m_tooltip->hide();
}

void PointStatisticsChartView::drawTrend(std::vector<int> years, std::vector<float> outputValues)
{

    trend->clear();
    if (chart()->series().contains(trend))
    {
        chart()->removeSeries(trend);
    }
    /*
    categories.clear();
    for (int year = years[0]; year < years.size(); year++)
    {
        categories.append(QString::number(year));
    }
    */

    float maxValue = NODATA;
    float minValue = -NODATA;
    for (unsigned int i = 0; i < years.size(); i++)
    {
        if (outputValues[i] != NODATA)
        {
            trend->append(QPointF(years[i], outputValues[i]));
            if (outputValues[i] > maxValue)
            {
                maxValue = outputValues[i];
            }
            if (outputValues[i] < minValue)
            {
                minValue = outputValues[i];
            }
        }
    }
    double yRange = maxValue - minValue;
    double deltaY = yRange/100;
    axisY->setMax(maxValue+3*deltaY);
    axisY->setMin(minValue-3*deltaY);
    axisXvalue->setRange(years[0], years[years.size()-1]);
    axisXvalue->setTickCount(years.size());
    axisXvalue->setLabelFormat("%d");
    //axisX->setCategories(categories);
    chart()->addSeries(trend);
    trend->attachAxis(axisXvalue);
    trend->attachAxis(axisY);
    connect(trend, &QScatterSeries::hovered, this, &PointStatisticsChartView::tooltipLineSeries);
}

void PointStatisticsChartView::tooltipLineSeries(QPointF point, bool state)
{

    auto serie = qobject_cast<QScatterSeries *>(sender());
    if (state)
    {
        double xValue = point.x();
        double yValue = point.y();

        m_tooltip->setText(QString("%1: %2").arg(xValue).arg(yValue, 0, 'f', 3));
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

/*
void PointStatisticsChartView::cleanScatterSeries()
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

void PointStatisticsChartView::drawScatterSeries(QList<QPointF> pointListSeries1, QList<QPointF> pointListSeries2, QList<QPointF> pointListSeries3)
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

    connect(series1, &QScatterSeries::hovered, this, &PointStatisticsChartView::tooltipScatterSeries);
    connect(series2, &QScatterSeries::hovered, this, &PointStatisticsChartView::tooltipScatterSeries);
    connect(series3, &QScatterSeries::hovered, this, &PointStatisticsChartView::tooltipScatterSeries);
}

void PointStatisticsChartView::cleanClimLapseRate()
{
    if (chart()->series().contains(climLapseRatelineSeries))
    {
        chart()->removeSeries(climLapseRatelineSeries);
        climLapseRatelineSeries->clear();
    }
}

void PointStatisticsChartView::drawClimLapseRate(QPointF firstPoint, QPointF lastPoint)
{
    climLapseRatelineSeries->append(firstPoint);
    climLapseRatelineSeries->append(lastPoint);
    chart()->addSeries(climLapseRatelineSeries);
    climLapseRatelineSeries->attachAxis(axisX);
    climLapseRatelineSeries->attachAxis(axisY);
}

void PointStatisticsChartView::cleanModelLapseRate()
{
    if (chart()->series().contains(modelLapseRatelineSeries))
    {
        chart()->removeSeries(modelLapseRatelineSeries);
        modelLapseRatelineSeries->clear();
    }
}

void PointStatisticsChartView::drawModelLapseRate(QList<QPointF> pointList)
{
    for (int i = 0; i < pointList.size(); i++)
    {
        modelLapseRatelineSeries->append(pointList[i]);
    }
    chart()->addSeries(modelLapseRatelineSeries);
    modelLapseRatelineSeries->attachAxis(axisX);
    modelLapseRatelineSeries->attachAxis(axisY);
}

void PointStatisticsChartView::setIdPointMap(const QMap<QString, QPointF> &valuePrimary, const QMap<QString, QPointF> &valueSecondary, const QMap<QString, QPointF> &valueSupplemental)
{
    idPointMap.clear();
    idPointMap2.clear();
    idPointMap3.clear();
    idPointMap = valuePrimary;
    idPointMap2 = valueSecondary;
    idPointMap3 = valueSupplemental;
}


*/

