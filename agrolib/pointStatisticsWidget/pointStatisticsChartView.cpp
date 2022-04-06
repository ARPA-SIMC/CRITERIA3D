#include "pointStatisticsChartView.h"
#include "commonConstants.h"
#include <QtCharts/QLegendMarker>
#include <QtGui/QImage>
#include <QtGui/QPainter>
#include <QtCore/QtMath>
#include <qdebug.h>

PointStatisticsChartView::PointStatisticsChartView(QWidget *parent) :
    QChartView(new QChart(), parent)
{
    trend = new QScatterSeries();
    trend->setName("Trend");
    trend->setColor(Qt::red);
    trend->setMarkerSize(10.0);
    setRenderHint(QPainter::Antialiasing);

    climaDaily = new QLineSeries();
    climaDaily->setName("Daily");
    climaDaily->setColor(Qt::black);

    climaDecadal = new QLineSeries();
    climaDecadal->setName("Decadal");
    climaDecadal->setColor(Qt::red);

    climaMonthly = new QLineSeries();
    climaMonthly->setName("Monthly");
    climaMonthly->setColor(Qt::green);

    distributionBar = new QBarSeries();
    distributionLine = new QLineSeries();
    climaMonthly->setColor(Qt::green);

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

    cleanTrendSeries();
    cleanClimaSeries();
    cleanDistribution();
    chart()->legend()->setVisible(true);

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
    if (maxValue != minValue)
    {
        double yRange = maxValue - minValue;
        double deltaY = yRange/100;
        axisY->setMax(maxValue+3*deltaY);
        axisY->setMin(minValue-3*deltaY);
    }
    else
    {
        axisY->setMax(maxValue+3);
        axisY->setMin(minValue-3);
    }
    axisXvalue->setRange(years[0], years[years.size()-1]);
    axisXvalue->setTickCount(years.size());
    axisXvalue->setLabelFormat("%d");
    chart()->addSeries(trend);
    trend->attachAxis(axisXvalue);
    trend->attachAxis(axisY);
    connect(trend, &QScatterSeries::hovered, this, &PointStatisticsChartView::tooltipTrendSeries);
}

void PointStatisticsChartView::cleanClimaSeries()
{
    if (chart()->series().contains(climaDaily))
    {
        chart()->removeSeries(climaDaily);
        climaDaily->clear();
    }
    if (chart()->series().contains(climaDecadal))
    {
        chart()->removeSeries(climaDecadal);
        climaDecadal->clear();
    }
    if (chart()->series().contains(climaMonthly))
    {
        chart()->removeSeries(climaMonthly);
        climaMonthly->clear();
    }
}

void PointStatisticsChartView::cleanTrendSeries()
{
    if (chart()->series().contains(trend))
    {
        chart()->removeSeries(trend);
        trend->clear();
    }
}

void PointStatisticsChartView::drawClima(QList<QPointF> dailyPointList, QList<QPointF> decadalPointList, QList<QPointF> monthlyPointList)
{
    cleanClimaSeries();
    cleanTrendSeries();
    cleanDistribution();
    chart()->legend()->setVisible(true);

    float maxValue = NODATA;
    float minValue = -NODATA;

    for (int i = 0; i < dailyPointList.size(); i++)
    {
        climaDaily->append(dailyPointList[i]);
        if(dailyPointList[i].y() != NODATA)
        {
            if (dailyPointList[i].y() > maxValue)
            {
                maxValue = dailyPointList[i].y();
            }
            if (dailyPointList[i].y() < minValue)
            {
                minValue = dailyPointList[i].y();
            }
        }
    }

    for (int i = 0; i < decadalPointList.size(); i++)
    {
        climaDecadal->append(decadalPointList[i]);
        if(decadalPointList[i].y() != NODATA)
        {
            if (decadalPointList[i].y() > maxValue)
            {
                maxValue = decadalPointList[i].y();
            }
            if (decadalPointList[i].y() < minValue)
            {
                minValue = decadalPointList[i].y();
            }
        }
    }

    for (int i = 0; i < monthlyPointList.size(); i++)
    {
        climaMonthly->append(monthlyPointList[i]);
        if(monthlyPointList[i].y() != NODATA)
        {
            if (monthlyPointList[i].y() > maxValue)
            {
                maxValue = monthlyPointList[i].y();
            }
            if (monthlyPointList[i].y() < minValue)
            {
                minValue = monthlyPointList[i].y();
            }
        }
    }
    axisY->setMax(maxValue);
    axisY->setMin(minValue);
    axisXvalue->setRange(1, 366);
    axisXvalue->setTickCount(20);
    axisXvalue->setLabelFormat("%d");
    axisY->setLabelFormat("%.3f");

    chart()->addSeries(climaDaily);
    chart()->addSeries(climaDecadal);
    chart()->addSeries(climaMonthly);
    climaDaily->attachAxis(axisXvalue);
    climaDaily->attachAxis(axisY);
    climaDecadal->attachAxis(axisXvalue);
    climaDecadal->attachAxis(axisY);
    climaMonthly->attachAxis(axisXvalue);
    climaMonthly->attachAxis(axisY);
    connect(climaDaily, &QLineSeries::hovered, this, &PointStatisticsChartView::tooltipClimaSeries);
    connect(climaDecadal, &QLineSeries::hovered, this, &PointStatisticsChartView::tooltipClimaSeries);
    connect(climaMonthly, &QLineSeries::hovered, this, &PointStatisticsChartView::tooltipClimaSeries);
}

void PointStatisticsChartView::drawDistribution(std::vector<float> barValues, QList<QPointF> lineValues, int minValue, int maxValue)
{
    cleanClimaSeries();
    cleanTrendSeries();
    cleanDistribution();
    chart()->legend()->setVisible(false);

    QBarSet *distributionSet = new QBarSet("Distribution");
    distributionSet->setColor(Qt::red);
    distributionSet->setBorderColor(Qt::red);

    float maxValueY = NODATA;
    float minValueY = -NODATA;

    for (int i = 0; i<barValues.size(); i++)
    {
        *distributionSet << barValues[i];
        if(barValues[i] != NODATA)
        {
            if (barValues[i] > maxValueY)
            {
                maxValueY = barValues[i];
            }
            if (barValues[i] < minValueY)
            {
                minValueY = barValues[i];
            }
        }
    }

    for (int i = 0; i<lineValues.size(); i++)
    {
        distributionLine->append(lineValues[i]);
        if(lineValues[i].y() != NODATA)
        {
            if (lineValues[i].y() > maxValueY)
            {
                maxValueY = lineValues[i].y();
            }
            if (lineValues[i].y() < minValueY)
            {
                minValueY = lineValues[i].y();
            }
        }
    }
    distributionBar->append(distributionSet);
    axisY->setMax(maxValueY);
    axisY->setMin(minValueY);
    axisXvalue->setRange(minValue, maxValue);

    distributionLine->attachAxis(axisXvalue);
    distributionLine->attachAxis(axisY);
    distributionBar->attachAxis(axisXvalue);
    distributionBar->attachAxis(axisY);
    chart()->addSeries(distributionBar);
    chart()->addSeries(distributionLine);
    connect(distributionLine, &QLineSeries::hovered, this, &PointStatisticsChartView::tooltipDistributionSeries);
    connect(distributionBar, &QBarSeries::hovered, this, &PointStatisticsChartView::tooltipBar);

}

void PointStatisticsChartView::cleanDistribution()
{
    if (chart()->series().contains(distributionLine))
    {
        chart()->removeSeries(distributionLine);
        distributionLine->clear();
    }
    if (chart()->series().contains(distributionBar))
    {
        chart()->removeSeries(distributionBar);
        distributionBar->clear();
    }
}

void PointStatisticsChartView::tooltipTrendSeries(QPointF point, bool state)
{

    auto serie = qobject_cast<QScatterSeries *>(sender());
    if (state)
    {
        double xValue = point.x();
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

void PointStatisticsChartView::tooltipClimaSeries(QPointF point, bool state)
{

    auto serie = qobject_cast<QLineSeries *>(sender());
    if (state)
    {
        int xValue = point.x();
        double yValue = point.y();

        m_tooltip->setText(QString("dOY %1: %2").arg(xValue).arg(yValue, 0, 'f', 3));
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

void PointStatisticsChartView::tooltipDistributionSeries(QPointF point, bool state)
{

    auto serie = qobject_cast<QLineSeries *>(sender());
    if (state)
    {
        int xValue = point.x();
        double yValue = point.y();

        m_tooltip->setText(QString("%1,%2").arg(xValue, 0, 'f', 1).arg(yValue, 0, 'f', 3));
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

void PointStatisticsChartView::tooltipBar(bool state, int index, QBarSet *barset)
{

    QBarSeries *series = qobject_cast<QBarSeries *>(sender());

    if (state && barset!=nullptr && index < barset->count())
    {

        QPoint CursorPoint = QCursor::pos();
        QPoint mapPoint = mapFromGlobal(CursorPoint);
        QPointF pointF = this->chart()->mapToValue(mapPoint,series);

        // check if bar is hiding QlineSeries
        if (  static_cast<int>( distributionLine->at(pointF.toPoint().x()).y() ) == pointF.toPoint().y())
        {
            tooltipDistributionSeries(pointF, true);
        }

        QString valueStr = QString("%1").arg(barset->at(index), 0, 'f', 1);
        m_tooltip->setSeries(series);
        m_tooltip->setText(valueStr);
        m_tooltip->setAnchor(pointF);
        m_tooltip->setZValue(11);
        m_tooltip->updateGeometry();
        m_tooltip->show();

    }
    else
    {
        m_tooltip->hide();
    }
}

