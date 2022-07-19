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
    axisX = new QBarCategoryAxis();
    axisX->hide();
    axisY = new QValueAxis();

    chart()->addAxis(axisXvalue, Qt::AlignBottom);
    chart()->addAxis(axisX, Qt::AlignBottom);
    chart()->addAxis(axisY, Qt::AlignLeft);
    chart()->setDropShadowEnabled(false);

    chart()->legend()->setVisible(true);
    chart()->legend()->setAlignment(Qt::AlignBottom);
    m_tooltip = new Callout(chart());
    m_tooltip->hide();
}

void PointStatisticsChartView::drawTrend(std::vector<int> years, std::vector<float> outputValues)
{

    if (chart()->series().size() > 0)
    {
        cleanClimaSeries();
        cleanDistribution();
        cleanTrendSeries();
    }
    chart()->legend()->setVisible(false);

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
    int nYears = years.size();
    if ( nYears <= 20)
    {
        axisXvalue->setTickCount(nYears);
    }
    else
    {
        int div = 0;
        for (int i = 2; i<=4; i++)
        {
            if ( (nYears-1) % i == 0 && (nYears-1)/i <= 20)
            {
                div = i;
                break;
            }
        }
        if (div == 0)
        {
            axisXvalue->setTickCount(2);
        }
        else
        {
            axisXvalue->setTickCount( (nYears-1)/div + 1);
        }
    }
    axisXvalue->setLabelFormat("%d");
    axisY->setLabelFormat("%.1f");
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
    if (chart()->series().size() > 0)
    {
        cleanClimaSeries();
        cleanDistribution();
        cleanTrendSeries();
    }
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
    axisXvalue->setTickCount(28);
    axisXvalue->setLabelFormat("%d");
    axisY->setLabelFormat("%.1f");

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

void PointStatisticsChartView::drawDistribution(std::vector<float> barValues, QList<QPointF> lineValues, int minValue, int maxValue, int classWidthValue)
{

    if (chart()->series().size() > 0)
    {
        cleanClimaSeries();
        cleanDistribution();
        cleanTrendSeries();
    }
    chart()->legend()->setVisible(false);
    categories.clear();
    widthValue = classWidthValue;


    QBarSet *distributionSet = new QBarSet("Distribution");
    distributionSet->setColor(Qt::red);
    distributionSet->setBorderColor(Qt::red);

    float maxValueY = NODATA;
    float minValueY = -NODATA;

    for (int i = 0; i<barValues.size(); i++)
    {
        categories.append(QString::number(i));
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
    axisY->setLabelFormat("%.3f");
    axisXvalue->setRange(minValue, maxValue);
    axisX->setCategories(categories);

    chart()->addSeries(distributionBar);
    chart()->addSeries(distributionLine);

    distributionLine->attachAxis(axisXvalue);

    distributionBar->attachAxis(axisX);
    distributionBar->attachAxis(axisY);

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
        double xValue = point.x();
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
        float xStart = axisXvalue->min() + (index*widthValue);
        float xEnd = axisXvalue->min() + ((index+1)*widthValue);


        // check if bar is hiding QlineSeries
        if (  static_cast<int>( distributionLine->at(pointF.toPoint().x()).y() ) == pointF.toPoint().y())
        {
            tooltipDistributionSeries(pointF, true);
        }

        QString valueStr = QString("[%1:%2] frequency %3").arg(xStart, 0, 'f', 1).arg(xEnd, 0, 'f', 1).arg(barset->at(index), 0, 'f', 3);
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

void PointStatisticsChartView::setYmax(float value)
{
    axisY->setMax(value);
}

void PointStatisticsChartView::setYmin(float value)
{
    axisY->setMin(value);
}

QList<QPointF> PointStatisticsChartView::exportTrend()
{
    return trend->points();
}

QList<QPointF> PointStatisticsChartView::exportClimaDaily()
{
    return climaDaily->points();
}

QList<QPointF> PointStatisticsChartView::exportClimaDecadal()
{
    return climaDecadal->points();
}

QList<QPointF> PointStatisticsChartView::exportClimaMonthly()
{
    return climaMonthly->points();
}

QList< QList<float> > PointStatisticsChartView::exportDistribution()
{
    QList< QList<float> > barValues;
    QList<QBarSet *> barSet = distributionBar->barSets();
    QList<float> tuple;
    float xStart;
    float xEnd;

    if (barSet.size() != 0)
    {
        for (int i = 0; i<barSet[0]->count(); i++)
        {
            tuple.clear();
            xStart = axisXvalue->min() + (i*widthValue);
            xEnd = axisXvalue->min() + ((i+1)*widthValue);
            tuple.append(xStart);
            tuple.append(xEnd);
            tuple.append(barSet[0]->at(i));
            barValues.append(tuple);
        }
    }
    return barValues;
}

