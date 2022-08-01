#include "interpolationChartView.h"
#include "commonConstants.h"
#include "QDate"
#include <QtCharts/QLegendMarker>
#include <QtGui/QImage>
#include <QtGui/QPainter>
#include <QtCore/QtMath>
#include <qdebug.h>

InterpolationChartView::InterpolationChartView(QWidget *parent) :
        QChartView(new QChart(), parent)
{
    axisX = new QDateTimeAxis();
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

void InterpolationChartView::setYmax(float value)
{
    axisY->setMax(value);
}

void InterpolationChartView::setYmin(float value)
{
    axisY->setMin(value);
}


void InterpolationChartView::drawGraphInterpolation(std::vector<float> values, QDate myStartDate, QString var, int lag, int smooth)
{
    chart()->legend()->setVisible(true);
    QString name = var+" lag="+QString::number(lag)+" smooth="+QString::number(smooth);
    for(int i = 0; i<interpolationGraphSeries.size(); i++)
    {
        if (interpolationGraphSeries[i]->name() == name)
        {
            return;
        }
    }
    QLineSeries* graphSeries = new QLineSeries();
    graphSeries->setName(name);
    for (unsigned int nValues = 0; nValues < values.size(); nValues++)
    {
        QDateTime myDate(myStartDate.addDays(nValues), QTime(1,0,0),Qt::UTC);
        if (values[nValues] != NODATA)
        {
            if (values[nValues] > maxValue)
            {
                maxValue = values[nValues];
            }
            if (values[nValues] < minValue)
            {
                minValue = values[nValues];
            }
            graphSeries->append(myDate.toMSecsSinceEpoch(), values[nValues]);
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
    axisX->setRange(QDateTime(myStartDate, QTime(1,0,0),Qt::UTC), QDateTime(myStartDate.addDays(values.size()-1), QTime(1,0,0),Qt::UTC));
    if ( values.size() <= 15)
    {
        axisX->setTickCount(values.size());
    }
    else
    {
        axisX->setTickCount(15);
    }
    axisX->setFormat("dd.MM.yyyy");
    axisY->setLabelFormat("%.1f");
    axisX->setTitleText("date");
    axisY->setTitleText("r2");
    chart()->addSeries(graphSeries);
    graphSeries->attachAxis(axisX);
    graphSeries->attachAxis(axisY);
    interpolationGraphSeries.push_back(graphSeries);
    connect(graphSeries, &QLineSeries::hovered, this, &InterpolationChartView::tooltipGraphInterpolationSeries);
}

void InterpolationChartView::clearInterpolationGraphSeries()
{
    if (chart()->series().size() > 0)
    {
        for(int i = 0; i<interpolationGraphSeries.size(); i++)
        {
            if (chart()->series().contains(interpolationGraphSeries[i]))
            {
                chart()->removeSeries(interpolationGraphSeries[i]);
                interpolationGraphSeries[i]->clear();
            }
        }
    }
    interpolationGraphSeries.clear();
    maxValue = NODATA;
    minValue = -NODATA;
}

void InterpolationChartView::tooltipGraphInterpolationSeries(QPointF point, bool state)
{

    auto serie = qobject_cast<QLineSeries *>(sender());
    if (state)
    {
        int xValue = point.x();
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
