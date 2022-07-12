#include "annualSeriesChartView.h"
#include "commonConstants.h"
#include <QtCharts/QLegendMarker>
#include <QtGui/QImage>
#include <QtGui/QPainter>
#include <QtCore/QtMath>
#include <qdebug.h>

AnnualSeriesChartView::AnnualSeriesChartView(QWidget *parent) :
    QChartView(new QChart(), parent)
{
    annualSeries = new QScatterSeries();
    annualSeries->setName("Annual Series");
    annualSeries->setColor(Qt::red);
    annualSeries->setMarkerSize(10.0);
    setRenderHint(QPainter::Antialiasing);

    axisX = new QValueAxis();
    axisX->setTitleText("years");
    axisY = new QValueAxis();

    chart()->addAxis(axisX, Qt::AlignBottom);
    chart()->addAxis(axisY, Qt::AlignLeft);
    chart()->setDropShadowEnabled(false);

    chart()->legend()->setVisible(true);
    chart()->legend()->setAlignment(Qt::AlignBottom);
    m_tooltip = new Callout(chart());
    m_tooltip->hide();
}

void AnnualSeriesChartView::draw(std::vector<int> years, std::vector<float> outputValues)
{

    float maxValue = NODATA;
    float minValue = -NODATA;
    for (unsigned int i = 0; i < years.size(); i++)
    {
        if (outputValues[i] != NODATA)
        {
            annualSeries->append(QPointF(years[i], outputValues[i]));
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
    axisX->setRange(years[0], years[years.size()-1]);
    if (years.size() <= 15)
    {
        axisX->setTickCount(years.size());
    }
    else
    {
        axisX->setTickCount(15);
    }
    axisX->setLabelFormat("%d");
    axisY->setLabelFormat("%.1f");
    chart()->addSeries(annualSeries);
    annualSeries->attachAxis(axisX);
    annualSeries->attachAxis(axisY);
    connect(annualSeries, &QScatterSeries::hovered, this, &AnnualSeriesChartView::tooltipAnnualSeries);
}

void AnnualSeriesChartView::clearSeries()
{
    if (chart()->series().size() > 0)
    {
        chart()->removeSeries(annualSeries);
        annualSeries->clear();
    }
}

void AnnualSeriesChartView::setYmax(float value)
{
    axisY->setMax(value);
}

void AnnualSeriesChartView::setYmin(float value)
{
    axisY->setMin(value);
}

void AnnualSeriesChartView::setYTitle(QString title)
{
    axisY->setTitleText(title);
}

void AnnualSeriesChartView::tooltipAnnualSeries(QPointF point, bool state)
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

QList<QPointF> AnnualSeriesChartView::exportAnnualValues()
{
    return annualSeries->points();
}
