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

    SNHT_T95Values = new QLineSeries();
    SNHT_T95Values->setName("SNHT_T95");
    SNHT_T95Values->setColor(Qt::black);

    axisX = new QValueAxis();
    axisY = new QValueAxis();

    chart()->addAxis(axisX, Qt::AlignBottom);
    chart()->addAxis(axisY, Qt::AlignLeft);
    chart()->setDropShadowEnabled(false);

    chart()->legend()->setVisible(true);
    chart()->legend()->setAlignment(Qt::AlignBottom);
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

/*
void SynchronicityChartView::drawGraphStation(std::vector<int> years, std::vector<float> outputValues)
{
    chart()->legend()->setVisible(true);
    float maxValue = NODATA;
    float minValue = -NODATA;
    for (unsigned int i = 0; i < years.size(); i++)
    {
        if (outputValues[i] != NODATA)
        {
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
        axisY->setMax(maxValue);
        axisY->setMin(minValue);
    }
    else
    {
        axisY->setMax(maxValue+3);
        axisY->setMin(minValue-3);
    }
    axisX->setRange(years[0], years[years.size()-1]);
    int nYears = years.size();
    if ( nYears <= 15)
    {
        axisX->setTickCount(nYears);
    }
    else
    {
        int div = 0;
        for (int i = 2; i<=4; i++)
        {
            if ( (nYears-1) % i == 0 && (nYears-1)/i <= 15)
            {
                div = i;
                break;
            }
        }
        if (div == 0)
        {
            axisX->setTickCount(2);
        }
        else
        {
            axisX->setTickCount( (nYears-1)/div + 1);
        }
    }
    axisX->setLabelFormat("%d");
    axisY->setLabelFormat("%.1f");
    axisX->setTitleText("years");
    axisY->setTitleText("");
    chart()->addSeries(tValues);
    tValues->attachAxis(axisX);
    tValues->attachAxis(axisY);
    connect(tValues, &QScatterSeries::hovered, this, &SynchronicityChartView::tooltipSNHTSeries);
    connect(SNHT_T95Values, &QScatterSeries::hovered, this, &SynchronicityChartView::tooltipSNHTSeries);

}

void SynchronicityChartView::drawCraddock(int myFirstYear, int myLastYear, std::vector<std::vector<float>> outputValues, std::vector<QString> refNames, meteoVariable myVar, double averageValue)
{

    clearCraddockSeries();
    float myMinValue = NODATA;
    float myMaxValue = NODATA;
    for (int refIndex = 0; refIndex<refNames.size(); refIndex++)
    {
        QLineSeries* refSeries = new QLineSeries();
        refSeries->setName(refNames[refIndex]);
        refSeries->append(myFirstYear - 1,0);
        for (int myYear = 0; myYear < outputValues[refIndex].size(); myYear++)
        {
            float myValue = outputValues[refIndex][myYear];
            int year = myFirstYear + myYear;
            if (myValue != NODATA)
            {
                if ((myMinValue == NODATA) || (myValue < myMinValue))
                {
                    myMinValue = myValue;
                }
                if ((myMaxValue == NODATA) || (myValue > myMaxValue))
                {
                    myMaxValue = myValue;
                }
                refSeries->append(year,myValue);
            }
            else
            {
                refSeries->append(year,NODATA);
            }
        }
        craddockSeries.push_back(refSeries);
    }
    float myThreshold;
    if (myVar == dailyPrecipitation)
    {
        myThreshold = averageValue * (myLastYear - myFirstYear) / 25;
        axisY->setTitleText("[mm]");

    }
    else if (myVar == dailyAirTemperatureAvg || myVar == dailyAirTemperatureRange)
    {
        myThreshold = averageValue * (myLastYear - myFirstYear) / 50;
        axisY->setTitleText("[°C]");
    }
    axisY->setMax(std::max(myThreshold, myMaxValue));
    axisY->setMin(std::min(-myThreshold, myMinValue));
    axisX->setLabelFormat("%d");
    axisY->setLabelFormat("%.1f");
    axisX->setTitleText("years");
    axisX->setRange(myFirstYear-1, myLastYear);
    if (myLastYear-myFirstYear+2 <= 15)
    {
        axisX->setTickCount(myLastYear-myFirstYear+2);
    }
    else
    {
        axisX->setTickCount(15);
    }
    for (int i = 0; i<craddockSeries.size(); i++)
    {
        chart()->addSeries(craddockSeries[i]);
        craddockSeries[i]->attachAxis(axisX);
        craddockSeries[i]->attachAxis(axisY);
        connect(craddockSeries[i], &QScatterSeries::hovered, this, &SynchronicityChartView::tooltipCraddockSeries);
    }

}

void SynchronicityChartView::clearSNHTSeries()
{
    if (chart()->series().contains(tValues))
    {
        chart()->removeSeries(tValues);
        tValues->clear();
    }
    if (chart()->series().contains(SNHT_T95Values))
    {
        chart()->removeSeries(SNHT_T95Values);
        SNHT_T95Values->clear();
    }
}

void SynchronicityChartView::clearCraddockSeries()
{
    if (chart()->series().size() > 0)
    {
        for(int i = 0; i<craddockSeries.size(); i++)
        {
            if (chart()->series().contains(craddockSeries[i]))
            {
                chart()->removeSeries(craddockSeries[i]);
                craddockSeries[i]->clear();
            }
        }
    }
    craddockSeries.clear();
}

void SynchronicityChartView::tooltipSNHTSeries(QPointF point, bool state)
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

void SynchronicityChartView::tooltipCraddockSeries(QPointF point, bool state)
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

QList<QPointF> SynchronicityChartView::exportSNHTValues()
{
    return tValues->points();
}

QList<QList<QPointF>> SynchronicityChartView::exportCraddockValues(QList<QString> &refNames)
{
    QList<QList<QPointF>> pointsAllSeries;
    for (int i = 0; i<craddockSeries.size(); i++)
    {
        QList<QPointF> pointsSerie = craddockSeries[i]->points();
        refNames.push_back(craddockSeries[i]->name());
        pointsAllSeries.push_back(pointsSerie);
    }
    return pointsAllSeries;
}
*/
