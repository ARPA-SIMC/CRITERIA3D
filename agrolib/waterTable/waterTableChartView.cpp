#include "waterTableChartView.h"

WaterTableChartView::WaterTableChartView(QWidget *parent) :
    QChartView(new QChart(), parent)
{
    obsDepthSeries = new QScatterSeries();
    obsDepthSeries->setName("Observed");
    obsDepthSeries->setColor(Qt::red);
    obsDepthSeries->setMarkerSize(8.0);

    hindcastSeries = new QLineSeries();
    hindcastSeries->setName("hindcast");
    hindcastSeries->setColor(Qt::green);

    interpolationSeries = new QLineSeries();
    interpolationSeries->setName("interpolation");
    interpolationSeries->setColor(QColor(0,0,1));

    axisX = new QDateTimeAxis();
    axisX->setFormat("yyyy/MM");
    axisY = new QValueAxis();
    axisY->setReverse(true);

    chart()->addAxis(axisX, Qt::AlignBottom);
    chart()->addAxis(axisY, Qt::AlignLeft);
    chart()->setDropShadowEnabled(false);

    chart()->legend()->setVisible(true);
    chart()->legend()->setAlignment(Qt::AlignBottom);
    m_tooltip = new Callout(chart());
    m_tooltip->hide();
}


void WaterTableChartView::drawWaterTable(std::vector<QDate> &myDates, std::vector<float> &myHindcastSeries, std::vector<float> &myInterpolateSeries,
                               QMap<QDate, float> obsDepths, float maximumObservedDepth)
{
    axisY->setMax(maximumObservedDepth);  // unit of observed watertable data, usually [cm]
    axisY->setMin(0);
    axisY->setLabelFormat("%d");
    axisY->setTickCount(16);

    QDateTime firstDate;
    firstDate.setDate(myDates[0]);
    QDateTime lastDate;
    lastDate.setDate(myDates[myDates.size()-1]);

    axisX->setRange(firstDate, lastDate);
    axisX->setTickCount(15);

    int nDays = int(myDates.size());
    QDateTime currentDateTime;
    for (int day = 0; day < nDays; day++)
    {
        currentDateTime.setDate(myDates[day]);
        hindcastSeries->append(currentDateTime.toMSecsSinceEpoch(), myHindcastSeries[day]);
        interpolationSeries->append(currentDateTime.toMSecsSinceEpoch(), myInterpolateSeries[day]);

        if(obsDepths.contains(myDates[day]))
        {
            int myDepth = obsDepths[myDates[day]];
            obsDepthSeries->append(currentDateTime.toMSecsSinceEpoch(), myDepth);
        }
    }

    chart()->addSeries(obsDepthSeries);
    chart()->addSeries(hindcastSeries);
    chart()->addSeries(interpolationSeries);

    obsDepthSeries->attachAxis(axisX);
    obsDepthSeries->attachAxis(axisY);
    hindcastSeries->attachAxis(axisX);
    hindcastSeries->attachAxis(axisY);
    interpolationSeries->attachAxis(axisX);
    interpolationSeries->attachAxis(axisY);

    connect(obsDepthSeries, &QScatterSeries::hovered, this, &WaterTableChartView::tooltipObsDepthSeries);
    connect(hindcastSeries, &QLineSeries::hovered, this, &WaterTableChartView::tooltipLineSeries);
    connect(interpolationSeries, &QLineSeries::hovered, this, &WaterTableChartView::tooltipLineSeries);
    foreach(QLegendMarker* marker, chart()->legend()->markers())
    {
        marker->setVisible(true);
        marker->series()->setVisible(true);
        QObject::connect(marker, &QLegendMarker::clicked, this, &WaterTableChartView::handleMarkerClicked);
    }
}


void WaterTableChartView::tooltipObsDepthSeries(QPointF point, bool state)
{

    auto serie = qobject_cast<QScatterSeries *>(sender());
    if (state)
    {
        QDateTime firstDate(QDate(1970,1,1), QTime(0,0,0));
        QDateTime xValue = firstDate.addMSecs(point.x());
        QDate myDate = xValue.date().addDays(1);
        int yValue = point.y();

        m_tooltip->setText(QString("%1: %2").arg(myDate.toString("yyyy/MM/dd")).arg(yValue));
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

void WaterTableChartView::tooltipLineSeries(QPointF point, bool state)
{

    auto serie = qobject_cast<QLineSeries *>(sender());
    if (state)
    {
        QDateTime firstDate(QDate(1970,1,1), QTime(0,0,0));
        QDateTime xValue = firstDate.addMSecs(point.x());
        QDate myDate = xValue.date().addDays(1);
        int yValue = point.y();

        m_tooltip->setText(QString("%1: %2").arg(myDate.toString("yyyy/MM/dd")).arg(yValue));
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

void WaterTableChartView::handleMarkerClicked()
{

    QLegendMarker* marker = qobject_cast<QLegendMarker*> (sender());

    // Toggle visibility of series
    bool isVisible = marker->series()->isVisible();
    marker->series()->setVisible(!isVisible);

    // Turn legend marker back to visible, since otherwise hiding series also hides the marker
    marker->setVisible(true);

    // change marker alpha, if series is not visible
    qreal alpha;
    if (isVisible)
    {
        alpha = 0.5;
    }
    else
    {
        alpha = 1.0;
    }

    QColor color;
    QBrush brush = marker->labelBrush();
    color = brush.color();
    color.setAlphaF(alpha);
    brush.setColor(color);
    marker->setLabelBrush(brush);

    brush = marker->brush();
    color = brush.color();
    color.setAlphaF(alpha);
    brush.setColor(color);
    marker->setBrush(brush);

    QPen pen = marker->pen();
    color = pen.color();
    color.setAlphaF(alpha);
    pen.setColor(color);
    marker->setPen(pen);

}

QList<QPointF> WaterTableChartView::exportInterpolationValues()
{
    QList<QPointF> pointsSerie = interpolationSeries->points();
    return pointsSerie;
}
