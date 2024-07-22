#include "waterTableChartView.h"


WaterTableChartView::WaterTableChartView(QWidget *parent) :
    QChartView(new QChart(), parent)
{
    obsDepthSeries = new QScatterSeries();
    obsDepthSeries->setName("Observed");
    obsDepthSeries->setColor(Qt::green);
    obsDepthSeries->setMarkerSize(8.0);

    hindcastSeries = new QLineSeries();
    hindcastSeries->setName("hindcast");
    hindcastSeries->setColor(Qt::red);

    interpolationSeries = new QLineSeries();
    interpolationSeries->setName("interpolation");
    interpolationSeries->setColor(QColor(0,0,1));

    climateSeries = new QLineSeries();
    climateSeries->setName("climate");
    climateSeries->setColor(Qt::green);

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


void WaterTableChartView::drawWaterTable(WaterTable &waterTable, float maximumObservedDepth)
{
    axisY->setMax(maximumObservedDepth);  // unit of observed watertable data, usually [cm]
    axisY->setMin(0);
    axisY->setLabelFormat("%d");
    axisY->setTickCount(16);

    QDateTime firstDateTime, lastDateTime;
    int nrDays = int(waterTable.interpolationSeries.size());
    firstDateTime.setDate(waterTable.firstDate);
    lastDateTime.setDate(waterTable.firstDate);
    lastDateTime = lastDateTime.addDays(nrDays-1);

    axisX->setRange(firstDateTime, lastDateTime);
    axisX->setTickCount(15);

    QDateTime currentDateTime = firstDateTime;
    for (int day = 0; day < nrDays; day++)
    {
        QDate firstJanuary;
        firstJanuary.setDate(currentDateTime.date().year(), 1, 1);
        int doy = firstJanuary.daysTo(currentDateTime.date()) + 1;

        hindcastSeries->append(currentDateTime.toMSecsSinceEpoch(), waterTable.hindcastSeries[day]);
        interpolationSeries->append(currentDateTime.toMSecsSinceEpoch(), waterTable.interpolationSeries[day]);
        climateSeries->append(currentDateTime.toMSecsSinceEpoch(), waterTable.WTClimateDaily[doy]);

        if(waterTable.getWell()->depths.contains(currentDateTime.date()))
        {
            int myDepth = waterTable.getWell()->depths[currentDateTime.date()];
            obsDepthSeries->append(currentDateTime.toMSecsSinceEpoch(), myDepth);
        }

        currentDateTime = currentDateTime.addDays(1);
    }

    chart()->addSeries(obsDepthSeries);
    chart()->addSeries(hindcastSeries);
    chart()->addSeries(interpolationSeries);
    chart()->addSeries(climateSeries);

    obsDepthSeries->attachAxis(axisX);
    obsDepthSeries->attachAxis(axisY);
    hindcastSeries->attachAxis(axisX);
    hindcastSeries->attachAxis(axisY);
    interpolationSeries->attachAxis(axisX);
    interpolationSeries->attachAxis(axisY);
    climateSeries->attachAxis(axisX);
    climateSeries->attachAxis(axisY);

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
