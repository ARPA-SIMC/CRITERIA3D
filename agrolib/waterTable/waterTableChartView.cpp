#include "waterTableChartView.h"

WaterTableChartView::WaterTableChartView(QWidget *parent) :
    QChartView(new QChart(), parent)
{
    obsDepthSeries = new QScatterSeries();
    obsDepthSeries->setName("Observed");
    obsDepthSeries->setColor(Qt::green);
    obsDepthSeries->setMarkerSize(10.0);

    hindcastSeries = new QLineSeries();
    hindcastSeries->setName("hindcast");
    hindcastSeries->setColor(Qt::red);

    interpolationSeries = new QLineSeries();
    interpolationSeries->setName("interpolation");
    interpolationSeries->setColor(Qt::black);

    axisX = new QDateTimeAxis();
    axisX->setFormat("yyyy/MM/dd");
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

void WaterTableChartView::draw(std::vector<QDate> myDates, std::vector<float> myHindcastSeries, std::vector<float> myInterpolateSeries, QMap<QDate, int> obsDepths)
{

    int nDays = myDates.size();
    QDateTime myDateTime;
    myDateTime.setTime(QTime(0,0));
    for (int day = 0; day < nDays; day++)
    {
        myDateTime.setDate(myDates[day]);
        hindcastSeries->append(myDateTime.toMSecsSinceEpoch(), myHindcastSeries[day]);
        interpolationSeries->append(myDateTime.toMSecsSinceEpoch(), myInterpolateSeries[day]);

        if(obsDepths.contains(myDates[day]))
        {
            int myDepth = obsDepths[myDates[day]];
            obsDepthSeries->append(myDateTime.toMSecsSinceEpoch(), myDepth);
        }
    }


    axisY->setMax(300);
    axisY->setMin(0);
    axisY->setLabelFormat("%d");

    QDateTime firstDateTime;
    firstDateTime.setDate(myDates[0]);
    firstDateTime.setTime(QTime(0,0));
    QDateTime lastDateTime;
    lastDateTime.setDate(myDates[myDates.size()-1]);
    lastDateTime.setTime(QTime(0,0));

    axisX->setTickCount(12);
    axisX->setMin(firstDateTime);
    axisX->setMax(lastDateTime);

    chart()->addSeries(obsDepthSeries);
    chart()->addSeries(hindcastSeries);
    chart()->addSeries(interpolationSeries);

    connect(obsDepthSeries, &QScatterSeries::hovered, this, &WaterTableChartView::tooltipObsDepthSeries);
    connect(hindcastSeries, &QLineSeries::hovered, this, &WaterTableChartView::tooltipLineSeries);
    connect(interpolationSeries, &QLineSeries::hovered, this, &WaterTableChartView::tooltipLineSeries);
    foreach(QLegendMarker* marker, chart()->legend()->markers())
    {
        marker->setVisible(true);
        marker->series()->setVisible(true);
        QObject::connect(marker, &QLegendMarker::clicked, this, &WaterTableChartView::handleMarkerClicked);
    }
    return;
}

void WaterTableChartView::tooltipObsDepthSeries(QPointF point, bool state)
{

    auto serie = qobject_cast<QScatterSeries *>(sender());
    if (state)
    {
        QDateTime firstDate(QDate(1970,1,1), QTime(0,0,0));
        QDateTime xValue = firstDate.addMSecs(point.x());
        double yValue = point.y();

        m_tooltip->setText(QString("%1: %2").arg(xValue.date().toString("yyyy/MM/dd")).arg(yValue, 0, 'd'));
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
        double yValue = point.y();

        m_tooltip->setText(QString("%1: %2").arg(xValue.date().toString("yyyy/MM/dd")).arg(yValue, 0, 'd'));
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
