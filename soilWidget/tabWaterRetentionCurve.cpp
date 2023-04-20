#include "tabWaterRetentionCurve.h"
#include "commonConstants.h"


TabWaterRetentionCurve::TabWaterRetentionCurve()
{

    QHBoxLayout *mainLayout = new QHBoxLayout;
    QVBoxLayout *plotLayout = new QVBoxLayout;

    chart = new QChart();
    chartView = new QChartView();
    chartView->setChart(chart);

    axisX = new QLogValueAxis();
    axisX->setTitleText(QString("Water potential [%1]").arg(QString("kPa")));
    axisX->setBase(10);
    axisX->setRange(xMin, xMax);
    axisY = new QValueAxis();
    axisY->setTitleText(QString("Volumetric water content [%1]").arg(QString("m3 m-3")));
    axisY->setRange(yMin, yMax);
    axisY->setTickCount(7);

    QFont font = axisY->titleFont();
    font.setPointSize(11);
    font.setBold(true);
    axisX->setTitleFont(font);
    axisY->setTitleFont(font);

    chart->addAxis(axisX, Qt::AlignBottom);
    chart->addAxis(axisY, Qt::AlignLeft);

    chart->legend()->setVisible(false);
    chart->legend()->setAlignment(Qt::AlignBottom);
    chart->setAcceptHoverEvents(true);

    m_tooltip = new Callout(chart);
    m_tooltip->hide();

    mainLayout->addWidget(barHorizons.groupBox);
    plotLayout->addWidget(chartView);
    mainLayout->addLayout(plotLayout);

    setLayout(mainLayout);
    setMouseTracking(true);
    fillElement = false;

}

void TabWaterRetentionCurve::resetAll()
{

    // delete all Widgets
    barHorizons.clear();

    if (!curveList.isEmpty())
    {
        qDeleteAll(curveList);
        curveList.clear();
    }

    if (!curveMarkerMap.isEmpty())
    {
        qDeleteAll(curveMarkerMap);
        curveMarkerMap.clear();
    }

    chart->removeAllSeries();
    delete m_tooltip;
    m_tooltip = new Callout(chart);
    m_tooltip->hide();
    fillElement = false;

}

bool TabWaterRetentionCurve::getFillElement() const
{
    return fillElement;
}

void TabWaterRetentionCurve::setFillElement(bool value)
{
    fillElement = value;
}

void TabWaterRetentionCurve::insertElements(soil::Crit3DSoil *soil)
{

    if (soil == nullptr)
    {
        return;
    }

    resetAll();

    barHorizons.draw(soil);
    fillElement = true;
    mySoil = soil;
    double x;
    double maxThetaSat = 0;

    for (unsigned int i = 0; i < mySoil->nrHorizons; i++)
    {
        QColor color = barHorizons.getColor(i);
        // insert Curves
        QLineSeries* curve = new QLineSeries();
        curve->setColor(color);
        double factor = 1.2;
        x = dxMin;
        while (x < dxMax*factor)
        {
            double y = soil::thetaFromSignPsi(-x, mySoil->horizon[i]);
            if (y != NODATA)
            {
                curve->append(x,y);
                maxThetaSat = MAXVALUE(maxThetaSat, y);
            }
            x *= factor;
        }
        curveList.push_back(curve);
        chart->addSeries(curve);
        curve->attachAxis(axisX);
        curve->attachAxis(axisY);
        connect(curve, &QXYSeries::clicked, this, &TabWaterRetentionCurve::curveClicked);
        connect(curve, &QLineSeries::hovered, this, &TabWaterRetentionCurve::tooltipLineSeries);

        // insert marker
        if (!mySoil->horizon[i].dbData.waterRetention.empty())
        {
            QScatterSeries *curveMarkers = new QScatterSeries();
            curveMarkers->setColor(color);
            curveMarkers->setMarkerSize(8);
            for (unsigned int j = 0; j < mySoil->horizon[i].dbData.waterRetention.size(); j++)
            {
                double x = mySoil->horizon[i].dbData.waterRetention[j].water_potential;
                double y = mySoil->horizon[i].dbData.waterRetention[j].water_content;
                if (x>0 && x != NODATA && y != NODATA)
                {
                    curveMarkers->append(x,y);
                }
            }
            curveMarkerMap[i] = curveMarkers;
            chart->addSeries(curveMarkers);
            curveMarkers->attachAxis(axisX);
            curveMarkers->attachAxis(axisY);
            connect(curveMarkers, &QXYSeries::clicked, this, &TabWaterRetentionCurve::markerClicked);
            connect(curveMarkers, &QScatterSeries::hovered, this, &TabWaterRetentionCurve::tooltipScatterSeries);
        }
    }

    // round maxThetaSat to first decimal
    maxThetaSat = ceil(maxThetaSat * 10) * 0.1;

    // rescale to maxThetaSat
    axisY->setMax(std::max(yMax, maxThetaSat));

    for (int i=0; i < barHorizons.barList.size(); i++)
    {
        connect(barHorizons.barList[i], SIGNAL(clicked(int)), this, SLOT(widgetClicked(int)));
    }
}


void TabWaterRetentionCurve::widgetClicked(int index)
{

    // check selection state
    if (barHorizons.barList[index]->getSelected())
    {
        barHorizons.deselectAll(index);

        // select the right curve
        indexSelected = index;
        highlightCurve(true);
        emit horizonSelected(index);
    }
    else
    {
        indexSelected = -1;
        highlightCurve(false);
        emit horizonSelected(-1);
    }

}

void TabWaterRetentionCurve::curveClicked()
{

    auto serie = qobject_cast<QLineSeries *>(sender());
    if (serie != nullptr)
    {
        int index = curveList.indexOf(serie);
        indexSelected = index;
        highlightCurve(true);
        barHorizons.selectItem(index);
        emit horizonSelected(index);
    }
}

void TabWaterRetentionCurve::markerClicked()
{

    auto serie = qobject_cast<QScatterSeries *>(sender());
    if (serie != nullptr)
    {
        int index = curveMarkerMap.key(serie);
        indexSelected = index;
        highlightCurve(true);
        barHorizons.selectItem(index);
        emit horizonSelected(index);
    }
}

void TabWaterRetentionCurve::highlightCurve( bool isHightlight )
{
    for ( int i = 0; i < curveList.size(); i++ )
    {
        QColor curveColor = curveList[i]->color();
        if ( isHightlight && i == indexSelected)
        {
            QPen pen = curveList[i]->pen();
            pen.setWidth(3);
            pen.setBrush(QBrush(curveColor));
            curveList[i]->setPen(pen);
        }
        else
        {
            QPen pen = curveList[i]->pen();
            pen.setWidth(1);
            pen.setBrush(QBrush(curveColor));
            curveList[i]->setPen(pen);
        }
    }

}

void TabWaterRetentionCurve::tooltipLineSeries(QPointF point, bool state)
{

    auto serie = qobject_cast<QLineSeries *>(sender());
    int index = curveList.indexOf(serie)+1;
    if (state)
    {
        double xValue = point.x();
        double yValue = point.y();

        m_tooltip->setText(QString("Horizon %1 \n%2 %3 ").arg(index).arg(xValue, 0, 'f', 1).arg(yValue, 0, 'f', 3));
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

void TabWaterRetentionCurve::tooltipScatterSeries(QPointF point, bool state)
{

    auto serie = qobject_cast<QScatterSeries *>(sender());
    int index = curveMarkerMap.key(serie)+1;
    if (state)
    {
        double xValue = point.x();
        double yValue = point.y();

        m_tooltip->setText(QString("Horizon %1 \n%2 %3 ").arg(index).arg(xValue, 0, 'f', 1).arg(yValue, 0, 'f', 3));
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

