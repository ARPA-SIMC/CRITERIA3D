#include "tabHydraulicConductivityCurve.h"
#include "commonConstants.h"

TabHydraulicConductivityCurve::TabHydraulicConductivityCurve()
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

    axisY = new QLogValueAxis();
    axisY->setTitleText(QString("Water conductivity [%1]").arg(QString("cm day-1")));
    axisY->setBase(10);
    axisY->setRange(yMin, yMax);
    axisY->setLabelFormat("%1.0E");

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
    fillElement = false;

}

void TabHydraulicConductivityCurve::resetAll()
{
    // delete all Widgets
    barHorizons.clear();

    if (!curveList.isEmpty())
    {
        qDeleteAll(curveList);
        curveList.clear();
    }

    chart->removeAllSeries();
    delete m_tooltip;
    m_tooltip = new Callout(chart);
    m_tooltip->hide();
    fillElement = false;

}

bool TabHydraulicConductivityCurve::getFillElement() const
{
    return fillElement;
}

void TabHydraulicConductivityCurve::setFillElement(bool value)
{
    fillElement = value;
}

void TabHydraulicConductivityCurve::insertElements(soil::Crit3DSoil *soil)
{
    if (soil == nullptr) return;

    resetAll();

    barHorizons.draw(soil);

    fillElement = true;
    mySoil = soil;
    double x;
    double maxValue = 0;

    for (unsigned int i = 0; i < mySoil->nrHorizons; i++)
    {
        // insert Curves
        QColor color = barHorizons.getColor(i);
        // insert Curves
        QLineSeries* curve = new QLineSeries();
        curve->setColor(color);
        double factor = 1.1;
        x = xMin;
        while (x <= (xMax * factor))
        {
            double y = soil::waterConductivityFromSignPsi(-x, mySoil->horizon[i]);
            if (y != NODATA)
            {
                curve->append(x,y);
                maxValue = std::max(maxValue, y);
            }
            x *= factor;
        }
        curveList.push_back(curve);
        chart->addSeries(curve);
        curve->attachAxis(axisX);
        curve->attachAxis(axisY);
        connect(curve, &QXYSeries::clicked, this, &TabHydraulicConductivityCurve::curveClicked);
        connect(curve, &QLineSeries::hovered, this, &TabHydraulicConductivityCurve::tooltipLineSeries);
    }

    // round maxValue to first decimal
    maxValue = ceil(maxValue * 10) * 0.1;

    // rescale to maxValue
    axisY->setMax(std::max(yMax, maxValue));

    for (int i=0; i < barHorizons.barList.size(); i++)
    {
        connect(barHorizons.barList[i], SIGNAL(clicked(int)), this, SLOT(widgetClicked(int)));
    }
}


void TabHydraulicConductivityCurve::widgetClicked(int index)
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

void TabHydraulicConductivityCurve::curveClicked()
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

void TabHydraulicConductivityCurve::highlightCurve( bool isHightlight )
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

void TabHydraulicConductivityCurve::tooltipLineSeries(QPointF point, bool state)
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

