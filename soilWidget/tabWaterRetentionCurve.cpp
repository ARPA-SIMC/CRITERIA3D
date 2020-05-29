#include "tabWaterRetentionCurve.h"
#include "commonConstants.h"


TabWaterRetentionCurve::TabWaterRetentionCurve()
{

    QHBoxLayout *mainLayout = new QHBoxLayout;
    QVBoxLayout *plotLayout = new QVBoxLayout;

    chart = new QChart();
    chartView = new QChartView(chart);
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

    // Left Button for panning
    /*
    Crit3DCurvePanner* panner = new Crit3DCurvePanner(myPlot, xlog, dxMin, dxMax, dyMin, dyMax);
    panner->setMouseButton(Qt::LeftButton);
    QwtPlotZoomer* zoomer = new QwtPlotZoomer( QwtPlot::xBottom, QwtPlot::yLeft, myPlot->canvas()  );
    zoomer->setRubberBandPen( QColor( Qt::black ) );
    zoomer->setTrackerPen( QColor( Qt::red ) );
    zoomer->setMaxStackDepth(5);
    // CTRL+LeftButton for the zooming
    zoomer->setMousePattern( QwtEventPattern::MouseSelect1, Qt::LeftButton, Qt::ControlModifier);
    // CTRL+RightButton back to full size
    zoomer->setMousePattern( QwtEventPattern::MouseSelect2, Qt::RightButton, Qt::ControlModifier);

    // grid
    QwtPlotGrid *grid = new QwtPlotGrid();
    grid->enableY(true);
    grid->enableYMin(true);
    grid->setMajorPen( Qt::darkGray, 0, Qt::SolidLine );
    grid->setMinorPen( Qt::gray, 0 , Qt::DotLine );
    grid->attach(myPlot);
    */

    chart->legend()->setVisible(false);
    chart->legend()->setAlignment(Qt::AlignBottom);
    chart->setAcceptHoverEvents(true);

    mainLayout->addWidget(barHorizons.groupBox);
    plotLayout->addWidget(chartView);
    mainLayout->addLayout(plotLayout);

    setLayout(mainLayout);
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

    /*
    if (pick != nullptr)
    {
        delete pick;
        pick = nullptr;
    }
    */

    chart->removeAllSeries();
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
        QString name = QString::number(i);
        curve->setName(name);
        double factor = 1.2;
        x = dxMin;
        while (x < dxMax*factor)
        {
            double y = soil::thetaFromSignPsi(-x, &mySoil->horizon[i]);
            if (y != NODATA)
            {
                curve->append(x,y);
                maxThetaSat = MAXVALUE(maxThetaSat, y);
            }
            x *= factor;
        }
        chart->addSeries(curve);
        curve->attachAxis(axisX);
        curve->attachAxis(axisY);
        curveList.push_back(curve);
        connect(curve, &QLineSeries::clicked, [=](){ this->curveClicked(); });

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
                if (x != NODATA && y != NODATA)
                {
                    curveMarkers->append(x,y);
                }
            }
            chart->addSeries(curveMarkers);
            curveMarkers->attachAxis(axisX);
            curveMarkers->attachAxis(axisY);
            curveMarkerMap[i] = curveMarkers;
            //connect(curveMarkers, &QScatterSeries::clicked, [=](){ this->curveClicked(); });
        }
    }

    // round maxThetaSat to first decimal
    maxThetaSat = ceil(maxThetaSat * 10) * 0.1;

    // rescale to maxThetaSat
    axisY->setMax(std::max(yMax, maxThetaSat));

    /*
    pick = new Crit3DCurvePicker(myPlot, curveList, curveMarkerMap);
    pick->setStateMachine(new QwtPickerClickPointMachine());
    connect(pick, SIGNAL(clicked(int)), this, SLOT(curveClicked(int)));

    */
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
    // TO DO
    qDebug() << "curveClicked";
    QLineSeries *series = qobject_cast<QLineSeries *>(sender());
    //int index = series->name().toInt();
    int index = curveList.indexOf(series);
    qDebug() << "index " << index;
    //barHorizons.selectItem(index);
    //emit horizonSelected(index);
}

void TabWaterRetentionCurve::highlightCurve( bool isHightlight )
{
    for ( int i = 0; i < curveList.size(); i++ )
    {
        QColor curveColor = curveList[i]->color();
        if ( isHightlight && i == indexSelected)
        {
            qreal alpha = 1.0;
            curveColor.setAlphaF(alpha);
            curveList[i]->setColor(curveColor);
            if (!curveMarkerMap.isEmpty() && i<curveMarkerMap.size())
            {
                curveMarkerMap[i]->setColor(curveColor);
            }
        }
        else
        {
            qreal alpha = 0.5;
            curveColor.setAlphaF(alpha);
            curveList[i]->setColor(curveColor);
            if (!curveMarkerMap.isEmpty() && i<curveMarkerMap.size())
            {
                curveMarkerMap[i]->setColor(curveColor);
            }
        }
    }

}
