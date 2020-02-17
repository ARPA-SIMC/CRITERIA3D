#include "tabHydraulicConductivityCurve.h"
#include "commonConstants.h"
#include "curvePanner.h"
#include <qwt_point_data.h>
#include <qwt_scale_engine.h>

#include <qwt_plot_grid.h>
#include <qwt_plot_panner.h>
#include <qwt_plot_zoomer.h>
#include <qwt_event_pattern.h>
#include <qwt_picker_machine.h>
#include <qwt_symbol.h>

TabHydraulicConductivityCurve::TabHydraulicConductivityCurve()
{
    pick = nullptr;
    QHBoxLayout *mainLayout = new QHBoxLayout;
    QVBoxLayout *plotLayout = new QVBoxLayout;

    myPlot = new QwtPlot;
    myPlot->setAxisScaleEngine(QwtPlot::xBottom, new QwtLogScaleEngine(10));
    myPlot->setAxisScaleEngine(QwtPlot::yLeft, new QwtLogScaleEngine(10));
    myPlot->setAxisTitle(QwtPlot::yLeft,QString("Water conductivity [%1]").arg(QString("cm day-1")));
    myPlot->setAxisTitle(QwtPlot::xBottom,QString("Water potential [%1]").arg(QString("kPa")));
    myPlot->setAxisScale(QwtPlot::xBottom,xMin, xMax);
    myPlot->setAxisScale(QwtPlot::yLeft,yMin, yMax);

    // Left Button for panning
    Crit3DCurvePanner* panner = new Crit3DCurvePanner(myPlot, xlogylog, dxMin, dxMax, yMin, yMax);
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

    mainLayout->addWidget(barHorizons.groupBox);
    plotLayout->addWidget(myPlot);
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

    if (pick != nullptr)
    {
        delete pick;
        pick = nullptr;
    }

    myPlot->detachItems( QwtPlotItem::Rtti_PlotCurve );
    myPlot->replot();
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
    // rescale
    myPlot->setAxisScale(QwtPlot::xBottom, xMin, xMax);

    if (soil == nullptr) return;

    resetAll();

    barHorizons.draw(soil);

    fillElement = true;
    mySoil = soil;
    QVector<double> xVector;
    QVector<double> yVector;
    double x;
    double maxThetaSat = 0;

    for (unsigned int i = 0; i < mySoil->nrHorizons; i++)
    {
        // insert Curves
        QwtPlotCurve *curve = new QwtPlotCurve;
        xVector.clear();
        yVector.clear();
        double factor = 1.1;
        x = dxMin;
        while (x < dxMax*factor)
        {
            double y = soil::waterConductivityFromSignPsi(-x, &mySoil->horizon[i]);
            if (y != NODATA)
            {
                xVector.push_back(x);
                yVector.push_back(y);
                maxThetaSat = MAXVALUE(maxThetaSat, y);
            }
            x *= factor;
        }
        QwtPointArrayData *data = new QwtPointArrayData(xVector,yVector);
        curve->setSamples(data);
        curve->attach(myPlot);
        curveList.push_back(curve);

    }

    // round maxThetaSat to first decimal
    maxThetaSat = ceil(maxThetaSat * 10) * 0.1;

    // rescale to maxThetaSat
    myPlot->setAxisScale(QwtPlot::yLeft, yMin, std::max(yMax, maxThetaSat));

    pick = new Crit3DCurvePicker(myPlot, curveList);
    pick->setStateMachine(new QwtPickerClickPointMachine());
    connect(pick, SIGNAL(clicked(int)), this, SLOT(curveClicked(int)));

    for (int i=0; i < barHorizons.barList.size(); i++)
    {
        connect(barHorizons.barList[i], SIGNAL(clicked(int)), this, SLOT(widgetClicked(int)));
    }

    myPlot->replot();
}


void TabHydraulicConductivityCurve::widgetClicked(int index)
{
    // check selection state

    if (barHorizons.barList[index]->getSelected())
    {
        barHorizons.deselectAll(index);

        // select the right curve
        pick->setSelectedCurveIndex(index);
        pick->highlightCurve(true);
        emit horizonSelected(index);
    }
    else
    {
        pick->highlightCurve(false);
        pick->setSelectedCurveIndex(-1);
        emit horizonSelected(-1);
    }


}

void TabHydraulicConductivityCurve::curveClicked(int index)
{
    barHorizons.selectItem(index);
    emit horizonSelected(index);
}
