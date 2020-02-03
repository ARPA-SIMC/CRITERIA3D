#include "curvePanner.h"
#include "qwt_scale_div.h"
#include "qwt_plot.h"
#include <math.h>


Crit3DCurvePanner::Crit3DCurvePanner(QwtPlot *plot, scaleType type, double dxMin, double dxMax, double dyMin, double dyMax) : QwtPlotPanner (plot->canvas()),
    qwtPlot(plot), type(type), dxMin(dxMin), dxMax(dxMax), dyMin(dyMin), dyMax(dyMax)
{

}

void Crit3DCurvePanner::moveCanvas(int dx, int dy)
{
    if ( dx == 0 && dy == 0 )
        return;

    if ( qwtPlot == nullptr )
        return;

    const bool doAutoReplot = qwtPlot->autoReplot();
    qwtPlot->setAutoReplot(false);

    switch (type) {
        case linear:
        {
            //TO DO
            break;
        }
        case xlog:
        {
            moveCanvasXlog(dx, dy);
            break;
        }
        case ylog:
        {
            //TO DO
            break;
        }
        case xlogylog:
        {
            moveCanvasXlogYlog(dx, dy);
            break;
        }
    }

    qwtPlot->setAutoReplot(doAutoReplot);
    qwtPlot->replot();
}

void Crit3DCurvePanner::moveCanvasXlog(int dx, int dy)
{
    for ( int axis = 0; axis < QwtPlot::axisCnt; axis++ )
    {
        const QwtScaleMap map = qwtPlot->canvasMap(axis);

        const QwtScaleDiv *scaleDiv;
        scaleDiv = &qwtPlot->axisScaleDiv(axis);
        double i1 = map.transform(scaleDiv->lowerBound());
        double i2 = map.transform(scaleDiv->upperBound());

        double d1, d2;
        if ( axis == QwtPlot::xBottom || axis == QwtPlot::xTop )
        {
            d1 = map.invTransform(i1 - dx);
            d2 = map.invTransform(i2 - dx);
        }
        else
        {
            d1 = map.invTransform(i1 - dy);
            d2 = map.invTransform(i2 - dy);
        }
        double range;
        if (axis == QwtPlot::xBottom)
        {
            range = log10(d2) - log10(d1);
            if(d1 < dxMin)
            {
                d1 = dxMin;
                d2 = pow(10, (range + log10(d1)));
            }
            if(d2 > dxMax)
            {
                d2 = dxMax;
                d1 = pow(10, (log10(d2) - range));
            }
        }
        if (axis == QwtPlot::yLeft)
        {
            range = d2 - d1;
            if(d1 < dyMin)
            {
                d1 = dyMin;
                d2 = range + d1;
            }
            if(d2 > dyMax)
            {
                d2 = dyMax;
                d1 = d2 - range;
            }
        }
        qwtPlot->setAxisScale(axis, d1, d2);
    }
}

void Crit3DCurvePanner::moveCanvasXlogYlog(int dx, int dy)
{
    for ( int axis = 0; axis < QwtPlot::axisCnt; axis++ )
    {
        const QwtScaleMap map = qwtPlot->canvasMap(axis);

        const QwtScaleDiv *scaleDiv;
        scaleDiv = &qwtPlot->axisScaleDiv(axis);
        double i1 = map.transform(scaleDiv->lowerBound());
        double i2 = map.transform(scaleDiv->upperBound());

        double d1, d2;
        if ( axis == QwtPlot::xBottom || axis == QwtPlot::xTop )
        {
            d1 = map.invTransform(i1 - dx);
            d2 = map.invTransform(i2 - dx);
        }
        else
        {
            d1 = map.invTransform(i1 - dy);
            d2 = map.invTransform(i2 - dy);
        }
        double range;
        if (axis == QwtPlot::xBottom)
        {
            range = log10(d2) - log10(d1);
            if(d1 < dxMin)
            {
                d1 = dxMin;
                d2 = pow(10, (range + log10(d1)));
            }
            if(d2 > dxMax)
            {
                d2 = dxMax;
                d1 = pow(10, (log10(d2) - range));
            }
        }
        if (axis == QwtPlot::yLeft)
        {
            range = log10(d2) - log10(d1);
            if(d1 < dyMin)
            {
                d1 = dyMin;
                d2 = pow(10, (range + log10(d1)));
            }
            if(d2 > dyMax)
            {
                d2 = dyMax;
                d1 = pow(10, (log10(d2) - range));
            }
        }
        qwtPlot->setAxisScale(axis, d1, d2);
    }
}

