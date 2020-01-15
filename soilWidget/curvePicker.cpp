#include "curvePicker.h"
#include "qwt_symbol.h"


Crit3DCurvePicker::Crit3DCurvePicker(QwtPlot *plot, QList<QwtPlotCurve *> allCurveList, QMap< int, QwtPlotCurve* > allCurveMarkerMap) :
    QwtPlotPicker(QwtPlot::xBottom, QwtPlot::yLeft, CrossRubberBand, QwtPicker::AlwaysOn, plot->canvas()),
    qwtPlot(plot),
    allCurveList(allCurveList),
    allCurveMarkerMap(allCurveMarkerMap)
{
    connect( this, SIGNAL( selected( const QPointF ) ), this, SLOT( slotSelected( const QPointF ) ) );
}

Crit3DCurvePicker::Crit3DCurvePicker(QwtPlot *plot, QList<QwtPlotCurve *> allCurveList) :
    QwtPlotPicker(QwtPlot::xBottom, QwtPlot::yLeft, CrossRubberBand, QwtPicker::AlwaysOn, plot->canvas()),
    qwtPlot(plot),
    allCurveList(allCurveList)
{
    connect( this, SIGNAL( selected( const QPointF ) ), this, SLOT( slotSelected( const QPointF ) ) );
}


void Crit3DCurvePicker::slotSelected( const QPointF &pos)
{

    QwtPlotCurve *curveFound = nullptr;
    double minDist = std::numeric_limits<double>::max();
    int curveIndex = -1;

    for ( int i = 0; i < allCurveList.size(); i++ )
    {
        QwtPlotCurve *c = allCurveList[i];
        double d = 0;
        if (closestPoint( *c, pos, &d ) != -1)
        {
            if ( d < minDist )
            {
                curveFound = c;
                curveIndex = i;
                minDist = d;
            }
        }

    }

    double tolerance;

    const QwtScaleMap map = qwtPlot->canvasMap(2);
    const QwtScaleDiv *scaleDiv;
    scaleDiv = &qwtPlot->axisScaleDiv(2);

    if (pos.x() < scaleDiv->upperBound())
    {
        tolerance = pos.x()/10;
    }
    else
    {
        tolerance = pos.x()/1000;
    }
    if ( curveFound && minDist < tolerance) // empiric tolerance
    {
        selectedCurveIndex = curveIndex;
        highlightCurve( true );
    }
    else
    {
        highlightCurve(false);  // clear previous selection
        selectedCurveIndex = -1;
    }
    emit clicked(selectedCurveIndex);
}

int Crit3DCurvePicker::getSelectedCurveIndex() const
{
    return selectedCurveIndex;
}

void Crit3DCurvePicker::setSelectedCurveIndex(int value)
{
    selectedCurveIndex = value;
}


void Crit3DCurvePicker::highlightCurve( bool isHightlight )
{
    for ( int i = 0; i < allCurveList.size(); i++ )
    {
        if ( isHightlight && i == selectedCurveIndex)
        {
            allCurveList[i]->setPen(Qt::red, 3);
        }
        else
        {
            allCurveList[i]->setPen(Qt::black, 1);
        }
    }
    if (isHightlight && !allCurveMarkerMap.isEmpty())
    {
        QMap< int, QwtPlotCurve* > ::Iterator iterator;
        for (iterator = allCurveMarkerMap.begin(); iterator!=allCurveMarkerMap.end(); ++iterator)
        {
            if (iterator.key() == selectedCurveIndex)
            {
                allCurveMarkerMap[iterator.key()]->setSymbol(new QwtSymbol( QwtSymbol::Ellipse, QBrush( Qt::red ), QPen( Qt::red, 0 ), QSize( 5, 5 ) ));
            }
            else
            {
                allCurveMarkerMap[iterator.key()]->setSymbol(new QwtSymbol( QwtSymbol::Ellipse, QBrush( Qt::black ), QPen( Qt::black, 0 ), QSize( 5, 5 ) ));
            }
        }
    }
    else if (!isHightlight && !allCurveMarkerMap.isEmpty())
    {
        QMap< int, QwtPlotCurve* > ::Iterator iterator;
        for (iterator = allCurveMarkerMap.begin(); iterator!=allCurveMarkerMap.end(); ++iterator)
        {
            if (iterator.key() == selectedCurveIndex)
            {
                allCurveMarkerMap[iterator.key()]->setSymbol(new QwtSymbol( QwtSymbol::Ellipse, QBrush( Qt::black ), QPen( Qt::black, 0 ), QSize( 5, 5 ) ));
            }
        }
    }
    qwtPlot->replot();
}


int Crit3DCurvePicker::closestPoint( QwtPlotCurve& curve, const QPointF &pos, double *dist )
{
    const size_t numSamples = curve.dataSize();

    if ( curve.plot() == nullptr || numSamples <= 0 )
        return -1;

    const QwtSeriesData<QPointF> *series = curve.data();
    const double xPos = pos.x();
    const double yPos = pos.y();

    int index = -1;
    double dmin = std::numeric_limits<double>::max();

    for ( uint i = 0; i < numSamples; i++ )
    {
        const QPointF sample = series->sample( i );
        const double cx = abs(sample.x()  - xPos);
        const double cy = abs(sample.y()  - yPos);

        const double dist = sqrt( pow(cx,2) + pow(cy,2) );
        if ( dist < dmin )
        {
            index = i;
            dmin = dist;
        }
    }
    if ( dist )
        *dist = dmin;

    return index;

}
