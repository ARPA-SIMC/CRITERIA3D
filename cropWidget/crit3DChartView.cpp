#include "crit3DChartView.h"

Crit3DChartView::Crit3DChartView(QChart *chart, QWidget *parent) :
    QChartView(chart, parent)
{
    setRubberBand(QChartView::HorizontalRubberBand);
}
