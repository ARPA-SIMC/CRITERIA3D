#ifndef WATERTABLEWIDGET_H
#define WATERTABLEWIDGET_H

#include <QtWidgets>
#include <QtCharts>

#include "waterTable.h"
#include "waterTableChartView.h"

class WaterTableWidget : public QWidget
{
    Q_OBJECT
public:
    WaterTableWidget(WaterTable myWaterTable);
private:
    WaterTableChartView *waterTableChartView;
};

#endif // WATERTABLEWIDGET_H
