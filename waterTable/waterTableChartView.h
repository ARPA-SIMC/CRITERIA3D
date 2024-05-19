#ifndef WATERTABLECHARTVIEW_H
#define WATERTABLECHARTVIEW_H

#include <QtWidgets>
#include <QtCharts>
#include "waterTable.h"

class WaterTableChartView : public QChartView
{
public:
    WaterTableChartView(QWidget *parent = 0);
};

#endif // WATERTABLECHARTVIEW_H
