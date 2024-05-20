#ifndef WATERTABLEWIDGET_H
#define WATERTABLEWIDGET_H

#include <QtWidgets>
#include <QtCharts>

#include "waterTableChartView.h"

class WaterTableWidget : public QWidget
{
    Q_OBJECT
public:
    WaterTableWidget(QString id, std::vector<QDate> myDates, std::vector<float> myHindcastSeries, std::vector<float> myInterpolateSeries, QMap<QDate, int> obsDepths);
    ~WaterTableWidget();
    void closeEvent(QCloseEvent *event);

private:
    WaterTableChartView *waterTableChartView;

};

#endif // WATERTABLEWIDGET_H
