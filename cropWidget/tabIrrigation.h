#ifndef TABIRRIGATION_H
#define TABIRRIGATION_H

#include <QtWidgets>
#include <QtCharts>

class TabIrrigation : public QWidget
{
    Q_OBJECT
public:
    TabIrrigation();
private:
    int year;
    QChartView *chartView;
    QChart *chart;
    QDateTimeAxis *axisX;
    QValueAxis *axisY;

};

#endif // TABIRRIGATION_H
