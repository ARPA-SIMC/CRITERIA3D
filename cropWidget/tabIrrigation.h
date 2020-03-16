#ifndef TABIRRIGATION_H
#define TABIRRIGATION_H

#include <QtWidgets>
#include <QtCharts>
#include "criteria1DCase.h"

class TabIrrigation : public QWidget
{
    Q_OBJECT
public:
    TabIrrigation();
    void computeIrrigation(Crit1DCase myCase, int currentYear);
private:
    int year;
    QChartView *chartView;
    QChart *chart;
    QBarCategoryAxis *axisX;
    QValueAxis *axisY;
    QValueAxis *axisYdx;
    QStringList categories;
    QLineSeries* seriesLAI;
    QLineSeries* seriesMaxTransp;
    QLineSeries* seriesRealTransp;
    QBarSeries *seriesPrec;
    QBarSeries *seriesIrr;
    QBarSet *setPrec;
    QBarSet *setIrrigation;

};

#endif // TABIRRIGATION_H
