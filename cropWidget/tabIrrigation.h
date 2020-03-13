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
    QBarCategoryAxis *axisYdx;
    QStringList categories;
    QLineSeries* seriesLAI;
    QLineSeries* seriesMaxTransp;
    QLineSeries* seriesRealTransp;
    QBarSeries *seriesPrec;
    QBarSeries *seriesIrrigation;
    QBarSet *setPrec;
    QBarSet *setIrrigation;

};

#endif // TABIRRIGATION_H
