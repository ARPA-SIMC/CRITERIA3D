#ifndef TABWATERCONTENT_H
#define TABWATERCONTENT_H


#include <QtWidgets>
#include <QtCharts>

#include "criteria1DCase.h"
#include "callout.h"

class TabWaterContent : public QWidget
{
    Q_OBJECT
public:
    TabWaterContent();
    void computeWaterContent(Crit1DCase myCase, int currentYear);

private:
    int year;
    QChartView *chartView;
    QChart *chart;
    QHorizontalBarSeries *seriesWaterContent;
    QBarSet *set;
    QDateTimeAxis *axisX;
    QBarCategoryAxis *axisY;
    QStringList categories;
    Callout *m_tooltip;

};

#endif // TABWATERCONTENT_H
