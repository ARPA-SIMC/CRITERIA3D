#ifndef TABROOTDENSITY_H
#define TABROOTDENSITY_H


#include <QtWidgets>
#include <QtCharts>

#include "callout.h"

#ifndef METEOPOINT_H
    #include "meteoPoint.h"
#endif
#ifndef CROP_H
    #include "crop.h"
#endif

class TabRootDensity : public QWidget
{
    Q_OBJECT
public:
    TabRootDensity();
    void tooltip(QPointF point, bool state);
    Callout *m_tooltip;
private:
    int year;
    QChartView *chartView;
    QChart *chart;
    QLineSeries *seriesRootDensity;
    QDateTimeAxis *axisX;
    QValueAxis *axisY;
};

#endif // TABROOTDENSITY_H
