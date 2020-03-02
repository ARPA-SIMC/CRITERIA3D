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
    void computeRootDensity(Crit3DCrop* myCrop, Crit3DMeteoPoint *meteoPoint, int currentYear, const std::vector<soil::Crit3DLayer> &soilLayers);
    void tooltip(QPointF point, bool state);
    Callout *m_tooltip;
private:
    int year;
    QChartView *chartView;
    QChart *chart;
    QHorizontalPercentBarSeries *seriesRootDensity;
    QBarSet *set;
    QValueAxis *axisX;
    QBarCategoryAxis *axisY;
    QStringList categories;

};

#endif // TABROOTDENSITY_H
