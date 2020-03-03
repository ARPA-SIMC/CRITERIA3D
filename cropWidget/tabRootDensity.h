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
    void updateRootDensity();

private:
    Crit3DCrop* crop;
    Crit3DMeteoPoint *mp;
    std::vector<soil::Crit3DLayer> layers;
    unsigned int nrLayers;
    int year;
    QDateEdit *currentDate;
    QChartView *chartView;
    QChart *chart;
    QHorizontalBarSeries *seriesRootDensity;
    QBarSet *set;
    QValueAxis *axisX;
    QBarCategoryAxis *axisY;
    QStringList categories;

};

#endif // TABROOTDENSITY_H
