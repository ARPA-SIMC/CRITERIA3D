#ifndef TABROOTDEPTH_H
#define TABROOTDEPTH_H

    #include <QtWidgets>
#ifndef METEOPOINT_H
    #include "meteoPoint.h"
#endif
#ifndef CROP_H
    #include "crop.h"
#endif
#include <QtCharts>
#include "crit3DChartView.h"
#include "callout.h"


    class TabRootDepth : public QWidget
    {
        Q_OBJECT
    public:
        TabRootDepth();
        void computeRootDepth(Crit3DCrop* myCrop, Crit3DMeteoPoint *meteoPoint, int currentYear, const std::vector<soil::Crit3DLayer> &soilLayers);
        void tooltip(QPointF point, bool state);
        Callout *m_tooltip;
    private:
        int year;
        Crit3DChartView *chartView;
        QChart *chart;
        QLineSeries *seriesRootDepth;
        QLineSeries *seriesRootDepthMin;
        QDateTimeAxis *axisX;
        QValueAxis *axisY;
    };

#endif // TABROOTDEPTH_H
