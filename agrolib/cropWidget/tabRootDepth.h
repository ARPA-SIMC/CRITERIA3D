#ifndef TABROOTDEPTH_H
#define TABROOTDEPTH_H

    #include <QtWidgets>
    #include <QtCharts>

    #include "cropCallout.h"

    #ifndef METEOPOINT_H
        #include "meteoPoint.h"
    #endif
    #ifndef CROP_H
        #include "crop.h"
    #endif

    class TabRootDepth : public QWidget
    {
        Q_OBJECT
    public:
        TabRootDepth();
        void computeRootDepth(Crit3DCrop* myCrop, Crit3DMeteoPoint *meteoPoint, int firstYear, int lastYear, QDate lastDBMeteoDate, const std::vector<soil::Crit3DLayer> &soilLayers);
        void tooltipRDM(QPointF point, bool state);
        void tooltipRD(QPointF point, bool state);
        CropCallout *m_tooltip;
    private:
        QChartView *chartView;
        QChart *chart;
        QLineSeries *seriesRootDepth;
        QLineSeries *seriesRootDepthMin;
        QDateTimeAxis *axisX;
        QValueAxis *axisY;
    };

#endif // TABROOTDEPTH_H
