#ifndef TABLAI_H
#define TABLAI_H

    #include <QtWidgets>
    #include <QtCharts>
    #include "callout.h"
    #include "soil.h"

    class Crit3DCrop;
    class Crit3DMeteoPoint;

    class TabLAI : public QWidget
    {
        Q_OBJECT
    public:
        TabLAI();
        void computeLAI(Crit3DCrop* myCrop, Crit3DMeteoPoint *meteoPoint, int firstYear, int lastYear, QDate lastDBMeteoDate, const std::vector<soil::Crit3DLayer>& soilLayers);
        void tooltipLAI(QPointF point, bool state);
        void tooltipPE(QPointF point, bool state);
        void tooltipME(QPointF point, bool state);
        void tooltipMT(QPointF point, bool state);
        void handleMarkerClicked();
    private:
        QChartView *chartView;
        QChart *chart;
        QLineSeries *seriesLAI;
        QLineSeries *seriesPotentialEvap;
        QLineSeries *seriesMaxEvap;
        QLineSeries *seriesMaxTransp;
        QDateTimeAxis *axisX;
        QValueAxis *axisY;
        QValueAxis *axisYdx;
        Callout *m_tooltip;
    };

#endif // TABLAI_H
