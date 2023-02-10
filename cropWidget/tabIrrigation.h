#ifndef TABIRRIGATION_H
#define TABIRRIGATION_H

    #include <QtWidgets>
    #include <QtCharts>
    #include "callout.h"

    class Crit1DCase;

    class TabIrrigation : public QWidget
    {
        Q_OBJECT

    public:
        TabIrrigation();
        void computeIrrigation(Crit1DCase myCase, int firstYear, int lastYear, QDate lastDBMeteoDate);
        void tooltipLAI(QPointF point, bool state);
        void tooltipMT(QPointF point, bool state);
        void tooltipRT(QPointF point, bool state);
        void tooltipPrecIrr(bool state, int index, QBarSet *barset);
        void handleMarkerClicked();

    private:
        int firstYear;
        QChartView *chartView;
        QChart *chart;
        QBarCategoryAxis *axisX;
        QDateTimeAxis *axisXvirtual;
        QValueAxis *axisY;
        QValueAxis *axisYdx;
        QList<QString> categories;
        QLineSeries* seriesLAI;
        QLineSeries* seriesMaxTransp;
        QLineSeries* seriesRealTransp;
        QBarSeries* seriesPrecIrr;
        QBarSet *setPrec;
        QBarSet *setIrrigation;
        Callout *m_tooltip;

    };

#endif // TABIRRIGATION_H
