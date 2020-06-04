#ifndef TABWATERRETENTIONCURVE_H
#define TABWATERRETENTIONCURVE_H

    #include <QtWidgets>
    #include <QtCharts>
    #include <QMap>
    #include "soil.h"
    #include "barHorizon.h"
    #include "callout.h"


    class TabWaterRetentionCurve: public QWidget
    {
        Q_OBJECT
    public:
        TabWaterRetentionCurve();
        void insertElements(soil::Crit3DSoil* soil);
        void resetAll();
        bool getFillElement() const;
        void setFillElement(bool value);
        void highlightCurve(bool isHightlight);
        void tooltipLineSeries(QPointF point, bool state);
        void tooltipScatterSeries(QPointF point, bool state);

    private:
        BarHorizonList barHorizons;

        soil::Crit3DSoil* mySoil;
        QChartView *chartView;
        QChart *chart;
        QList<QLineSeries*> curveList;
        QValueAxis *axisY;
        QLogValueAxis *axisX;
        QMap< int, QScatterSeries* > curveMarkerMap;
        Callout *m_tooltip;
        bool fillElement;
        int indexSelected;

        double dxMin = 0.001;
        double dxMax = 10000000;
        double dyMin = 0.0;
        double dyMax = 1.0;

        double xMin = (dxMin * 100);
        double xMax = (dxMax / 100);
        double yMin = 0.0;
        double yMax = 0.6;

    private slots:
        void widgetClicked(int index);
        void curveClicked();
        void markerClicked();

    signals:
        void horizonSelected(int nHorizon);

    };

#endif // WATERRETENTIONCURVE_H
