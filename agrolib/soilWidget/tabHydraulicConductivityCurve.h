#ifndef TABHYDRAULICCONDUCTIVITYCURVE_H
#define TABHYDRAULICCONDUCTIVITYCURVE_H

    #include <QtWidgets>
    #include <QtCharts>
    #include "soil.h"
    #include "barHorizon.h"
    #include "soilCallout.h"

    class TabHydraulicConductivityCurve: public QWidget
    {
        Q_OBJECT
    public:
        TabHydraulicConductivityCurve();
        void insertElements(soil::Crit3DSoil* soil);
        void resetAll();
        bool getFillElement() const;
        void setFillElement(bool value);
        void highlightCurve( bool isHightlight );
        void tooltipLineSeries(QPointF point, bool state);

    private:
        BarHorizonList barHorizons;
        soil::Crit3DSoil* mySoil;
        QChartView *chartView;
        QChart *chart;
        QList<QLineSeries*> curveList;
        QLogValueAxis *axisX;
        QLogValueAxis *axisY;
        SoilCallout *m_tooltip;
        bool fillElement;
        int indexSelected;

        double dxMin = 0.001;
        double dxMax = 10000000;

        double xMin = (dxMin * 100);
        double xMax = (dxMax / 100);
        double yMin = pow(10, -12);
        double yMax = 1000;

    private slots:
        void widgetClicked(int index);
        void curveClicked();

    signals:
        void horizonSelected(int nHorizon);

    };

#endif // HYDRAULICCONDUCTIVITYCURVE_H
