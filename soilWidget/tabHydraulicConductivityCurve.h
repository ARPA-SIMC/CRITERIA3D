#ifndef TABHYDRAULICCONDUCTIVITYCURVE_H
#define TABHYDRAULICCONDUCTIVITYCURVE_H

    #include <QtWidgets>
    #include <qwt_plot_curve.h>
    #include <qwt_plot.h>
    #include "soil.h"
    #include "barHorizon.h"
    #include "curvePicker.h"

    class TabHydraulicConductivityCurve: public QWidget
    {
        Q_OBJECT
    public:
        TabHydraulicConductivityCurve();
        void insertElements(soil::Crit3DSoil* soil);
        void resetAll();
        bool getFillElement() const;
        void setFillElement(bool value);

    private:
        BarHorizonList barHorizons;
        soil::Crit3DSoil* mySoil;
        QwtPlot *myPlot;
        QList<QwtPlotCurve*> curveList;
        Crit3DCurvePicker *pick;
        bool fillElement;

        double dxMin = 0.001;
        double dxMax = 10000000;

        double xMin = (dxMin * 100);
        double xMax = (dxMax / 100);
        double yMin = pow(10, -12);
        double yMax = 1000;

    private slots:
        void widgetClicked(int index);
        void curveClicked(int index);

    signals:
        void horizonSelected(int nHorizon);

    };

#endif // HYDRAULICCONDUCTIVITYCURVE_H
