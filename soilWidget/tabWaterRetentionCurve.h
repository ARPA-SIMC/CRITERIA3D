#ifndef TABWATERRETENTIONCURVE_H
#define TABWATERRETENTIONCURVE_H

    #include <QtWidgets>
    #include <QMap>
    #include <qwt_plot_curve.h>
    #include <qwt_plot.h>
    #include "soil.h"
    #include "barHorizon.h"
    #include "curvePicker.h"


    class TabWaterRetentionCurve: public QWidget
    {
        Q_OBJECT
    public:
        TabWaterRetentionCurve();
        void insertElements(soil::Crit3DSoil* soil);
        void resetAll();
        bool getFillElement() const;
        void setFillElement(bool value);

    private:
        BarHorizonList barHorizons;

        soil::Crit3DSoil* mySoil;
        QwtPlot *myPlot;
        QList<QwtPlotCurve*> curveList;
        QMap< int, QwtPlotCurve* > curveMarkerMap;
        Crit3DCurvePicker *pick;
        bool fillElement;

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
        void curveClicked(int index);

    signals:
        void horizonSelected(int nHorizon);

    };

#endif // WATERRETENTIONCURVE_H
