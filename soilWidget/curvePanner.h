#ifndef CURVEPANNER_H
#define CURVEPANNER_H

    #include <QWidget>
    #include "qwt_plot_panner.h"

    enum scaleType {linear, xlog, ylog, xlogylog};

    class Crit3DCurvePanner : public QwtPlotPanner
    {
        Q_OBJECT
    public:
        Crit3DCurvePanner(QwtPlot *plot, scaleType type, double dxMin, double dxMax, double dyMin, double dyMax);
        void moveCanvas(int dx, int dy);
        void moveCanvasXlog(int dx, int dy);
        void moveCanvasXlogYlog(int dx, int dy);

    private:
        QwtPlot *qwtPlot;
        scaleType type;
        double dxMin;
        double dxMax;
        double dyMin;
        double dyMax;

    };

#endif // CURVEPANNER_H
