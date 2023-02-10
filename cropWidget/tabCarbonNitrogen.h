#ifndef TABCARBONNITROGEN_H
#define TABCARBONNITROGEN_H

    #include <QtWidgets>
    #include "qcustomplot.h"

    class Crit1DCase;
    class Crit1DCarbonNitrogenProfile;

    class TabCarbonNitrogen : public QWidget
    {
        Q_OBJECT
    public:
        TabCarbonNitrogen();

        void computeCarbonNitrogen(Crit1DCase myCase, Crit1DCarbonNitrogenProfile myCarbonNitrogen, int firstYear, int lastYear, QDate lastDBMeteoDate);

    private:
        QString title;
        QCustomPlot *graphic;
        QCPColorMap *colorMap;
        QCPColorScale *colorScale;
        QCPColorGradient gradient;
        int nx;
        int ny;
    };

#endif // TABCARBONNITROGEN_H
