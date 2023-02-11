#ifndef TABCARBONNITROGEN_H
#define TABCARBONNITROGEN_H

    #include <QtWidgets>
    #include "qcustomplot.h"

    class Crit1DProject;

    enum carbonNitrogenVariable {NH3, NH4, N_HUMUS, N_LITTER, C_HUMUS, C_LITTER};

    class TabCarbonNitrogen : public QWidget
    {
        Q_OBJECT
    public:
        TabCarbonNitrogen();

        void computeCarbonNitrogen(Crit1DProject &myProject, carbonNitrogenVariable currentVariable,
                                   int firstYear, int lastYear);

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
