#ifndef TABWATERCONTENT_H
#define TABWATERCONTENT_H


#include <QtWidgets>
#include <QtCharts>

#include "criteria1DCase.h"
#include "qcustomplot.h"

class TabWaterContent : public QWidget
{
    Q_OBJECT
public:
    TabWaterContent();
    void computeWaterContent(Crit1DCase myCase, int currentYear, bool isVolumetricWaterContent);

private:
    int year;
    bool isVolumetricWaterContent;
    QString title;
    QCustomPlot *graphic;
    QCPColorMap *colorMap;
    QCPColorScale *colorScale;
    QCPColorGradient gradient;
    int nx;
    int ny;
    //QList<double> depthLayers;

};

#endif // TABWATERCONTENT_H
