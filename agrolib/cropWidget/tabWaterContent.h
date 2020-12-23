#ifndef TABWATERCONTENT_H
#define TABWATERCONTENT_H

#include <QtWidgets>
#include "criteria1DCase.h"
#include "qcustomplot.h"

class TabWaterContent : public QWidget
{
    Q_OBJECT
public:
    TabWaterContent();
    void computeWaterContent(Crit1DCase myCase, int firstYear, int lastYear, QDate lastDBMeteoDate, bool isVolumetricWaterContent);

private:
    bool isVolumetricWaterContent;
    QString title;
    QCustomPlot *graphic;
    QCPColorMap *colorMap;
    QCPColorScale *colorScale;
    QCPColorGradient gradient;
    int nx;
    int ny;

};

#endif // TABWATERCONTENT_H
