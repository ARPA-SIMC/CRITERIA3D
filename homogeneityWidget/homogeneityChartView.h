#ifndef HomogeneityChartView_H
#define HomogeneityChartView_H

#include <QtCharts/QChartView>
#include <QtCharts/QScatterSeries>
#include <QtCharts/QLineSeries>
#include <QtCharts/QValueAxis>
#include "meteo.h"
#include "callout.h"

#if (QT_VERSION < QT_VERSION_CHECK(6, 0, 0))
    QT_CHARTS_USE_NAMESPACE
#endif

class HomogeneityChartView : public QChartView
{
    Q_OBJECT
public:
    explicit HomogeneityChartView(QWidget *parent = 0);
    void setYmax(float value);
    void setYmin(float value);
    void drawSNHT(std::vector<int> years, std::vector<float> tvalues, QList<QPointF> t95Points);
    void drawCraddock(int myFirstYear, int myLastYear, std::vector<std::vector<float>> outputValues, std::vector<QString> refNames, meteoVariable myVar, double averageValue);
    void clearSNHTSeries();
    void tooltipSNHTSeries(QPointF point, bool state);
    void tooltipCraddockSeries(QPointF point, bool state);
    QList<QPointF> exportSNHTValues();
    QList<QList<QPointF>> exportCraddockValues(QList<QString> &refNames);

private:
    QScatterSeries* tValues;
    QLineSeries* SNHT_T95Values;
    QValueAxis* axisX;
    QValueAxis* axisY;
    Callout *m_tooltip;
    QList<QLineSeries*> craddockSeries;
};

#endif // HomogeneityChartView_H
