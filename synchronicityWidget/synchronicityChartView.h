#ifndef SYNCHRONICITYCHARTVIEW_H
#define SYNCHRONICITYCHARTVIEW_H

#include <QtCharts/QChartView>
#include <QtCharts/QScatterSeries>
#include <QtCharts/QLineSeries>
#include <QtCharts/QValueAxis>
#include "meteo.h"
#include "callout.h"

#if (QT_VERSION < QT_VERSION_CHECK(6, 0, 0))
    QT_CHARTS_USE_NAMESPACE
#endif

class SynchronicityChartView : public QChartView
{
    Q_OBJECT
public:
    explicit SynchronicityChartView(QWidget *parent = 0);
    void setYmax(float value);
    void setYmin(float value);
    void drawGraphStation(int firstYear, std::vector<float> outputValues);
    void clearStationGraphSeries();
    void tooltipGraphStationSeries(QPointF point, bool state);


private:
    QValueAxis* axisX;
    QValueAxis* axisY;
    Callout *m_tooltip;
    QList<QLineSeries*> stationGraphSeries;
};

#endif // SYNCHRONICITYCHARTVIEW_H
