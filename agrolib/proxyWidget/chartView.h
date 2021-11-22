#ifndef CHARTVIEW_H
#define CHARTVIEW_H

#include <QtCharts/QChartView>
#include <QtCharts/QScatterSeries>
#include <QtCharts/QLineSeries>
#include <QtCharts/QValueAxis>
#include "proxyCallout.h"

QT_CHARTS_USE_NAMESPACE

class ChartView : public QChartView
{
    Q_OBJECT
public:
    explicit ChartView(QWidget *parent = 0);
    void cleanScatterSeries();
    void drawScatterSeries(QList<QPointF> pointListSeries1, QList<QPointF> pointListSeries2, QList<QPointF> pointListSeries3);
    void cleanClimLapseRate();
    void drawClimLapseRate(QPointF firstPoint, QPointF lastPoint);
    void cleanModelLapseRate();
    void drawModelLapseRate(QList<QPointF> pointList);
    void tooltipScatterSeries(QPointF point, bool state);
    void setIdPointMap(const QMap<QString, QPointF> &valuePrimary, const QMap<QString, QPointF> &valueSecondary, const QMap<QString, QPointF> &valueSupplemental);

private:
    QScatterSeries *series1;
    QScatterSeries *series2;
    QScatterSeries *series3;
    QLineSeries* climLapseRatelineSeries;
    QLineSeries* modelLapseRatelineSeries;
    QValueAxis* axisX;
    QValueAxis* axisY;
    ProxyCallout *m_tooltip;
    QMap< QString, QPointF > idPointMap;
    QMap< QString, QPointF > idPointMap2;
    QMap< QString, QPointF > idPointMap3;
};

#endif // CHARTVIEW_H
