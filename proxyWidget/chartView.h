#ifndef CHARTVIEW_H
#define CHARTVIEW_H

#include <QtCharts/QChartView>
#include <QtCharts/QScatterSeries>
#include <QtCharts/QLineSeries>
#include <QtCharts/QValueAxis>
#include "callout.h"

#if (QT_VERSION < QT_VERSION_CHECK(6, 0, 0))
    QT_CHARTS_USE_NAMESPACE
#endif

class ChartView : public QChartView
{
    Q_OBJECT
public:
    explicit ChartView(QWidget *parent = 0);
    QValueAxis* axisX;
    QValueAxis* axisY;

    void cleanScatterSeries();
    void drawScatterSeries(const QList<QPointF> &pointListPrimary, const QList<QPointF> &pointListSecondary,
                           const QList<QPointF> &pointListSupplemental, const QList<QPointF> &pointListMarked);
    void cleanClimLapseRate();
    void drawClimLapseRate(QPointF firstPoint, QPointF lastPoint);
    void cleanModelLapseRate();
    void drawModelLapseRate(QList<QPointF> pointList);
    void tooltipScatterSeries(QPointF point, bool state);
    void setIdPointMap(const QMap<QString, QPointF> &valuePrimary, const QMap<QString, QPointF> &valueSecondary, const QMap<QString, QPointF> &valueSupplemental, const QMap<QString, QPointF> &valueMarked);
    void setProvince(const std::string &province);
private:
    QScatterSeries *series1;
    QScatterSeries *series2;
    QScatterSeries *series3;
    QScatterSeries *seriesMarked;
    QLineSeries* climLapseRatelineSeries;
    QLineSeries* modelLapseRatelineSeries;
    Callout *m_tooltip;
    QMap< QString, QPointF > idPointMap1;
    QMap< QString, QPointF > idPointMap2;
    QMap< QString, QPointF > idPointMap3;
    QMap< QString, QPointF > idPointMapMarked;
    QMap< QString, QPointF > _province;
};

#endif // CHARTVIEW_H
