#ifndef PointStatisticsChartView_H
#define PointStatisticsChartView_H

#include <QtCharts/QChartView>
#include <QtCharts/QScatterSeries>
#include <QtCharts/QLineSeries>
#include <QtCharts/QValueAxis>
#include <QtCharts/QBarCategoryAxis>
#include "pointStatisticsCallout.h"

#if (QT_VERSION < QT_VERSION_CHECK(6, 0, 0))
    QT_CHARTS_USE_NAMESPACE
#endif

class PointStatisticsChartView : public QChartView
{
    Q_OBJECT
public:
    explicit PointStatisticsChartView(QWidget *parent = 0);
    void drawTrend(std::vector<int> years, std::vector<float> outputValues);
    void tooltipLineSeries(QPointF point, bool state);
    /*
    void cleanScatterSeries();
    void drawScatterSeries(QList<QPointF> pointListSeries1, QList<QPointF> pointListSeries2, QList<QPointF> pointListSeries3);
    void cleanClimLapseRate();
    void drawClimLapseRate(QPointF firstPoint, QPointF lastPoint);
    void cleanModelLapseRate();
    void drawModelLapseRate(QList<QPointF> pointList);
    void tooltipScatterSeries(QPointF point, bool state);
    void setIdPointMap(const QMap<QString, QPointF> &valuePrimary, const QMap<QString, QPointF> &valueSecondary, const QMap<QString, QPointF> &valueSupplemental);
    */

private:
    QScatterSeries* trend;
    //QBarCategoryAxis *axisX;
    QValueAxis* axisXvalue;
    QValueAxis* axisY;
    //QList<QString> categories;
    PointStatisticsCallout *m_tooltip;
    /*
    QScatterSeries *series1;
    QScatterSeries *series2;
    QScatterSeries *series3;
    QLineSeries* climLapseRatelineSeries;
    QLineSeries* modelLapseRatelineSeries; 
    QMap< QString, QPointF > idPointMap;
    QMap< QString, QPointF > idPointMap2;
    QMap< QString, QPointF > idPointMap3;
    */
};

#endif // PointStatisticsChartView_H
