#ifndef PointStatisticsChartView_H
#define PointStatisticsChartView_H

#include <QtCharts/QChartView>
#include <QtCharts/QScatterSeries>
#include <QtCharts/QLineSeries>
#include <QtCharts/QValueAxis>
#include <QtCharts/QBarCategoryAxis>
#include <QtCharts/QBarSeries>
#include <QtCharts/QBarSet>
#include "callout.h"

#if (QT_VERSION < QT_VERSION_CHECK(6, 0, 0))
    QT_CHARTS_USE_NAMESPACE
#endif

class PointStatisticsChartView : public QChartView
{
    Q_OBJECT
public:
    explicit PointStatisticsChartView(QWidget *parent = 0);
    void drawTrend(std::vector<int> years, std::vector<float> outputValues);
    void drawClima(QList<QPointF> dailyPointList, QList<QPointF> decadalPointList, QList<QPointF> monthlyPointList);
    void drawDistribution(std::vector<float> barValues, QList<QPointF> lineValues, int minValue, int maxValue, int classWidthValue);
    void tooltipTrendSeries(QPointF point, bool state);
    void tooltipClimaSeries(QPointF point, bool state);
    void tooltipDistributionSeries(QPointF point, bool state);
    void tooltipBar(bool state, int index, QBarSet *barset);
    void cleanTrendSeries();
    void cleanClimaSeries();
    void cleanDistribution();
    void setYmax(float value);
    void setYmin(float value);
    QList<QPointF> exportTrend();
    QList<QPointF> exportClimaDaily();
    QList<QPointF> exportClimaDecadal();
    QList<QPointF> exportClimaMonthly();
    QList<QList<float> > exportDistribution();

private:
    QScatterSeries* trend;
    QLineSeries* climaDaily;
    QLineSeries* climaDecadal;
    QLineSeries* climaMonthly;
    QBarSeries *distributionBar;
    QLineSeries *distributionLine;
    int widthValue;
    QBarCategoryAxis *axisX;
    QValueAxis* axisXvalue;
    QValueAxis* axisY;
    QList<QString> categories;
    Callout *m_tooltip;
};

#endif // PointStatisticsChartView_H
