#ifndef CHARTVIEW_H
#define CHARTVIEW_H

#include <QtCharts/QChartView>
#include <QtCharts/QScatterSeries>
#include <QtCharts/QLineSeries>

QT_CHARTS_USE_NAMESPACE

class ChartView : public QChartView
{
    Q_OBJECT
public:
    explicit ChartView(QWidget *parent = 0);
    void drawPointSeriesPrimary(QList<QPointF> pointList);
    void drawPointSeriesSecondary(QList<QPointF> pointList);
    void drawPointSeriesSupplemental(QList<QPointF> pointList);
    void cleanClimLapseRate();
    void drawClimLapseRate();
private:
    QScatterSeries *series1;
    QScatterSeries *series2;
    QScatterSeries *series3;
    QLineSeries* climLapseRatelineSeries;
};

#endif // CHARTVIEW_H
