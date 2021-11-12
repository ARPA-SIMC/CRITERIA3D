#ifndef CHARTVIEW_H
#define CHARTVIEW_H

#include <QtCharts/QChartView>
#include <QtCharts/QScatterSeries>
#include <QtCharts/QLineSeries>
#include <QtCharts/QValueAxis>

QT_CHARTS_USE_NAMESPACE

class ChartView : public QChartView
{
    Q_OBJECT
public:
    explicit ChartView(QWidget *parent = 0);
    void drawScatterSeries(QList<QPointF> pointListSeries1, QList<QPointF> pointListSeries2, QList<QPointF> pointListSeries3);
    void cleanClimLapseRate();
    void drawClimLapseRate();
private:
    QScatterSeries *series1;
    QScatterSeries *series2;
    QScatterSeries *series3;
    QLineSeries* climLapseRatelineSeries;
    QValueAxis* axisX;
    QValueAxis* axisY;
};

#endif // CHARTVIEW_H
