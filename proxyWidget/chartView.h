#ifndef CHARTVIEW_H
#define CHARTVIEW_H

#include <QtCharts/QChartView>
#include <QtCharts/QScatterSeries>

QT_CHARTS_USE_NAMESPACE

class ChartView : public QChartView
{
    Q_OBJECT
public:
    explicit ChartView(QWidget *parent = 0);
    void appendPointSeriesPrimary(QList<QPointF> pointList);
    void appendPointSeriesSecondary(QList<QPointF> pointList);
private:
    QScatterSeries *series1;
    QScatterSeries *series2;
};

#endif // CHARTVIEW_H
