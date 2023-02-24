#ifndef INTERPOLATIONCHARTVIEW_H
#define INTERPOLATIONCHARTVIEW_H

#include <QtCharts/QChartView>
#include <QtCharts/QScatterSeries>
#include <QtCharts/QLineSeries>
#include <QtCharts/QValueAxis>
#include <QtCharts/QDateTimeAxis>
#include "meteo.h"
#include "callout.h"

#if (QT_VERSION < QT_VERSION_CHECK(6, 0, 0))
    QT_CHARTS_USE_NAMESPACE
#endif

class InterpolationChartView : public QChartView
{
    Q_OBJECT
public:
    explicit InterpolationChartView(QWidget *parent = 0);
    void setYmax(float value);
    void setYmin(float value);
    void drawGraphInterpolation(std::vector<float> values, QDate myStartDate, QString var, int lag, int smooth, QString elabType);
    void clearInterpolationGraphSeries();
    void tooltipGraphInterpolationSeries(QPointF point, bool state);


private:
    QDateTimeAxis* axisX;
    QValueAxis* axisY;
    Callout *m_tooltip;
    QList<QLineSeries*> interpolationGraphSeries;
    float maxValue;
    float minValue;
};

#endif // INTERPOLATIONCHARTVIEW_H
