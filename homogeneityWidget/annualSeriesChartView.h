#ifndef ANNUALSERIESCHARTVIEW_H
#define ANNUALSERIESCHARTVIEW_H

#include <QtCharts/QChartView>
#include <QtCharts/QScatterSeries>
#include <QtCharts/QLineSeries>
#include <QtCharts/QValueAxis>
#include "callout.h"

#if (QT_VERSION < QT_VERSION_CHECK(6, 0, 0))
    QT_CHARTS_USE_NAMESPACE
#endif

class AnnualSeriesChartView : public QChartView
{
    Q_OBJECT
public:
    explicit AnnualSeriesChartView(QWidget *parent = 0);
    void draw(std::vector<int> years, std::vector<float> outputValues);
    void tooltipAnnualSeries(QPointF point, bool state);
    void setYmax(float value);
    void setYmin(float value);
    void setYTitle(QString title);
    void clearSeries();
    QList<QPointF> exportAnnualValues();

private:
    QScatterSeries* annualSeries;
    QValueAxis* axisX;
    QValueAxis* axisY;
    Callout *m_tooltip;
};

#endif // ANNUALSERIESCHARTVIEW_H
