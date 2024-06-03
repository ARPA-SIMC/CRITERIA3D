#ifndef WATERTABLEWIDGET_H
#define WATERTABLEWIDGET_H

    #include <QtWidgets>
    #include <QtCharts>

    #include "waterTableChartView.h"

    class WaterTableWidget : public QWidget
    {
        Q_OBJECT
    public:
        WaterTableWidget(const QString &id, std::vector<QDate> myDates, std::vector<float> myHindcastSeries,
                         std::vector<float> myInterpolateSeries, QMap<QDate, float> obsDepths, float maxObservedDepth);
        ~WaterTableWidget();
        void closeEvent(QCloseEvent *event);
        void on_actionExportInterpolationData();

    private:
        WaterTableChartView *waterTableChartView;

    };

#endif // WATERTABLEWIDGET_H
