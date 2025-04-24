#ifndef WATERTABLEWIDGET_H
#define WATERTABLEWIDGET_H

    #include <QtWidgets>
    #include <QtCharts>

    #include "waterTableChartView.h"

    class WaterTableWidget : public QWidget
    {
        Q_OBJECT
    public:
        WaterTableWidget(const QString &id, const WaterTable &waterTable, float maxObservedDepth);

        ~WaterTableWidget() { ; }

        void closeEvent(QCloseEvent *event)
        { event->accept(); }

    private:
        WaterTableChartView *waterTableChartView;

        void on_actionExportInterpolationData();
        void on_actionChangeXAxis();

    };

#endif // WATERTABLEWIDGET_H
