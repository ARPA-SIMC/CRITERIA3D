#ifndef WATERTABLECHARTVIEW_H
#define WATERTABLECHARTVIEW_H

    #include <QtWidgets>
    #include <QtCharts>
    #include "waterTable.h"
    #include "callout.h"

    class WaterTableChartView : public QChartView
    {
        public:
            WaterTableChartView(QWidget *parent = 0);

            void drawWaterTable(const WaterTable &waterTable, float maximumObservedDepth);

            void tooltipObsDepthSeries(QPointF point, bool state);
            void tooltipLineSeries(QPointF point, bool state);
            void handleMarkerClicked();
            QList<QPointF> exportInterpolationValues();

            QDateTimeAxis* axisX;

        private:
            QScatterSeries* obsDepthSeries;
            QLineSeries* hindcastSeries;
            QLineSeries* interpolationSeries;
            QLineSeries* climateSeries;

            QValueAxis* axisY;
            Callout *m_tooltip;
    };

#endif // WATERTABLECHARTVIEW_H
