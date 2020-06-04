/*
 * zoom functionality from: https://github.com/martonmiklos/qt_zoomable_chart_widget
*/

#ifndef CHARTVIEW_H
#define CHARTVIEW_H

#include <QtCharts/QChartView>
#include <QtWidgets/QRubberBand>
#include "rangelimitedvalueaxis.h"


QT_CHARTS_USE_NAMESPACE
#define MAXZOOM 10

//![1]
class ZoomableChartView : public QChartView
        //![1]
{
public:
    enum ZoomMode {
        Pan,
        RectangleZoom,
        VerticalZoom,
        HorizontalZoom
    };

    ZoomableChartView(QWidget *parent = nullptr);

    void zoomX(qreal factor, qreal xcenter);
    void zoomX(qreal factor);

    void zoomY(qreal factor, qreal ycenter);
    void zoomY(qreal factor);

    //![2]
    ZoomMode zoomMode() const;
    void setMaxZoomIteration(int max);
    void setZoomMode(const ZoomMode &zoomMode);
    void setRangeX(double dxMin, double dxMax);
    void setRangeY(double dyMin, double dyMax);


protected:
    void mousePressEvent(QMouseEvent *event);
    //void mouseMoveEvent(QMouseEvent *event);
    void mouseReleaseEvent(QMouseEvent *event);
    void keyPressEvent(QKeyEvent *event);
    void wheelEvent(QWheelEvent *event);
    //![2]

private:

    bool m_isTouching = false;
    QPointF m_lastMousePos;
    ZoomMode m_zoomMode = RectangleZoom;
    RangeLimitedValueAxis *rangeXAxis;
    RangeLimitedValueAxis *rangeYAxis;
    int maxZoom;
    int nZoomIterations;
    double dxMin;
    double dxMax;
    double dyMin;
    double dyMax;

    static bool isAxisTypeZoomableWithMouse(const QAbstractAxis::AxisType type);
    QPointF getSeriesCoordFromChartCoord(const QPointF & mousePos, QAbstractSeries *series) const;
    QPointF getChartCoordFromSeriesCoord(const QPointF & seriesPos, QAbstractSeries *series) const;

};

#endif
