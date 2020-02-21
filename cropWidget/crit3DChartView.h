#ifndef CRIT3DCHARTVIEW_H
#define CRIT3DCHARTVIEW_H

#include <QtCharts>
#include <QChartView>

class Crit3DChartView : public QChartView
{

public:
    Crit3DChartView(QChart *chart, QWidget *parent = nullptr);

protected:
//    bool viewportEvent(QEvent *event);
//    void mousePressEvent(QMouseEvent *event);
//    void mouseMoveEvent(QMouseEvent *event);
//    void mouseReleaseEvent(QMouseEvent *event);
//    void keyPressEvent(QKeyEvent *event);
};

#endif // CRIT3DCHARTVIEW_H
