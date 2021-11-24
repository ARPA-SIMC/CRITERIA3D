#ifndef PROXYCALLOUT_H
#define PROXYCALLOUT_H


#include <QtCharts/QChartGlobal>
#include <QtWidgets/QGraphicsItem>
#include <QtGui/QFont>

QT_BEGIN_NAMESPACE
class QGraphicsSceneMouseEvent;
QT_END_NAMESPACE

#if (QT_VERSION >= QT_VERSION_CHECK(6, 0, 0))
    class QChart;
    class QAbstractSeries;
#else
    QT_CHARTS_BEGIN_NAMESPACE
    class QChart;
    class QAbstractSeries;
    QT_CHARTS_END_NAMESPACE
    QT_CHARTS_USE_NAMESPACE
#endif

class ProxyCallout : public QGraphicsItem
{
public:
    ProxyCallout(QChart *parent);

    void setText(const QString &text);
    void setAnchor(QPointF point);
    void updateGeometry();

    QRectF boundingRect() const;
    void paint(QPainter *painter, const QStyleOptionGraphicsItem *option,QWidget *widget);
    void setSeries(QAbstractSeries *series);

protected:
    void mousePressEvent(QGraphicsSceneMouseEvent *event);
    void mouseMoveEvent(QGraphicsSceneMouseEvent *event);

private:
    QString m_text;
    QRectF m_textRect;
    QRectF m_rect;
    QPointF m_anchor;
    QFont m_font;
    QChart *m_chart;
    QAbstractSeries *m_series;
};


#endif // PROXYCALLOUT_H
