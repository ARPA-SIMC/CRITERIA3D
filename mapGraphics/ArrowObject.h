#ifndef ARROWOBJECT_H
#define ARROWOBJECT_H

#include "MapGraphics_global.h"
#include "Position.h"
#include "MapGraphicsObject.h"

class MAPGRAPHICSSHARED_EXPORT ArrowObject : public MapGraphicsObject
{
    Q_OBJECT
public:
    explicit ArrowObject(qreal dx, qreal dy,
                        QColor color = QColor(0,0,0,0),
                        MapGraphicsObject *parent = nullptr);
    virtual ~ArrowObject() override;

    // pure-virtual from MapGraphicsObject
    QRectF boundingRect() const override;

    // pure-virtual from MapGraphicsObject
    void paint(QPainter *painter, const QStyleOptionGraphicsItem *option, QWidget *widget) override;

    void setColor(const QColor& color);

signals:

public slots:

private slots:

private:
    qreal _dx, _dy;
    QColor _color;
};

#endif // ARROWOBJECT_H
