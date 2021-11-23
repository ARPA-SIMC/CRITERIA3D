#ifndef SQUAREOBJECT_H
#define SQUAREOBJECT_H

#include "MapGraphics_global.h"
#include "MapGraphicsObject.h"

class MAPGRAPHICSSHARED_EXPORT SquareObject : public MapGraphicsObject
{
    Q_OBJECT
public:
    explicit SquareObject(qreal side, bool sizeIsZoomInvariant=true,
                          QColor fillColor = QColor(0,0,0,0), MapGraphicsObject *parent = nullptr);
    virtual ~SquareObject();

    //pure-virtual from MapGraphicsObject
    QRectF boundingRect() const;

    //pure-virtual from MapGraphicsObject
    void paint(QPainter *painter, const QStyleOptionGraphicsItem *option, QWidget *widget);

    qreal side() const;
    QColor color() const;
    void setSide(qreal radius);
    void setFillColor(const QColor& color);
    
signals:
    
public slots:

protected:
    //virtual from MapGraphicsObject
    virtual void keyReleaseEvent(QKeyEvent *event);

private:
    qreal _side;
    QColor _fillColor;
    
};

#endif // SQUAREOBJECT_H
