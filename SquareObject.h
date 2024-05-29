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

    qreal side() const { return _side; }
    QColor color() const { return _fillColor; }
    qreal currentValue() const { return _currentValue; }

    void showText(bool isShowText);

    void setSide(qreal side);
    void setFillColor(const QColor& color);
    void setCurrentValue(qreal currentValue);
    
signals:
    
public slots:

protected:
    //virtual from MapGraphicsObject
    virtual void keyReleaseEvent(QKeyEvent *event);

private:
    qreal _side;
    qreal _currentValue;
    QColor _fillColor;

    bool _isText;
};

#endif // SQUAREOBJECT_H
