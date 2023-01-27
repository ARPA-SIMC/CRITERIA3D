#ifndef CIRCLEOBJECT_H
#define CIRCLEOBJECT_H

#include "MapGraphics_global.h"
#include "MapGraphicsObject.h"

class MAPGRAPHICSSHARED_EXPORT CircleObject : public MapGraphicsObject
{
    Q_OBJECT
public:
    explicit CircleObject(qreal radius,
                          bool sizeIsZoomInvariant=true,
                          QColor fillColor = QColor(0,0,0,0),
                          MapGraphicsObject *parent = nullptr);
    virtual ~CircleObject() override;

    //pure-virtual from MapGraphicsObject
    QRectF boundingRect() const override;

    //pure-virtual from MapGraphicsObject
    void paint(QPainter *painter, const QStyleOptionGraphicsItem *option, QWidget *widget) override;

    qreal radius() const;
    qreal currentValue() const;
    QColor color() const;
    void setRadius(qreal radius);
    void setFillColor(const QColor& color);
    void setCurrentValue(qreal currentValue);
    void setShowText(bool isShowText);
    void setMultiColorText(bool isMultiColorText);
    void setMarked(bool isMarked);

    bool isMarked(){ return _isMarked; }
    
signals:
    
public slots:

protected:
    //virtual from MapGraphicsObject
    virtual void keyReleaseEvent(QKeyEvent *event) override;

private:
    qreal _radius;
    qreal _currentValue;
    QColor _fillColor;
    bool _isText;
    bool _isMultiColorText;
    bool _isMarked;
    
};

#endif // CIRCLEOBJECT_H
