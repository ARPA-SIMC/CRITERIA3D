#include "CircleObject.h"

#include <QtDebug>
#include <QKeyEvent>
#define NODATA -9999

CircleObject::CircleObject(qreal radius, bool sizeIsZoomInvariant, QColor fillColor, MapGraphicsObject *parent) :
    MapGraphicsObject(sizeIsZoomInvariant, parent), _fillColor(fillColor)
{
    _radius = qMax<qreal>(radius, 0.01);
    _currentValue = NODATA;
    _isShowValue = false;

    this->setFlag(MapGraphicsObject::ObjectIsSelectable);
    this->setFlag(MapGraphicsObject::ObjectIsMovable);
    this->setFlag(MapGraphicsObject::ObjectIsFocusable);
}

CircleObject::~CircleObject()
{
}

QRectF CircleObject::boundingRect() const
{
    return QRectF(-3*_radius,
                  -1*_radius,
                  6*_radius,
                  2*_radius);
}

void CircleObject::paint(QPainter *painter, const QStyleOptionGraphicsItem *option, QWidget *widget)
{
    Q_UNUSED(option)
    Q_UNUSED(widget)
    painter->setRenderHint(QPainter::Antialiasing,true);
    if (_isShowValue)
    {
        QString valueStr = "";
        if (_currentValue != NODATA)
        {
            valueStr = QString::number(_currentValue);
        }
        painter->scale(1,-1);
        painter->drawText(-3*_radius, _radius, valueStr);
    }
    else
    {
        painter->setBrush(_fillColor);
        painter->drawEllipse(QPointF(0,0), _radius, _radius);
    }
}

qreal CircleObject::radius() const
{
    return _radius;
}

void CircleObject::setRadius(qreal radius)
{
    _radius = radius;
    emit this->redrawRequested();
}

qreal CircleObject::currentValue() const
{
    return _currentValue;
}

void CircleObject::setCurrentValue(qreal currentValue)
{
    _currentValue = currentValue;
    emit this->redrawRequested();
}

void CircleObject::setShowValue(bool isShowValue)
{
    _isShowValue = isShowValue;
    emit this->redrawRequested();
}

QColor CircleObject::color() const
{
    return _fillColor;
}

void CircleObject::setFillColor(const QColor &color)
{
    if (_fillColor == color)
        return;

    _fillColor = color;
    emit this->redrawRequested();
}


//protected
//virtual from MapGraphicsObject
void CircleObject::keyReleaseEvent(QKeyEvent *event)
{
    if (event->matches(QKeySequence::Delete))
    {
        this->deleteLater();
        event->accept();
    }
    else
        event->ignore();
}


