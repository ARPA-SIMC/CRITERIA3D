#include "CircleObject.h"

#include <QtDebug>
#include <QStaticText>
#include <QKeyEvent>
#include <QBrush>
#define NODATA -9999

CircleObject::CircleObject(qreal radius, bool sizeIsZoomInvariant, QColor fillColor, MapGraphicsObject *parent) :
    MapGraphicsObject(sizeIsZoomInvariant, parent), _fillColor(fillColor)
{
    _radius = qMax<qreal>(radius, 0.01);
    _currentValue = NODATA;
    _isText = false;
    _isMultiColorText = false;
    _isMarked = false;

    this->setFlag(MapGraphicsObject::ObjectIsSelectable);
    this->setFlag(MapGraphicsObject::ObjectIsMovable);
    this->setFlag(MapGraphicsObject::ObjectIsFocusable);
}

CircleObject::~CircleObject()
{
}

QRectF CircleObject::boundingRect() const
{
    return QRectF(-4*_radius,
                  -2*_radius,
                  8*_radius,
                  4*_radius);
}

void CircleObject::paint(QPainter *painter, const QStyleOptionGraphicsItem *option, QWidget *widget)
{
    Q_UNUSED(option)
    Q_UNUSED(widget)
    painter->setRenderHint(QPainter::Antialiasing, true);

    if (_isText || _isMultiColorText)
    {
        if (_currentValue != NODATA)
        {
            QString valueStr = QString::number(_currentValue, 'f', 1);
            QStaticText myText = QStaticText(valueStr);
            myText.setTextWidth(_radius * 8);
            painter->scale(1,-1);
            if (_isMultiColorText)
            {
                QPen myPen;
                myPen.setColor(_fillColor);
                painter->setPen(myPen);
            }
            painter->drawStaticText(-int(myText.textWidth() / 2), -int(_radius*2), myText);
        }
    }
    else
    {
        painter->setBrush(_fillColor);
        painter->drawEllipse(QPointF(0,0), _radius, _radius);
    }

    if (_isMarked)
    {
        painter->setPen(QPen(QBrush(QColor(Qt::black)),2));
        painter->setBrush(Qt::transparent);
        painter->drawEllipse(QPointF(0,0), _radius*2, _radius*2);
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


void CircleObject::setMarked(bool isMarked)
{
    if (_isMarked == isMarked)
        return;

    _isMarked = isMarked;
    emit this->redrawRequested();
}


qreal CircleObject::currentValue() const
{
    return _currentValue;
}

void CircleObject::setCurrentValue(qreal currentValue)
{
    if (_currentValue == currentValue)
        return;

    _currentValue = currentValue;
    emit this->redrawRequested();
}

void CircleObject::setShowText(bool isShowText)
{
    if (_isText == isShowText)
        return;

    _isText = isShowText;
    emit this->redrawRequested();
}


void CircleObject::setMultiColorText(bool isMultiColorText)
{
    if (_isMultiColorText == isMultiColorText)
        return;

    _isMultiColorText = isMultiColorText;
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


