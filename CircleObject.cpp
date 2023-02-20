#include "CircleObject.h"

#include <QStaticText>
#include <QKeyEvent>
#include <QBrush>

#define NODATA -9999
#define RADIUS_WIDTH 3


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
    return QRectF(-RADIUS_WIDTH *_radius,
                  -2 * _radius,
                  RADIUS_WIDTH * 2 * _radius,
                  4 * _radius);
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
            QStaticText staticText = QStaticText(valueStr);
            staticText.setTextWidth(_radius * RADIUS_WIDTH * 2);

            QTextOption myOption;
            myOption.setAlignment(Qt::AlignCenter);
            staticText.setTextOption(myOption);

            if (_isMultiColorText)
            {
                QPen myPen;
                myPen.setColor(_fillColor);
                painter->setPen(myPen);
            }

            painter->scale(1,-1);
            painter->drawStaticText(-_radius * RADIUS_WIDTH, -_radius * 2, staticText);
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

void CircleObject::setRadius(qreal radius)
{
    if (_radius == radius) return;

    _radius = radius;
    emit this->redrawRequested();
}


void CircleObject::setMarked(bool isMarked)
{
    if (_isMarked == isMarked) return;

    _isMarked = isMarked;
    emit this->redrawRequested();
}


void CircleObject::setCurrentValue(qreal currentValue)
{
    if (_currentValue == currentValue) return;

    _currentValue = currentValue;
    emit this->redrawRequested();
}


void CircleObject::setShowText(bool isShowText)
{
    if (_isText == isShowText) return;

    _isText = isShowText;
    emit this->redrawRequested();
}


void CircleObject::setMultiColorText(bool isMultiColorText)
{
    if (_isMultiColorText == isMultiColorText) return;

    _isMultiColorText = isMultiColorText;
    emit this->redrawRequested();
}

void CircleObject::setFillColor(const QColor &color)
{
    if (_fillColor == color) return;

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

