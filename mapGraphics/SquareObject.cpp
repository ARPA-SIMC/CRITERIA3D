#include "SquareObject.h"

#include <QKeyEvent>

SquareObject::SquareObject(qreal side, bool sizeIsZoomInvariant, QColor fillColor, MapGraphicsObject *parent) :
    MapGraphicsObject(sizeIsZoomInvariant,parent), _fillColor(fillColor)
{
    _side = qMax<qreal>(side, 0.01);

    this->setFlag(MapGraphicsObject::ObjectIsSelectable);
    this->setFlag(MapGraphicsObject::ObjectIsMovable);
    this->setFlag(MapGraphicsObject::ObjectIsFocusable);
}

SquareObject::~SquareObject()
{
}

QRectF SquareObject::boundingRect() const
{
    return QRectF(-0.5*_side,
                  -0.5*_side,
                  _side,
                  _side);
}

void SquareObject::paint(QPainter *painter, const QStyleOptionGraphicsItem *option, QWidget *widget)
{
    Q_UNUSED(option)
    Q_UNUSED(widget)
    painter->setRenderHint(QPainter::Antialiasing,true);
    painter->setBrush(_fillColor);
    painter->drawRect(boundingRect());
}

qreal SquareObject::side() const
{
    return _side;
}

void SquareObject::setSide(qreal side)
{
    _side = side;
    emit this->redrawRequested();
}

QColor SquareObject::color() const
{
    return _fillColor;
}

void SquareObject::setFillColor(const QColor &color)
{
    if (_fillColor == color)
        return;

    _fillColor = color;
    emit this->redrawRequested();
}


//virtual from MapGraphicsObject
void SquareObject::keyReleaseEvent(QKeyEvent *event)
{
    if (event->matches(QKeySequence::Delete))
    {
        this->deleteLater();
        event->accept();
    }
    else
        event->ignore();
}


