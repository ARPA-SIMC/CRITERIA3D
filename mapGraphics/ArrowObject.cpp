#include "ArrowObject.h"
#include <algorithm>


ArrowObject::ArrowObject(qreal dx, qreal dy,
                        QColor color,
                        MapGraphicsObject *parent) :
    MapGraphicsObject(true, parent),
    _dx(dx), _dy(dy), _color(color)
{ }

ArrowObject::~ArrowObject()
{
}

// pure-virtual from MapGraphicsObject
QRectF ArrowObject::boundingRect() const
{
    qreal step = 5;
    qreal left = std::min(0., _dx) -step;
    qreal width = abs(_dx) + step*2;
    qreal top = std::max(0., _dy) +step;
    qreal height = abs(_dy) + step*2;

    return QRectF(left, -top, width, height);
}

// pure-virtual from MapGraphicsObject
void ArrowObject::paint(QPainter *painter,
                       const QStyleOptionGraphicsItem *option,
                       QWidget *widget)
{
    Q_UNUSED(option)
    Q_UNUSED(widget)

    painter->setRenderHint(QPainter::Antialiasing, true);
    QPen pen = painter->pen();
    pen.setColor(_color);
    painter->setPen(pen);

    painter->drawLine(0, 0, int(_dx), int(_dy));
}

void ArrowObject::setColor(const QColor &color)
{
    if (_color == color)
        return;

    _color = color;
    emit this->redrawRequested();
}
