#include "rubberBand.h"

RubberBand::RubberBand(QRubberBand::Shape s, QWidget *p) :
    QRubberBand(s,p)
{
    isActive = false;
}

void RubberBand::setOrigin(QPoint origin)
{
    this->origin = origin;
}

QPoint RubberBand::getOrigin()
{
    return this->origin;
}

void RubberBand::paintEvent(QPaintEvent *event)
{
    QRubberBand::paintEvent(event);

    QPainter p(this);
    p.setPen(QPen(Qt::black, 2));
}


void RubberBand::resizeEvent(QResizeEvent *)
{
  this->resize(size());
}
