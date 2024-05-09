#include "well.h"

Well::Well()
{

}

QString Well::getId() const
{
    return id;
}

void Well::setId(const QString &newId)
{
    id = newId;
}

double Well::getUtmX() const
{
    return utmX;
}

void Well::setUtmX(double newUtmX)
{
    utmX = newUtmX;
}

double Well::getUtmY() const
{
    return utmY;
}

void Well::setUtmY(double newUtmY)
{
    utmY = newUtmY;
}
