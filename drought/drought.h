#ifndef DROUGHT_H
#define DROUGHT_H

#ifndef METEOPOINT_H
    #include "meteoPoint.h"
#endif

enum droughtIndex {INDEX_SPI, INDEX_SPEI, INDEX_DECILES};

class Drought
{
public:
    Drought();

private:
    Crit3DMeteoPoint meteoPoint;
    droughtIndex index;
    int timeScale;
    int firstYear;
    int lastYear;

};

#endif // DROUGHT_H



