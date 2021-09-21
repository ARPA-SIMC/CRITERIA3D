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

    meteoVariable getVar() const;
    void setVar(const meteoVariable &value);

    droughtIndex getIndex() const;
    void setIndex(const droughtIndex &value);

    int getTimeScale() const;
    void setTimeScale(int value);

    int getFirstYear() const;
    void setFirstYear(int value);

    int getLastYear() const;
    void setLastYear(int value);

    bool getUseClima() const;
    void setUseClima(bool value);

    bool getComputeAll() const;
    void setComputeAll(bool value);

private:
    Crit3DMeteoPoint* meteoPoint;
    meteoVariable var;
    droughtIndex index;
    int timeScale;
    int firstYear;
    int lastYear;
    bool useClima;
    bool computeAll;

};

#endif // DROUGHT_H



