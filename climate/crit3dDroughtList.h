#ifndef CRIT3DDROUGHTLIST_H
#define CRIT3DDROUGHTLIST_H

#ifndef METEO_H
    #include "meteo.h"
#endif

#include <QDate>

class Crit3DDroughtList
{
public:
    Crit3DDroughtList();
    void reset();
    bool addDrought(unsigned int index);
    void eraseElement(unsigned int index);

    void setIsMeteoGrid(bool isMeteoGrid);
    void insertYearStart(int yearStart);
    void insertYearEnd(int yearEnd);
    void insertIndex(droughtIndex index);
    void insertFileName(QString filename);
    void insertDate(QDate date);
    void insertTimescale(int timescale);
    void insertVariable(meteoVariable variable);
    void updateVariable(meteoVariable variable, int index);

    bool isMeteoGrid() const;
    std::vector<int> listYearStart() const;
    std::vector<int> listYearEnd() const;
    std::vector<droughtIndex> listIndex() const;
    std::vector<QDate> listDate() const;
    std::vector<int> listTimescale() const;
    std::vector<meteoVariable> listVariable() const;
    std::vector<QString> listFileName() const;
    std::vector<QString> listAll() const;

private:
    bool _isMeteoGrid;
    std::vector<QString> _listAll;
    std::vector<int> _listYearStart;
    std::vector<int> _listYearEnd;
    std::vector<droughtIndex> _listIndex;
    std::vector<QDate> _listDate;
    std::vector<int> _listTimescale;
    std::vector<QString> _listFileName;
    std::vector<meteoVariable> _listVariable;

};

#endif // CRIT3DDROUGHTLIST_H
