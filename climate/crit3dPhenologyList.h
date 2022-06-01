#ifndef CRIT3DPHENOLOGYLIST_H
#define CRIT3DPHENOLOGYLIST_H

#ifndef METEO_H
    #include "meteo.h"
#endif

#ifndef FENOLOGIA_H
    #include "fenologia.h"
#endif

#include <QDate>

class Crit3DPhenologyList
{
public:
    Crit3DPhenologyList();
    void reset();
    bool addPhenology(unsigned int index);
    void eraseElement(unsigned int index);

    void setIsMeteoGrid(bool isMeteoGrid);
    void insertFileName(QString filename);
    void insertDateStart(QDate dateStart);
    void insertDateEnd(QDate dateEnd);
    void insertComputation(phenoComputation computation);
    void insertCrop(phenoCrop crop);
    void insertVariety(phenoVariety variety);
    void insertVernalization(int vernalization);
    void insertScale(phenoScale scale);

    bool isMeteoGrid() const;
    std::vector<QString> listAll() const;
    std::vector<QString> listFileName() const;
    std::vector<QDate> listDateStart() const;
    std::vector<QDate> listDateEnd() const;
    std::vector<phenoComputation> listComputation() const;
    std::vector<phenoCrop> listCrop() const;
    std::vector<phenoVariety> listVariety() const;
    std::vector<int> listVernalization() const;
    std::vector<phenoScale> listScale() const;

private:
    bool _isMeteoGrid;
    std::vector<QString> _listAll;
    std::vector<QDate> _listDateStart;
    std::vector<QDate> _listDateEnd;
    std::vector<QString> _listFileName;
    std::vector<phenoComputation> _listComputation;
    std::vector<phenoCrop> _listCrop;
    std::vector<phenoVariety> _listVariety;
    std::vector<int> _listVernalization;
    std::vector<phenoScale> _listScale;
};

#endif // CRIT3DPHENOLOGYLIST_H
