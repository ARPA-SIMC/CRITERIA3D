#include <QDate>
#include <QString>

#include "formInfo.h"
#include "gis.h"
#include "utilities.h"
#include "interpolation.h"
#include "interpolationCmd.h"

void Crit3DProxyGridSeries::initialize()
{
    gridName.clear();
    gridYear.clear();
    proxyName = "";
}

std::vector<QString> Crit3DProxyGridSeries::getGridName() const
{
    return gridName;
}

std::vector<int> Crit3DProxyGridSeries::getGridYear() const
{
    return gridYear;
}

QString Crit3DProxyGridSeries::getProxyName() const
{
    return proxyName;
}

Crit3DProxyGridSeries::Crit3DProxyGridSeries()
{
    initialize();
}

Crit3DProxyGridSeries::Crit3DProxyGridSeries(QString name_)
{
    initialize();
    proxyName = name_;
}

void Crit3DProxyGridSeries::addGridToSeries(QString name_, int year_)
{
    gridName.push_back(name_);
    gridYear.push_back(year_);
}

bool interpolateProxyGridSeries(const Crit3DProxyGridSeries& mySeries, QDate myDate, const gis::Crit3DRasterGrid& gridBase, gis::Crit3DRasterGrid* gridOut)
{
    std::string myError;
    std::vector <QString> gridNames = mySeries.getGridName();
    std::vector <int> gridYears = mySeries.getGridYear();
    size_t nrGrids = gridNames.size();

    if (gridOut == nullptr) return false;

    gis::Crit3DRasterGrid tmpGrid;

    if (nrGrids == 1)
    {
        if (! gis::readEsriGrid(gridNames[0].toStdString(), &tmpGrid, &myError)) return false;
        gis::resampleGrid(tmpGrid, gridOut, *gridBase.header, aggrAverage, 0);
        return true;
    }

    unsigned first, second;

    for (second = 0; second < nrGrids; second++)
        if (myDate.year() <= gridYears[second]) break;

    if (second == 0)
    {
        second = 1;
        first = 0;
    }
    else if (second == nrGrids)
    {
        first = unsigned(nrGrids) - 2;
        second = unsigned(nrGrids) - 1;
    }
    else
        first = second - 1;

    // load grids
    gis::Crit3DRasterGrid firstGrid, secondGrid;
    if (! gis::readEsriGrid(gridNames[first].toStdString(), &firstGrid, &myError)) return false;
    if (! gis::readEsriGrid(gridNames[second].toStdString(), &secondGrid, &myError)) return false;

    firstGrid.setMapTime(getCrit3DTime(QDate(gridYears[first],1,1), 0));
    secondGrid.setMapTime(getCrit3DTime(QDate(gridYears[second],1,1), 0));

    // use first as reference if different resolution when resampling
    if (! gis::compareGrids(firstGrid, secondGrid))
    {
        tmpGrid = secondGrid;
        secondGrid.clear();
        gis::resampleGrid(tmpGrid, &secondGrid, *firstGrid.header, aggrAverage, 0);
        tmpGrid.initializeGrid();
    }

    float myMin = MINVALUE(firstGrid.minimum, secondGrid.minimum);
    float myMax = MAXVALUE(firstGrid.maximum, secondGrid.maximum);

    if (! gis::temporalYearlyInterpolation(firstGrid, secondGrid, myDate.year(), myMin, myMax, &tmpGrid)) return false;

    gis::resampleGrid(tmpGrid, gridOut, *gridBase.header, aggrAverage, 0);

    gridOut->setMapTime(tmpGrid.getMapTime());

    firstGrid.clear();
    secondGrid.clear();
    tmpGrid.clear();

    return true;
}

bool checkProxyGridSeries(Crit3DInterpolationSettings* mySettings, const gis::Crit3DRasterGrid& gridBase, std::vector <Crit3DProxyGridSeries> mySeries, QDate myDate)
{
    unsigned i,j;
    gis::Crit3DRasterGrid* gridOut;

    for (i=0; i < mySettings->getProxyNr(); i++)
    {
        for (j=0; j < mySeries.size(); j++)
        {
            if (mySeries[j].getProxyName() == QString::fromStdString(mySettings->getProxyName(i)))
            {
                if (mySeries[j].getGridName().size() > 0)
                {
                    gridOut = new gis::Crit3DRasterGrid();
                    interpolateProxyGridSeries(mySeries[j], myDate, gridBase, gridOut);
                    mySettings->getProxy(i)->setGrid(gridOut);
                    return true;
                }

            }
        }
    }

    return false;
}


bool interpolationRaster(std::vector <Crit3DInterpolationDataPoint> &myPoints, Crit3DInterpolationSettings* mySettings,
                        gis::Crit3DRasterGrid* myGrid, const gis::Crit3DRasterGrid& raster, meteoVariable myVar, bool showInfo)
{
    if (! myGrid->initializeGrid(raster))
        return false;

    FormInfo myInfo;
    int infoStep = 1;
    QString infoStr;

    if (showInfo)
    {
        infoStr = "Interpolating ";
        infoStr += QString::fromStdString(getVariableString(myVar));
        infoStr += " on DEM...";
        infoStep = myInfo.start(infoStr, myGrid->header->nrRows);
    }

    float myX, myY;
    std::vector <float> proxyValues;
    proxyValues.resize(unsigned(mySettings->getProxyNr()));

    for (long myRow = 0; myRow < myGrid->header->nrRows ; myRow++)
    {
        if (showInfo && (myRow % infoStep) == 0)
            myInfo.setValue(myRow);

        for (long myCol = 0; myCol < myGrid->header->nrCols; myCol++)
        {
            gis::getUtmXYFromRowColSinglePrecision(*myGrid, myRow, myCol, &myX, &myY);
            float myZ = raster.value[myRow][myCol];
            if (int(myZ) != int(myGrid->header->flag))
            {
                if (getUseDetrendingVar(myVar)) getProxyValuesXY(myX, myY, mySettings, proxyValues);
                myGrid->value[myRow][myCol] = interpolate(myPoints, mySettings, myVar, myX, myY, myZ, proxyValues, true);
            }
        }
    }

    if (showInfo) myInfo.close();

    if (! gis::updateMinMaxRasterGrid(myGrid))
        return false;

    return true;
}
