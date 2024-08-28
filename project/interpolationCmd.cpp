#include <QDate>
#include <QString>

#include "basicMath.h"
#include "gis.h"
#include "utilities.h"
#include "interpolation.h"
#include "interpolationCmd.h"
#include "interpolationSettings.h"


float crossValidationStatistics::getMeanAbsoluteError() const
{
    return meanAbsoluteError;
}

void crossValidationStatistics::setMeanAbsoluteError(float newMeanAbsoluteError)
{
    meanAbsoluteError = newMeanAbsoluteError;
}

float crossValidationStatistics::getRootMeanSquareError() const
{
    return rootMeanSquareError;
}

void crossValidationStatistics::setRootMeanSquareError(float newRootMeanSquareError)
{
    rootMeanSquareError = newRootMeanSquareError;
}

float crossValidationStatistics::getCompoundRelativeError() const
{
    return compoundRelativeError;
}

void crossValidationStatistics::setCompoundRelativeError(float newCompoundRelativeError)
{
    compoundRelativeError = newCompoundRelativeError;
}

float crossValidationStatistics::getMeanBiasError() const
{
    return meanBiasError;
}

void crossValidationStatistics::setMeanBiasError(float newMeanBiasError)
{
    meanBiasError = newMeanBiasError;
}

const Crit3DTime &crossValidationStatistics::getRefTime() const
{
    return refTime;
}

void crossValidationStatistics::setRefTime(const Crit3DTime &newRefTime)
{
    refTime = newRefTime;
}

const Crit3DProxyCombination &crossValidationStatistics::getProxyCombination() const
{
    return proxyCombination;
}

void crossValidationStatistics::setProxyCombination(const Crit3DProxyCombination &newProxyCombination)
{
    proxyCombination = newProxyCombination;
}

float crossValidationStatistics::getR2() const
{
    return R2;
}

void crossValidationStatistics::setR2(float newR2)
{
    R2 = newR2;
}

crossValidationStatistics::crossValidationStatistics()
{
    initialize();
}

void crossValidationStatistics::initialize()
{
    meanAbsoluteError = NODATA;
    rootMeanSquareError = NODATA;
    compoundRelativeError = NODATA;
    meanBiasError = NODATA;
    R2 = NODATA;
}

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


bool interpolateProxyGridSeries(const Crit3DProxyGridSeries& mySeries, QDate myDate, const gis::Crit3DRasterGrid& gridBase,
                                gis::Crit3DRasterGrid *gridOut, QString &errorStr)
{
    errorStr = "";
    std::vector <QString> gridNames = mySeries.getGridName();
    std::vector <int> gridYears = mySeries.getGridYear();
    size_t nrGrids = gridNames.size();

    gis::Crit3DRasterGrid tmpGrid;
    std::string myError;

    if (nrGrids == 1)
    {
        if (! gis::readEsriGrid(gridNames[0].toStdString(), &tmpGrid, myError))
        {
            errorStr = QString::fromStdString(myError);
            return false;
        }

        gis::resampleGrid(tmpGrid, gridOut, gridBase.header, aggrAverage, 0);
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
    if (! gis::readEsriGrid(gridNames[first].toStdString(), &firstGrid, myError))
    {
        errorStr = QString::fromStdString(myError);
        return false;
    }

    if (! gis::readEsriGrid(gridNames[second].toStdString(), &secondGrid, myError))
    {
        errorStr = QString::fromStdString(myError);
        return false;
    }

    firstGrid.setMapTime(getCrit3DTime(QDate(gridYears[first],1,1), 0));
    secondGrid.setMapTime(getCrit3DTime(QDate(gridYears[second],1,1), 0));

    // use first as reference if different resolution when resampling
    if (! gis::compareGrids(firstGrid, secondGrid))
    {
        tmpGrid = secondGrid;
        secondGrid.clear();
        gis::resampleGrid(tmpGrid, &secondGrid, firstGrid.header, aggrAverage, 0);
        tmpGrid.initializeGrid();
    }

    float myMin = MINVALUE(firstGrid.minimum, secondGrid.minimum);
    float myMax = MAXVALUE(firstGrid.maximum, secondGrid.maximum);

    if (! gis::temporalYearlyInterpolation(firstGrid, secondGrid, myDate.year(), myMin, myMax, &tmpGrid))
    {
        errorStr = "Error interpolatinn proxy grid series";
        return false;
    }

    gis::resampleGrid(tmpGrid, gridOut, gridBase.header, aggrAverage, 0);

    gridOut->setMapTime(tmpGrid.getMapTime());

    firstGrid.clear();
    secondGrid.clear();
    tmpGrid.clear();

    return true;
}


bool checkProxyGridSeries(Crit3DInterpolationSettings* mySettings, const gis::Crit3DRasterGrid& gridBase,
                          std::vector <Crit3DProxyGridSeries> myProxySeries, QDate myDate, QString &errorStr)
{
    unsigned i,j;
    gis::Crit3DRasterGrid* gridOut;
    errorStr = "";

    for (i=0; i < mySettings->getProxyNr(); i++)
    {
        for (j=0; j < myProxySeries.size(); j++)
        {
            if (myProxySeries[j].getProxyName() == QString::fromStdString(mySettings->getProxyName(i)))
            {
                if (myProxySeries[j].getGridName().size() > 0)
                {
                    gridOut = new gis::Crit3DRasterGrid();
                    if (! interpolateProxyGridSeries(myProxySeries[j], myDate, gridBase, gridOut, errorStr))
                    {
                        errorStr = "Error in interpolate proxy gris series: " + errorStr;
                        return false;
                    }

                    mySettings->getProxy(i)->setGrid(gridOut);
                    return true;
                }

            }
        }
    }

    return true;
}


bool interpolationRaster(std::vector <Crit3DInterpolationDataPoint> &myPoints, Crit3DInterpolationSettings* mySettings,
                         Crit3DMeteoSettings* meteoSettings, gis::Crit3DRasterGrid* outputGrid,
                         gis::Crit3DRasterGrid& raster, meteoVariable myVar)
{
    if (! outputGrid->initializeGrid(raster))
    {
        return false;
    }

    float myX, myY;
    std::vector <double> proxyValues;
    proxyValues.resize(unsigned(mySettings->getProxyNr()));

    for (long myRow = 0; myRow < outputGrid->header->nrRows ; myRow++)
    {
        for (long myCol = 0; myCol < outputGrid->header->nrCols; myCol++)
        {
            gis::getUtmXYFromRowColSinglePrecision(*outputGrid, myRow, myCol, &myX, &myY);
            float myZ = raster.value[myRow][myCol];
            if (! isEqual(myZ, outputGrid->header->flag))
            {
                if (getUseDetrendingVar(myVar))
                {
                    getProxyValuesXY(myX, myY, mySettings, proxyValues);
                }

                outputGrid->value[myRow][myCol] = interpolate(myPoints, mySettings, meteoSettings,
                                                              myVar, myX, myY, myZ, proxyValues, true);
            }
        }
    }

    if (! gis::updateMinMaxRasterGrid(outputGrid))
        return false;

    return true;
}

bool topographicIndex(const gis::Crit3DRasterGrid& DEM, std::vector <float> windowWidths, gis::Crit3DRasterGrid& outGrid)
{

    if (! outGrid.initializeGrid(DEM))
        return false;

    if (windowWidths.size() == 0)
        return false;

    float threshold = float(EPSILON);

    float z, value, cellNr, cellDelta;
    int r1, r2, c1, c2, windowRow, windowCol;
    float higherSum, lowerSum, equalSum, weightSum;

    for (auto width : windowWidths)
    {
        cellNr = round(width / DEM.header->cellSize);

        for (int row = 0; row < outGrid.header->nrRows ; row++)
        {
            for (int col = 0; col < outGrid.header->nrCols; col++)
            {

                z = DEM.value[row][col];
                if (! isEqual(z, DEM.header->flag))
                {
                    r1 = row - cellNr;
                    r2 = row + cellNr;
                    c1 = col - cellNr;
                    c2 = col + cellNr;

                    higherSum = 0;
                    lowerSum = 0;
                    equalSum = 0;
                    weightSum = 0;

                    for (windowRow = r1; windowRow <= r2; windowRow++)
                    {
                        for (windowCol = c1; windowCol <= c2; windowCol++)
                        {
                            if (! gis::isOutOfGridRowCol(windowRow, windowCol, DEM))
                            {
                                value = DEM.value[windowRow][windowCol];

                                if (! isEqual(value, DEM.header->flag))
                                {
                                    if (windowRow != row && windowCol != col)
                                    {
                                        cellDelta = gis::computeDistance(windowRow, windowCol, row, col);

                                        if (cellDelta <= cellNr)
                                        {
                                            float weight = 1 - (cellDelta / cellNr);

                                            if (value - z > threshold)
                                                higherSum += weight;
                                            else if (value - z < -threshold)
                                                lowerSum += weight;
                                            else
                                                equalSum += weight;

                                            weightSum += weight;
                                        }
                                    }
                                }
                            }
                        }
                    }

                    if (weightSum > 0)
                    {
                        if (isEqual(outGrid.value[row][col], outGrid.header->flag))
                            outGrid.value[row][col] = (lowerSum - higherSum - equalSum * 0.5) / weightSum;
                        else
                            outGrid.value[row][col] += (lowerSum - higherSum - equalSum * 0.5) / weightSum;
                    }
                }
            }
        }
    }

    gis::updateMinMaxRasterGrid(&outGrid);

    return true;
}
