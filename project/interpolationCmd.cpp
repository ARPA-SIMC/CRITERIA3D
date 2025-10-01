#include <QDate>
#include <QString>
#include <math.h>
#include <omp.h>

#include "basicMath.h"
#include "gis.h"
#include "utilities.h"
#include "interpolation.h"
#include "interpolationCmd.h"
#include "interpolationSettings.h"


float Crit3DCrossValidationStatistics::getMeanAbsoluteError() const
{
    return meanAbsoluteError;
}

void Crit3DCrossValidationStatistics::setMeanAbsoluteError(float newMeanAbsoluteError)
{
    meanAbsoluteError = newMeanAbsoluteError;
}

float Crit3DCrossValidationStatistics::getRootMeanSquareError() const
{
    return rootMeanSquareError;
}

void Crit3DCrossValidationStatistics::setRootMeanSquareError(float newRootMeanSquareError)
{
    rootMeanSquareError = newRootMeanSquareError;
}

float Crit3DCrossValidationStatistics::getNashSutcliffeEfficiency() const
{
    return NashSutcliffeEfficiency;
}

void Crit3DCrossValidationStatistics::setNashSutcliffeEfficiency(float newNashSutcliffeEfficiency)
{
    NashSutcliffeEfficiency = newNashSutcliffeEfficiency;
}

float Crit3DCrossValidationStatistics::getMeanBiasError() const
{
    return meanBiasError;
}

void Crit3DCrossValidationStatistics::setMeanBiasError(float newMeanBiasError)
{
    meanBiasError = newMeanBiasError;
}

const Crit3DTime &Crit3DCrossValidationStatistics::getRefTime() const
{
    return refTime;
}

void Crit3DCrossValidationStatistics::setRefTime(const Crit3DTime &newRefTime)
{
    refTime = newRefTime;
}

const Crit3DProxyCombination &Crit3DCrossValidationStatistics::getProxyCombination() const
{
    return proxyCombination;
}

void Crit3DCrossValidationStatistics::setProxyCombination(const Crit3DProxyCombination &newProxyCombination)
{
    proxyCombination = newProxyCombination;
}

float Crit3DCrossValidationStatistics::getR2() const
{
    return R2;
}

void Crit3DCrossValidationStatistics::setR2(float newR2)
{
    R2 = newR2;
}

Crit3DCrossValidationStatistics::Crit3DCrossValidationStatistics()
{
    initialize();
}

void Crit3DCrossValidationStatistics::initialize()
{
    meanAbsoluteError = NODATA;
    rootMeanSquareError = NODATA;
    NashSutcliffeEfficiency = NODATA;
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


bool checkProxyGridSeries(Crit3DInterpolationSettings &interpolationSettings, const gis::Crit3DRasterGrid& gridBase,
                          std::vector <Crit3DProxyGridSeries> myProxySeries, QDate myDate, QString &errorStr)
{
    unsigned i,j;
    gis::Crit3DRasterGrid* gridOut;
    errorStr = "";

    for (i=0; i < interpolationSettings.getProxyNr(); i++)
    {
        for (j=0; j < myProxySeries.size(); j++)
        {
            if (myProxySeries[j].getProxyName() == QString::fromStdString(interpolationSettings.getProxyName(i)))
            {
                if (myProxySeries[j].getGridName().size() > 0)
                {
                    gridOut = new gis::Crit3DRasterGrid();
                    if (! interpolateProxyGridSeries(myProxySeries[j], myDate, gridBase, gridOut, errorStr))
                    {
                        errorStr = "Error in interpolate proxy gris series: " + errorStr;
                        gridOut->clear();
                        return false;
                    }

                    interpolationSettings.getProxy(i)->setGrid(gridOut);
                    return true;
                }

            }
        }
    }

    return true;
}


bool interpolationRaster(std::vector <Crit3DInterpolationDataPoint> &dataPoints, Crit3DInterpolationSettings &interpolationSettings,
                         Crit3DMeteoSettings* meteoSettings, gis::Crit3DRasterGrid* outputGrid,
                         gis::Crit3DRasterGrid& raster, meteoVariable variable, bool isParallelComputing)
{
    if (! outputGrid->initializeGrid(raster))
    {
        return false;
    }

    unsigned int maxThreads = 1;
    if (isParallelComputing)
    {
        maxThreads = omp_get_max_threads();
    }
    omp_set_num_threads(static_cast<int>(maxThreads));

    #pragma omp parallel
    {
        std::vector<double> proxyValues(interpolationSettings.getProxyNr());
        #pragma omp for
        for (long row = 0; row < outputGrid->header->nrRows ; row++)
        {
            for (long col = 0; col < outputGrid->header->nrCols; col++)
            {
                float z = raster.value[row][col];
                if (! isEqual(z, outputGrid->header->flag))
                {
                    float x, y;
                    gis::getUtmXYFromRowColSinglePrecision(*(outputGrid->header), row, col, &x, &y);

                    if (getUseDetrendingVar(variable))
                    {
                        getProxyValuesXY(x, y, interpolationSettings, proxyValues);
                    }

                    outputGrid->value[row][col] = interpolate(dataPoints, interpolationSettings, meteoSettings,
                                                              variable, x, y, z, proxyValues, true);
                }
            }
        }
    }

    return gis::updateMinMaxRasterGrid(outputGrid);
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

