/*!
    \file watershed.cpp

    \abstract functions to extract and remove spikes of river basins

    \copyright
    This file is part of CRITERIA3D.
    CRITERIA3D has been developed by ARPAE Emilia-Romagna.

    CRITERIA3D is free software: you can redistribute it and/or modify
    it under the terms of the GNU Lesser General Public License as published by
    the Free Software Foundation, either version 3 of the License, or
    (at your option) any later version.

    CRITERIA3D is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
    GNU Lesser General Public License for more details.

    You should have received a copy of the GNU Lesser General Public License
    along with CRITERIA3D.  If not, see <http://www.gnu.org/licenses/>.

    \authors
    Fausto Tomei ftomei@arpae.it
    Gabriele Antolini gantolini@arpae.it
*/

#include <math.h>
#include <algorithm>
#include <queue>
#include <vector>

#include "commonConstants.h"
#include "basicMath.h"
#include "statistics.h"
#include "watershed.h"


namespace gis
{

    /*!
     * \brief extractBasin_singleStep
     * extract a basin from a digital terrain model, starting from the closure point (xClosure, yClosure)
     */
    bool extractBasin_singleStep(Crit3DRasterGrid& inputRaster, Crit3DRasterGrid& outputRaster, double xClosure, double yClosure)
    {
        // check closure point
        float refValue = inputRaster.getValueFromXY(xClosure, yClosure);
        if (isEqual(refValue, inputRaster.header->flag))
            return false;

        // initialize new raster (basin)
        Crit3DRasterGrid basinRaster;
        basinRaster.initializeGrid(*inputRaster.header);

        // set first value
        int rowClosure, colClosure;
        inputRaster.getRowCol(xClosure, yClosure, rowClosure, colClosure);
        basinRaster.value[rowClosure][colClosure] = refValue;

        // initialize queue
        std::vector<int> rowList, colList, newRowList, newColList;
        rowList.push_back(rowClosure);
        colList.push_back(colClosure);

        // *** step 1: adds points with higher topographic elevation

        float rasterValue, basinValue;
        int side = 5;
        while (! rowList.empty())
        {
            for (int i=0; i < (int)rowList.size(); i++)
            {
                int row = rowList[i];
                int col = colList[i];
                refValue = basinRaster.value[row][col];
                if (! isEqual(refValue, basinRaster.header->flag))
                {
                    for (int r = -side; r <= side; r++)
                    {
                        for (int c = -side; c <= side; c++)
                        {
                            if (r != 0 || c != 0)
                            {
                                rasterValue = inputRaster.getValueFromRowCol(row+r, col+c);
                                if (! isEqual(rasterValue, inputRaster.header->flag) && (rasterValue >= refValue))
                                {
                                    basinValue = basinRaster.getValueFromRowCol(row+r, col+c);
                                    if (isEqual(basinValue, basinRaster.header->flag))
                                    {
                                        newRowList.push_back(row+r);
                                        newColList.push_back(col+c);
                                        basinRaster.value[row+r][col+c] = rasterValue;
                                    }
                                }
                            }
                        }
                    }
                }
            }

            rowList = newRowList;
            colList = newColList;
            newRowList.clear();
            newColList.clear();
        }

        rowList.clear();
        colList.clear();

        // *** step 2: adds terrain depressions

        for (int row = 0; row < basinRaster.header->nrRows; row++)
        {
            // left and right edge
            if (isEqual(basinRaster.value[row][0], basinRaster.header->flag))
            {
                rowList.push_back(row);
                colList.push_back(0);
            }
            if (isEqual(basinRaster.value[row][basinRaster.header->nrCols-1], basinRaster.header->flag))
            {
                rowList.push_back(row);
                colList.push_back(basinRaster.header->nrCols-1);
            }
        }

        for (int col = 0; col < basinRaster.header->nrCols; col++)
        {
            // top and bottom edge
            if (isEqual(basinRaster.value[0][col], basinRaster.header->flag))
            {
                rowList.push_back(0);
                colList.push_back(col);
            }
            if (isEqual(basinRaster.value[basinRaster.header->nrRows-1][col], basinRaster.header->flag))
            {
                rowList.push_back(basinRaster.header->nrRows-1);
                colList.push_back(col);
            }
        }

        // initialize new raster (boundaries)
        Crit3DRasterGrid boundariesRaster;
        boundariesRaster.initializeGrid(*inputRaster.header);

        for (int i=0; i < (int)rowList.size(); i++)
        {
            boundariesRaster.value[rowList[i]][colList[i]] = 1;
        }

        // adds empty points
        float boundaryValue;
        while (! rowList.empty())
        {
            for (int i=0; i < (int)rowList.size(); i++)
            {
                int row = rowList[i];
                int col = colList[i];
                for (int r = -1; r <= 1; r++)
                {
                    for (int c = -1; c <= 1; c++)
                    {
                        if (r != 0 || c != 0)
                        {
                            if (! basinRaster.isOutOfGrid(row+r, col+c))
                            {
                                basinValue = basinRaster.value[row+r][col+c];
                                if (isEqual(basinValue, basinRaster.header->flag))
                                {
                                    boundaryValue = boundariesRaster.value[row+r][col+c];
                                    if (isEqual(boundaryValue, boundariesRaster.header->flag))
                                    {
                                        newRowList.push_back(row+r);
                                        newColList.push_back(col+c);
                                        boundariesRaster.value[row+r][col+c] = 1;
                                    }
                                }
                            }
                        }
                    }
                }
            }
            rowList = newRowList;
            colList = newColList;
            newRowList.clear();
            newColList.clear();
        }

        // adds terrain depressions
        for (int row = 0; row < basinRaster.header->nrRows; row++)
        {
            for (int col = 0; col < basinRaster.header->nrCols; col++)
            {
                basinValue = basinRaster.value[row][col];
                boundaryValue = boundariesRaster.value[row][col];
                if (isEqual(basinValue, basinRaster.header->flag) && isEqual(boundaryValue, boundariesRaster.header->flag))
                {
                    rasterValue = inputRaster.value[row][col];
                    basinRaster.value[row][col] = rasterValue;
                }
            }
        }

        // *** step 3: clean the basin

        // a) removes points relating to other basins
        cleanBasin_simple(inputRaster, basinRaster, xClosure, yClosure);

        // b) delete empty frame
        std::string errorStr;
        if (! resizeRasterCutEmptyFrame(&basinRaster, &outputRaster, errorStr))
            return false;

        return true;
    }


    // returns downstream receiver cell using D8 steepest descent
    static D8Cell computeFlowDirectionD8(const Crit3DRasterGrid& dem, int row, int col)
    {
        float center = dem.getValueFromRowCol(row, col);

        if (isEqual(center, dem.header->flag))
            return {};

        static const int dr[8] = {-1,-1,-1, 0,0, 1,1,1};
        static const int dc[8] = {-1, 0, 1,-1,1,-1,0,1};

        float bestSlope = 0.f;
        D8Cell bestCell;

        for (int i = 0; i < 8; ++i)
        {
            int nr = row + dr[i];
            int nc = col + dc[i];

            if (nr < 0 || nc < 0 ||
                nr >= dem.header->nrRows ||
                nc >= dem.header->nrCols)
            {
                continue;
            }

            float neigh = dem.getValueFromRowCol(nr, nc);

            if (isEqual(neigh, dem.header->flag))
                continue;

            double dist =
                (dr[i] != 0 && dc[i] != 0)
                    ? dem.header->cellSize * std::sqrt(2.0)
                    : dem.header->cellSize;

            float slope = (center - neigh) / float(dist);

            if (slope > bestSlope)
            {
                bestSlope = slope;
                bestCell.row = nr;
                bestCell.col = nc;
            }
        }

        return bestCell;
    }


    /*!
    * \brief cleanBasin
    * Keeps only cells draining to closure point
    */
    bool cleanBasin(const Crit3DRasterGrid& dem, Crit3DRasterGrid& outputRaster,
                    double xClosure, double yClosure)
    {
        const int nrRows = dem.header->nrRows;
        const int nrCols = dem.header->nrCols;

        // ---------------------------------------------------------
        // 1. check closure cell
        // ---------------------------------------------------------
        int closureRow, closureCol;

        dem.getRowCol(xClosure, yClosure,
                      closureRow, closureCol);

        if (closureRow < 0 || closureCol < 0 ||
            closureRow >= nrRows || closureCol >= nrCols)
        {
            return false;
        }

        // initialize basin
        gis::Crit3DRasterGrid basinRaster;
        basinRaster.copyGrid(dem);

        // ---------------------------------------------------------
        // 2. compute downstream receiver for each cell
        // ---------------------------------------------------------

        std::vector<std::vector<D8Cell>> flow(
            nrRows,
            std::vector<D8Cell>(nrCols));

        for (int r = 0; r < nrRows; ++r)
        {
            for (int c = 0; c < nrCols; ++c)
            {
                if (! isEqual(basinRaster.value[r][c],
                             basinRaster.header->flag))
                {
                    flow[r][c] =
                        computeFlowDirectionD8(dem, r, c);
                }
            }
        }

        // ---------------------------------------------------------
        // 3. reverse adjacency list
        //    (who drains into me?)
        // ---------------------------------------------------------

        std::vector<std::vector<std::vector<Cell>>> upstream(
            nrRows,
            std::vector<std::vector<Cell>>(nrCols));

        for (int r = 0; r < nrRows; ++r)
        {
            for (int c = 0; c < nrCols; ++c)
            {
                D8Cell dst = flow[r][c];

                if (dst.isValid())
                {
                    upstream[dst.row][dst.col]
                        .push_back({r, c});
                }
            }
        }

        // ---------------------------------------------------------
        // 4. upstream flood-fill from closure
        // ---------------------------------------------------------

        std::vector<std::vector<bool>> keep(
            nrRows,
            std::vector<bool>(nrCols, false));

        std::queue<Cell> q;

        q.push({closureRow, closureCol});
        keep[closureRow][closureCol] = true;

        while (!q.empty())
        {
            Cell current = q.front();
            q.pop();

            for (const Cell& up :
                 upstream[current.row][current.col])
            {
                if (!keep[up.row][up.col])
                {
                    keep[up.row][up.col] = true;
                    q.push(up);
                }
            }
        }

        // ---------------------------------------------------------
        // 5. remove cells outside watershed
        // ---------------------------------------------------------

        for (int r = 0; r < nrRows; ++r)
        {
            for (int c = 0; c < nrCols; ++c)
            {
                if (! keep[r][c])
                {
                    basinRaster.value[r][c] =
                        basinRaster.header->flag;
                }
            }
        }

        // delete empty frame
        std::string errorStr;
        if (! resizeRasterCutEmptyFrame(&basinRaster, &outputRaster, errorStr))
        {
            return false;
        }

        return true;
    }


    // removes points relating to other basins
    void cleanBasin_simple(const Crit3DRasterGrid& dem, Crit3DRasterGrid& outputRaster,
                           double xClosure, double yClosure)
    {
        double threshold = outputRaster.header->cellSize * 3.0;

        for (int row = 0; row < outputRaster.header->nrRows; row++)
        {
            for (int col = 0; col < outputRaster.header->nrCols; col++)
            {
                if (! isEqual(outputRaster.value[row][col], outputRaster.header->flag))
                {
                    float refValue = outputRaster.value[row][col];

                    int lastRow = row;
                    int lastCol = col;
                    bool isNewPoint = true;
                    double x, y;

                    // descends following the maximum slope
                    while (isNewPoint)
                    {
                        isNewPoint = false;
                        int currentRow = lastRow;
                        int currentCol = lastCol;
                        dem.getXY(currentRow, currentCol, x, y);

                        if (computeDistance(x, y, xClosure, yClosure) > threshold)
                        {
                            for (int r = -1; r <= 1; r++)
                            {
                                for (int c = -1; c <= 1; c++)
                                {
                                    if (r != 0 || c != 0)
                                    {
                                        float rasterValue = dem.getValueFromRowCol(currentRow+r, currentCol+c);
                                        if (! isEqual(rasterValue, dem.header->flag) && (rasterValue < refValue))
                                        {
                                            refValue = rasterValue;
                                            lastRow = currentRow+r;
                                            lastCol = currentCol+c;
                                            isNewPoint = true;
                                        }
                                    }
                                }
                            }
                        }
                    }

                    // remove the origin point if the last point of path is outside the basin
                    if (isEqual(outputRaster.value[lastRow][lastCol], outputRaster.header->flag))
                    {
                        outputRaster.value[row][col] = outputRaster.header->flag;
                    }
                }
            }
        }
    }


    /*!
     * \brief extractBasin
     * extract a basin from a digital terrain model, starting from the closure point (xClosure, yClosure)
     */
    bool extractBasin(const Crit3DRasterGrid& dem, Crit3DRasterGrid& outputRaster, double xClosure, double yClosure)
    {
        // initialize new raster (basin)
        Crit3DRasterGrid basinRaster;
        basinRaster.copyGrid(dem);

        for (int i = 0; i < 3; i++)
        {
            if (! extractBasin_singleStep(basinRaster, outputRaster, xClosure, yClosure))
                return false;

            basinRaster.copyGrid(outputRaster);
        }

        return true;
    }


    bool computeWaterRunoffPath(const Crit3DRasterGrid& inputRaster, Crit3DRasterGrid& outputRaster, double xStart, double yStart)
    {
        outputRaster.initializeGrid(inputRaster);

        // set first value
        int row, col;
        inputRaster.getRowCol(xStart, yStart, row, col);
        if (! inputRaster.isOutOfGrid(row, col))
            outputRaster.value[row][col] = 0.;

        // todo

        return true;
    }


}  // end gis namespace
