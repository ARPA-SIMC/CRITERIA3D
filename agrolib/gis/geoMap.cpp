/*!
    \copyright 2016 Fausto Tomei, Gabriele Antolini,
    Alberto Pistocchi, Marco Bittelli, Antonio Volta, Laura Costantini

    This file is part of CRITERIA3D.
    CRITERIA3D has been developed under contract issued by ARPAE Emilia-Romagna

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

    contacts:
    ftomei@arpae.it
    gantolini@arpae.it
*/


#include "commonConstants.h"
#include "geoMap.h"
#include "math.h"

namespace gis
{

    Crit3DRasterWindow::Crit3DRasterWindow()
    {}

    Crit3DRasterWindow::Crit3DRasterWindow(int row0, int col0, int row1, int col1)
    {
        if (row0 > row1)
        {
            int tmp = row0;
            row0 = row1;
            row1 = tmp;
        }

        if (col0 > col1)
        {
            int tmp = col0;
            col0 = col1;
            col1 = tmp;
        }

        this->v[0].row = row0;
        this->v[0].col = col0;
        this->v[1].row = row1;
        this->v[1].col = col1;
    }

    int Crit3DRasterWindow::nrRows() const
    {
        return (this->v[1].row - this->v[0].row + 1);
    }

    int Crit3DRasterWindow::nrCols() const
    {
        return (this->v[1].col - this->v[0].col + 1);
    }


    Crit3DUtmWindow::Crit3DUtmWindow() {}

    Crit3DUtmWindow::Crit3DUtmWindow(const Crit3DUtmPoint& v0, const Crit3DUtmPoint& v1)
    {
        this->v0 = v0;
        this->v1 = v1;
    }

    double Crit3DUtmWindow::width()
    {
        return fabs(this->v1.x - this->v0.x);
    }

    double Crit3DUtmWindow::height()
    {
        return fabs(this->v1.y - this->v0.y);
    }

    Crit3DPixelWindow::Crit3DPixelWindow() {}

    Crit3DPixelWindow::Crit3DPixelWindow(const Crit3DPixel& v0, const Crit3DPixel& v1)
    {
        this->v0 = v0;
        this->v1 = v1;
    }

    int Crit3DPixelWindow::width()
    {
        return abs(this->v1.x - this->v0.x);
    }

    int Crit3DPixelWindow::height()
    {
        return abs(this->v1.y - this->v0.y);
    }

    Crit3DGeoMap::Crit3DGeoMap()
    {
        this->isDrawing = false;
        this->isChanged = false;
        this->isSelecting = false;
    }


    bool updateColorScale(Crit3DRasterGrid* myGrid, const Crit3DRasterWindow& myWindow)
    {
        return updateColorScale(myGrid, myWindow.v[0].row, myWindow.v[0].col, myWindow.v[1].row, myWindow.v[1].col);
    }

    bool getUtmWindow(const Crit3DGridHeader& latLonHeader, const Crit3DRasterHeader& utmHeader,
                      const Crit3DRasterWindow& latLonWindow, Crit3DRasterWindow* utmWindow, int utmZone)
    {
        Crit3DGeoPoint p[2];
        Crit3DUtmPoint utmPoint[2];

        getLatLonFromRowCol(latLonHeader, latLonWindow.v[0], &(p[0]));
        getUtmFromLatLon(utmZone, p[0], &(utmPoint[0]));
        getRowColFromXY(utmHeader, utmPoint[0], &(utmWindow->v[0]));

        getLatLonFromRowCol(latLonHeader, latLonWindow.v[1], &(p[1]));
        getUtmFromLatLon(utmZone, p[1], &(utmPoint[1]));
        getRowColFromXY(utmHeader, utmPoint[1], &(utmWindow->v[1]));

        return true;
    }

}
