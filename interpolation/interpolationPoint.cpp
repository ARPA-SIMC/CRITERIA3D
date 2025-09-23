/*!
    \copyright 2016 Fausto Tomei, Gabriele Antolini,
    Alberto Pistocchi, Marco Bittelli, Antonio Volta, Laura Costantini

    This file is part of CRITERIA3D.
    CRITERIA3D has been developed under contract issued by A.R.P.A. Emilia-Romagna

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
    fausto.tomei@gmail.com
    ftomei@arpae.it
*/


#include "interpolationPoint.h"
#include "gis.h"
#include "commonConstants.h"


Crit3DInterpolationDataPoint::Crit3DInterpolationDataPoint()
{
    index = NODATA;
    isActive = false;
    isMarked = false;

    value = NODATA;
    regressionWeight = NODATA;

    lapseRateCode = primary;

    topographicDistance = nullptr;

    point = new gis::Crit3DPoint();
    proxyValues.clear();
}

float Crit3DInterpolationDataPoint::getProxyValue(unsigned int pos)
{
    if (pos < proxyValues.size())
        return proxyValues[pos];
    else
        return NODATA;
}
