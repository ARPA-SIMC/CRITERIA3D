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

#include <algorithm>
#include <math.h>

#include "commonConstants.h"
#include "basicMath.h"
#include "color.h"


Crit3DColor::Crit3DColor()
{
    red = 0;
    green = 0;
    blue = 0;
}

Crit3DColor::Crit3DColor(short myRed, short myGreen, short myBlue)
{
    red = myRed;
    green = myGreen;
    blue = myBlue;
}


Crit3DColorScale::Crit3DColorScale()
{
    _nrKeyColors = 1;
    _nrColors = 1;
    keyColor.resize(_nrKeyColors);
    color.resize(_nrColors);

    _minimum = NODATA;
    _maximum = NODATA;
    _isFixedRange = false;
    _isHideOutliers = false;
    _isTransparent = false;

    _classification = classificationMethod::EqualInterval;
}


void Crit3DColorScale::initialize(unsigned int nrKeyColors, unsigned int nrColors)
{
    _nrKeyColors = nrKeyColors;
    _nrColors = nrColors;

    keyColor.clear();
    keyColor.resize(_nrKeyColors);

    color.clear();
    color.resize(_nrColors);
}


bool Crit3DColorScale::setRange(float minimum, float maximum)
{
    if (maximum < minimum)
        return false;

    _minimum = minimum;
    _maximum = maximum;
    return true;
}


bool Crit3DColorScale::classify()
{
    unsigned int i, j, n, nrIntervals, nrStep;
    float dRed, dGreen, dBlue;

    if (_classification == classificationMethod::EqualInterval)
    {
        nrIntervals = MAXVALUE(_nrKeyColors -1, 1);
        nrStep = _nrColors / nrIntervals;

        for (i = 0; i < nrIntervals; i++)
        {
            dRed = float(keyColor[i+1].red - keyColor[i].red) / float(nrStep);
            dGreen = float(keyColor[i+1].green - keyColor[i].green) / float(nrStep);
            dBlue = float(keyColor[i+1].blue - keyColor[i].blue) / float(nrStep);

            for (j = 0; j < nrStep; j++)
            {
                n = nrStep * i + j;
                color[n].red = keyColor[i].red + short(dRed * float(j));
                color[n].green = keyColor[i].green + short(dGreen * float(j));
                color[n].blue = keyColor[i].blue + short(dBlue * float(j));
            }
        }
        color[_nrColors-1] = keyColor[_nrKeyColors -1];
    }

    return true;
}


Crit3DColor* Crit3DColorScale::getColor(float value)
{
    unsigned int index = 0;

    if (value <= _minimum)
    {
        index = 0;
    }
    else if (value >= _maximum)
    {
        index = _nrColors-1;
    }
    else
    {
        if (_classification == classificationMethod::EqualInterval)
        {
            index = unsigned(float(_nrColors-1) * ((value - _minimum) / (_maximum - _minimum)));
        }
    }

    return &color[index];
}


unsigned int Crit3DColorScale::getColorIndex(float value)
{
    if (value <= _minimum)
        return 0;
    else if (value >= _maximum)
        return _nrColors-1;
    else if (_classification == classificationMethod::EqualInterval)
        return unsigned(float(_nrColors-1) * ((value - _minimum) / (_maximum - _minimum)));
    else return 0;
}


bool setDefaultScale(Crit3DColorScale* myScale)
{
    myScale->initialize(4, 256);

    myScale->keyColor[0] = Crit3DColor(0, 0, 255);         /*!< blue */
    myScale->keyColor[1] = Crit3DColor(64, 196, 64);       /*!< green */
    myScale->keyColor[2] = Crit3DColor(255, 255, 0);       /*!< yellow */
    myScale->keyColor[3] = Crit3DColor(255, 0, 0);         /*!< red */

    return(myScale->classify());
}


bool setDTMScale(Crit3DColorScale* myScale)
{
    myScale->initialize(4, 256);

    myScale->keyColor[0] = Crit3DColor(32, 160, 32);        /*!<  green */
    myScale->keyColor[1] = Crit3DColor(224, 224, 0);        /*!<  yellow */
    myScale->keyColor[2] = Crit3DColor(160, 64, 0);         /*!<  red */
    myScale->keyColor[3] = Crit3DColor(224, 224, 224);      /*!<  light gray */

    return(myScale->classify());
}


bool setLAIScale(Crit3DColorScale* myScale)
{
    myScale->initialize(4, 256);

    myScale->keyColor[0] = Crit3DColor(200, 160, 0);        /*!<  ocra */
    myScale->keyColor[1] = Crit3DColor(160, 160, 0);        /*!<  yellow */
    myScale->keyColor[2] = Crit3DColor(32, 160, 32);        /*!<  dark green */
    myScale->keyColor[3] = Crit3DColor(0, 255, 0);          /*!<  green */

    return(myScale->classify());
}


bool setTemperatureScale(Crit3DColorScale* myScale)
{
    myScale->initialize(5, 256);

    myScale->keyColor[0] = Crit3DColor(0, 0, 255);         /*!< blue */
    myScale->keyColor[1] = Crit3DColor(64, 196, 64);       /*!< green */
    myScale->keyColor[2] = Crit3DColor(255, 255, 0);       /*!< yellow */
    myScale->keyColor[3] = Crit3DColor(255, 0, 0);         /*!< red */
    myScale->keyColor[4] = Crit3DColor(128, 0, 128);       /*!< violet */

    return(myScale->classify());
}


bool setSlopeStabilityScale(Crit3DColorScale* myScale)
{
    myScale->initialize(5, 256);

    myScale->keyColor[0] = Crit3DColor(0, 0, 0);           /*!< black */
    myScale->keyColor[1] = Crit3DColor(128, 0, 128);       /*!< violet */
    myScale->keyColor[2] = Crit3DColor(255, 0, 0);         /*!< red */
    myScale->keyColor[3] = Crit3DColor(255, 255, 0);       /*!< yellow */
    myScale->keyColor[4] = Crit3DColor(64, 196, 64);       /*!< green */

    return(myScale->classify());
}


bool setAnomalyScale(Crit3DColorScale* myScale)
{
    myScale->initialize(5, 256);

    myScale->keyColor[0] = Crit3DColor(0, 0, 255);         /*!< blue */
    myScale->keyColor[1] = Crit3DColor(64, 196, 64);       /*!< green */
    myScale->keyColor[2] = Crit3DColor(255, 255, 255);     /*!< white */
    myScale->keyColor[3] = Crit3DColor(255, 0, 0);         /*!< red */
    myScale->keyColor[4] = Crit3DColor(128, 0, 128);       /*!< violet */

    return(myScale->classify());
}


bool setPrecipitationScale(Crit3DColorScale* myScale)
{
    myScale->initialize(6, 252);

    myScale->keyColor[0] = Crit3DColor(255, 255, 255);      /*!< white */
    myScale->keyColor[1] = Crit3DColor(0, 0, 255);          /*!< blue */
    myScale->keyColor[2] = Crit3DColor(64, 196, 64);        /*!< green */
    myScale->keyColor[3] = Crit3DColor(255, 255, 0);        /*!< yellow */
    myScale->keyColor[4] = Crit3DColor(255, 0, 0);          /*!< red */
    myScale->keyColor[5] = Crit3DColor(128, 0, 128);        /*!< violet */

    return(myScale->classify());
}


bool setCenteredScale(Crit3DColorScale* myScale)
{
    myScale->initialize(5, 256);

    myScale->keyColor[0] = Crit3DColor(0, 0, 255);         /*!< blue */
    myScale->keyColor[1] = Crit3DColor(64, 196, 64);       /*!< green */
    myScale->keyColor[2] = Crit3DColor(255, 255, 255);     /*!< white */
    myScale->keyColor[3] = Crit3DColor(255, 255, 0);       /*!< yellow */
    myScale->keyColor[4] = Crit3DColor(255, 0, 0);         /*!< red */

    return(myScale->classify());
}


bool setCircolarScale(Crit3DColorScale* myScale)
{
    myScale->initialize(5, 256);

    myScale->keyColor[0] = Crit3DColor(0, 0, 255);         /*!< blue */
    myScale->keyColor[1] = Crit3DColor(255, 255, 0);       /*!< yellow */
    myScale->keyColor[2] = Crit3DColor(255, 0, 0);         /*!< red */
    myScale->keyColor[3] = Crit3DColor(0, 255, 0);         /*!< green */
    myScale->keyColor[4] = Crit3DColor(0, 0, 255);         /*!< blue */

    return(myScale->classify());
}


bool setRelativeHumidityScale(Crit3DColorScale* myScale)
{
    myScale->initialize(3, 256);

    myScale->keyColor[0] = Crit3DColor(128, 0, 0);          /*!< dark red */
    myScale->keyColor[1] = Crit3DColor(255, 255, 0);        /*!< yellow */
    myScale->keyColor[2] = Crit3DColor(0, 0, 255);          /*!< blue */

    return(myScale->classify());
}


bool setWindIntensityScale(Crit3DColorScale* myScale)
{
    myScale->initialize(3, 256);

    myScale->keyColor[0] = Crit3DColor(32, 128, 32);        /*!< dark green */
    myScale->keyColor[1] = Crit3DColor(255, 255, 0);        /*!< yellow */
    myScale->keyColor[2] = Crit3DColor(255, 0, 0);          /*!< red */

    return(myScale->classify());
}


bool setRadiationScale(Crit3DColorScale* myScale)
{
    myScale->initialize(4, 256);

    myScale->keyColor[0] = Crit3DColor(0, 0, 255);          /*!< blue */
    myScale->keyColor[1] = Crit3DColor(255, 255, 0);        /*!< yellow */
    myScale->keyColor[2] = Crit3DColor(255, 0, 0);          /*!< red */
    myScale->keyColor[3] = Crit3DColor(128, 0, 128);        /*!< violet */

    return myScale->classify();
}


bool setSurfaceWaterScale(Crit3DColorScale* myScale)
{
    myScale->initialize(3, 256);

    myScale->keyColor[0] = Crit3DColor(255, 255, 255);      /*!< white */
    myScale->keyColor[1] = Crit3DColor(0, 255, 255);        /*!< cyan */
    myScale->keyColor[2] = Crit3DColor(0, 0, 255);          /*!< blue */

    return(myScale->classify());
}


bool setGrayScale(Crit3DColorScale* myScale)
{
    myScale->initialize(2, 256);

    myScale->keyColor[0] = Crit3DColor(0, 0, 0);
    myScale->keyColor[1] = Crit3DColor(255, 255, 255);

    return myScale->classify();
}


bool setBlackScale(Crit3DColorScale* myScale)
{
    myScale->initialize(2, 256);

    myScale->keyColor[0] = Crit3DColor(0, 0, 0);
    myScale->keyColor[1] = Crit3DColor(0, 0, 0);

    return myScale->classify();
}


bool reverseColorScale(Crit3DColorScale* myScale)
{
    // copy key colors
    std::vector<Crit3DColor> oldKeyColor = myScale->keyColor;

    // reverse key colors
    unsigned int lastIndex = myScale->nrKeyColors() - 1;
    for (unsigned int i = 0; i < myScale->nrKeyColors(); i++)
    {
        myScale->keyColor[i] = oldKeyColor[lastIndex - i];
    }
    oldKeyColor.clear();

    // reclassify
    return myScale->classify();
}


void mixColor(const Crit3DColor &backColor, const Crit3DColor &foreColor, Crit3DColor &colorOut, float alpha)
{
    if (alpha == 0)
    {
        colorOut.red = backColor.red;
        colorOut.green = backColor.green;
        colorOut.blue = backColor.blue;
    }
    else
    {
        colorOut.red = std::min(255, int(backColor.red * (1. - alpha)) + int(foreColor.red * alpha));
        colorOut.green = std::min(255, int(backColor.green * (1. - alpha)) + int(foreColor.green * alpha));
        colorOut.blue = std::min(255, int(backColor.blue * (1. - alpha)) +int( foreColor.blue * alpha));
    }
}


/*!
 * \brief roundColorScale round colorScale values on the second (or third) digit of each range.
 * It requires that nrColors is a multiply of nrIntervals for a correct visualization in the colors legend.
 * It is projected for a legend of nrIntervals+1 levels (i.e 4 intervals = 5 levels)
 * \param myScale
 * \param nrIntervals
 * \param lessRounded if true the round is on more digit
 * \return
 */
bool roundColorScale(Crit3DColorScale* myScale, int nrIntervals, bool lessRounded)
{
    if (myScale == nullptr) return false;

    if (isEqual(myScale->minimum(), NODATA)
        || isEqual(myScale->maximum(), NODATA)) return false;

    if (nrIntervals < 1) return false;

    if (isEqual(myScale->minimum(), myScale->maximum()))
    {
        myScale->setMaximum(myScale->maximum() + 0.1f);

        if (! isEqual(myScale->minimum(), 0))
        {
            myScale->setMinimum(myScale->minimum() - 0.1f);
        }
        return true;
    }

    double avg = double(myScale->minimum()) + double(myScale->maximum() - myScale->minimum()) / 2;
    double step = double(myScale->maximum() - myScale->minimum()) / double(nrIntervals);
    double logStep = log10(step);

    double myExp;
    double roundAvg = avg;

    if (isEqual(avg, 0))
    {
        myExp = floor(logStep)-1;
    }
    else
    {
        double logAvg = log10(abs(avg));
        if (lessRounded)
        {
            myExp = int((floor(logStep) + floor(logAvg))/2 - 1);
        }
        else
        {
            myExp = std::max(floor(logStep)-1, floor(logAvg)-1);
        }
    }

    double pow10 = pow(10, myExp);
    double newStep = step / pow10;
    double roundStep;
    if (isEqual(newStep, floor(newStep)))
        roundStep = newStep * pow10;
    else
        roundStep = ceil(newStep) * pow10;

    if (! isEqual(avg, 0))
    {
        roundAvg = round(avg / pow10) * pow10;
    }

    if (myScale->minimum() == 0.f)
    {
        //precipitation
        myScale->setMaximum(float(roundStep * nrIntervals));
    }
    else
    {
        myScale->setMinimum(float(roundAvg - roundStep*(nrIntervals/2)));
        myScale->setMaximum(float(roundAvg + roundStep*(nrIntervals/2)));
    }

    return true;
}


