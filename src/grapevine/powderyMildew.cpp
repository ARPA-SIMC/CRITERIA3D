#include "powderyMildew.h"
#include "physics.h"
#include <math.h>


using namespace std;

////////////////////////////////////////////////////////////////////////////////////////////////////
/// <summary> powdery mildew model (grapevine) </summary>
///
/// <remarks> Author: Laura Costantini, 30/08/2013.
///
///		<para>Read the TmildewInput structure and implement the model mildew,
///     filling the TmildewOutput structure and updating TmildewState.</para>
///
///	</remarks>
///
/// <param name="mildewCore"> Pointer to a Tmildew structure. </param>
////////////////////////////////////////////////////////////////////////////////////////////////////


// constants
const double delta = 0.969;
const double lambda = 0.0004;
const double fi = 7.391;
const double nu = 2.403;
const double csi = 0.892;
const double upsilon = 0.221;
const double gammaConst = 44.7;
const double psi = 0.067;
const double theta = 3.244;


void powderyMildew(Tmildew* mildewCore, bool isBudBreak){

    // initialization
    int och = 1;       // och (Overwintered chasmothecia) ascospore population from the previous years, the model set it to 1 by convention
    float tavg = mildewCore->input.tavg;
    float rain = mildewCore->input.rain;
    int leafWetness = mildewCore->input.leafWetness;     

    mildewCore->output.aol = 0;    // Ascospores on grape leaves
    mildewCore->output.col = 0;    // Colony-forming ascospores on grape leaves
    mildewCore->output.infectionRate = 0;
    mildewCore->output.infectionRisk = 0;
    mildewCore->output.dayInfection = false;  // day where the colony germinates
    mildewCore->output.daySporulation = false; // day where the colony begins to sporulate, after infection has been developed

    // if it is the first time, assign default parameters
    if (isBudBreak)
    {
        mildewCore->state.degreeDays = 0.0;
        // Amount of ascospores already ready before the opening of the buds (0.14)
        mildewCore->state.aic = ascosporesReadyFraction(mildewCore->state.degreeDays);
        mildewCore->state.currentColonies = 0.0;
        mildewCore->state.totalSporulatingColonies = 0.0;
    }

    //*****************************************************ALGORITHM*******************************************************//
    // degree days of the current day
    float currentDegreeDay = computeDegreeDay(tavg);

    // compute the vapour pressure deficit
    float vpre = float(vapourPressureDeficit(tavg, mildewCore->input.relativeHumidity));

    // calculate the cumulative of mature ascospore
    mildewCore->state.aic += och*(ascosporesReadyFraction(mildewCore->state.degreeDays+currentDegreeDay)
                                   - ascosporesReadyFraction(mildewCore->state.degreeDays));

    mildewCore->output.infectionRate = infectionRate(tavg, vpre);

    mildewCore->output.infectionRisk = mildewCore->output.infectionRate * mildewCore->state.aic;

    // ascospore discharged onto leaves
    mildewCore->output.aol  =  mildewCore->state.aic * ascosporeDischargeRate(tavg, rain, leafWetness);

    // update ascospore ready for discharge
    mildewCore->state.aic -=  mildewCore->output.aol;

    // amount of colonies developed
    mildewCore->output.col = mildewCore->output.aol * mildewCore->output.infectionRate;

    // dayInfection: the day on which germinate colonies
    if (mildewCore->output.col > 0.001f)
            mildewCore->output.dayInfection = true;

    // current latency
    float latency = latencyProgress(tavg);      //[0,1]

    float dailySporulating = mildewCore->state.currentColonies * latency;
    mildewCore->state.totalSporulatingColonies += dailySporulating;

    // update currentColonies
    mildewCore->state.currentColonies += (mildewCore->output.col - dailySporulating);

    mildewCore->state.degreeDays += currentDegreeDay;
}


//*****************************************************auxiliary functions*******************************************************//

////////////////////////////////////////////////////////////////////////////////////////////////////
/// <summary> Compute degree day</summary>
///
/// <remarks> Author: Laura Costantini, 30/08/2013.
///
///		<para> Compute degree day.</para>
///
///	</remarks>
///
/// <param name="temp"> Value of temperature </param>
///
/// <returns> Degree day if temperature is more than 10 degree, 0 otherwise.</returns>
///
////////////////////////////////////////////////////////////////////////////////////////////////////
float computeDegreeDay(float temp){

    if (temp > 10.f)
        return (temp - 10.f);
    else
        return 0.0;
}



////////////////////////////////////////////////////////////////////////////////////////////////////
/// <summary> Fraction of ascospores ready for discharge </summary>
///
/// <remarks> Author: Laura Costantini, 30/08/2013.
///
///		<para> Compute the fration of ascospores ready for discharge.</para>
///
///	</remarks>
///
/// <param name="degreeDay"> Value of degree day </param>
///
/// <returns> Fraction of ascospores ready for discharge.</returns>
///
////////////////////////////////////////////////////////////////////////////////////////////////////
///
float ascosporesReadyFraction(float degreeDay){

        return expf(-1.95f * expf(-1.91f * degreeDay / 100.f));

}


////////////////////////////////////////////////////////////////////////////////////////////////////
/// <summary> Ascospore discharge rate </summary>
///
/// <remarks> Author: Laura Costantini, 30/08/2013.
///
///		<para> Compute the ascospore discharge rate.</para>
///
///	</remarks>
///
/// <param name="temp"> Value of temperature </param>
/// <param name="rain"> mm of rain </param>
/// <param name="leafWetness"> Value of leaf wetness </param>
///
/// <returns> Ascospore discharge rate.</returns>
///
////////////////////////////////////////////////////////////////////////////////////////////////////
float ascosporeDischargeRate(float temp,float rain,int leafWetness){

    if  ((rain < 2) || ( temp < 4) || ( temp > 30))
        return 0;

    else
        return float(1.0 -delta * exp(-lambda * pow(temp, 2.0) * leafWetness));

}


////////////////////////////////////////////////////////////////////////////////////////////////////
/// <summary> Infection rate </summary>
///
/// <remarks> Author: Laura Costantini, 30/08/2013.
///
///		<para> Determine the proportion of colony-forming ascospores on leaves.</para>
///
///	</remarks>
///
/// <param name="temp"> Value of temperature </param>
/// <param name="tempEqui"> Value of temperature equivalent </param>
/// <param name="vapourPressure"> Value of vapour pressure deficit </param>
///
/// <returns> 0 if temperature is < 5 or > 31, infection rate otherwise.</returns>
///
////////////////////////////////////////////////////////////////////////////////////////////////////
float infectionRate(float temp, float vapourPressure){

    if ((temp < 5) || ( temp > 31))
        return 0.0;

    else
    {
        float Tequivalent = (temp - 5.f) / (31.f - 5.f);
        return float((pow((fi*pow(Tequivalent, nu)*(1.0-double(Tequivalent))), csi))
                     *exp(-upsilon*double(vapourPressure)));
    }

}



////////////////////////////////////////////////////////////////////////////////////////////////////
/// <summary> Daily progress of latency </summary>
///
/// <remarks> Author: Laura Costantini, 30/08/2013.
///
///		<para> Compute the daily progress of latency.</para>
///
///	</remarks>
///
/// <param name="temp"> Value of temperature </param>
///
/// <returns> Daily progress of latency.</returns>
///
////////////////////////////////////////////////////////////////////////////////////////////////////
float latencyProgress(float temp){

        return float(1.0 / (gammaConst + psi*pow(temp,2.0) - theta*double(temp)));

}


float max_vector(const vector<float>& v)
{
    float maximum = v[0];
    for(unsigned int i = 1; i < v.size(); i++)
        if (v[i] > maximum) maximum = v[i];
    return maximum;
}
