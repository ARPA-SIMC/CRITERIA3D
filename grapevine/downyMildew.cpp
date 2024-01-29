#include "downyMildew.h"
#include "physics.h"
#include <math.h>

////////////////////////////////////////////////////////////////////////////////////////////////////
/// <summary> downy mildew model (grapevine) </summary>
///
/// <remarks> Author: Laura Costantini, 27/11/2013.
///
///		<para>Read the TdownyMildewInput structure and implement the model mildew, filling the
///     TdownyMildewOutput structure and updating TdownyMildewState.</para>
///
///	</remarks>
///
/// <param name="mildewCore"> Pointer to a Tmildew structure. </param>
///
///
////////////////////////////////////////////////////////////////////////////////////////////////////

void downyMildew(TdownyMildew* downyMildewCore, bool isFirstJanuary){

    TdownyMildewState newState;
    int i, last, nrEvents;

    // initialization
    downyMildewCore->output.isInfection = false;
    downyMildewCore->output.oilSpots = 0.0;
    downyMildewCore->output.infectionRate = 0.0;
    downyMildewCore->output.mmo = 0.0;

    // algorithm start from first january
    if (isFirstJanuary)
    {
        downyMildewCore->state.clear();
        downyMildewCore->htt = 0.0;
        downyMildewCore->currentPmo = 0.0;
        downyMildewCore->isGermination = false;
    }

    float tair = downyMildewCore->input.tair;
    float rain = downyMildewCore->input.rain;
    int leafWetness = downyMildewCore->input.leafWetness;
    float relativeHumidity = downyMildewCore->input.relativeHumidity;

    float avgT = 0.0;
    float wdtwd = 0.0;

    float vpd = float(vapourPressureDeficit(tair, relativeHumidity));
    int litterMoist = leafLitterMoisture(rain, vpd);

    // total matured oospores are given each hour with the rate of dormancy breaking of MMO (MMO=1)
    float previousSumPMO = dormancyBreaking(downyMildewCore->htt);

    // hydrothermal time
    downyMildewCore->htt += hydrothermalTime(tair, litterMoist);

    // every time we have a rate of breakage of dormancy, which expresses the proportion
    // of spores which turn from the stadium MMO (dormant) to PMO (physiologically mature)
    float sumPMO = dormancyBreaking(downyMildewCore->htt);

    float hourlyPmo = sumPMO - previousSumPMO;
    if (hourlyPmo < 0.0) hourlyPmo = 0.0;

    if ((downyMildewCore->htt >= 1.3) && (downyMildewCore->htt < 8.6))
    {
        // germination starts with rain and presence of physiological mature oospore
        if  ((rain >= 0.2) && (downyMildewCore->currentPmo >= 0.01) && (!downyMildewCore->isGermination))
        {
            downyMildewCore->isGermination = true;
            downyMildewCore->state.push_back(newState);
            last = (int)downyMildewCore->state.size() - 1;

            // initialize
            downyMildewCore->state[last].stage = 1;
            downyMildewCore->state[last].cohort = downyMildewCore->currentPmo;
            downyMildewCore->state[last].rate = 0.0;
            downyMildewCore->state[last].wetDuration = 0;
            downyMildewCore->state[last].sumT = 0.0;
            downyMildewCore->state[last].nrHours = 0;

            downyMildewCore->currentPmo = 0.0;
        }
    }

    // first dry hour - germination event is complete
    if ((litterMoist == 0) && downyMildewCore->isGermination)
        downyMildewCore->isGermination = false;

    downyMildewCore->currentPmo += hourlyPmo;      // update mature oospores
    downyMildewCore->output.mmo = 1.f - sumPMO;

    nrEvents = (int)downyMildewCore->state.size();

    i = 0;
    while(i < nrEvents)
    {
        // STAGE 1 - germination in the litter
        if (downyMildewCore->state[i].stage == 1)
        {
            // accumulation of germination of oospores
            downyMildewCore->state[i].rate += hydrothermalTime(tair, litterMoist);

            // the oospores form sporangia
            if (downyMildewCore->state[i].rate >= 1.0)
            {
                downyMildewCore->state[i].stage = 2;
                downyMildewCore->state[i].wetDuration = 0;
                downyMildewCore->state[i].rate = 0.0;
                downyMildewCore->state[i].sumT = 0.0;
                downyMildewCore->state[i].nrHours = 0;
            }
        }

        // STAGE 2 - SPORANGIA
        else if (downyMildewCore->state[i].stage == 2)
        {
            // SURVIVAL of sporangia
            downyMildewCore->state[i].rate += survivalRateSporangia(tair, relativeHumidity);

            if (downyMildewCore->state[i].rate > 1.0)
            {
                 // death of sporangia
                 downyMildewCore->state.erase(downyMildewCore->state.begin()+i);
                 i--;
                 nrEvents--;
            }
            else if (leafWetness > 0)
            {
                 downyMildewCore->state[i].nrHours++;
                 // wetDuration is the number of hours of wetness from day of germsporangia
                 downyMildewCore->state[i].wetDuration += leafWetness;
                 // avgtair is the average tair since day of germination of sporagia until now
                 downyMildewCore->state[i].sumT += tair;
                 avgT = downyMildewCore->state[i].sumT / downyMildewCore->state[i].nrHours;

                 // condition for zoospore release (ZRE)
                 // ftomei: modified (error in paper?)
                 float functionT = (float)exp(-1.022 + (19.634 / avgT));
                 if (downyMildewCore->state[i].wetDuration >= functionT)
                 {
                        downyMildewCore->state[i].stage = 3;
                        downyMildewCore->state[i].wetDuration = 0;
                        downyMildewCore->state[i].nrHours = 0;
                 }
            }

         }

         // STAGE 3 - zoospores released
         else if (downyMildewCore->state[i].stage == 3)
         {
            downyMildewCore->state[i].wetDuration += leafWetness;
            downyMildewCore->state[i].nrHours++;

            // zoospore in ZRE stage survive until is wet (we tolerate 1 dry hour for measure errors)
            if ((downyMildewCore->state[i].nrHours - downyMildewCore->state[i].wetDuration) > 1)
            {
                // death of zoospore
                downyMildewCore->state.erase(downyMildewCore->state.begin()+i);
                i--;
                nrEvents--;
            }
            // if rain, the zoospores splash to leaves
            else if (rain > 0.2)
            {
                downyMildewCore->state[i].stage = 4;
                downyMildewCore->state[i].wetDuration = 1;
                downyMildewCore->state[i].sumT = tair;
                downyMildewCore->state[i].nrHours = 1;
            }
        }

        // STAGE 4 - zoospores on leaves cause infection
        else if (downyMildewCore->state[i].stage == 4)
        {
            downyMildewCore->state[i].wetDuration += leafWetness;
            downyMildewCore->state[i].nrHours++;

            // zoospore survive until is wet (we tolerate 1 dry hour for measure errors)
            if ((downyMildewCore->state[i].nrHours - downyMildewCore->state[i].wetDuration) > 1)
            {
                downyMildewCore->state.erase(downyMildewCore->state.begin()+i);
                i--;
                nrEvents--;
            }
            else
            {
                downyMildewCore->state[i].sumT += tair;
                avgT = downyMildewCore->state[i].sumT / downyMildewCore->state[i].nrHours;
                wdtwd = (avgT * downyMildewCore->state[i].wetDuration);

                // when the product of avgT and wetness duration are > 60 zoospores cause infection
                if (wdtwd >= 60.0)
                {
                    downyMildewCore->state[i].stage = 5;
                    downyMildewCore->output.isInfection = true;

                    // infectionRate assumes the relative value of the number of spores that were in the cohort of i spores
                    downyMildewCore->output.infectionRate += downyMildewCore->state[i].cohort;
                    downyMildewCore->state[i].rate = 0.0;    //reset rate variable
                }
            }
       }

       // STAGE 5 - symptoms
       else if (downyMildewCore->state[i].stage == 5)
       {
             downyMildewCore->state[i].rate += incubation(tair);
             if (downyMildewCore->state[i].rate > 1)
             {
                 //  Oil spots on leaves (the symptoms that appears at the end)
                 downyMildewCore->output.oilSpots = downyMildewCore->state[i].cohort;
                 downyMildewCore->state.erase(downyMildewCore->state.begin()+i);
                 i--;
                 nrEvents--;
             }
             else
             {
                downyMildewCore->output.infectionRate += downyMildewCore->state[i].cohort;
             }
        }

        i++;
     }

}


//*****************************************************auxiliary functions*******************************************************//



////////////////////////////////////////////////////////////////////////////////////////////////////
/// <summary> Leaf litter moisture </summary>
///
/// <remarks> Author: Laura Costantini, 27/11/2013.
///
///		<para> Compute the leaf litter moisture.</para>
///
///	</remarks>
///
/// <param name="rain"> mm of rain </param>
/// <param name="vpd"> Value of vapour pressure deficit </param>
///
/// <returns> Returns a dichotomic variable accounting for the moisture of the leaf litter holding oospores. </returns>
///
////////////////////////////////////////////////////////////////////////////////////////////////////

int leafLitterMoisture(float rain,float vpd){

    if ((rain > 0) || (vpd <= 4.5))
        return 1;
    else
        return 0;
}

////////////////////////////////////////////////////////////////////////////////////////////////////
/// <summary> hydrothermal time </summary>
///
/// <remarks> Author: Laura Costantini, 27/11/2013.
///
///		<para> Compute hydrothermal time.</para>
///
///	</remarks>
///
/// <param name="tair"> Value of tairerature </param>
/// <param name="llm"> Leaf litter moisture </param>
///
/// <returns> Hydrothermal time. </returns>
///
////////////////////////////////////////////////////////////////////////////////////////////////////

float hydrothermalTime(float tair, int llm) {
    if (tair <= 0.0)
        return 0.0;
    else
        return (float)(llm/(1330.1 - 116.19*tair + 2.6256*pow(tair,2.0)));
}

////////////////////////////////////////////////////////////////////////////////////////////////////
/// <summary> dormancy breaking </summary>
///
/// <remarks> Author: Laura Costantini, 27/11/2013.
///
///		<para> Compute progress of dormancy breaking.</para>
///
///	</remarks>
///
/// <param name="htime"> Value of hydrothermal time </param>
///
/// <returns> Progress of dormancy breaking. </returns>
///
////////////////////////////////////////////////////////////////////////////////////////////////////

float dormancyBreaking(float htime) {

    return (float)( exp(-15.891*exp(-0.653*(htime+1.0))) );
}

////////////////////////////////////////////////////////////////////////////////////////////////////
/// <summary> survival rate of sporangia </summary>
///
/// <remarks> Author: Laura Costantini, 27/11/2013.
///
///		<para> Compute survival rate of sporangia.</para>
///
///	</remarks>
///
/// <param name="tair"> Value of tairerature </param>
/// <param name="relativeHumidity"> Value of relative Humidity </param>
///
/// <returns> Survival rate of sporangia. </returns>
///
////////////////////////////////////////////////////////////////////////////////////////////////////

float survivalRateSporangia(float tair, float relativeHumidity) {

    //check relativeHumidity
    if (relativeHumidity < 1) relativeHumidity = 1;
    if (relativeHumidity > 100) relativeHumidity = 100;
    relativeHumidity /= 100.f;

    return 1.f / (24.f*(5.67f-0.47f*(tair*(1.f-relativeHumidity))+0.01f * powf((tair*(1.f-relativeHumidity)), 2.f)));
}

////////////////////////////////////////////////////////////////////////////////////////////////////
/// <summary> progress of incubation </summary>
///
/// <remarks> Author: Laura Costantini, 27/11/2013.
///
///		<para> Compute hourly progress of incubation of oil spots.</para>
///
///	</remarks>
///
/// <param name="tair"> Value of tairerature </param>
///
/// <returns> Hourly progress of incubation. </returns>
///
////////////////////////////////////////////////////////////////////////////////////////////////////

float incubation(float tair) {

    return (float)(1/(24*(45.1-3.45*tair+0.073*pow(tair,2.0))) );
}

