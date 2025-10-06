/*! ----------------------------------------------------------------------------
* $Id: g_solposition.c,v 2.2 2006/02/09 03:09:03 glynn Exp $
*
*   G_calc_solar_position() calculates solar position parameters from
*   given position, date and time
*
*   Written by Markus Neteler <neteler@geog.uni-hannover.de>
*   with kind help from Morten Hulden
*
*----------------------------------------------------------------------------
*   using solpos.c with permission from
*   From rredc@nrel.gov Wed Mar 21 18:37:25 2001
*   Message-Id: <v04220805b6de9b1ad6ff@[192.174.39.30]>
*   Mary Anderberg
*   http://rredc.nrel.gov
*   National Renewable Energy Laboratory
*   1617 Cole Boulevard
*   Golden, Colorado, USA 80401
*
*   http://rredc.nrel.gov/solar/codes_algs/solpos/
*
*   G_calc_solar_position is based on: soltest.c
*   by
*    Martin Rymes
*    National Renewable Energy Laboratory
*    25 March 1998
*----------------------------------------------------------------------------*/


#include "solPos.h"
#include "sunPosition.h"


long RSUN_compute_solar_position (struct SolPosData &pdat, float longitude, float latitude, int myTimezone,
                int year, int month, int day, int hour, int minute, int second,
                float temp, float press, float aspect, float tilt,
                float sbwid, float sbrad, float sbsky)
{

    /*! Note: this code is valid from year 1950 to 2050 (solpos restriction)
     - the algorithm will compensate for leap year.
     - longitude, latitude: decimal degree
     - timezone: DO NOT ADJUST FOR DAYLIGHT SAVINGS TIME.
     - timezone: negative for zones west of Greenwich
     - lat/long: east and north positive
     - the atmospheric refraction is calculated for 1013hPa, 15 C
     - time: local time from your watch
     */

    long retval;            /*!< to capture S_solpos return codes */

    /*! Initialize structure to default values. (Optional only if ALL input
       parameters are initialized in the calling code, which they are not
       in this example.) */

    S_init(&pdat);

    pdat.longitude = longitude;             /*!< Note that latitude and longitude are  */
    pdat.latitude  = latitude;              /*!< in DECIMAL DEGREES, not Deg/Min/Sec   */
    pdat.timezone  = float(myTimezone);     /*!< DO NOT ADJUST FOR DAYLIGHT SAVINGS TIME. */

    pdat.year      = year;                  /*!< The year */
    pdat.function &= ~S_DOY;
    pdat.month     = month;
    pdat.day       = day;                   /*!< the algorithm will compensate for leap year, so
                                            you just count days). S_solpos can be
                                            configured to accept day-of-the year */

    /*! The time of day (STANDARD (GMT) time) */

    pdat.hour      = hour;
    pdat.minute    = minute;
    pdat.second    = second;

    /*! The temperature is used for the
       atmospheric refraction correction, and the pressure is used for the
       refraction correction and the pressure-corrected airmass. */

    pdat.temp       = temp;
    pdat.press      = press;

    pdat.aspect     = aspect;
    pdat.tilt		= tilt;

    pdat.sbwid		= sbwid;
    pdat.sbrad		= sbrad;
    pdat.sbsky		= sbsky;

    /*! compute solar position */
    retval = S_solpos(&pdat);

    return retval;
}


void RSUN_get_results (struct SolPosData &pdat, float &amass, float &ampress,
                      float &azim, float &cosinc, float &coszen,
                      float &elevetr, float &elevref,
                      float &etr, float &etrn, float &etrtilt,
                      float &prime, float &sbcf,
                      float &sunrise, float &sunset,
                      float &unprime, float &zenref)
{
    amass		= pdat.amass;
    ampress     = pdat.ampress;
    azim		= pdat.azim;
    cosinc		= pdat.cosinc;
    coszen		= pdat.coszen;
    elevetr     = pdat.elevetr;
    elevref     = pdat.elevref;
    etr         = pdat.etr;
    etrn		= pdat.etrn;
    etrtilt     = pdat.etrtilt;
    prime		= pdat.prime;
    sbcf		= pdat.sbcf;
    sunrise     = pdat.sretr;
    sunset		= pdat.ssetr;
    unprime     = pdat.unprime;
    zenref		= pdat.zenref;
}
