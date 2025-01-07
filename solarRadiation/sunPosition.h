#ifndef SUNPOSITION_H
#define SUNPOSITION_H

/*!
 * \brief RSUN_compute_solar_position
 * \param longitude
 * \param latitude
 * \param timezone
 * \param year
 * \param month
 * \param day
 * \param hour
 * \param minute
 * \param second
 * \param temp
 * \param press
 * \param aspect
 * \param tilt
 * \param sbwid
 * \param sbrad
 * \param sbsky
 * \return
 */

long RSUN_compute_solar_position (
                    float longitude, float latitude, int timezone,
                    int year, int month, int day, int hour, int minute, int second,
                    float temp, float press, float aspect, float tilt,
                    float sbwid, float sbrad, float sbsky);

/*!
 * \brief RSUN_get_results
 * \param amass Relative optical airmass
 * \param ampress Pressure-corrected airmass
 * \param azim Solar azimuth angle:  N=0, E=90, S=180, W=270
 * \param cosinc Cosine of solar incidence angle on panel
 * \param coszen Cosine of refraction corrected solar zenith angle
 * \param elevetr Solar elevation, no atmospheric correction (= ETR)
 * \param elevref Solar elevation angle, deg. from horizon, refracted
 * \param etr Extraterrestrial (top-of-atmosphere) - W/sq m global horizontal solar irradiance
 * \param etrn Extraterrestrial (top-of-atmosphere) - W/sq m direct normal solar irradiance
 * \param etrtilt Extraterrestrial (top-of-atmosphere) - W/sq m global irradiance on a tilted surface
 * \param prime Factor that normalizes Kt, Kn, etc.
 * \param sbcf Shadow-band correction factor
 * \param sunrise Sunrise time, minutes from midnight, local, WITHOUT refraction
 * \param sunset Sunset time, minutes from midnight, local, WITHOUT refraction
 * \param unprime Factor that denormalizes Kt', Kn', etc.
 * \param zenref Solar zenith angle, deg. from zenith, refracted
 */
void RSUN_get_results (
                    float *amass, float *ampress, float *azim,
                    float *cosinc, float *coszen, float *elevetr, float *elevref,
                    float *etr, float *etrn, float *etrtilt, float *prime, float *sbcf,
                    float *sunrise, float *sunset, float *unprime, float *zenref);

#endif // SUNPOSITION_H
