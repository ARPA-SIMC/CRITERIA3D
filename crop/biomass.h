#ifndef BIOMASS_H
#define BIOMASS_H

    /*!
    * Assign physical and miscellaneous constants
    */
    #define      CARBONFACTOR 0.5           /*!< coeff for conversion of carbon into DM, kgC kgDM-1  */
    #define      GAMMA  66.2                /*!< psychrometer constant, Pa K-1  */
    #define      LATENT  43956              /*!< latent heat of vaporization, J mol-1  */
    #define      H2OMOLECULARWEIGHT  0.018  /*!< molecular weight of H2O, kg mol-1  */
    #define      OSS 21176                  /*!< oxygen part pressure in the atmosphere, Pa  */

    /*!
    * Define additional photosynthetic parameters
    */
    #define      HARD       46.39           /*!< activation energy of RD0 (kJ mol-1)   */
    #define      HAVCM      65.33           /*!< activation energy of VCMOP (kJ mol-1)  */
    #define      HAJM       43.9            /*!< activation energy of JMOP (kJ mol-1 e-)  */
    #define      HAKC       79.43           /*!< activation energy of KCT0 (kJ mol-1)  */
    #define      HAKO       36.38           /*!< activation energy of KOT0 (kJ mol-1)  */
    #define      HAGSTAR    37.83           /*!< activation energy of Gamma_star (kJ mol-1)  */
    #define      HDEACTIVATION  200         /*!< deactivation energy from Kattge & Knorr 2007 (kJ mol-1)  */

    #define      CRD        18.72           /*!< scaling factor in RD0 response to temperature (-)  */
    #define      CVCM       26.35           /*!< scaling factor in VCMOP response to temperature (-)  */
    #define      CVOM       22.98           /*!< scaling factor in VOMOP response to temperature (-)  */
    #define      CGSTAR     19.02           /*!< scaling factor in Gamma_star response to temperature (-)  */
    #define      CKC        38.05           /*!< scaling factor in KCT0 response to temperature (-)  */
    #define      CKO        20.30           /*!< scaling factor in KOT0 response to temperature (-)  */
    #define      CJM        17.7            /*!< scaling factor in JMOP response to temperature (-)  */             

    /*!
     * Define additional functional and structural parameters for stand and understorey
     */
    #define      CONV       0.8             //    dry matter conversion efficiency (growth resp.)(-)
    #define      MERCH      0.85            //    merchantable wood as fraction of stem biomass (-)
    #define      RADRT      1.E-3           //    root radius (m)
    #define      STH0       0.8561          //    intercept in self-thinning eq. (log(TREES) vs log(WST)) (m-2)
    #define      STH1       1.9551          //    slope in self-thinning eq. (log(TREES) vs log(WST)) (kgDM-1)
    //#define      ALLRUND    0.5             //    coeff of allocation to roots in understorey (-)

    /*!
     * Define soil respiration parameters, partition soil C into young and old components
     * Note: initial steady-state conditions are assumed (Andren & Katterer 1997)
     * Reference respiration rates are for a 'tropical' soil (Andren & Katterer 1997, p. 1231).
    */
    #define      HUMCOEF    0.125           /*!< humification coefficient (-)  */
    #define      R0SLO      0.04556         /*!< resp per unit old soil carbon at ref conditions (kgDM kgDM-1 d-1) */
    #define      R0SLY      4.228           /*!< resp per unit young soil carbon at ref conditions (kgDM kgDM-1 d-1) */
    #define      CHLDEFAULT 500             /*!< [g cm-2]  */
    #define      RUEGRASS   1.0             /*!< maize: 1.5-2.0, vine: 0.6-1.0  */

    #define      SHADEDGRASS true
    #define      SUNLITGRASS false
	

#endif // BIOMASS_H
