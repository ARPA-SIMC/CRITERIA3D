#ifndef TYPESEXTRA_H
#define TYPESEXTRA_H

    struct TboundaryHeat
    {
        double temperature;                     /*!< [K] temperature of the boundary (ex. air temperature) */
        double relativeHumidity;                /*!< [%] relative humidity */
        double windSpeed;                       /*!< [m s-1] wind speed */
        double netIrradiance;                   /*!< [W m-2] net irradiance */
        double heightWind;                      /*!< [m] reference height for wind measurement */
        double heightTemperature;               /*!< [m] reference height for temperature and humidity measurement */
        double roughnessHeight;                 /*!< [m] surface roughness height */
        double sensibleFlux;                    /*!< [W m-2] boundary sensible heat flux density */
        double latentFlux;                      /*!< [W m-2] boundary latent heat flux density */
        double radiativeFlux;                   /*!< [W m-2] boundary net radiative flux density */
        double advectiveHeatFlux;               /*!< [W m-2] boundary advective heat flux density  */
        double aerodynamicConductance;          /*!< [m s-1] aerodynamic conductance for heat */
        double soilConductance;                 /*!< [m s-1] soil conductance */
        double fixedTemperature;                /*!< [K] fixed temperature */
        double fixedTemperatureDepth;           /*!< [m] depth of fixed temperature layer */
    };

    struct THeatFlux
    {
        float waterFlux;                    /*!< [m3 s-1]   */
        float vaporFlux;                    /*!< [kg s-1]   */
        float* fluxes;                      /*!< [W] for heat fluxes; [m3 s-1] for water fluxes */
    };

    struct TCrit3DNodeHeat
    {
        double T;                           /*!< [K] node temperature */
        double oldT;                        /*!< [K] old node temperature */
        double Qh;                          /*!< [W] heat flow */
        double sinkSource;                  /*!< [W] heat sink/source */
    };

    struct TCrit3DLinkedNodeExtra
    {
        THeatFlux* heatFlux;
    };

    struct TCrit3DnodeExtra
    {
        TCrit3DNodeHeat *Heat;              /*!< heat pointer */
    };

    void initializeExtra(TCrit3DnodeExtra *myNodeExtra, bool computeHeat, bool computeSolutes);
    void initializeLinkExtra(TCrit3DLinkedNodeExtra* myLinkedNodeExtra, bool computeHeat, bool computeSolutes);
    void initializeNodeHeatFlux(TCrit3DLinkedNodeExtra* myLinkExtra, bool initHeat, bool initWater);

#endif // TYPESEXTRA_H
