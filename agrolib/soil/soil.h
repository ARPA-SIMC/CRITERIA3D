#ifndef SOIL_H
#define SOIL_H

    #ifndef _STRING_
        #include <string>
    #endif
    #ifndef _VECTOR_
        #include <vector>
    #endif

    #define MINIMUM_ORGANIC_MATTER 0.005

    namespace soil {

        enum units {METER, KPA, CM};

        enum soilVariable {soilWaterContentSat, soilWaterContentFC,
                           soilWaterContentWP, soilWaterContentUI,
                           soilWaterPotentialFC, soilWaterPotentialWP
                          };

        struct Crit3DWaterRetention
        {
            double water_potential;             /*!<  [kPa]      */
            double water_content;               /*!<  [m^3 m^-3] */
        };


        class Crit3DHorizonDbData
        {
        public:
            int horizonNr;
            double upperDepth, lowerDepth;      /*!<   [cm]         */
            double sand, silt, clay;            /*!<   [%]          */
            double coarseFragments;             /*!<   [%]          */
            double organicMatter;               /*!<   [%]          */
            double bulkDensity;                 /*!<   [g cm^-3]    */
            double thetaSat;                    /*!<   [m^3 m^-3]   */
            double kSat;                        /*!<   [cm day^-1]  */
            double effectiveCohesion;           /*!<   [kPa]        */
            double frictionAngle;               /*!<   [degrees]    */

            std::vector <Crit3DWaterRetention> waterRetention;

            Crit3DHorizonDbData();
        };


        class Crit3DTexture
        {
        public:
            double sand;                        /*!<   [%]         */
            double silt;                        /*!<   [%]         */
            double clay;                        /*!<   [%]         */
            int classUSDA;
            int classNL;
            std::string classNameUSDA;
            // Unified Soil Classification System (USCS)
            int classUSCS;

            Crit3DTexture();
            /*!
             * \brief Crit3DTexture class constructor
             * \param sand fraction of sand [-]
             * \param silt fraction of silt [-]
             * \param clay fraction of clay [-]
             */
            Crit3DTexture (double sand, double silt, double clay);
        };


        class Crit3DVanGenuchten
        {
        public:
            double alpha;                   /*!<  [kPa^-1] Van Genuchten parameter */
            double n;                       /*!<  [-] Van Genuchten parameter */
            double m;                       /*!<  [-] Van Genuchten parameter (restricted: 1-1/n) */
            double he;                      /*!<  [kPa] air potential (modified VG - Ippisch, 2006) */
            double sc;                      /*!<  [-] reduction factor (modified VG - Ippisch, 2006) */
            double thetaR;                  /*!<  [m^3 m^-3] */
            double thetaS;                  /*!<  [m^3 m^-3] volumetric water content at saturation */
            double refThetaS;               /*!<  [m^3 m^-3] reference volumetric water content at saturation */

            Crit3DVanGenuchten();
        };


        class Crit3DGeotechnicsClass
        {
        public:
            double effectiveCohesion;      /*!<  [kPa] soil effective cohesion */
            double frictionAngle;          /*!<  [°] soil failure angle */

            Crit3DGeotechnicsClass();
        };


        /*! Driessen parameters for empirical infiltration model
         * van Keulen, Wolf, 1986 */
        class Crit3DDriessen
        {
        public:
            double k0;                      /*!<   [cm day^-1] saturated hydraulic conductivity */
            double maxSorptivity;           /*!<   [cm day^-1/2] maximum sorptivity (sorptivity of a completely dry matrix) */
            double gravConductivity;        /*!<   [cm day^-1] infiltration due to gravitational force */

            Crit3DDriessen();
        };


        class Crit3DWaterConductivity
        {
        public:
            double kSat;                    /*!<   [cm day^-1] saturated conductivity  */
            double l;                       /*!<   [-] tortuosity parameter (Van Genuchten - Mualem)  */

            Crit3DWaterConductivity();
        };


        class Crit3DTextureClass
        {
        public:
            Crit3DVanGenuchten vanGenuchten;
            Crit3DWaterConductivity waterConductivity;
            Crit3DDriessen Driessen;
            std::string classNameUSDA;
        };


        class Crit3DHorizon
        {
        public:
            double upperDepth, lowerDepth;      /*!<  [m]       */
            double coarseFragments;             /*!<  [-] 0-1   */
            double organicMatter;               /*!<  [-] 0-1   */
            double bulkDensity;                 /*!<  [g/cm^3]  */
            double effectiveCohesion;           /*!<  [kPa]     */
            double frictionAngle;               /*!<  [degrees] */

            double fieldCapacity;               /*!<  [kPa]     */
            double wiltingPoint;                /*!<  [kPa]     */
            double waterContentFC;              /*!<  [m^3 m^-3]*/
            double waterContentWP;              /*!<  [m^3 m^-3]*/
            double PH;                          /*!<  [-]       */
            double CEC;                         /*!<  [meq/100g]*/

            Crit3DHorizonDbData dbData;
            Crit3DTexture texture;
            Crit3DVanGenuchten vanGenuchten;
            Crit3DWaterConductivity waterConductivity;
            Crit3DDriessen Driessen;

            Crit3DHorizon();
        };


        class Crit3DLayer
        {
        public:
            double depth;               /*!<   [m] */
            double thickness;           /*!<   [m] */
            double soilFraction;        /*!<   [-] fraction of soil (1 - coarse fragment fraction) */
            double waterContent;        /*!<   [mm] */
            double SAT;                 /*!<   [mm] water content at saturation  */
            double FC;                  /*!<   [mm] water content at field capacity */
            double WP;                  /*!<   [mm] water content at wilting point  */
            double HH;                  /*!<   [mm] water content at hygroscopic humidity */
            double critical;            /*!<   [mm] water content at critical point for water movement (typical FC)  */
            double maxInfiltration;     /*!<   [mm]  */
            double flux;                /*!<   [mm]  */

            Crit3DHorizon *horizonPtr;

            Crit3DLayer();

            bool setLayer(Crit3DHorizon *horizonPointer);
            double getVolumetricWaterContent();
            double getDegreeOfSaturation();
            double getWaterPotential();
            double getWaterConductivity();
        };


        class Crit3DSoil
        {
        public:
            int id;
            std::string code;
            std::string name;
            unsigned int nrHorizons;
            double totalDepth;                  /*!<   [m]  */
            std::vector <Crit3DHorizon> horizon;

            Crit3DSoil();

            void initialize(const std::string &soilCode, int nrHorizons);
            void cleanSoil();
            void addHorizon(int nHorizon, const Crit3DHorizon &newHorizon);
            void deleteHorizon(int nHorizon);
            int getHorizonIndex(double depth);

            bool setSoilLayers(double layerThicknessMin, double geometricFactor,
                               std::vector<Crit3DLayer> &soilLayers, std::string &myError);
        };


        class Crit3DFittingOptions
        {
        public:
            int waterRetentionCurve;
            bool useWaterRetentionData;
            bool airEntryFixed;
            bool mRestriction;

            Crit3DFittingOptions();
        };


        int getUSDATextureClass(Crit3DTexture texture);
        int getNLTextureClass(Crit3DTexture texture);
        int getUSDATextureClass(double sand, double silt, double clay);
        int getNLTextureClass(double sand, double silt, double clay);

        int getUSCSClass(const Crit3DHorizon &horizon);

        int getHorizonIndex(const Crit3DSoil &soil, double depth);
        int getSoilLayerIndex(const std::vector<Crit3DLayer> &soilLayers, double depth);

        double getFieldCapacity(const Crit3DHorizon &horizon, soil::units unit);
        double getWiltingPoint(soil::units unit);

        double kPaToMeters(double value);
        double metersTokPa(double value);

        double kPaToCm(double value);
        double cmTokPa(double value);

        double SeFromTheta(double theta, const Crit3DHorizon &horizon);
        double psiFromTheta(double theta, const Crit3DHorizon &horizon);
        double degreeOfSaturationFromSignPsi(double signPsi, const Crit3DHorizon &horizon);
        double thetaFromSignPsi(double signPsi, const Crit3DHorizon &horizon);
        double waterConductivityFromSignPsi(double psi, const Crit3DHorizon &horizon);

        double waterConductivity(double Se, const Crit3DHorizon &horizon);

        double estimateOrganicMatter(double upperDepth);
        double estimateSpecificDensity(double organicMatter);
        double estimateBulkDensity(const Crit3DHorizon &horizon, double totalPorosity, bool increaseWithDepth);
        double estimateSaturatedConductivity(const Crit3DHorizon &horizon, double bulkDensity);
        double estimateTotalPorosity(const Crit3DHorizon &horizon, double bulkDensity);
        double estimateThetaSat(const Crit3DHorizon &horizon, double bulkDensity);

        double getWaterContentFromPsi(double signPsi, const Crit3DLayer &layer);
        double getWaterContentFromAW(double availableWater, const Crit3DLayer &layer);

        bool setHorizon(Crit3DHorizon &horizon, const std::vector<Crit3DTextureClass> &textureClassList,
                        const std::vector<Crit3DGeotechnicsClass> &geotechnicsClassList,
                        const Crit3DFittingOptions &fittingOptions, std::string &errorStr);

        bool fittingWaterRetentionCurve(Crit3DHorizon &horizon, const Crit3DFittingOptions &fittingOptions);

        bool sortWaterPotential(soil::Crit3DWaterRetention first, soil::Crit3DWaterRetention second);
    }


#endif // SOIL_H
