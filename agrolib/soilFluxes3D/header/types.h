#ifndef SOILFLUXES3DTYPES
#define SOILFLUXES3DTYPES

    #ifndef PARAMETERS_H
        #include "parameters.h"
    #endif
    #ifndef TYPESEXTRA_H
        #include "extra.h"
    #endif

    #include "commonConstants.h"

    struct Tboundary
    {
        short type;
        float slope;                        /*!< [m m-1]   */
        float boundaryArea;                 /*!< [m2] (only for surface runoff [m]) */
        double waterFlow;                   /*!< [m3 s-1]   */
        double sumBoundaryWaterFlow;        /*!< [m3] sum of boundary water flow */
        double prescribedTotalPotential;	/*!< [m] imposed total soil-water potential (H) */

        TboundaryHeat *Heat;                /*!< extra variables for heat flux */
    };

    struct TCrit3DStructure
    {
        long nrLayers;
        long nrNodes;
        int nrLateralLinks;
        int maxNrColumns;

        bool computeWater;
        bool computeHeat;
        bool computeSolutes;
        bool computeHeatVapor;
        bool computeHeatAdvection;

        int saveHeatFluxesType;

        void initialize()
        {
            nrLayers = 0;
            nrNodes = 0;
            nrLateralLinks = 0;
            maxNrColumns = 0;
            computeWater = true;
            computeHeat = false;
            computeHeatAdvection = false;
            computeHeatVapor = false;
            computeSolutes = false;
            saveHeatFluxesType = SAVE_HEATFLUXES_NONE;
        }
    };


    struct TlinkedNode
    {
        long index;                             /*!< index of linked elements */
        float area;                             /*!< interface area in soil nodes [m2] - interface side on surface [m] */
        float sumFlow;                          /*!< [m3] sum of flow(i,j) */

        TCrit3DLinkedNodeExtra* linkedExtra;    /*!< extra variables for heat flux */
    };


    struct Tsoil
    {
        double VG_alpha;            /*!< [m-1] Van Genutchen alpha parameter */
        double VG_n;                /*!< [-] Van Genutchen n parameter */
        double VG_m;                /*!< [-] Van Genutchen m parameter  ]0. , 1.[ */
        double VG_he;               /*!< [m] air-entry potential for modified VG formulation */
        double VG_Sc;               /*!< [-] reduction factor for modified VG formulation */
        double Theta_s;             /*!< [m3 m-3] saturated water content */
        double Theta_r;             /*!< [m3 m-3] residual water content */
        double K_sat;               /*!< [m sec-1] saturated hydraulic conductivity */
        double Mualem_L;            /*!< [-] Mualem tortuosity parameter */

        double roughness;           /*!< [s m-1/3] surface: Manning roughness */

        // for heat flux
        double organicMatter;       /*!< [-] fraction of organic matter */
        double clay;                /*!< [-] fraction of clay */
    };


    struct TCrit3Dnode
    {
        double Se;					/*!< [-] degree of saturation */
        double k;                   /*!< [m s-1] soil water conductivity */
        double H;                   /*!< [m] pressure head */
        double oldH;				/*!< [m] previous pressure head */
        double bestH;				/*!< [m] pressure head of best iteration */
        double waterSinkSource;     /*!< [m3 s-1] water sink source */
        double Qw;                  /*!< [m3 s-1] water flow */

        double volume_area;         /*!< [m3] sub-surface: volume of voxel   */
                                    /*!< [m2] surface: area of voxel   */
        float x, y;                 /*!< [m] coordinates of the center of the voxel */
        double z;                   /*!< [m] heigth of the center of the voxel */

        double pond;                /*!< [m] only surface: height of immobilized water */

        Tsoil *Soil;                /*!< soil pointer */
        Tboundary *boundary;        /*!< boundary pointer */
        TlinkedNode up;				/*!< upper link */
        TlinkedNode down;			/*!< lower link */
        TlinkedNode *lateral;       /*!< lateral link */

        TCrit3DnodeExtra* extra;    /*!< extra variables for heat and solutes */

        bool isSurface;
    };


    struct TmatrixElement
    {
        long index;
        double val;
    };


    struct Tbalance
    {
        double storageWater;
        double sinkSourceWater;
        double waterMBE, waterMBR;

        double storageHeat;
        double sinkSourceHeat;
        double heatMBE = 0.0;
        double heatMBR = 0.0;
    };

    struct Tculvert
    {
		long index = NOLINK;
		double width;				/*!< [m] */
		double height;				/*!< [m] */
        double roughness;			/*!< [s m-1/3] Manning roughness */
		double slope;				/*!< [-] */
    };

    extern TCrit3DStructure myStructure;
    extern TParameters myParameters;
    extern TCrit3Dnode *nodeList;
    extern TmatrixElement **A;
    extern Tculvert myCulvert;
    extern double *b, *C, *X, *X1;
    extern double *invariantFlux;         // array accessorio per flussi avvettivi e latenti
    extern double CourantWater;

    extern Tbalance balanceCurrentTimeStep, balancePreviousTimeStep, balanceCurrentPeriod, balanceWholePeriod;

#endif // SOILFLUXES3DTYPES
