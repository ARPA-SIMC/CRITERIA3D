#ifndef VINE3DPROJECT_H
#define VINE3DPROJECT_H

    #ifndef QSTRING_H
        #include <QString>
    #endif

    #ifndef QSQLDATABASE_H
        #include <QSqlDatabase>
    #endif

    #ifndef GRAPEVINE_H
        #include "grapevine.h"
    #endif

    #ifndef METEOMAPS_H
        #include "meteoMaps.h"
    #endif

    #ifndef PLANT_H
        #include "plant.h"
    #endif

    #ifndef WATERBALANCE_H
        #include "waterBalance.h"
    #endif

    struct TVine3DOperation {
        QDate operationDate;
        TfieldOperation operation;
        int idField;
        float quantity;         //irrigation hours, thinning percentage, tartaric acid value
    };


    class Vine3DHourlyMaps
    {
    public:
        gis::Crit3DRasterGrid* mapHourlyIrrigation;

        Vine3DHourlyMaps(const gis::Crit3DRasterGrid& DEM);
        ~Vine3DHourlyMaps();

    };

    class Vine3DProject : public Project3D
    {

    public:
        QString dbVine3DFileName;
        QSqlDatabase dbVine3D;

        Crit3DTime currentTime;

        QString dailyOutputPath;
        QString hourlyOutputPath;

        bool computeDiseases;

        gis::Crit3DRasterGrid dataRaster;

        std::vector <TVineCultivar> cultivar;
        std::vector <TtrainingSystem> trainingSystems;

        std::vector <Crit3DModelCase> inputModelCases;
        std::vector <Crit3DModelCase> modelCases;
        std::vector <TVine3DOperation> fieldBook;

        bool isObsDataLoaded;
        int* varCodes;
        int* aggrVarCodes;
        int nrAggrVar;

        Crit3DQuality qualityParameters;

        Vine3DHourlyMaps* vine3DMapsH;
        Crit3DDailyMeteoMaps* vine3DMapsD;

        Crit3DWaterBalanceMaps* outputWaterBalanceMaps;
        Crit3DStatePlantMaps* statePlantMaps;
        Crit3DOutputPlantMaps* outputPlantMaps;

        TstatePlant statePlant;
        Vine3D_Grapevine grapevine;

        QDate lastDateTransmissivity;

        Vine3DProject();

        void initializeVine3DProject();
        void clearVine3DProject();

        void loadVine3DSettings();

        bool loadFieldsProperties();

        //bool loadDBPoints();
        //void findVine3DLastMeteoDate();
        //int queryFieldPoint(double x, double y);

        bool loadGrapevineParameters();
        bool loadTrainingSystems();

        bool loadFieldBook();

        bool loadVine3DProject(QString projectFileName);
        bool openVine3DDatabase(QString fileName);

        bool initializeGrapevine();

        bool setModelCasesMap();

        bool readFieldQuery(QSqlQuery &myQuery, int &idField, GrapevineLanduse &landuse, int &vineIndex, int &trainingIndex, float &maxLaiGrass,  float &maxIrrigationRate);
        bool setField(int fieldIndex, int fieldId, GrapevineLanduse landuse, int soilIndex, int vineIndex, int trainingIndex,
                            float maxLaiGrass,  float maxIrrigationRate);
        bool getFieldBookIndex(int firstIndex, QDate myQDate, int fieldIndex, int* outputIndex);

        //int getAggregatedVarCode(int rawVarCode);
        bool getMeteoVarIndexRaw(meteoVariable myVar, int *nrVarIndices, int **varIndices);

        bool loadObsDataHourlyVar(int indexPoint, meteoVariable myVar, QDate d1, QDate d2, QString tableName, bool useAggrCodes);
        bool loadObsDataAllPointsVar(meteoVariable myVar, QDate d1, QDate d2);

        bool loadStates(QDate myDate);
        bool saveStateAndOutput(QDate myDate);

        float getTimeStep();

        int getModelCaseIndex(int row, int col);

        bool isVineyard(unsigned row, unsigned col);

        bool computeVine3DWaterSinkSource();

        bool runModels(QDateTime firstTime, QDateTime lastTime, bool saveOutput);

        bool vine3dShell();
        bool vine3dBatch(QString scriptFileName);
        bool executeCommand(QStringList argumentList);
        bool executeVine3DCommand(QStringList argumentList, bool* isCommandFound);
        void cmdVine3dList();
        bool cmdOpenVine3DProject(QStringList argumentList);
        bool cmdRunModels(QStringList argumentList);

        void resetWaterBalanceMap();
        void updateWaterBalanceMaps();
    };


#endif // PROJECT_H
