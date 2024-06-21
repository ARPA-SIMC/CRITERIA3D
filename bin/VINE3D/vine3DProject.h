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
        QString fieldMapName;

        bool computeDiseases;

        gis::Crit3DRasterGrid dataRaster;
        gis::Crit3DRasterGrid modelCaseIndexMap;

        std::vector <TVineCultivar> cultivar;
        std::vector <TtrainingSystem> trainingSystems;

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

        bool loadVine3DSettings();
        bool loadVine3DProjectSettings(QString projectFile);

        bool loadFieldsProperties();
        bool loadDBPoints();
        bool loadGrapevineParameters();
        bool loadTrainingSystems();

        bool loadFieldBook();
        float findSoilMaxDepth();
        soil::Crit3DSoil *loadHorizons(int idSoil, QString soil_code);

        void initializeVine3DProject();
        void clearVine3DProject();
        bool loadVine3DProject(QString myFileName);
        bool openVine3DDatabase(QString fileName);

        int queryFieldPoint(double x, double y);
        bool loadFieldShape();
        bool loadFieldMap(QString myFileName);

        bool readFieldQuery(QSqlQuery &myQuery, int &idField, Crit3DLanduse &landuse, int &vineIndex, int &trainingIndex,
                            int &soilIndex, float &maxLaiGrass,  float &maxIrrigationRate);
        bool setField(int fieldIndex, int fieldId, Crit3DLanduse landuse, int soilIndex, int vineIndex, int trainingIndex,
                            float maxLaiGrass,  float maxIrrigationRate);
        bool getFieldBookIndex(int firstIndex, QDate myQDate, int fieldIndex, int* outputIndex);

        int getAggregatedVarCode(int rawVarCode);
        bool getMeteoVarIndexRaw(meteoVariable myVar, int *nrVarIndices, int **varIndices);

        bool loadObsDataHourlyVar(int indexPoint, meteoVariable myVar, QDate d1, QDate d2, QString tableName, bool useAggrCodes);
        bool loadObsDataAllPointsVar(meteoVariable myVar, QDate d1, QDate d2);

        bool isMeteoDataLoaded(const Crit3DTime& myTimeIni, const Crit3DTime& myTimeFin);
        float meteoDataConsistency(meteoVariable myVar, const Crit3DTime& myTimeIni, const Crit3DTime& myTimeFin);

        //bool loadObsDataSubHourly(int indexPoint, meteoVariable myVar, QDateTime d1, QDateTime d2, QString tableName);
        //bool loadObsDataHourly(int indexPoint, QDate d1, QDate d2, QString tableName, bool useAggrCodes);
        //bool loadObsDataFilled(QDateTime firstTime, QDateTime lastTime);
         //bool loadObsDataAllPoints(QDate d1, QDate d2, bool showInfo);
        void findVine3DLastMeteoDate();

        bool loadStates(QDate myDate);
        bool saveStateAndOutput(QDate myDate);

        int getIndexPointFromId(QString myId);

        float getTimeStep();

        int getModelCaseIndex(unsigned row, unsigned col);

        bool isVineyard(unsigned row, unsigned col);

        int getVine3DSoilIndex(long row, long col);

        bool setVine3DSoilIndexMap();
        bool computeVine3DWaterSinkSource();

        soil::Crit3DHorizon* getSoilHorizon(long row, long col, int layer);

        bool runModels(QDateTime firstTime, QDateTime lastTime, bool saveOutput);

        bool executeVine3DCommand(QStringList argumentList, bool* isCommandFound);
    };

#endif // PROJECT_H
