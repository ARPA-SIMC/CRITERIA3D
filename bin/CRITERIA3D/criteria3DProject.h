#ifndef CRITERIA3DPROJECT_H
#define CRITERIA3DPROJECT_H

    #ifndef SOIL_H
        #include "soil.h"
    #endif
    #ifndef PROJECT_H
        #include "project.h"
    #endif
    #ifndef PROJECT3D_H
        #include "project3D.h"
    #endif

    #include <QString>

    class Crit3DProject : public Project3D
    {
    private:
        void clearCriteria3DProject();

        bool setSoilIndexMap();

    public:
        // same header of DEM
        gis::Crit3DRasterGrid soilMap;
        gis::Crit3DRasterGrid soilUseMap;

        Crit3DProject();

        bool initializeCriteria3DModel();

        bool loadCriteria3DProject(QString myFileName);
        bool loadCriteria3DSettings();
        bool loadSoilMap(QString fileName);

        double getSoilVar(int soilIndex, int layerIndex, soil::soilVariable myVar);
        double* getSoilVarProfile(int row, int col, soil::soilVariable myVar);

        int getCrit3DSoilId(double x, double y);
        int getCrit3DSoilIndex(double x, double y);
        QString getCrit3DSoilCode(double x, double y);

        bool computeAllMeteoMaps(const QDateTime& myTime, bool showInfo);

        void setAllHourlyMeteoMapsComputed(bool value);

        bool saveDailyOutput(QDate myDate, const QString& outputPathHourly);
    };


#endif // CRITERIA3DPROJECT_H
