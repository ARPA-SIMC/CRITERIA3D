#ifndef RADIATIONSETTINGS_H
#define RADIATIONSETTINGS_H

    #ifndef CRIT3DDATE_H
        #include "crit3dDate.h"
    #endif
    #ifndef RADIATIONDEFINITIONS_H
        #include "radiationDefinitions.h"
    #endif

    class Crit3DRadiationSettings
    {
    private:
        TradiationAlgorithm algorithm;
        TradiationRealSkyAlgorithm realSkyAlgorithm;
        TparameterMode linkeMode;
        TparameterMode albedoMode;
        TlandUse landUse;
        TtiltMode tiltMode;

        bool realSky;
        bool shadowing;
        float linke;
        float albedo;
        float tilt;
        float aspect;
        float clearSky;

        std::vector <float> LinkeMonthly;
        std::vector <float> AlbedoMonthly;

        std::string linkeMapName;
        std::string albedoMapName;
        gis::Crit3DRasterGrid* linkeMap;
        gis::Crit3DRasterGrid* albedoMap;

    public:
        gis::Crit3DGisSettings* gisSettings;

        Crit3DRadiationSettings();
        ~Crit3DRadiationSettings();

        void initialize();
        void setGisSettings(const gis::Crit3DGisSettings *myGisSettings);

        TradiationRealSkyAlgorithm getRealSkyAlgorithm() const;
        void setRealSkyAlgorithm(const TradiationRealSkyAlgorithm &value);
        float getClearSky() const;
        void setClearSky(float value);
        bool getRealSky() const;
        void setRealSky(bool value);
        bool getShadowing() const;
        void setShadowing(bool value);
        float getLinke() const;
        float getLinke(int row, int col) const;
        float getLinke(const gis::Crit3DPoint &myPoint) const;
        void setLinke(float value);
        float getMonthlyLinke(int month);
        float getAlbedo() const;
        float getAlbedo(int row, int col) const;
        float getAlbedo(const gis::Crit3DPoint& myPoint) const;
        void setAlbedo(float value);
        float getTilt() const;
        void setTilt(float value);
        float getAspect() const;
        void setAspect(float value);
        TradiationAlgorithm getAlgorithm() const;
        void setAlgorithm(const TradiationAlgorithm &value);
        TparameterMode getLinkeMode() const;
        void setLinkeMode(const TparameterMode &value);
        TparameterMode getAlbedoMode() const;
        void setAlbedoMode(const TparameterMode &value);
        TtiltMode getTiltMode() const;
        void setTiltMode(const TtiltMode &value);
        TlandUse getLandUse() const;
        void setLandUse(const TlandUse &value);
        gis::Crit3DRasterGrid *getLinkeMap() const;
        void setLinkeMap(gis::Crit3DRasterGrid *value);
        gis::Crit3DRasterGrid *getAlbedoMap() const;
        void setAlbedoMap(gis::Crit3DRasterGrid *value);
        std::string getLinkeMapName() const;
        void setLinkeMapName(const std::string &value);
        std::string getAlbedoMapName() const;
        void setAlbedoMapName(const std::string &value);
        void setLinkeMonthly(std::vector <float> myLinke);
        void setAlbedoMonthly(std::vector<float> myAlbedo);
        std::vector<float> getLinkeMonthly() const;
    } ;

    std::string getKeyStringRadAlgorithm(TradiationAlgorithm value);
    std::string getKeyStringRealSky(TradiationRealSkyAlgorithm value);
    std::string getKeyStringParamMode(TparameterMode value);
    std::string getKeyStringTiltMode(TtiltMode value);
    std::string getKeyStringLandUse(TlandUse value);

#endif // RADIATIONSETTINGS_H
