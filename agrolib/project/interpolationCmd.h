#ifndef INTERPOLATIONCMD_H
#define INTERPOLATIONCMD_H

    #ifndef GIS_H
        #include "gis.h"
    #endif
    #ifndef METEO_H
        #include "meteo.h"
    #endif
    #ifndef INTERPOLATIONSETTINGS_H
        #include "interpolationSettings.h"
    #endif
    #ifndef INTERPOLATIONPOINT_H
        #include "interpolationPoint.h"
    #endif
    #ifndef QSTRING_H
        #include <QString>
    #endif

    class QDate;

    class Crit3DCrossValidationStatistics {
    private:
        Crit3DTime refTime;
        Crit3DProxyCombination proxyCombination;
        float meanAbsoluteError;
        float rootMeanSquareError;
        float NashSutcliffeEfficiency;
        float meanBiasError;
        float R2;

    public:
        Crit3DCrossValidationStatistics();
        void initialize();

        const Crit3DProxyCombination &getProxyCombination() const;
        void setProxyCombination(const Crit3DProxyCombination &newProxyCombination);
        float getMeanAbsoluteError() const;
        void setMeanAbsoluteError(float newMeanAbsoluteError);
        float getRootMeanSquareError() const;
        void setRootMeanSquareError(float newRootMeanSquareError);
        float getNashSutcliffeEfficiency() const;
        void setNashSutcliffeEfficiency(float newNashSutcliffeEfficiency);
        float getMeanBiasError() const;
        void setMeanBiasError(float newMeanBiasError);
        const Crit3DTime &getRefTime() const;
        void setRefTime(const Crit3DTime &newRefTime);
        float getR2() const;
        void setR2(float newR2);
    };

    class Crit3DProxyGridSeries
    {
    private:
        std::vector <QString> gridName;
        std::vector <int> gridYear;
        QString proxyName;

    public:
        Crit3DProxyGridSeries();
        Crit3DProxyGridSeries(QString name_);

        void initialize();
        void addGridToSeries(QString name_, int year_);
        std::vector<QString> getGridName() const;
        std::vector<int> getGridYear() const;
        QString getProxyName() const;
    };

    bool checkProxyGridSeries(Crit3DInterpolationSettings &interpolationSettings, const gis::Crit3DRasterGrid &gridBase,
                              std::vector <Crit3DProxyGridSeries> mySeries, QDate myDate, QString &errorStr);

    bool interpolationRaster(std::vector <Crit3DInterpolationDataPoint> &dataPoints, Crit3DInterpolationSettings &interpolationSettings,
                             Crit3DMeteoSettings *meteoSettings, gis::Crit3DRasterGrid* outputGrid,
                             gis::Crit3DRasterGrid &raster, meteoVariable variable, bool isParallelComputing);

    bool interpolateProxyGridSeries(const Crit3DProxyGridSeries& mySeries, QDate myDate, const gis::Crit3DRasterGrid& gridBase,
                                    gis::Crit3DRasterGrid *gridOut, QString &errorStr);

    bool topographicIndex(const gis::Crit3DRasterGrid &DEM, std::vector <float> windowWidths, gis::Crit3DRasterGrid& outGrid);

#endif // INTERPOLATIONCMD_H
