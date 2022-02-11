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

    bool checkProxyGridSeries(Crit3DInterpolationSettings* mySettings, const gis::Crit3DRasterGrid &gridBase,
                              std::vector <Crit3DProxyGridSeries> mySeries, QDate myDate);

    bool interpolationRaster(std::vector <Crit3DInterpolationDataPoint> &myPoints, Crit3DInterpolationSettings* mySettings, Crit3DMeteoSettings *meteoSettings,
                            gis::Crit3DRasterGrid* outputGrid, const gis::Crit3DRasterGrid& raster, meteoVariable myVar);


#endif // INTERPOLATIONCMD_H
