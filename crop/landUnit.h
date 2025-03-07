#ifndef LANDUNIT_H
#define LANDUNIT_H

    #include <QString>
    #include <vector>
    #include <map>

    class QSqlDatabase;

    enum landUseTypeList {LANDUSE_BARESOIL, LANDUSE_FALLOW, LANDUSE_HERBACEOUS, LANDUSE_HORTICULTURAL, LANDUSE_GRASS,
                      LANDUSE_ORCHARD, LANDUSE_FOREST, LANDUSE_URBAN, LANDUSE_ROAD, LANDUSE_WATERBODIES };

    const std::map<std::string, landUseTypeList> MapLandUseFromString = {
        { "BARE", LANDUSE_BARESOIL },
        { "FALLOW", LANDUSE_FALLOW },
        { "HERBACEOUS", LANDUSE_HERBACEOUS },
        { "HORTI", LANDUSE_HORTICULTURAL },
        { "GRASS", LANDUSE_GRASS },
        { "ORCHARD", LANDUSE_ORCHARD },
        { "FOREST", LANDUSE_FOREST },
        { "URBAN", LANDUSE_URBAN },
        { "ROAD", LANDUSE_ROAD },
        { "WATER", LANDUSE_WATERBODIES },
    };

    class Crit3DLandUnit
    {
        public:
            int id;

            QString name;
            QString description;
            QString idCrop;
            QString idLandUse;
            int landUseType;

            double roughness;            // Gauckler–Manning roughness coefficient [s m-1/3]
            double pond;                 // surface pond (immobilized water) [m]

            Crit3DLandUnit();
    };

    bool loadLandUnitList(const QSqlDatabase &dbCrop, std::vector<Crit3DLandUnit> &landUnitList, QString &errorStr);


#endif // LANDUNIT_H
