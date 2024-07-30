#ifndef LANDUNIT_H
#define LANDUNIT_H

    #include <QString>
    #include <vector>
    class QSqlDatabase;

    class Crit3DLandUnit
    {
        public:
            int id;

            QString name;
            QString description;
            QString idCrop;
            QString idLandUse;

            double roughness;            // Gauckler–Manning roughness coefficient [s m-1/3]
            double pond;                 // surface pond (immobilized water) [m]

            Crit3DLandUnit();
    };

    bool loadLandUnitList(const QSqlDatabase &dbCrop, std::vector<Crit3DLandUnit> &landUnitList, QString &errorStr);


#endif // LANDUNIT_H
