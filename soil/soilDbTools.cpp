#include "soil.h"
#include "soilDbTools.h"
#include "commonConstants.h"
#include "utilities.h"

#include <math.h>

#include <QSqlDatabase>
#include <QSqlQuery>
#include <QSqlError>
#include <QUuid>
#include <QVariant>
#include <QFile>


bool openDbSoil(const QString &dbSoilName, QSqlDatabase &dbSoil, QString &errorStr)
{
    if (! QFile(dbSoilName).exists())
    {
        errorStr = "Soil database doesn't exist:\n" + dbSoilName;
        return false;
    }

    dbSoil = QSqlDatabase::addDatabase("QSQLITE", QUuid::createUuid().toString());
    dbSoil.setDatabaseName(dbSoilName);

    if (! dbSoil.open())
    {
       errorStr = "Connection with database fail";
       return false;
    }

    return true;
}


bool loadGeotechnicsParameters(const QSqlDatabase &dbSoil, std::vector<soil::Crit3DGeotechnicsClass> &geotechnicsClassList, QString &errorStr)
{
    QString queryString = "SELECT id_class, effective_cohesion, friction_angle ";
    queryString        += "FROM geotechnics ORDER BY id_class";

    QSqlQuery query = dbSoil.exec(queryString);
    if (query.lastError().text() != "")
    {
        errorStr = query.lastError().text();
        return false;
    }

    query.last();
    int tableSize = query.at() + 1;     // SQLITE doesn't support SIZE

    if (tableSize == 0)
    {
        errorStr = "Table geotechnics: missing data.";
        return false;
    }
    else if (tableSize != 18)
    {
        errorStr = "Table geotechnics: wrong number of soil classes (must be 18).";
        return false;
    }

    query.first();
    do
    {
        bool isOk;
        int id = query.value("id_class").toInt(&isOk);
        if (! isOk)
        {
            errorStr = "Table geotechnics: \nWrong id_class: " + query.value("id_class").toString();
            return false;
        }

        getValue(query.value("effective_cohesion"), &(geotechnicsClassList[id].effectiveCohesion));     // [kPa]
        getValue(query.value("friction_angle"), &(geotechnicsClassList[id].frictionAngle));             // [°]
    }
    while (query.next());

    return true;
}


bool loadVanGenuchtenParameters(const QSqlDatabase &dbSoil, std::vector<soil::Crit3DTextureClass> &textureClassList, QString &errorStr)
{
    QString queryString = "SELECT id_texture, texture, alpha, n, he, theta_r, theta_s, k_sat, l ";
    queryString        += "FROM van_genuchten ORDER BY id_texture";

    QSqlQuery query = dbSoil.exec(queryString);
    query.last();
    int tableSize = query.at() + 1;     //SQLITE doesn't support SIZE

    if (tableSize == 0)
    {
        errorStr = "Table van_genuchten\n" + query.lastError().text();
        return false;
    }
    else if (tableSize != 12)
    {
        errorStr = "Table van_genuchten: wrong number of soil textures (must be 12)";
        return false;
    }

    //read values
    int id, j;
    float myValue;
    query.first();
    do
    {
        bool isOk;
        id = query.value(0).toInt(&isOk);
        if (! isOk)
        {
            errorStr = "Table van_genuchten: \nWrong ID: " + query.value(0).toString();
            return false;
        }

        //check data
        for (j = 2; j <= 8; j++)
            if (! getValue(query.value(j), &myValue))
            {
                errorStr = "Table van_genuchten: missing data in soil texture:" + QString::number(id);
                return false;
            }

        textureClassList[id].classNameUSDA = query.value(1).toString().toStdString();
        textureClassList[id].vanGenuchten.alpha = query.value(2).toDouble();    //[kPa^-1]
        textureClassList[id].vanGenuchten.n = query.value(3).toDouble();
        textureClassList[id].vanGenuchten.he = query.value(4).toDouble();       //[kPa]

        double m = 1 - 1 / textureClassList[id].vanGenuchten.n;
        textureClassList[id].vanGenuchten.m = m;
        textureClassList[id].vanGenuchten.sc = pow(1.0 + pow(textureClassList[id].vanGenuchten.alpha
                                        * textureClassList[id].vanGenuchten.he, textureClassList[id].vanGenuchten.n), -m);

        textureClassList[id].vanGenuchten.thetaR = query.value(5).toDouble();

        //reference theta at saturation
        textureClassList[id].vanGenuchten.refThetaS = query.value(6).toDouble();
        textureClassList[id].vanGenuchten.thetaS = textureClassList[id].vanGenuchten.refThetaS;

        textureClassList[id].waterConductivity.kSat = query.value(7).toDouble();
        textureClassList[id].waterConductivity.l = query.value(8).toDouble();

    } while(query.next());

    return true;
}


bool loadDriessenParameters(const QSqlDatabase &dbSoil, std::vector<soil::Crit3DTextureClass> &textureClassList, QString &errorStr)
{
    QString queryString = "SELECT id_texture, k_sat, grav_conductivity, max_sorptivity";
    queryString += " FROM driessen ORDER BY id_texture";

    QSqlQuery query = dbSoil.exec(queryString);
    query.last();
    int tableSize = query.at() + 1;     //SQLITE doesn't support SIZE

    if (tableSize == 0)
    {
        errorStr = "Table soil_driessen\n" + query.lastError().text();
        return false;
    }
    else if (tableSize != 12)
    {
        errorStr = "Table soil_driessen: wrong number of soil textures (must be 12)";
        return false;
    }

    //read values
    int id, j;
    float myValue;
    query.first();
    do
    {
        id = query.value(0).toInt();
        //check data
        for (j = 0; j <= 3; j++)
            if (! getValue(query.value(j), &myValue))
            {
                errorStr = "Table soil_driessen: missing data in soil texture:" + QString::number(id);
                return false;
            }

        textureClassList[id].Driessen.k0 = query.value("k_sat").toDouble();
        textureClassList[id].Driessen.gravConductivity = query.value("grav_conductivity").toDouble();
        textureClassList[id].Driessen.maxSorptivity = query.value("max_sorptivity").toDouble();

    } while(query.next());

    return true;
}


bool loadSoilInfo(const QSqlDatabase &dbSoil, const QString &soilCode, soil::Crit3DSoil &mySoil, QString &errorStr)
{
    if (soilCode.isEmpty())
    {
        errorStr = "soilCode missing";
        return false;
    }

    QSqlQuery qry(dbSoil);
    qry.prepare( "SELECT * FROM soils WHERE soil_code = :soil_code");
    qry.bindValue(":soil_code", soilCode);

    if( !qry.exec() )
    {
        errorStr = qry.lastError().text();
        return false;
    }
    else
    {
        QString name;
        if (qry.next())
        {
            getValue(qry.value("name"), &name);
            mySoil.name = name.toStdString();
            mySoil.code = soilCode.toStdString();
            return true;
        }
        else
        {
            errorStr = "soilCode not found";
            return false;
        }
    }
}


bool loadSoilData(const QSqlDatabase &dbSoil, const QString &soilCode, soil::Crit3DSoil &mySoil, QString &errorStr)
{
    QString queryString = "SELECT * FROM horizons ";
    queryString += "WHERE soil_code='" + soilCode + "' ORDER BY horizon_nr";

    QSqlQuery query = dbSoil.exec(queryString);
    query.last();

    if (! query.isValid())
    {
        if (query.lastError().type() != QSqlError::NoError)
        {
            errorStr = "dbSoil error: "+ query.lastError().text();
            return false;
        }
        else
        {
            // missing data
            mySoil.initialize(soilCode.toStdString(), 0);
            errorStr = "soil_code:" + soilCode + " has no horizons.";
            return false;
        }
    }

    int nrHorizons = query.at() + 1;     // SQLITE doesn't support SIZE
    mySoil.initialize(soilCode.toStdString(), nrHorizons);

    unsigned int i = 0;
    double sand, silt, clay;
    double organicMatter, coarseFragments, lowerDepth, upperDepth, bulkDensity, theta_sat, ksat;

    query.first();
    do
    {
        // horizon number
        mySoil.horizon[i].dbData.horizonNr = query.value("horizon_nr").toInt();

        // upper and lower depth [cm]
        getValue(query.value("upper_depth"), &upperDepth);
        getValue(query.value("lower_depth"), &lowerDepth);
        mySoil.horizon[i].dbData.upperDepth = upperDepth;
        mySoil.horizon[i].dbData.lowerDepth = lowerDepth;

        // sand silt clay [%]
        getValue(query.value("sand"), &sand);
        getValue(query.value("silt"), &silt);
        getValue(query.value("clay"), &clay);
        mySoil.horizon[i].dbData.sand = sand;
        mySoil.horizon[i].dbData.silt = silt;
        mySoil.horizon[i].dbData.clay = clay;

        // coarse fragments and organic matter [%]
        getValue(query.value("coarse_fragment"), &coarseFragments);
        getValue(query.value("organic_matter"), &organicMatter);
        mySoil.horizon[i].dbData.coarseFragments = coarseFragments;
        mySoil.horizon[i].dbData.organicMatter = organicMatter;

        // bulk density [g/cm3]
        getValue(query.value("bulk_density"), &bulkDensity);
        mySoil.horizon[i].dbData.bulkDensity = bulkDensity;

        // theta sat [m3/m3]
        getValue(query.value("theta_sat"), &theta_sat);
        mySoil.horizon[i].dbData.thetaSat = theta_sat;

        // saturated conductivity [cm day-1]
        getValue(query.value("k_sat"), &ksat);
        mySoil.horizon[i].dbData.kSat = ksat;

        // NEW fields for soil stability, not present in old databases
        QList<QString> fieldList = getFields(query);

        double value = NODATA;
        if (fieldList.contains("effective_cohesion"))
            if(getValue(query.value("effective_cohesion"), &value))
                mySoil.horizon[i].dbData.effectiveCohesion = value;

        if (fieldList.contains("friction_angle"))
            if(getValue(query.value("friction_angle"), &value))
                mySoil.horizon[i].dbData.frictionAngle = value;

        i++;

    } while(query.next());

    query.clear();

    // Read water retention data
    queryString = "SELECT * FROM water_retention ";
    queryString += "WHERE soil_code='" + soilCode + "' ORDER BY horizon_nr, water_potential";
    query = dbSoil.exec(queryString);

    query.last();
    int nrData = query.at() + 1;     //SQLITE doesn't support SIZE
    if (nrData <= 0) return true;

    soil::Crit3DWaterRetention waterRetention;
    query.first();
    do
    {
        unsigned int horizonNr = unsigned(query.value("horizon_nr").toInt());
        if (horizonNr > 0 && horizonNr <= mySoil.nrHorizons)
        {
            // TODO: check data
            waterRetention.water_potential = query.value("water_potential").toDouble();  // [kPa]
            waterRetention.water_content = query.value("water_content").toDouble();      // [m3 m-3]

            i = horizonNr-1;
            mySoil.horizon[i].dbData.waterRetention.push_back(waterRetention);
        }
    } while(query.next());

    query.clear();

    return true;
}


bool loadSoil(const QSqlDatabase &dbSoil, const QString &soilCode, soil::Crit3DSoil &mySoil,
              const std::vector<soil::Crit3DTextureClass> &textureClassList,
              const std::vector<soil::Crit3DGeotechnicsClass> &geotechnicsClassList,
              const soil::Crit3DFittingOptions &fittingOptions, QString& errorStr)
{
    if (!loadSoilInfo(dbSoil, soilCode, mySoil, errorStr))
    {
        return false;
    }
    if (!loadSoilData(dbSoil, soilCode, mySoil, errorStr))
    {
        return false;
    }

    errorStr = "";
    bool isFirstError = true;
    int firstWrongIndex = NODATA;
    for (unsigned int i = 0; i < mySoil.nrHorizons; i++)
    {
        std::string horizonError;
        if (! soil::setHorizon(mySoil.horizon[i], textureClassList, geotechnicsClassList, fittingOptions, horizonError))
        {
            if (isFirstError)
            {
                firstWrongIndex = i;
                isFirstError = false;
            }
            else
            {
                if (horizonError != "")
                    errorStr += "\n";
            }

            if (horizonError != "")
            {
                errorStr += "soil_code: " + soilCode
                        + " horizon nr." + QString::number(mySoil.horizon[i].dbData.horizonNr)
                        + " " + QString::fromStdString(horizonError);
            }
        }
    }

    // check total depth
    // errors on the last horizon is tolerated (bedrock)
    if (mySoil.nrHorizons > 0)
    {
        if (firstWrongIndex != NODATA)
        {
            if (mySoil.nrHorizons == 1 || firstWrongIndex == 0)
            {
                return false;
            }
            else
            {
                mySoil.nrHorizons = firstWrongIndex;
                mySoil.horizon.resize(mySoil.nrHorizons);
            }
        }

        mySoil.totalDepth = mySoil.horizon[mySoil.nrHorizons-1].lowerDepth;

    }
    else
    {
        mySoil.totalDepth = 0;
        errorStr = "soil_code: " + soilCode + " has no horizons.";
    }

    return true;
}


bool updateSoilData(const QSqlDatabase &dbSoil, const QString &soilCode, soil::Crit3DSoil &mySoil, QString &errorStr)
{
    QSqlQuery qry(dbSoil);
    if (soilCode.isEmpty())
    {
        errorStr = "soilCode missing";
        return false;
    }

    // check slopeStability
    bool isSlopeStability = false;
    qry.prepare("SELECT * FROM horizons WHERE soil_code = :soil_code");
    qry.bindValue(":soil_code", soilCode);
    qry.exec();
    QList<QString> fieldList = getFields(qry);
    if (fieldList.contains("effective_cohesion") && fieldList.contains("friction_angle"))
    {
        isSlopeStability = true;
    }

    // delete all row from table horizons of soil:soilCode
    qry.prepare( "DELETE FROM horizons WHERE soil_code = :soil_code");
    qry.bindValue(":soil_code", soilCode);

    if( !qry.exec() )
    {
        errorStr = qry.lastError().text();
        return false;
    }

    // insert new rows
    if (isSlopeStability)
    {
        qry.prepare( "INSERT INTO horizons (soil_code, horizon_nr, upper_depth, lower_depth, coarse_fragment, organic_matter,"
                " sand, silt, clay, bulk_density, theta_sat, k_sat, effective_cohesion, friction_angle)"
                " VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)" );
    }
    else
    {
        qry.prepare( "INSERT INTO horizons (soil_code, horizon_nr, upper_depth, lower_depth, coarse_fragment, organic_matter,"
                    " sand, silt, clay, bulk_density, theta_sat, k_sat)"
                    " VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)" );
    }

    QVariantList soil_code;
    QVariantList horizon_nr;
    QVariantList upper_depth;
    QVariantList lower_depth;
    QVariantList coarse_fragment;
    QVariantList organic_matter;
    QVariantList sand;
    QVariantList silt;
    QVariantList clay;
    QVariantList bulk_density;
    QVariantList theta_sat;
    QVariantList k_sat;
    QVariantList effective_cohesion;
    QVariantList friction_angle;

    for (unsigned int i=0; i < mySoil.nrHorizons; i++)
    {
        soil_code << soilCode;
        horizon_nr << i+1;
        upper_depth << mySoil.horizon[i].dbData.upperDepth;
        lower_depth << mySoil.horizon[i].dbData.lowerDepth;
        sand << mySoil.horizon[i].dbData.sand;
        silt << mySoil.horizon[i].dbData.silt;
        clay << mySoil.horizon[i].dbData.clay;

        if (mySoil.horizon[i].dbData.coarseFragments != NODATA)
            coarse_fragment << mySoil.horizon[i].dbData.coarseFragments;
        else
            coarse_fragment << "";

        if (mySoil.horizon[i].dbData.organicMatter != NODATA)
            organic_matter << mySoil.horizon[i].dbData.organicMatter;
        else
            organic_matter << "";

        if (mySoil.horizon[i].dbData.bulkDensity != NODATA)
            bulk_density << mySoil.horizon[i].dbData.bulkDensity;
        else
            bulk_density << "";

        if (mySoil.horizon[i].dbData.thetaSat != NODATA)
            theta_sat << mySoil.horizon[i].dbData.thetaSat;
        else
            theta_sat << "";

        if (mySoil.horizon[i].dbData.kSat != NODATA)
            k_sat << mySoil.horizon[i].dbData.kSat;
        else
            k_sat << "";

        if (mySoil.horizon[i].dbData.effectiveCohesion != NODATA)
            effective_cohesion << mySoil.horizon[i].dbData.effectiveCohesion;
        else
            effective_cohesion << "";

        if (mySoil.horizon[i].dbData.frictionAngle != NODATA)
            friction_angle << mySoil.horizon[i].dbData.frictionAngle;
        else
            friction_angle << "";
    }

    qry.addBindValue(soil_code);
    qry.addBindValue(horizon_nr);
    qry.addBindValue(upper_depth);
    qry.addBindValue(lower_depth);
    qry.addBindValue(coarse_fragment);
    qry.addBindValue(organic_matter);
    qry.addBindValue(sand);
    qry.addBindValue(silt);
    qry.addBindValue(clay);
    qry.addBindValue(bulk_density);
    qry.addBindValue(theta_sat);
    qry.addBindValue(k_sat);
    if (isSlopeStability)
    {
        qry.addBindValue(effective_cohesion);
        qry.addBindValue(friction_angle);
    }

    if( !qry.execBatch() )
    {
        errorStr = qry.lastError().text();
        return false;
    }
    else
        return true;
}


bool updateWaterRetentionData(QSqlDatabase &dbSoil, const QString &soilCode, soil::Crit3DSoil &mySoil, int horizon, QString &errorStr)
{
    QSqlQuery qry(dbSoil);
    if (soilCode.isEmpty())
    {
        errorStr = "soilCode missing";
        return false;
    }

    // delete all row from table horizons of soil:soilCode
    qry.prepare( "DELETE FROM water_retention WHERE soil_code = :soil_code AND horizon_nr = :horizon_nr");
    qry.bindValue(":soil_code", soilCode);
    qry.bindValue(":horizon_nr", horizon);

    if( !qry.exec() )
    {
        errorStr = qry.lastError().text();
        return false;
    }

    // insert new rows
    qry.prepare( "INSERT INTO water_retention (soil_code, horizon_nr, water_potential, water_content)"
                                              " VALUES (?, ?, ?, ?)" );

    QVariantList soil_code;
    QVariantList horizon_nr;
    QVariantList water_potential;
    QVariantList water_content;

    unsigned int horizon_index = unsigned(horizon-1);
    for (unsigned int i=0; i < mySoil.horizon[horizon_index].dbData.waterRetention.size(); i++)
    {
        soil_code << soilCode;
        horizon_nr << horizon;
        water_potential << mySoil.horizon[horizon_index].dbData.waterRetention[i].water_potential;
        water_content << mySoil.horizon[horizon_index].dbData.waterRetention[i].water_content;
    }

    qry.addBindValue(soil_code);
    qry.addBindValue(horizon_nr);
    qry.addBindValue(water_potential);
    qry.addBindValue(water_content);

    if( !qry.execBatch() )
    {
        errorStr = qry.lastError().text();
        return false;
    }
    else
        return true;
}


bool insertSoilData(QSqlDatabase &dbSoil, int soilID, const QString &soilCode, const QString &soilName, const QString &soilInfo, QString &errorStr)
{
    QSqlQuery qry(dbSoil);
    if (soilID == NODATA)
    {
        errorStr = "soilID missing";
        return false;
    }
    if (soilCode.isEmpty())
    {
        errorStr = "soilCode missing";
        return false;
    }
    if (soilName.isEmpty())
    {
        errorStr = "soilName missing";
        return false;
    }

    qry.prepare( "INSERT INTO soils (id_soil, soil_code, name, info) VALUES (:id_soil, :soil_code, :name, :info)" );
    qry.bindValue(":id_soil", soilID);
    qry.bindValue(":soil_code", soilCode);
    qry.bindValue(":name", soilName);
    qry.bindValue(":info", soilInfo);

    if( !qry.exec() )
    {
        errorStr = "Insert new soil failed:\n" + qry.lastError().text();
        return false;
    }

    return true;
}


bool deleteSoilData(QSqlDatabase &dbSoil, const QString &soilCode, QString &errorStr)
{
    QSqlQuery qry(dbSoil);
    if (soilCode.isEmpty())
    {
        errorStr = "soilCode missing";
        return false;
    }

    // delete all row from table horizons of soil:soilCode
    qry.prepare( "DELETE FROM horizons WHERE soil_code = :soil_code");
    qry.bindValue(":soil_code", soilCode);

    if( !qry.exec() )
    {
        errorStr = qry.lastError().text();
        return false;
    }

    // delete all row from table soils of soil:soilCode
    qry.prepare( "DELETE FROM soils WHERE soil_code = :soil_code");
    qry.bindValue(":soil_code", soilCode);

    if( !qry.exec() )
    {
        errorStr = qry.lastError().text();
        return false;
    }

    return true;
}


QString getIdSoilString(const QSqlDatabase &dbSoil, int idSoilNumber, QString &errorStr)
{
    errorStr = "";
    QString queryString = "SELECT * FROM soils WHERE id_soil='" + QString::number(idSoilNumber) +"'";

    QSqlQuery query = dbSoil.exec(queryString);
    query.last();

    if (! query.isValid())
    {
        errorStr = query.lastError().text();
        return "";
    }

    QString idSoilStr;
    getValue(query.value("soil_code"), &idSoilStr);

    return idSoilStr;
}


int getIdSoilNumeric(const QSqlDatabase &dbSoil, QString soilCode, QString &errorStr)
{
    errorStr = "";
    QString queryString = "SELECT * FROM soils WHERE soil_code=" + soilCode;

    QSqlQuery query = dbSoil.exec(queryString);
    query.last();

    if (! query.isValid())
    {
        errorStr = query.lastError().text();
        return NODATA;
    }

    int idSoilNumeric;
    getValue(query.value("id_soil"), &idSoilNumeric);

    return idSoilNumeric;
}


bool getSoilList(const QSqlDatabase &dbSoil, QList<QString> &soilList, QString &errorStr)
{
    // query soil list
    QString queryString = "SELECT DISTINCT soil_code FROM soils ORDER BY soil_code";
    QSqlQuery query = dbSoil.exec(queryString);

    query.first();
    if (! query.isValid())
    {
        errorStr = query.lastError().text();
        return false;
    }

    QString soilCode;
    do
    {
        getValue(query.value("soil_code"), &soilCode);
        if (soilCode != "")
        {
            soilList.append(soilCode);
        }
    }
    while(query.next());

    return true;
}


bool loadAllSoils(const QString &dbSoilName, std::vector <soil::Crit3DSoil> &soilList,
                  std::vector<soil::Crit3DTextureClass> &textureClassList,
                  std::vector<soil::Crit3DGeotechnicsClass> &geotechnicsClassList,
                  const soil::Crit3DFittingOptions &fittingOptions, QString &errorStr)
{
    QSqlDatabase dbSoil;
    if (! openDbSoil(dbSoilName, dbSoil, errorStr)) return false;

    bool result = loadAllSoils(dbSoil, soilList, textureClassList, geotechnicsClassList, fittingOptions, errorStr);
    dbSoil.close();

    return result;
}


bool loadAllSoils(const QSqlDatabase &dbSoil, std::vector <soil::Crit3DSoil> &soilList,
                  std::vector<soil::Crit3DTextureClass> &textureClassList,
                  std::vector<soil::Crit3DGeotechnicsClass> &geotechnicsClassList,
                  const soil::Crit3DFittingOptions &fittingOptions, QString& errorStr)
{
    soilList.clear();

    if (! loadVanGenuchtenParameters(dbSoil, textureClassList, errorStr))
        return false;

    // geotechnics table is not mandatory
    loadGeotechnicsParameters(dbSoil, geotechnicsClassList, errorStr);
    errorStr = "";

    // query soil list
    QString queryString = "SELECT id_soil, soil_code, name FROM soils";
    QSqlQuery query = dbSoil.exec(queryString);

    query.first();
    if (! query.isValid())
    {
        errorStr = query.lastError().text();
        return false;
    }

    // load soil properties
    QString soilCode;
    int idSoil;
    QString soilName;
    QString wrongSoilsStr = "";

    do
    {
        getValue(query.value("id_soil"), &idSoil);
        getValue(query.value("soil_code"), &soilCode);
        getValue(query.value("name"), &soilName);
        if (idSoil != NODATA && soilCode != "")
        {
            soil::Crit3DSoil mySoil;
            if (loadSoil(dbSoil, soilCode, mySoil, textureClassList, geotechnicsClassList, fittingOptions, errorStr))
            {
                mySoil.id = idSoil;
                mySoil.code = soilCode.toStdString();
                mySoil.name = soilName.toStdString();
                soilList.push_back(mySoil);
            }
            if (errorStr != "")
            {
                wrongSoilsStr += soilCode + ": " + errorStr + "\n";
            }
        }
    }
    while(query.next());

    if (soilList.size() == 0)
    {
       errorStr += "\nMissing soil properties";
       return false;
    }
    else if (wrongSoilsStr != "")
    {
        errorStr = wrongSoilsStr;
    }

    return true;
}
