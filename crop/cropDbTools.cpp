#include "cropDbTools.h"
#include "commonConstants.h"
#include "crop.h"
#include "utilities.h"

#include <QString>
#include <QSqlQuery>
#include <QSqlError>
#include <QUuid>
#include <QVariant>

bool openDbCrop(QString dbName, QSqlDatabase* dbCrop, QString* error)
{

    *dbCrop = QSqlDatabase::addDatabase("QSQLITE", QUuid::createUuid().toString());
    dbCrop->setDatabaseName(dbName);

    if (!dbCrop->open())
    {
       *error = "Connection with database fail";
       return false;
    }

    return true;
}

bool getCropNameList(QSqlDatabase* dbCrop, QStringList* cropNameList, QString* error)
{
    // query crop list
    QString queryString = "SELECT crop_name FROM crop";
    QSqlQuery query = dbCrop->exec(queryString);

    query.first();
    if (! query.isValid())
    {
        *error = query.lastError().text();
        return false;
    }

    QString cropName;
    do
    {
        getValue(query.value("crop_name"), &cropName);
        if (cropName != "")
        {
            cropNameList->append(cropName);
        }
    }
    while(query.next());

    return true;
}

QString getIdCropFromName(QSqlDatabase* dbCrop, QString cropName, QString *myError)
{
    *myError = "";
    QString queryString = "SELECT * FROM crop WHERE crop_name='" + cropName +"'";

    QSqlQuery query = dbCrop->exec(queryString);
    query.last();

    if (! query.isValid())
    {
        *myError = query.lastError().text();
        return "";
    }

    QString idCrop;
    getValue(query.value("id_crop"), &idCrop);

    return idCrop;
}



bool loadCropParameters(QString idCrop, Crit3DCrop* myCrop, QSqlDatabase* dbCrop, QString *myError)
{
    QString idCropString = idCrop;

    QString queryString = "SELECT * FROM crop WHERE id_crop = '" + idCrop + "'";

    QSqlQuery query = dbCrop->exec(queryString);
    query.last();

    if (! query.isValid())
    {
        if (query.lastError().isValid())
            *myError = "Error in reading crop parameters of " + idCropString + "\n" + query.lastError().text();
        else
            *myError = "Missing crop: " + idCropString;
        return(false);
    }

    myCrop->idCrop = idCropString.toStdString();
    myCrop->type = getCropType(query.value("type").toString().toStdString());
    myCrop->plantCycle = query.value("plant_cycle_max_duration").toInt();
    getValue(query.value("sowing_doy"), &(myCrop->sowingDoy));

    // LAI
    myCrop->LAImax = query.value("lai_max").toDouble();
    myCrop->LAImin = query.value("lai_min").toDouble();
    myCrop->LAIcurve_a = query.value("lai_curve_factor_a").toDouble();
    myCrop->LAIcurve_b = query.value("lai_curve_factor_b").toDouble();

    // THERMAL THRESHOLDS
    myCrop->thermalThreshold = query.value("thermal_threshold").toDouble();
    myCrop->upperThermalThreshold = query.value("upper_thermal_threshold").toDouble();
    myCrop->degreeDaysIncrease = query.value("degree_days_lai_increase").toInt();
    myCrop->degreeDaysDecrease = query.value("degree_days_lai_decrease").toInt();
    myCrop->degreeDaysEmergence = query.value("degree_days_emergence").toInt();

    // ROOT
    myCrop->roots.rootShape = root::getRootDistributionType(query.value("root_shape").toInt());
    myCrop->roots.shapeDeformation = query.value("root_shape_deformation").toDouble();
    myCrop->roots.rootDepthMin = query.value("root_depth_zero").toDouble();
    myCrop->roots.rootDepthMax = query.value("root_depth_max").toDouble();
    getValue(query.value("degree_days_root_increase"), &(myCrop->roots.degreeDaysRootGrowth));

    // WATER NEEDS
    myCrop->kcMax = query.value("kc_max").toDouble();
    // [cm]
    if (! getValue(query.value("psi_leaf"), &(myCrop->psiLeaf)))
        myCrop->psiLeaf = 16000;

    myCrop->stressTolerance = query.value("stress_tolerance").toDouble();

    // fraction of Readily Available Water
    if (! getValue(query.value("raw_fraction"), &(myCrop->fRAW)))
    {
        // old version
        if (! getValue(query.value("frac_read_avail_water_max"), &(myCrop->fRAW)))
        {
            *myError = "Missing RAW_fraction for crop: " + idCropString;
            return(false);
        }
    }

    // IRRIGATION
    getValue(query.value("irrigation_shift"), &(myCrop->irrigationShift));
    getValue(query.value("degree_days_start_irrigation"), &(myCrop->degreeDaysStartIrrigation));
    getValue(query.value("degree_days_end_irrigation"), &(myCrop->degreeDaysEndIrrigation));
    getValue(query.value("doy_start_irrigation"), &(myCrop->doyStartIrrigation));
    getValue(query.value("doy_end_irrigation"), &(myCrop->doyEndIrrigation));

    // key value for irrigation
    if (! getValue(query.value("irrigation_volume"), &(myCrop->irrigationVolume)))
        myCrop->irrigationVolume = 0;

    // LAI grass
    if (! getValue(query.value("lai_grass"), &(myCrop->LAIgrass)))
        myCrop->LAIgrass = 0;

    // max surface puddle
    if (! getValue(query.value("max_height_surface_puddle"), &(myCrop->maxSurfacePuddle)))
        myCrop->maxSurfacePuddle = 0;

    return true;
}


QString getCropFromClass(QSqlDatabase* dbCrop, QString cropClassTable, QString cropClassField, QString idCropClass, QString *myError)
{
    *myError = "";
    QString queryString = "SELECT * FROM " + cropClassTable + " WHERE " + cropClassField + " = '" + idCropClass + "'";

    QSqlQuery query = dbCrop->exec(queryString);
    query.last();

    if (! query.isValid())
    {
        if (query.lastError().isValid())
            *myError = query.lastError().text();
        return "";
    }

    QString myCrop;
    getValue(query.value("id_crop"), &myCrop);

    return myCrop;
}


QString getCropFromId(QSqlDatabase* dbCrop, QString cropClassTable, QString cropIdField, int cropId, QString *myError)
{
    *myError = "";
    QString queryString = "SELECT * FROM " + cropClassTable + " WHERE " + cropIdField + " = " + QString::number(cropId);

    QSqlQuery query = dbCrop->exec(queryString);
    query.last();

    if (! query.isValid())
    {
        if (query.lastError().isValid())
            *myError = query.lastError().text();
        return "";
    }

    QString myCrop;
    getValue(query.value("id_crop"), &myCrop);

    return myCrop;
}


float getIrriRatioFromClass(QSqlDatabase* dbCrop, QString cropClassTable, QString cropClassField, QString idCrop, QString *myError)
{
    *myError = "";

    QString queryString = "SELECT irri_ratio FROM " + cropClassTable + " WHERE " + cropClassField + " = '" + idCrop + "'";

    QSqlQuery query = dbCrop->exec(queryString);
    query.last();

    if (! query.isValid())
    {
        if (query.lastError().isValid())
            *myError = query.lastError().text();
        return(NODATA);
    }

    float myRatio = 0;

    if (getValue(query.value("irri_ratio"), &(myRatio)))
        return myRatio;
    else
        return NODATA;
}


float getIrriRatioFromId(QSqlDatabase* dbCrop, QString cropClassTable, QString cropIdField, int cropId, QString *myError)
{
    *myError = "";

    QString queryString = "SELECT irri_ratio FROM " + cropClassTable + " WHERE " + cropIdField + " = " + QString::number(cropId);

    QSqlQuery query = dbCrop->exec(queryString);
    query.last();

    if (! query.isValid())
    {
        if (query.lastError().isValid())
            *myError = query.lastError().text();
        return(NODATA);
    }

    float myRatio = 0;

    if (getValue(query.value("irri_ratio"), &(myRatio)))
        return myRatio;
    else
        return NODATA;
}

bool deleteCropData(QSqlDatabase* dbCrop, QString cropName, QString *error)
{

    QSqlQuery qry(*dbCrop);
    if (cropName.isEmpty())
    {
        *error = "crop_name missing";
        return false;
    }

    // delete all row from table crop of crop:crop_name
    qry.prepare( "DELETE FROM crop WHERE crop_name = :crop_name");
    qry.bindValue(":crop_name", cropName);

    if( !qry.exec() )
    {
        *error = qry.lastError().text();
        return false;
    }
    return true;
}


