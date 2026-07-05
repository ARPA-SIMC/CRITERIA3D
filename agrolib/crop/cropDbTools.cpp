#include "cropDbTools.h"
#include "commonConstants.h"
#include "crop.h"
#include "utilities.h"

#include <QString>
#include <QtSql>


bool openDbCrop(QSqlDatabase &dbCrop, const QString &dbName, QString &errorStr)
{
    dbCrop = QSqlDatabase::addDatabase("QSQLITE", QUuid::createUuid().toString());
    dbCrop.setDatabaseName(dbName);

    if (!dbCrop.open())
    {
       errorStr = "Connection with database fail";
       return false;
    }

    return true;
}


bool loadCropParameters(const QSqlDatabase &dbCrop, QString idCrop, Crit3DCrop &myCrop, QString& errorStr)
{
    myCrop.clear();
    QString idCropString = idCrop;
    QString queryString = "SELECT * FROM crop WHERE id_crop = '" + idCrop + "'";

    QSqlQuery query = dbCrop.exec(queryString);
    query.last();

    if (! query.isValid())
    {
        if (query.lastError().isValid())
            errorStr = "Error in reading crop parameters of " + idCropString + "\n" + query.lastError().text();
        else
            errorStr = "Missing crop: " + idCropString;

        return false;
    }

    myCrop.idCrop = idCropString.toStdString();
    myCrop.name = query.value("crop_name").toString().toStdString();
    myCrop.type = getCropType(query.value("type").toString().toStdString());
    myCrop.plantCycle = query.value("plant_cycle_max_duration").toInt();
    getValue(query.value("sowing_doy"), &(myCrop.sowingDoy));

    // LAI
    myCrop.LAImax = query.value("lai_max").toDouble();
    myCrop.LAImin = query.value("lai_min").toDouble();
    myCrop.LAIcurve_a = query.value("lai_curve_factor_a").toDouble();
    myCrop.LAIcurve_b = query.value("lai_curve_factor_b").toDouble();

    // THERMAL THRESHOLDS
    myCrop.thermalThreshold = query.value("thermal_threshold").toDouble();
    myCrop.upperThermalThreshold = query.value("upper_thermal_threshold").toDouble();
    myCrop.degreeDaysIncrease = query.value("degree_days_lai_increase").toInt();
    myCrop.degreeDaysDecrease = query.value("degree_days_lai_decrease").toInt();
    myCrop.degreeDaysEmergence = query.value("degree_days_emergence").toInt();

    // ROOT
    myCrop.roots.rootShape = root::getRootDistributionType(query.value("root_shape").toInt());
    myCrop.roots.shapeDeformation = query.value("root_shape_deformation").toDouble();
    myCrop.roots.rootDepthMin = query.value("root_depth_zero").toDouble();
    myCrop.roots.rootDepthMax = query.value("root_depth_max").toDouble();

    if (fieldExists(query, "roots_additional_cohesion"))
    {
        getValue(query.value("roots_additional_cohesion"), &(myCrop.roots.rootsAdditionalCohesion));
    }
    else
    {
        // default: no mechanical effect of roots
        myCrop.roots.rootsAdditionalCohesion = 0;
    }

    getValue(query.value("degree_days_root_increase"), &(myCrop.roots.degreeDaysRootGrowth));
    if (myCrop.roots.degreeDaysRootGrowth == NODATA)
    {
        myCrop.roots.degreeDaysRootGrowth = myCrop.degreeDaysIncrease;
    }

    // WATER NEEDS
    myCrop.kcMax = query.value("kc_max").toDouble();
    // [cm]
    if (! getValue(query.value("psi_leaf"), &(myCrop.psiLeaf)))
    {
        // default
        myCrop.psiLeaf = 16000;          // Attenzione Giulia! [cm]
    }

    myCrop.stressTolerance = query.value("stress_tolerance").toDouble();

    // fraction of Readily Available Water
    if (! getValue(query.value("raw_fraction"), &(myCrop.fRAW)))
    {
        // default
        myCrop.fRAW = 0.6;
    }

    // IRRIGATION
    getValue(query.value("irrigation_shift"), &(myCrop.irrigationShift));
    getValue(query.value("degree_days_start_irrigation"), &(myCrop.degreeDaysStartIrrigation));
    getValue(query.value("degree_days_end_irrigation"), &(myCrop.degreeDaysEndIrrigation));
    getValue(query.value("doy_start_irrigation"), &(myCrop.doyStartIrrigation));
    getValue(query.value("doy_end_irrigation"), &(myCrop.doyEndIrrigation));

    // irrigation volume [mm day-1]
    if (! getValue(query.value("irrigation_volume"), &(myCrop.irrigationVolume)))
    {
        // default: no irrigation
        myCrop.irrigationVolume = 0;
    }

    // LAI grass
    if (! getValue(query.value("lai_grass"), &(myCrop.LAIgrass)))
    {
        myCrop.LAIgrass = 0;
    }

    // max surface puddle [mm]
    if (! getValue(query.value("max_height_surface_puddle"), &(myCrop.maxSurfacePuddle)))
    {
        // default: 5 mm
        myCrop.maxSurfacePuddle = 5;
    }

    return true;
}


bool deleteCropData(QSqlDatabase &dbCrop, const QString &cropName, QString& errorStr)
{
    QSqlQuery qry(dbCrop);
    if (cropName.isEmpty())
    {
        errorStr = "crop_name missing";
        return false;
    }

    // delete all row from table crop with crop_name
    qry.prepare( "DELETE FROM crop WHERE crop_name = :crop_name");
    qry.bindValue(":crop_name", cropName);

    if( !qry.exec() )
    {
        errorStr = qry.lastError().text();
        return false;
    }
    return true;
}


bool updateCropLAIparam(QSqlDatabase &dbCrop, const Crit3DCrop &myCrop, QString &errorStr)
{
    QSqlQuery qry(dbCrop);
    qry.prepare( "UPDATE crop SET sowing_doy = :sowing_doy, plant_cycle_max_duration = :max_cycle, lai_min = :lai_min, lai_max = :lai_max, lai_grass = :lai_grass, "
                 "thermal_threshold = :thermal_threshold, upper_thermal_threshold = :upper_thermal_threshold, degree_days_emergence = :degree_days_emergence, "
                 "degree_days_lai_increase = :degree_days_lai_increase, degree_days_lai_decrease = :degree_days_lai_decrease, "
                 "lai_curve_factor_a = :lai_curve_factor_a, lai_curve_factor_b = :lai_curve_factor_b, "
                 "kc_max = :kc_max WHERE id_crop = :id_crop");

    if (myCrop.isBareSoil())
    {
        qry.bindValue(":sowing_doy", "");
        qry.bindValue(":max_cycle", "");
        qry.bindValue(":lai_grass", "");
        qry.bindValue(":lai_min", "");
        qry.bindValue(":lai_max", "");
        qry.bindValue(":thermal_threshold", "");
        qry.bindValue(":upper_thermal_threshold", "");
        qry.bindValue(":degree_days_lai_increase", "");
        qry.bindValue(":degree_days_lai_decrease", "");
        qry.bindValue(":lai_curve_factor_a", "");
        qry.bindValue(":lai_curve_factor_b", "");
        qry.bindValue(":kc_max", "");
    }
    else
    {
        if (myCrop.sowingDoy != NODATA)
        {
            qry.bindValue(":sowing_doy", myCrop.sowingDoy);
            qry.bindValue(":max_cycle", myCrop.plantCycle);
        }
        else
        {
            qry.bindValue(":sowing_doy", "");
            qry.bindValue(":max_cycle", 365);
        }

        if (myCrop.LAIgrass != NODATA)
        {
            qry.bindValue(":lai_grass", myCrop.LAIgrass);
        }
        else
        {
            qry.bindValue(":lai_grass", "");
        }

        if (myCrop.degreeDaysEmergence != 0 && myCrop.degreeDaysEmergence != NODATA)
        {
            qry.bindValue(":degree_days_emergence", myCrop.degreeDaysEmergence);
        }
        else
        {
            qry.bindValue(":degree_days_emergence", "");
        }

        qry.bindValue(":lai_min", myCrop.LAImin);
        qry.bindValue(":lai_max", myCrop.LAImax);

        qry.bindValue(":thermal_threshold", myCrop.thermalThreshold);
        qry.bindValue(":upper_thermal_threshold", myCrop.upperThermalThreshold);

        qry.bindValue(":degree_days_lai_increase", myCrop.degreeDaysIncrease);
        qry.bindValue(":degree_days_lai_decrease", myCrop.degreeDaysDecrease);

        qry.bindValue(":lai_curve_factor_a", myCrop.LAIcurve_a);
        qry.bindValue(":lai_curve_factor_b", myCrop.LAIcurve_b);
        qry.bindValue(":kc_max", myCrop.kcMax);
    }

    qry.bindValue(":id_crop", QString::fromStdString(myCrop.idCrop));

    if( ! qry.exec() )
    {
        errorStr = qry.lastError().text();
        return false;
    }
    return true;
}


bool updateCropRootparam(QSqlDatabase &dbCrop, const Crit3DCrop &myCrop, QString &errorStr)
{
    QSqlQuery qry(dbCrop);

    qry.prepare( "UPDATE crop SET root_shape = :root_shape,"
                 " root_depth_zero = :root_depth_zero,"
                 " root_depth_max = :root_depth_max,"
                 " root_shape_deformation = :root_shape_deformation,"
                 " degree_days_root_increase = :degree_days_root_increase"
                 " WHERE id_crop = :id_crop");

    if (myCrop.isBareSoil())
    {
        qry.bindValue(":root_shape", "");
        qry.bindValue(":root_depth_zero", "");
        qry.bindValue(":root_depth_max", "");
        qry.bindValue(":root_shape_deformation", "");
        qry.bindValue(":degree_days_root_increase", "");
    }
    else
    {
        qry.bindValue(":root_shape", root::getRootDistributionNumber(myCrop.roots.rootShape));
        qry.bindValue(":root_depth_zero", myCrop.roots.rootDepthMin);
        qry.bindValue(":root_depth_max", myCrop.roots.rootDepthMax);
        qry.bindValue(":root_shape_deformation", myCrop.roots.shapeDeformation);

        if (myCrop.roots.degreeDaysRootGrowth != NODATA)
        {
            qry.bindValue(":degree_days_root_increase", myCrop.roots.degreeDaysRootGrowth);
        }
        else
        {
            qry.bindValue(":degree_days_root_increase", "");
        }
    }

    qry.bindValue(":id_crop", QString::fromStdString(myCrop.idCrop));

    if( !qry.exec() )
    {
        errorStr = qry.lastError().text();
        return false;
    }

    return true;
}


bool updateCropIrrigationparam(QSqlDatabase &dbCrop, const Crit3DCrop &myCrop, QString &errorStr)
{
    QSqlQuery qry(dbCrop);

    qry.prepare( "UPDATE crop SET irrigation_shift = :irrigation_shift, irrigation_volume = :irrigation_volume, "
                 "degree_days_start_irrigation = :degree_days_start_irrigation, degree_days_end_irrigation = :degree_days_end_irrigation, "
                 "psi_leaf = :psi_leaf, raw_fraction = :raw_fraction, stress_tolerance = :stress_tolerance"
                 " WHERE id_crop = :id_crop");

    if (myCrop.irrigationVolume == 0 || myCrop.isBareSoil())
    {
        qry.bindValue(":irrigation_shift", "");
        qry.bindValue(":irrigation_volume", 0);
        qry.bindValue(":degree_days_start_irrigation", "");
        qry.bindValue(":degree_days_end_irrigation", "");
    }
    else
    {
        qry.bindValue(":irrigation_shift", myCrop.irrigationShift);
        qry.bindValue(":irrigation_volume", myCrop.irrigationVolume);
        qry.bindValue(":degree_days_start_irrigation", myCrop.degreeDaysStartIrrigation);
        qry.bindValue(":degree_days_end_irrigation", myCrop.degreeDaysEndIrrigation);
    }

    if (myCrop.isBareSoil() )
    {
        qry.bindValue(":psi_leaf", "");
        qry.bindValue(":raw_fraction", "");
        qry.bindValue(":stress_tolerance", "");
    }
    else
    {
        qry.bindValue(":psi_leaf", myCrop.psiLeaf);
        qry.bindValue(":raw_fraction", myCrop.fRAW);
        qry.bindValue(":stress_tolerance", myCrop.stressTolerance);
    }

    qry.bindValue(":id_crop", QString::fromStdString(myCrop.idCrop));

    if( !qry.exec() )
    {
        errorStr = qry.lastError().text();
        return false;
    }

    return true;
}

