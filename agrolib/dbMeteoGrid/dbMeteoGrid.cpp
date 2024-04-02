#include "dbMeteoGrid.h"
#include "crit3dDate.h"
#include "meteoGrid.h"
#include "basicMath.h"
#include "utilities.h"
#include "meteoPoint.h"
#include "meteo.h"
#include "commonConstants.h"

#include <iostream>
#include <QtSql>


Crit3DMeteoGridDbHandler::Crit3DMeteoGridDbHandler()
{
    _meteoGrid = new Crit3DMeteoGrid();
}

Crit3DMeteoGridDbHandler::~Crit3DMeteoGridDbHandler()
{
    closeDatabase();
    delete _meteoGrid;
}

bool Crit3DMeteoGridDbHandler::parseXMLFile(QString xmlFileName, QDomDocument* xmlDoc, QString *error)
{
    if (xmlFileName == "")
    {
        *error = "Missing XML file.";
        return false;
    }

    QFile myFile(xmlFileName);
    if (!myFile.open(QIODevice::ReadOnly))
    {
        *error = "Open XML failed:\n" + xmlFileName + "\n" + myFile.errorString();
        return (false);
    }

    QString myError;
    int myErrLine, myErrColumn;
    if (!xmlDoc->setContent(&myFile, &myError, &myErrLine, &myErrColumn))
    {
       *error = "Parse xml failed:" + xmlFileName
                + " Row: " + QString::number(myErrLine)
                + " - Column: " + QString::number(myErrColumn)
                + "\n" + myError;
        myFile.close();
        return(false);
    }

    myFile.close();
    return true;
}


bool Crit3DMeteoGridDbHandler::parseXMLGrid(QString xmlFileName, QString *myError)
{
    QDomDocument xmlDoc;

    if (! parseXMLFile(xmlFileName, &xmlDoc, myError)) return false;

    QDomNode child;
    QDomNode secondChild;
    TXMLvar varTable;

    QDomNode ancestor = xmlDoc.documentElement().firstChild();
    QString myTag;
    QString mySecondTag;
    int nRow = 0;
    int nCol = 0;

    _tableDaily.exists = false;
    _tableHourly.exists = false;
    _tableMonthly.exists = false;

    while(! ancestor.isNull())
    {
        if (ancestor.toElement().tagName().toUpper() == "CONNECTION")
        {
            child = ancestor.firstChild();
            while(! child.isNull())
            {
                myTag = child.toElement().tagName().toUpper();
                if (myTag == "PROVIDER")
                {
                    if (child.toElement().text().isEmpty())
                    {
                        *myError = "Missing provider";
                        return false;
                    }
                    _connection.provider = child.toElement().text();
                    // remove white spaces
                    _connection.provider = _connection.provider.simplified();
                }
                else if (myTag == "SERVER")
                {
                    if (child.toElement().text().isEmpty())
                    {
                        *myError = "Missing server";
                        return false;
                    }
                    _connection.server = child.toElement().text();
                    // remove white spaces
                    _connection.server = _connection.server.simplified();
                }
                else if (myTag == "NAME")
                {
                    if (child.toElement().text().isEmpty())
                    {
                        *myError = "Missing name";
                        return false;
                    }
                    _connection.name = child.toElement().text();
                    // remove white spaces
                    _connection.server = _connection.server.simplified();
                }
                else if (myTag == "USER")
                {
                    if (child.toElement().text().isEmpty())
                    {
                        *myError = "Missing user";
                        return false;
                    }
                    _connection.user = child.toElement().text();
                    // remove white spaces
                    _connection.user = _connection.user.simplified();
                }
                else if (myTag == "PASSWORD")
                {
                    if (child.toElement().text().isEmpty())
                    {
                        *myError = "Missing password";
                        return false;
                    }
                    _connection.password = child.toElement().text();
                    // remove white spaces
                    _connection.password = _connection.password.simplified();
                }

                child = child.nextSibling();
            }
        }
        else if (ancestor.toElement().tagName().toUpper() == "GRIDSTRUCTURE")
        {
            if (ancestor.toElement().attribute("isregular").toUpper() == "TRUE")
            {
                _gridStructure.setIsRegular(true);
            }
            else if (ancestor.toElement().attribute("isregular").toUpper() == "FALSE")
            {
                _gridStructure.setIsRegular(false);
            }
            else
            {
                *myError = "Invalid isRegular attribute";
                return false;
            }

            if (ancestor.toElement().attribute("isutm").toUpper() == "TRUE")
            {
                _gridStructure.setIsUTM(true);
            }
            else if (ancestor.toElement().attribute("isutm").toUpper() == "FALSE")
            {
                _gridStructure.setIsUTM(false);
            }
            else
            {
                *myError = "Invalid isutm attribute";
                return false;
            }

            if (ancestor.toElement().attribute("istin").toUpper() == "TRUE")
            {
                _gridStructure.setIsTIN(true);
            }
            else if (ancestor.toElement().attribute("istin").toUpper() == "FALSE")
            {
                _gridStructure.setIsTIN(false);
            }
            else
            {
                *myError = "Invalid istin attribute";
                return false;
            }

            if (ancestor.toElement().attribute("isfixedfields").toUpper() == "TRUE")
            {
                _gridStructure.setIsFixedFields(true);
                initMapMySqlVarType();
            }
            else if (ancestor.toElement().attribute("isfixedfields").toUpper() == "FALSE")
            {
                _gridStructure.setIsFixedFields(false);
            }
            else
            {
                *myError = "Invalid isfixedfields attribute";
                return false;
            }

            if (ancestor.toElement().attribute("isensemble").toUpper() == "TRUE")
            {
                _gridStructure.setIsEnsemble(true);
            }
            else
            {
                _gridStructure.setIsEnsemble(false);
            }

            int nrmembers = ancestor.toElement().attribute("nrmembers").toInt();
            if (nrmembers!=0)
            {
                _gridStructure.setNrMembers(nrmembers);
            }
            else
            {
                _gridStructure.setNrMembers(1);
            }

            child = ancestor.firstChild();
            gis::Crit3DLatLonHeader header;
            /* init */
            header.llCorner.longitude = NODATA;
            header.llCorner.latitude = NODATA;
            header.nrRows = NODATA;
            header.nrCols = NODATA;
            header.dx = NODATA;
            header.dy = NODATA;

            while( !child.isNull())
            {
                myTag = child.toElement().tagName().toUpper();
                if (myTag == "XLL")
                {
                    if (child.toElement().text().isEmpty())
                    {
                        *myError = "Missing XLL";
                        return false;
                    }
                    header.llCorner.longitude = child.toElement().text().toFloat();
                }
                if (myTag == "YLL")
                {
                    if (child.toElement().text().isEmpty())
                    {
                        *myError = "Missing YLL";
                        return false;
                    }
                    header.llCorner.latitude = child.toElement().text().toFloat();
                }
                if (myTag == "NROWS")
                {
                    if (child.toElement().text().isEmpty())
                    {
                        *myError = "Missing NROWS";
                        return false;
                    }
                    header.nrRows = child.toElement().text().toInt();
                    nRow = header.nrRows;
                }
                if (myTag == "NCOLS")
                {
                    if (child.toElement().text().isEmpty())
                    {
                        *myError = "Missing NCOLS";
                        return false;
                    }
                    header.nrCols = child.toElement().text().toInt();
                    nCol = header.nrCols;
                }
                if (myTag == "XWIDTH")
                {
                    if (child.toElement().text().isEmpty())
                    {
                        *myError = "Missing XWIDTH";
                        return false;
                    }
                    header.dx = child.toElement().text().toFloat();
                }
                if (myTag == "YWIDTH")
                {
                    if (child.toElement().text().isEmpty())
                    {
                        *myError = "Missing YWIDTH";
                        return false;
                    }
                    header.dy = child.toElement().text().toFloat();
                }
                child = child.nextSibling();
            }
            _gridStructure.setHeader(header);

        }

        else if (ancestor.toElement().tagName().toUpper() == "TABLEDAILY")
        {
            _tableDaily.exists = true;

            child = ancestor.firstChild();
            while( !child.isNull())
            {
                myTag = child.toElement().tagName().toUpper();
                if (myTag == "FIELDTIME")
                {
                    _tableDaily.fieldTime = child.toElement().text();
                    // remove white spaces
                    _tableDaily.fieldTime = _tableDaily.fieldTime.simplified();
                }
                if (myTag == "PREFIX")
                {
                    _tableDaily.prefix = child.toElement().text();
                    // remove white spaces
                    _tableDaily.prefix = _tableDaily.prefix.simplified();
                }
                if (myTag == "POSTFIX")
                {
                    _tableDaily.postFix = child.toElement().text();
                    // remove white spaces
                    _tableDaily.postFix = _tableDaily.postFix.simplified();
                }
                if (myTag == "VARCODE")
                {
                    secondChild = child.firstChild();
                    _tableDaily.varcode.push_back(varTable);

                    while( !secondChild.isNull())
                    {
                        mySecondTag = secondChild.toElement().tagName().toUpper();


                        if (mySecondTag == "VARFIELD")
                        {
                            _tableDaily.varcode[_tableDaily.varcode.size()-1].varField = secondChild.toElement().text();

                        }

                        else if (mySecondTag == "VARCODE")
                        {
                            _tableDaily.varcode[_tableDaily.varcode.size()-1].varCode = secondChild.toElement().text().toInt();

                        }

                        else if (mySecondTag == "VARPRAGANAME")
                        {
                            _tableDaily.varcode[_tableDaily.varcode.size()-1].varPragaName = secondChild.toElement().text();
                            // remove white spaces
                            _tableDaily.varcode[_tableDaily.varcode.size()-1].varPragaName = _tableDaily.varcode[_tableDaily.varcode.size()-1].varPragaName.simplified();
                        }
                        else
                        {
                            _tableDaily.varcode[_tableDaily.varcode.size()-1].varCode = NODATA;
                        }

                        secondChild = secondChild.nextSibling();
                    }
                }
                child = child.nextSibling();
            }
        }

        else if (ancestor.toElement().tagName().toUpper() == "TABLEHOURLY")
        {
            _tableHourly.exists = true;

            child = ancestor.firstChild();
            while( !child.isNull())
            {
                myTag = child.toElement().tagName().toUpper();
                if (myTag == "FIELDTIME")
                {
                    _tableHourly.fieldTime = child.toElement().text();
                    // remove white spaces
                    _tableHourly.fieldTime = _tableHourly.fieldTime.simplified();
                }
                if (myTag == "PREFIX")
                {
                    _tableHourly.prefix = child.toElement().text();
                    // remove white spaces
                    _tableHourly.prefix = _tableHourly.prefix.simplified();
                }
                if (myTag == "POSTFIX")
                {
                    _tableHourly.postFix = child.toElement().text();
                    // remove white spaces
                    _tableHourly.postFix = _tableHourly.postFix.simplified();
                }
                if (myTag == "VARCODE")
                {
                    secondChild = child.firstChild();
                    _tableHourly.varcode.push_back(varTable);

                    while( !secondChild.isNull())
                    {
                        mySecondTag = secondChild.toElement().tagName().toUpper();


                        if (mySecondTag == "VARFIELD")
                        {
                            _tableHourly.varcode[_tableHourly.varcode.size()-1].varField = secondChild.toElement().text();

                        }

                        else if (mySecondTag == "VARCODE")
                        {
                            _tableHourly.varcode[_tableHourly.varcode.size()-1].varCode = secondChild.toElement().text().toInt();

                        }

                        else if (mySecondTag == "VARPRAGANAME")
                        {
                            _tableHourly.varcode[_tableHourly.varcode.size()-1].varPragaName = secondChild.toElement().text();
                            // remove white spaces
                            _tableHourly.varcode[_tableHourly.varcode.size()-1].varPragaName = _tableHourly.varcode[_tableHourly.varcode.size()-1].varPragaName.simplified();
                        }
                        else
                        {
                            _tableHourly.varcode[_tableHourly.varcode.size()-1].varCode = NODATA;
                        }

                        secondChild = secondChild.nextSibling();
                    }
                }

                child = child.nextSibling();
            }

        }
        else if (ancestor.toElement().tagName().toUpper() == "TABLEMONTHLY")
        {
            _tableMonthly.exists = true;

            child = ancestor.firstChild();
            while( !child.isNull())
            {
                myTag = child.toElement().tagName().toUpper();
                if (myTag == "VARCODE")
                {
                    secondChild = child.firstChild();
                    _tableMonthly.varcode.push_back(varTable);

                    while( !secondChild.isNull())
                    {
                        mySecondTag = secondChild.toElement().tagName().toUpper();


                        if (mySecondTag == "VARFIELD")
                        {
                            _tableMonthly.varcode[_tableMonthly.varcode.size()-1].varField = secondChild.toElement().text();

                        }

                        else if (mySecondTag == "VARCODE")
                        {
                            _tableMonthly.varcode[_tableMonthly.varcode.size()-1].varCode = secondChild.toElement().text().toInt();

                        }

                        else if (mySecondTag == "VARPRAGANAME")
                        {
                            _tableMonthly.varcode[_tableMonthly.varcode.size()-1].varPragaName = secondChild.toElement().text();
                            // remove white spaces
                            _tableMonthly.varcode[_tableMonthly.varcode.size()-1].varPragaName = _tableMonthly.varcode[_tableMonthly.varcode.size()-1].varPragaName.simplified();
                        }
                        else
                        {
                            _tableMonthly.varcode[_tableMonthly.varcode.size()-1].varCode = NODATA;
                        }

                        secondChild = secondChild.nextSibling();
                    }
                }
                child = child.nextSibling();
            }
        }

        ancestor = ancestor.nextSibling();
    }
    xmlDoc.clear();

    if (!checkXML(myError))
    {
        return false;
    }

    // create variable maps
    for (unsigned int i=0; i < _tableDaily.varcode.size(); i++)
    {
        try
        {
            meteoVariable gridMeteoKey = MapDailyMeteoVar.at(_tableDaily.varcode[i].varPragaName.toStdString());
            _gridDailyVar.insert(gridMeteoKey, _tableDaily.varcode[i].varCode);
            _gridDailyVarField.insert(gridMeteoKey, _tableDaily.varcode[i].varField);
        }
        catch (const std::out_of_range& oor)
        {
            QString errMess = QString("%1 does not exist" ).arg(_tableDaily.varcode[i].varPragaName);
            *myError = oor.what() + errMess;
        }

    }

    for (unsigned int i=0; i < _tableHourly.varcode.size(); i++)
    {
        try
        {
            meteoVariable gridMeteoKey = MapHourlyMeteoVar.at(_tableHourly.varcode[i].varPragaName.toStdString());
            _gridHourlyVar.insert(gridMeteoKey, _tableHourly.varcode[i].varCode);
            _gridHourlyVarField.insert(gridMeteoKey, _tableHourly.varcode[i].varField);
        }
        catch (const std::out_of_range& oor)
        {
            QString errMess = QString("%1 does not exist" ).arg(_tableHourly.varcode[i].varPragaName);
            *myError = oor.what() + errMess;
        }
    }

    for (unsigned int i=0; i < _tableMonthly.varcode.size(); i++)
    {
        try
        {
            meteoVariable gridMeteoKey = MapMonthlyMeteoVar.at(_tableMonthly.varcode[i].varPragaName.toStdString());
            _gridMonthlyVar.insert(gridMeteoKey, _tableMonthly.varcode[i].varCode);
            _gridMonthlyVarField.insert(gridMeteoKey, _tableMonthly.varcode[i].varField);
        }
        catch (const std::out_of_range& oor)
        {
            QString errMess = QString("%1 does not exist" ).arg(_tableMonthly.varcode[i].varPragaName);
            *myError = oor.what() + errMess;
        }
    }

    _meteoGrid->setGridStructure(_gridStructure);

    _meteoGrid->initMeteoPoints(nRow, nCol);

    return true;
}

void Crit3DMeteoGridDbHandler::initMapMySqlVarType()
{
    _mapDailyMySqlVarType[dailyAirTemperatureMin] = "float(4,1)";
    _mapDailyMySqlVarType[dailyAirTemperatureMax] = "float(4,1)";
    _mapDailyMySqlVarType[dailyAirTemperatureAvg] = "float(4,1)";
    _mapDailyMySqlVarType[dailyPrecipitation] = "float(4,1) UNSIGNED";
    _mapDailyMySqlVarType[dailyAirRelHumidityMin] = "tinyint(3) UNSIGNED";
    _mapDailyMySqlVarType[dailyAirRelHumidityMax] = "tinyint(3) UNSIGNED";
    _mapDailyMySqlVarType[dailyAirRelHumidityAvg] = "tinyint(3) UNSIGNED";
    _mapDailyMySqlVarType[dailyGlobalRadiation] = "float(5,2) UNSIGNED";
    _mapDailyMySqlVarType[dailyWindScalarIntensityAvg] = "float(3,1) UNSIGNED";
    _mapDailyMySqlVarType[dailyWindScalarIntensityMax] = "float(3,1) UNSIGNED";
    _mapDailyMySqlVarType[dailyWindVectorIntensityAvg] = "float(3,1) UNSIGNED";
    _mapDailyMySqlVarType[dailyWindVectorIntensityMax] = "float(3,1) UNSIGNED";
    _mapDailyMySqlVarType[dailyWindVectorDirectionPrevailing] = "smallint(3) UNSIGNED";
    _mapDailyMySqlVarType[dailyReferenceEvapotranspirationHS] = "float(3,1) UNSIGNED";
    _mapDailyMySqlVarType[dailyReferenceEvapotranspirationPM] = "float(3,1) UNSIGNED";
    _mapDailyMySqlVarType[dailyHeatingDegreeDays] = "float(3,1) UNSIGNED";
    _mapDailyMySqlVarType[dailyCoolingDegreeDays] = "float(3,1) UNSIGNED";
    _mapDailyMySqlVarType[dailyLeafWetness] = "tinyint(3) UNSIGNED";
    _mapDailyMySqlVarType[dailyWaterTableDepth] = "tinyint(3) UNSIGNED";

    _mapHourlyMySqlVarType[airTemperature] = "float(4,1)";
    _mapHourlyMySqlVarType[precipitation] = "float(4,1) UNSIGNED";
    _mapHourlyMySqlVarType[airRelHumidity] = "tinyint(3) UNSIGNED";
    _mapHourlyMySqlVarType[globalIrradiance] = "float(5,1) UNSIGNED";
    _mapHourlyMySqlVarType[netIrradiance] = "float(5,1) UNSIGNED";
    _mapHourlyMySqlVarType[windScalarIntensity] = "float(3,1) UNSIGNED";
    _mapHourlyMySqlVarType[windVectorIntensity] = "float(3,1) UNSIGNED";
    _mapHourlyMySqlVarType[windVectorDirection] = "smallint(3) UNSIGNED";
    _mapHourlyMySqlVarType[referenceEvapotranspiration] = "float(3,1) UNSIGNED";
    _mapHourlyMySqlVarType[leafWetness] = "tinyint(3) UNSIGNED";

}

bool Crit3DMeteoGridDbHandler::checkXML(QString *myError)
{

    /* connection */
    if (_connection.provider.isNull() || _connection.provider.isEmpty())
    {
        *myError = "Missing connection provider";
        return false;
    }
    if (_connection.server.isNull() || _connection.server.isEmpty())
    {
        *myError = "Missing connection server";
        return false;
    }
    if (_connection.name.isNull() || _connection.name.isEmpty())
    {
        *myError = "Missing connection name";
        return false;
    }
    if (_connection.user.isNull() || _connection.user.isEmpty())
    {
        *myError = "Missing connection user";
        return false;
    }
    if (_connection.password.isNull() || _connection.password.isEmpty())
    {
        *myError = "Missing connection password";
        return false;
    }

    /* grid structure */

    if (_gridStructure.header().llCorner.longitude == NODATA)
    {
        *myError = "Error missing xll tag";
        return false;
    }
    if (_gridStructure.header().llCorner.latitude == NODATA)
    {
        *myError = "Error missing yll tag";
        return false;
    }
    if (_gridStructure.header().nrRows == NODATA)
    {
        *myError = "Error missing nrows tag";
        return false;
    }
    if (_gridStructure.header().nrCols == NODATA)
    {
        *myError = "Error missing ncols tag";
        return false;
    }
    if (_gridStructure.header().dx == NODATA)
    {
        *myError = "Error missing xwidth tag";
        return false;
    }
    if (_gridStructure.header().dy == NODATA)
    {
        *myError = "Error missing ywidth tag";
        return false;
    }

    if (_gridStructure.isUTM() == true && _gridStructure.header().dx != _gridStructure.header().dy )
    {
        *myError = "UTM grid with dx != dy";
        return false;
    }

    /* table daily */
    if (_tableDaily.exists)
    {
        if (_tableDaily.fieldTime.isNull() || _tableDaily.fieldTime.isEmpty())
        {
            *myError = "Missing table Daily fieldTime";
            return false;
        }

        if (_tableDaily.varcode.size() < 1 && _tableHourly.varcode.size() < 1)
        {
            *myError = "Missing daily and hourly var code";
            return false;
        }

        for (unsigned int i=0; i < _tableDaily.varcode.size(); i++)
        {
            if (_tableDaily.varcode[i].varCode == NODATA)
            {
                *myError = "Missing daily var code";
                return false;
            }
            if (_tableDaily.varcode[i].varPragaName.isNull() || _tableDaily.varcode[i].varPragaName.isEmpty())
            {
                *myError = "Missing daily varPragaName";
                return false;
            }
            if (_gridStructure.isFixedFields() == true && (_tableDaily.varcode[i].varField.isNull() || _tableDaily.varcode[i].varField.isEmpty()) )
            {
                *myError = "Fixed Field: Missing daily varField";
                return false;
            }
        }
    }

    /* table hourly */
    if (_tableHourly.exists)
    {
        if (_tableHourly.fieldTime.isNull() || _tableHourly.fieldTime.isEmpty())
        {
            *myError = "Missing table Hourly fieldTime";
            return false;
        }

        for (unsigned int i=0; i < _tableHourly.varcode.size(); i++)
        {
            if (_tableHourly.varcode[i].varCode == NODATA)
            {
                *myError = "Missing daily var code";
                return false;
            }
            if (_tableHourly.varcode[i].varPragaName.isNull() || _tableHourly.varcode[i].varPragaName.isEmpty())
            {
                *myError = "Missing daily varPragaName";
                return false;
            }
            if (_gridStructure.isFixedFields() == true && (_tableHourly.varcode[i].varField.isNull() || _tableHourly.varcode[i].varField.isEmpty()) )
            {
                *myError = "Fixed Field: Missing daily varField";
                return false;
            }
        }
    }

    /* table monthly */
    if (_tableMonthly.exists)
    {
        for (unsigned int i=0; i < _tableMonthly.varcode.size(); i++)
        {
            if (_tableMonthly.varcode[i].varCode == NODATA)
            {
                *myError = "Missing monthly var code";
                return false;
            }
            if (_tableMonthly.varcode[i].varPragaName.isNull() || _tableMonthly.varcode[i].varPragaName.isEmpty())
            {
                *myError = "Missing monthly varPragaName";
                return false;
            }
        }
    }

    return true;
}

int Crit3DMeteoGridDbHandler::getDailyVarCode(meteoVariable meteoGridDailyVar)
{

    int varCode = NODATA;
    //check
    if (meteoGridDailyVar == noMeteoVar)
    {
        return varCode;
    }
    if (_gridDailyVar.empty())
    {
        qDebug() << "_gridDailyVar is empty";
        return varCode;
    }
    if(_gridDailyVar.contains(meteoGridDailyVar))
    {
        varCode = _gridDailyVar[meteoGridDailyVar];
    }

    return varCode;

}

QString Crit3DMeteoGridDbHandler::getDailyVarField(meteoVariable meteoGridDailyVar)
{

    QString varField = "";
    //check
    if (meteoGridDailyVar == noMeteoVar)
    {
        return varField;
    }
    if (_gridDailyVarField.empty())
    {
        return varField;
    }
    if(_gridDailyVarField.contains(meteoGridDailyVar))
    {
        varField = _gridDailyVarField[meteoGridDailyVar];
    }

    return varField;

}

meteoVariable Crit3DMeteoGridDbHandler::getDailyVarEnum(int varCode)
{
    if (varCode == NODATA)
    {
        return noMeteoVar;
    }

    QMapIterator<meteoVariable, int> i(_gridDailyVar);
    while (i.hasNext()) {
        i.next();
        if (i.value() == varCode)
        {
            return i.key();
        }
    }

    return noMeteoVar;
}


meteoVariable Crit3DMeteoGridDbHandler::getDailyVarFieldEnum(QString varField)
{

    if (varField == "")
    {
        return noMeteoVar;
    }

    QMapIterator<meteoVariable, QString> i(_gridDailyVarField);
    while (i.hasNext()) {
        i.next();
        if (i.value() == varField)
        {
            return i.key();
        }
    }

    return noMeteoVar;
}

int Crit3DMeteoGridDbHandler::getHourlyVarCode(meteoVariable meteoGridHourlyVar)
{

    int varCode = NODATA;
    //check
    if (meteoGridHourlyVar == noMeteoVar)
    {
        return varCode;
    }
    if (_gridHourlyVar.empty())
    {
        return varCode;
    }
    if(_gridHourlyVar.contains(meteoGridHourlyVar))
    {
        varCode = _gridHourlyVar[meteoGridHourlyVar];
    }

    return varCode;

}

QString Crit3DMeteoGridDbHandler::getHourlyVarField(meteoVariable meteoGridHourlyVar)
{

    QString varField = "";
    //check
    if (meteoGridHourlyVar == noMeteoVar)
    {
        return varField;
    }
    if (_gridHourlyVarField.empty())
    {
        return varField;
    }
    if(_gridHourlyVarField.contains(meteoGridHourlyVar))
    {
        varField = _gridHourlyVarField[meteoGridHourlyVar];
    }

    return varField;

}

meteoVariable Crit3DMeteoGridDbHandler::getHourlyVarEnum(int varCode)
{

    if (varCode == NODATA)
    {
        return noMeteoVar;
    }

    QMapIterator<meteoVariable, int> i(_gridHourlyVar);
    while (i.hasNext()) {
        i.next();
        if (i.value() == varCode)
        {
            return i.key();
        }
    }

    return noMeteoVar;

}

meteoVariable Crit3DMeteoGridDbHandler::getHourlyVarFieldEnum(QString varField)
{

    if (varField == "")
    {
        return noMeteoVar;
    }

    QMapIterator<meteoVariable, QString> i(_gridHourlyVarField);
    while (i.hasNext()) {
        i.next();
        if (i.value() == varField)
        {
            return i.key();
        }
    }

    return noMeteoVar;

}

int Crit3DMeteoGridDbHandler::getMonthlyVarCode(meteoVariable meteoGridMonthlyVar)
{

    int varCode = NODATA;
    //check
    if (meteoGridMonthlyVar == noMeteoVar)
    {
        return varCode;
    }
    if (_gridMonthlyVar.empty())
    {
        return varCode;
    }
    if(_gridMonthlyVar.contains(meteoGridMonthlyVar))
    {
        varCode = _gridMonthlyVar[meteoGridMonthlyVar];
    }

    return varCode;

}

QString Crit3DMeteoGridDbHandler::getMonthlyVarField(meteoVariable meteoGridMonthlyVar)
{

    QString varField = "";
    //check
    if (meteoGridMonthlyVar == noMeteoVar)
    {
        return varField;
    }
    if (_gridMonthlyVar.empty())
    {
        return varField;
    }
    if(_gridMonthlyVar.contains(meteoGridMonthlyVar))
    {
        varField = QString::number(_gridMonthlyVar[meteoGridMonthlyVar]);
    }

    return varField;

}

meteoVariable Crit3DMeteoGridDbHandler::getMonthlyVarEnum(int varCode)
{

    if (varCode == NODATA)
    {
        return noMeteoVar;
    }

    QMapIterator<meteoVariable, int> i(_gridMonthlyVar);
    while (i.hasNext()) {
        i.next();
        if (i.value() == varCode)
        {
            return i.key();
        }
    }

    return noMeteoVar;

}

meteoVariable Crit3DMeteoGridDbHandler::getMonthlyVarFieldEnum(QString varField)
{

    if (varField == "")
    {
        return noMeteoVar;
    }

    QMapIterator<meteoVariable, QString> i(_gridMonthlyVarField);
    while (i.hasNext()) {
        i.next();
        if (i.value() == varField)
        {
            return i.key();
        }
    }

    return noMeteoVar;

}

std::string Crit3DMeteoGridDbHandler::getDailyPragaName(meteoVariable meteoVar)
{

    std::map<std::string, meteoVariable>::const_iterator it;
    std::string key = "";

    for (it = MapDailyMeteoVar.begin(); it != MapDailyMeteoVar.end(); ++it)
    {
        if (it->second == meteoVar)
        {
            key = it->first;
            break;
        }
    }
    return key;
}

std::string Crit3DMeteoGridDbHandler::getHourlyPragaName(meteoVariable meteoVar)
{

    std::map<std::string, meteoVariable>::const_iterator it;
    std::string key = "";

    for (it = MapHourlyMeteoVar.begin(); it != MapHourlyMeteoVar.end(); ++it)
    {
        if (it->second == meteoVar)
        {
            key = it->first;
            break;
        }
    }
    return key;
}

std::string Crit3DMeteoGridDbHandler::getMonthlyPragaName(meteoVariable meteoVar)
{

    std::map<std::string, meteoVariable>::const_iterator it;
    std::string key = "";

    for (it = MapMonthlyMeteoVar.begin(); it != MapMonthlyMeteoVar.end(); ++it)
    {
        if (it->second == meteoVar)
        {
            key = it->first;
            break;
        }
    }
    return key;
}


bool Crit3DMeteoGridDbHandler::openDatabase(QString *myError)
{

    if (_connection.provider.toUpper() == "MYSQL")
    {
        _db = QSqlDatabase::addDatabase("QMYSQL", "grid");
    }

    _db.setHostName(_connection.server);
    _db.setDatabaseName(_connection.name);
    _db.setUserName(_connection.user);
    _db.setPassword(_connection.password);

    if (!_db.open())
    {
       *myError = "Connection with database fail.\n" + _db.lastError().text();
       return false;
    }
    else
       return true;
}

bool Crit3DMeteoGridDbHandler::newDatabase(QString *myError)
{

    if (_connection.provider.toUpper() == "MYSQL")
    {
        _db = QSqlDatabase::addDatabase("QMYSQL");
    }

    _db.setHostName(_connection.server);
    _db.setUserName(_connection.user);
    _db.setPassword(_connection.password);
    _db.open();

    QSqlQuery query(_db);

    query.exec( "CREATE DATABASE IF NOT EXISTS "+_connection.name);

    if (!query.exec())
    {
       *myError = "MySQL error:" + query.lastError().text();
       return false;
    }
    _db.setDatabaseName(_connection.name);
    if (!_db.open())
    {
       *myError = "Connection with database fail.\n" + _db.lastError().text();
       return false;
    }
    else
       return true;
}

bool Crit3DMeteoGridDbHandler::deleteDatabase(QString *myError)
{
    QSqlQuery query(_db);

    query.exec( "DROP DATABASE IF EXISTS "+_connection.name);

    if (!query.exec())
    {
       *myError = "MySQL error:" + query.lastError().text();
       return false;
    }
    return true;
}

bool Crit3DMeteoGridDbHandler::newDatabase(QString *myError, QString connectionName)
{

    if (_connection.provider.toUpper() == "MYSQL")
    {
        _db = QSqlDatabase::addDatabase("QMYSQL", connectionName);
    }

    _db.setHostName(_connection.server);
    _db.setUserName(_connection.user);
    _db.setPassword(_connection.password);
    _db.open();

    QSqlQuery query(_db);

    query.exec( "CREATE DATABASE IF NOT EXISTS "+_connection.name);

    if (!query.exec())
    {
       *myError = "MySQL error:" + query.lastError().text();
       return false;
    }
    _db.setDatabaseName(_connection.name);
    if (!_db.open())
    {
       *myError = "Connection with database fail.\n" + _db.lastError().text();
       return false;
    }
    else
       return true;
}

bool Crit3DMeteoGridDbHandler::openDatabase(QString *myError, QString connectionName)
{

    if (_connection.provider.toUpper() == "MYSQL")
    {
        _db = QSqlDatabase::addDatabase("QMYSQL", connectionName);
    }

    _db.setHostName(_connection.server);
    _db.setDatabaseName(_connection.name);
    _db.setUserName(_connection.user);
    _db.setPassword(_connection.password);

    if (!_db.open())
    {
       *myError = "Connection with database fail.\n" + _db.lastError().text();
       return false;
    }
    else
       return true;
}


void Crit3DMeteoGridDbHandler::closeDatabase()
{
    if ((_db.isValid()) && (_db.isOpen()))
    {
        QString connection = _db.connectionName();
        _db.close();
        _db = QSqlDatabase();
        _db.removeDatabase(connection);
    }
}

bool Crit3DMeteoGridDbHandler::loadCellProperties(QString *myError)
{
    QSqlQuery qry(_db);
    int row, col, active, height;
    QString code, name, tableCellsProp;

    qry.prepare( "SHOW TABLES LIKE '%ells%roperties'" );
    if( !qry.exec() )
    {
        *myError = qry.lastError().text();
        return false;
    }
    else
    {
        qry.next();
        tableCellsProp = qry.value(0).toString();
    }

    QString statement = QString("SELECT * FROM `%1` ORDER BY Code").arg(tableCellsProp);

    if( !qry.exec(statement) )
    {
        *myError = qry.lastError().text();
        return false;
    }
    else
    {
        bool hasHeight = true;
        while (qry.next())
        {

            if (! getValue(qry.value("Code"), &code))
            {
                *myError = "Missing data: Code";
                return false;
            }

            // facoltativa
            if (! getValue(qry.value("Name"), &name))
            {
                name = code;
            }

            if (! getValue(qry.value("Row"), &row))
            {
                *myError = "Missing data: Row";
                return false;
            }

            if (! getValue(qry.value("Col"), &col))
            {
                *myError = "Missing data: Col";
                return false;
            }

            // height: facoltativa
            height = NODATA;
            if (hasHeight)
            {
                if (! qry.value("Height").isValid())
                    hasHeight = false;
                else
                    getValue(qry.value("Height"), &height);
            }

            if (! getValue(qry.value("Active"), &active))
            {
                *myError = "Missing data: Active";
                return false;
            }

            if (row < _meteoGrid->gridStructure().header().nrRows
                && col < _meteoGrid->gridStructure().header().nrCols)
            {
                _meteoGrid->fillMeteoPoint(row, col, code.toStdString(), name.toStdString(), height, active);
            }
            else
            {
                *myError = "Row or Col > nrRows or nrCols";
                return false;
            }
        }
    }
    return true;
}


bool Crit3DMeteoGridDbHandler::newCellProperties(QString *myError)
{
    QSqlQuery qry(_db);
    QString table = "CellsProperties";
    QString statement = QString("CREATE TABLE `%1`"
                                "(`Code` varchar(6) NOT NULL PRIMARY KEY, `Name` varchar(50), "
                                "`Row` INT, `Col` INT, `X` DOUBLE(16,2) DEFAULT 0.00, `Y` DOUBLE(16,2) DEFAULT 0.00, `Height` DOUBLE(16,2) DEFAULT 0.00, `Active` INT)").arg(table);

    if( !qry.exec(statement) )
    {
        *myError = qry.lastError().text();
        return false;
    }
    return true;
}


bool Crit3DMeteoGridDbHandler::writeCellProperties(QString *myError, int nRow, int nCol)
{
    QSqlQuery qry(_db);
    QString table = "CellsProperties";
    int id = 0;
    QString statement = QString(("INSERT INTO `%1` (`Code`, `Name`, `Row`, `Col`, `Active`) VALUES ")).arg(table);
    // standard QGis: first value at top left
    for (int c = 0; c<nCol; c++)
    {
        for (int r = nRow-1; r>=0; r--)
        {
            id = id + 1;
            statement += QString(" ('%1','%2','%3','%4',1),").arg(id, 6, 10, QChar('0')).arg(id, 6, 10, QChar('0')).arg(r).arg(c);
            _meteoGrid->fillMeteoPoint(r, c, QString("%1").arg(id, 6, 10, QChar('0')).toStdString(), QString("%1").arg(id, 6, 10, QChar('0')).toStdString(), 0, 1);
        }
    }

    statement = statement.left(statement.length() - 1);

    if( !qry.exec(statement) )
    {
        *myError = qry.lastError().text();
        return false;
    }
    return true;
}


bool Crit3DMeteoGridDbHandler::activeAllCells(QString *myError)
{
    QSqlQuery qry(_db);

    qry.prepare( "UPDATE CellsProperties SET Active = 1" );
    if( !qry.exec() )
    {
        *myError = qry.lastError().text();
        return false;
    }
    else
    {
        return true;
    }
}

bool Crit3DMeteoGridDbHandler::setActiveStateCellsInList(QString *myError, QList<QString> idList, bool activeState)
{
    QSqlQuery qry(_db);
    QString statement = QString("UPDATE CellsProperties SET Active = %1 WHERE `Code` IN ('%2')").arg(activeState).arg(idList.join("','"));

    if( !qry.exec(statement) )
    {
        *myError = qry.lastError().text();
        return false;
    }
    else
    {
        return true;
    }
}

bool Crit3DMeteoGridDbHandler::loadIdMeteoProperties(QString *myError, QString idMeteo)
{
    QSqlQuery qry(_db);
    int row, col, active, height;
    QString code, name, tableCellsProp;

    qry.prepare( "SHOW TABLES LIKE '%ells%roperties'" );
    if( !qry.exec() )
    {
        *myError = qry.lastError().text();
        return false;
    }
    else
    {
        qry.next();
        tableCellsProp = qry.value(0).toString();
    }

    QString statement = QString("SELECT * FROM `%1` WHERE `Code` = '%2'").arg(tableCellsProp).arg(idMeteo);

    if( !qry.exec(statement) )
    {
        *myError = qry.lastError().text();
        return false;
    }
    else
    {
        bool hasHeight = true;
        while (qry.next())
        {

            if (! getValue(qry.value("Code"), &code))
            {
                *myError = "Missing data: Code";
                return false;
            }

            // facoltativa
            if (! getValue(qry.value("Name"), &name))
            {
                name = code;
            }

            if (! getValue(qry.value("Row"), &row))
            {
                *myError = "Missing data: Row";
                return false;
            }

            if (! getValue(qry.value("Col"), &col))
            {
                *myError = "Missing data: Col";
                return false;
            }

            // height: facoltativa
            height = NODATA;
            if (hasHeight)
            {
                if (! qry.value("Height").isValid())
                    hasHeight = false;
                else
                    getValue(qry.value("Height"), &height);
            }

            if (! getValue(qry.value("Active"), &active))
            {
                *myError = "Missing data: Active";
                return false;
            }

            if (row < _meteoGrid->gridStructure().header().nrRows
                && col < _meteoGrid->gridStructure().header().nrCols)
            {
                _meteoGrid->fillMeteoPoint(row, col, code.toStdString(), name.toStdString(), height, active);
            }
            else
            {
                *myError = "Row or Col > nrRows or nrCols";
                return false;
            }
        }
    }
    return true;
}


bool Crit3DMeteoGridDbHandler::updateMeteoGridDate(QString &myError)
{
    QList<QString> tableList = _db.tables(QSql::Tables);
    if (tableList.size() <= 1)
    {
        myError = "No data.";
        return false;
    }

    int row = 0;
    int col = 0;
    QString tableNotFoundError = "1146";
    std::string id;

    if (!_meteoGrid->findFirstActiveMeteoPoint(&id, &row, &col))
    {
        myError = "No active cells.";
        return false;
    }

    QDate noDate = QDate(1800, 1, 1);

    _lastDailyDate = noDate;
    _firstDailyDate= noDate;
    _lastHourlyDate = noDate;
    _firstHourlyDate = noDate;
    _lastMonthlyDate = noDate;
    _firstMonthlyDate = noDate;

    QString tableD = _tableDaily.prefix + QString::fromStdString(id) + _tableDaily.postFix;
    QString tableH = _tableHourly.prefix + QString::fromStdString(id) + _tableHourly.postFix;
    QString tableM = "MonthlyData";

    QSqlQuery qry(_db);

    if (_tableDaily.exists)
    {
        QString statement = QString("SELECT MIN(%1) as minDate, MAX(%1) as maxDate FROM `%2`").arg(_tableDaily.fieldTime, tableD);
        if(! qry.exec(statement) )
        {
            while( qry.lastError().nativeErrorCode() == tableNotFoundError
                   && (col < _gridStructure.header().nrCols-1
                       || row < _gridStructure.header().nrRows-1))
            {
                if ( col < _gridStructure.header().nrCols-1)
                {
                    col = col + 1;
                }
                else if( row < _gridStructure.header().nrRows-1)
                {
                    row = row + 1;
                    col = 0;
                }

                if (!_meteoGrid->findFirstActiveMeteoPoint(&id, &row, &col))
                {
                    myError = "active cell not found";
                    return false;
                }
                tableD = _tableDaily.prefix + QString::fromStdString(id) + _tableDaily.postFix;

                statement = QString("SELECT MIN(%1) as minDate, MAX(%1) as maxDate FROM `%2`").arg(_tableDaily.fieldTime, tableD);
                qry.exec(statement);
            }
        }

        if ( qry.lastError().type() != QSqlError::NoError )
        {
            myError = qry.lastError().text();
            return false;
        }
        else
        {
            if (qry.next())
            {
                QDate temp;
                if (getValue(qry.value("minDate"), &temp))
                {
                    _firstDailyDate = temp;
                }

                if (getValue(qry.value("maxDate"), &temp))
                {
                    _lastDailyDate = temp;
                }
            }
            else
            {
                myError = "Daily time field not found: " + _tableDaily.fieldTime;
                return false;
            }
        }
    }

    if (_tableHourly.exists)
    {
        tableH = _tableHourly.prefix + QString::fromStdString(id) + _tableHourly.postFix;
        QString statement = QString("SELECT MIN(%1) as minDate, MAX(%1) as maxDate FROM `%2`").arg(_tableHourly.fieldTime, tableH);
        if(! qry.exec(statement) )
        {
            while( qry.lastError().nativeErrorCode() == tableNotFoundError)
            {
                if ( col < _gridStructure.header().nrCols-1)
                {
                    col = col + 1;
                }
                else if( row < _gridStructure.header().nrRows-1)
                {
                    row = row + 1;
                    col = 0;
                }

                if (! _meteoGrid->findFirstActiveMeteoPoint(&id, &row, &col))
                {
                    myError = "active cell not found";
                    return false;
                }

                tableH = _tableHourly.prefix + QString::fromStdString(id) + _tableHourly.postFix;

                statement = QString("SELECT MIN(%1) as minDate, MAX(%1) as maxDate FROM `%2`").arg(_tableHourly.fieldTime, tableH);
                qry.exec(statement);
            }
        }

        if ( qry.lastError().type() != QSqlError::NoError && qry.lastError().nativeErrorCode() != tableNotFoundError)
        {
            myError = qry.lastError().text();
            return false;
        }
        else
        {
            if (qry.next())
            {
                QDate temp;
                if (getValue(qry.value("minDate"), &temp))
                {
                    _firstHourlyDate = temp;
                }

                if (getValue(qry.value("maxDate"), &temp))
                {
                    // the last hourly day is always incomplete, there is just 00.00 value
                    _lastHourlyDate = temp.addDays(-1);
                }
            }
            else
            {
                myError = "Hourly time field not found: " + _tableHourly.fieldTime;
                return false;
            }
        }
    }

    if (_tableMonthly.exists)
    {
        QString table = "MonthlyData";
        QString statement = QString("CREATE TABLE IF NOT EXISTS `%1`"
                                    "(PragaYear smallint(4) UNSIGNED, PragaMonth tinyint(2) UNSIGNED, PointCode VARCHAR(6), "
                                    "VariableCode tinyint(3) UNSIGNED, Value float(6,1), PRIMARY KEY(PragaYear,PragaMonth,PointCode,VariableCode))").arg(table);

        if(! qry.exec(statement) )
        {
            myError = qry.lastError().text();
            return false;
        }

        int minPragaYear;
        int maxPragaYear;
        int minPragaMonth;
        int maxPragaMonth;
        statement = QString("SELECT MIN(%1) as minYear, MAX(%1) as maxYear FROM `%2`").arg("PragaYear", tableM);
        qry.exec(statement);

        if ( qry.lastError().type() != QSqlError::NoError )
        {
            myError = qry.lastError().text();
            return false;
        }
        else
        {
            if (qry.next())
            {
                getValue(qry.value("minYear"), &minPragaYear);
                getValue(qry.value("maxYear"), &maxPragaYear);
            }
            else
            {
                myError = "PragaYear field not found";
                return false;
            }
        }
        statement = QString("SELECT MIN(%1) as minMonth FROM `%2` WHERE PragaYear=%3 ").arg("PragaMonth", tableM).arg(minPragaYear);
        qry.exec(statement);

        if ( qry.lastError().type() != QSqlError::NoError )
        {
            myError = qry.lastError().text();
            return false;
        }
        else
        {
            if (qry.next())
            {
                getValue(qry.value("minMonth"), &minPragaMonth);
            }
            else
            {
                myError = "PragaMonth field not found";
                return false;
            }
        }

        statement = QString("SELECT MAX(%1) as maxMonth FROM `%2` WHERE PragaYear=%3 ").arg("PragaMonth", tableM).arg(maxPragaYear);
        qry.exec(statement);

        if ( qry.lastError().type() != QSqlError::NoError )
        {
            myError = qry.lastError().text();
            return false;
        }
        else
        {
            if (qry.next())
            {
                getValue(qry.value("maxMonth"), &maxPragaMonth);
            }
            else
            {
                myError = "PragaMonth field not found";
                return false;
            }
        }

        if (minPragaYear != NODATA && maxPragaYear != NODATA &&
                minPragaMonth != NODATA && maxPragaMonth != NODATA) {

            _lastMonthlyDate.setDate(maxPragaYear, maxPragaMonth, getDaysInMonth(maxPragaMonth, maxPragaYear));
            _firstMonthlyDate.setDate(minPragaYear, minPragaMonth, 1);
        }
    }

    // FIRST DATE
    _firstDate = noDate;
    if (_firstDailyDate != noDate)
    {
        _firstDate = _firstDailyDate;
    }
    if (_firstHourlyDate != noDate && (_firstDate == noDate || _firstHourlyDate < _firstDate))
    {
        _firstDate = _firstHourlyDate;
    }
    if (_firstMonthlyDate != noDate && (_firstDate == noDate || _firstMonthlyDate < _firstDate))
    {
        _firstDate = _firstMonthlyDate;
    }

    // LAST DATE
    _lastDate = noDate;
    if (_lastDailyDate != noDate)
    {
        _lastDate = _lastDailyDate;
    }
    if (_lastHourlyDate != noDate && (_lastDate == noDate || _lastHourlyDate > _lastDate))
    {
        _lastDate = _lastHourlyDate;
    }
    if (_lastMonthlyDate != noDate && (_lastDate == noDate || _lastMonthlyDate > _lastDate))
    {
        _lastDate = _lastMonthlyDate;
    }

    if (_firstDate == noDate || _lastDate == noDate)
    {
        myError = "Missing data.";
        return false;
    }

    return true;
}


bool Crit3DMeteoGridDbHandler::loadGridDailyData(QString &myError, const QString &meteoPointId, const QDate &firstDate, const QDate &lastDate)
{
    myError = "";
    QString tableD = _tableDaily.prefix + meteoPointId + _tableDaily.postFix;

    unsigned row, col;
    if ( !_meteoGrid->findMeteoPointFromId(&row, &col, meteoPointId.toStdString()) )
    {
        myError = "Missing meteoPoint id: " + meteoPointId;
        return false;
    }

    int numberOfDays = firstDate.daysTo(lastDate) + 1;
    _meteoGrid->meteoPointPointer(row, col)->initializeObsDataD(numberOfDays, getCrit3DDate(firstDate));

    if (_firstDailyDate.isValid() && _lastDailyDate.isValid())
    {
        if (_firstDailyDate.year() != 1800 && _lastDailyDate.year() != 1800)
        {
            if (firstDate > _lastDailyDate || lastDate < _firstDailyDate)
            {
                myError = "Missing data in this time interval.";
                return false;
            }
        }
    }

    QSqlQuery qry(_db);
    QString statement;
    bool isSingleDate = false;
    QDate date;
    if (firstDate == lastDate)
    {
        statement = QString("SELECT * FROM `%1` WHERE %2 = '%3'").arg(tableD, _tableDaily.fieldTime, firstDate.toString("yyyy-MM-dd"));
        isSingleDate = true;
        date = firstDate;
    }
    else
    {
        statement = QString("SELECT * FROM `%1` WHERE %2 >= '%3' AND %2 <= '%4' ORDER BY %2")
                            .arg(tableD, _tableDaily.fieldTime, firstDate.toString("yyyy-MM-dd"), lastDate.toString("yyyy-MM-dd"));
    }
    qry.prepare(statement);

    if(! qry.exec())
    {
        myError = qry.lastError().text();
        return false;
    }

    int varCode;
    float value;
    while (qry.next())
    {
        getValue(qry.value("Value"), &value);

        if (value != NODATA)
        {
            if (! isSingleDate)
            {
                if (! getValue(qry.value(_tableDaily.fieldTime), &date))
                {
                    myError = "Missing " + _tableDaily.fieldTime;
                    return false;
                }
            }

            if (! getValue(qry.value("VariableCode"), &varCode))
            {
                myError = "Missing VariableCode";
                return false;
            }

            meteoVariable variable = getDailyVarEnum(varCode);

            if (! _meteoGrid->meteoPointPointer(row, col)->setMeteoPointValueD(getCrit3DDate(date), variable, value))
            {
                myError = "Error in setMeteoPointValueD";
                return false;
            }
        }
    }

    return true;
}


bool Crit3DMeteoGridDbHandler::loadGridDailyDataEnsemble(QString &myError, QString meteoPoint, int memberNr, QDate first, QDate last)
{
    myError = "";

    if (!_meteoGrid->gridStructure().isEnsemble())
    {
        myError = "Grid structure has not ensemble field";
        return false;
    }
    QSqlQuery qry(_db);
    QString tableD = _tableDaily.prefix + meteoPoint + _tableDaily.postFix;
    QDate date;
    int varCode;
    float value;

    unsigned row;
    unsigned col;

    if (!_meteoGrid->findMeteoPointFromId(&row, &col, meteoPoint.toStdString()) )
    {
        myError = "Missing MeteoPoint id";
        return false;
    }

    int numberOfDays = first.daysTo(last) + 1;
    _meteoGrid->meteoPointPointer(row,col)->initializeObsDataD(numberOfDays, getCrit3DDate(first));

    QString statement = QString("SELECT * FROM `%1` WHERE `%2`>= '%3' AND `%2`<= '%4' AND MemberNr = '%5' ORDER BY `%2`").arg(tableD).arg(_tableDaily.fieldTime).arg(first.toString("yyyy-MM-dd")).arg(last.toString("yyyy-MM-dd")).arg(memberNr);
    if( !qry.exec(statement) )
    {
        myError = qry.lastError().text();
        return false;
    }
    else
    {
        while (qry.next())
        {
            if (!getValue(qry.value(_tableDaily.fieldTime), &date))
            {
                myError = "Missing fieldTime";
                return false;
            }

            if (!getValue(qry.value("VariableCode"), &varCode))
            {
                myError = "Missing VariableCode";
                return false;
            }

            if (!getValue(qry.value("Value"), &value))
            {
                myError = "Missing Value";
            }

            meteoVariable variable = getDailyVarEnum(varCode);

            if (! _meteoGrid->meteoPointPointer(row,col)->setMeteoPointValueD(getCrit3DDate(date), variable, value))
                return false;
        }
    }

    return true;
}


bool Crit3DMeteoGridDbHandler::loadGridDailyDataFixedFields(QString &myError, QString meteoPoint, QDate first, QDate last)
{
    myError = "";

    QSqlQuery qry(_db);
    QString tableD = _tableDaily.prefix + meteoPoint + _tableDaily.postFix;
    QDate date;
    int varCode;
    float value;

    unsigned row, col;

    if (!_meteoGrid->findMeteoPointFromId(&row, &col, meteoPoint.toStdString()) )
    {
        myError = "Missing MeteoPoint id";
        return false;
    }

    int numberOfDays = first.daysTo(last) + 1;
    _meteoGrid->meteoPointPointer(row,col)->initializeObsDataD(numberOfDays, getCrit3DDate(first));

    QString statement = QString("SELECT * FROM `%1` WHERE `%2` >= '%3' AND `%2` <= '%4' ORDER BY `%2`").arg(tableD,
                                    _tableDaily.fieldTime, first.toString("yyyy-MM-dd"), last.toString("yyyy-MM-dd"));
    if( !qry.exec(statement) )
    {
        myError = qry.lastError().text();
        return false;
    }
    else
    {
        while (qry.next())
        {
            if (!getValue(qry.value(_tableDaily.fieldTime), &date))
            {
                myError = "Missing fieldTime";
                return false;
            }

            for (unsigned int i=0; i < _tableDaily.varcode.size(); i++)
            {
                varCode = _tableDaily.varcode[i].varCode;
                if (!getValue(qry.value(_tableDaily.varcode[i].varField), &value))
                {
                    myError = "Missing VarField";
                }

                meteoVariable variable = getDailyVarEnum(varCode);

                if (! _meteoGrid->meteoPointPointer(row,col)->setMeteoPointValueD(getCrit3DDate(date), variable, value))
                    return false;
            }
        }
    }

    return true;
}


bool Crit3DMeteoGridDbHandler::loadGridHourlyData(QString &myError, QString meteoPoint, QDateTime firstDate, QDateTime lastDate)
{
    myError = "";
    QString tableH = _tableHourly.prefix + meteoPoint + _tableHourly.postFix;

    unsigned row, col;
    if (!_meteoGrid->findMeteoPointFromId(&row, &col, meteoPoint.toStdString()) )
    {
        myError = "Missing MeteoPoint id";
        return false;
    }

    int numberOfDays = firstDate.date().daysTo(lastDate.date());
    _meteoGrid->meteoPointPointer(row, col)->initializeObsDataH(1, numberOfDays, getCrit3DDate(firstDate.date()));

    if (firstDate.date() > _lastHourlyDate || lastDate.date() < _firstHourlyDate)
    {
        myError = "missing data";
        return false;
    }

    QSqlQuery qry(_db);
    QDateTime date;
    int varCode;
    float value;
    QString statement = QString("SELECT * FROM `%1` WHERE `%2` >= '%3' AND `%2` <= '%4' ORDER BY `%2`")
                                .arg(tableH, _tableHourly.fieldTime, firstDate.toString("yyyy-MM-dd hh:mm"),
                                 lastDate.toString("yyyy-MM-dd hh:mm") );

    if( !qry.exec(statement) )
    {
        myError = qry.lastError().text();
    }
    else
    {
        while (qry.next())
        {
            getValue(qry.value("Value"), &value);

            if (value != NODATA)
            {
                if (! getValue(qry.value(_tableHourly.fieldTime), &date))
                {
                    myError = "Missing fieldTime";
                    return false;
                }

                if (! getValue(qry.value("VariableCode"), &varCode))
                {
                    myError = "Missing VariableCode";
                    return false;
                }
                meteoVariable variable = getHourlyVarEnum(varCode);

                if (! _meteoGrid->meteoPointPointer(row,col)->setMeteoPointValueH(getCrit3DDate(date.date()),
                                                     date.time().hour(), date.time().minute(), variable, value))
                {
                    myError = "Error in setMeteoPointValueH";
                    return false;
                }
            }
        }
    }

    return true;
}


bool Crit3DMeteoGridDbHandler::loadGridHourlyDataEnsemble(QString &myError, QString meteoPoint, int memberNr, QDateTime first, QDateTime last)
{
    myError = "";

    if (!_meteoGrid->gridStructure().isEnsemble())
    {
        myError = "Grid structure has not ensemble field";
        return false;
    }

    QSqlQuery qry(_db);
    QString tableH = _tableHourly.prefix + meteoPoint + _tableHourly.postFix;
    QDateTime date;
    int varCode;
    float value;

    unsigned row;
    unsigned col;

    if (!_meteoGrid->findMeteoPointFromId(&row, &col, meteoPoint.toStdString()) )
    {
        myError = "Missing MeteoPoint id";
        return false;
    }

    int numberOfDays = first.date().daysTo(last.date());
    _meteoGrid->meteoPointPointer(row, col)->initializeObsDataH(1, numberOfDays, getCrit3DDate(first.date()));

    QString statement = QString("SELECT * FROM `%1` WHERE `%2` >= '%3' AND `%2` <= '%4' AND MemberNr = '%5' ORDER BY `%2`")
                                .arg(tableH).arg(_tableHourly.fieldTime).arg(first.toString("yyyy-MM-dd hh:mm")).arg(last.toString("yyyy-MM-dd hh:mm")).arg(memberNr);

    if( !qry.exec(statement) )
    {
        myError = qry.lastError().text();
    }
    else
    {
        while (qry.next())
        {
            if (!getValue(qry.value(_tableHourly.fieldTime), &date))
            {
                myError = "Missing fieldTime";
                return false;
            }

            if (!getValue(qry.value("VariableCode"), &varCode))
            {
                myError = "Missing VariableCode";
                return false;
            }

            if (!getValue(qry.value("Value"), &value))
            {
                myError = "Missing Value";
            }

            meteoVariable variable = getHourlyVarEnum(varCode);

            if (! _meteoGrid->meteoPointPointer(row,col)->setMeteoPointValueH(getCrit3DDate(date.date()), date.time().hour(), date.time().minute(), variable, value))
                return false;
        }
    }

    return true;
}


bool Crit3DMeteoGridDbHandler::loadGridHourlyDataFixedFields(QString &myError, QString meteoPoint, QDateTime first, QDateTime last)
{
    myError = "";

    QSqlQuery qry(_db);
    QString tableH = _tableHourly.prefix + meteoPoint + _tableHourly.postFix;
    QDateTime date;
    int varCode;
    float value;

    unsigned row;
    unsigned col;

    if (!_meteoGrid->findMeteoPointFromId(&row, &col, meteoPoint.toStdString()) )
    {
        myError = "Missing MeteoPoint id";
        return false;
    }

    int numberOfDays = first.date().daysTo(last.date());
    _meteoGrid->meteoPointPointer(row, col)->initializeObsDataH(1, numberOfDays, getCrit3DDate(first.date()));

    QString statement = QString("SELECT * FROM `%1` WHERE `%2` >= '%3' AND `%2`<= '%4' ORDER BY `%2`").arg(tableH).arg(_tableHourly.fieldTime).arg(first.toString("yyyy-MM-dd hh:mm")).arg(last.toString("yyyy-MM-dd hh:mm"));
    if( !qry.exec(statement) )
    {
        myError = qry.lastError().text();
    }
    else
    {
        while (qry.next())
        {
            if (!getValue(qry.value(_tableHourly.fieldTime), &date))
            {
                myError = "Missing fieldTime";
                return false;
            }

            for (unsigned int i=0; i < _tableHourly.varcode.size(); i++)
            {
                varCode = _tableHourly.varcode[i].varCode;

                if (!getValue(qry.value(_tableHourly.varcode[i].varField), &value))
                {
                    myError = "Missing fieldTime";
                }
                meteoVariable variable = getHourlyVarEnum(varCode);

                if (! _meteoGrid->meteoPointPointer(row,col)->setMeteoPointValueH(getCrit3DDate(date.date()), date.time().hour(), date.time().minute(), variable, value))
                    return false;
            }

        }

    }

    return true;
}


bool Crit3DMeteoGridDbHandler::loadGridMonthlySingleDate(QString &myError, const QString &meteoPoint, const QDate &myDate)
{
    myError = "";

    unsigned row, col;
    if (! _meteoGrid->findMeteoPointFromId(&row, &col, meteoPoint.toStdString()) )
    {
        myError = "Missing MeteoPoint id: " + meteoPoint;
        return false;
    }

    int year = myDate.year();
    short month = myDate.month();
    _meteoGrid->meteoPointPointer(row, col)->initializeObsDataM(1, month, year);

    if (myDate > _lastMonthlyDate || myDate < _firstMonthlyDate)
    {
        return false;
    }

    QSqlQuery qry(_db);
    QString statement = QString("SELECT * FROM MonthlyData WHERE `PointCode` = '%1' AND `PragaYear`= %2 AND `PragaMonth`= %3")
                            .arg(meteoPoint).arg(year).arg(month);
    if(! qry.exec(statement) )
    {
        myError = qry.lastError().text();
        return false;
    }

    int varCode;
    float value;
    while (qry.next())
    {
        if (! getValue(qry.value("VariableCode"), &varCode))
        {
            myError = "Missing VariableCode.";
            return false;
        }
        meteoVariable variable = getMonthlyVarEnum(varCode);

        getValue(qry.value("Value"), &value);

        if (! _meteoGrid->meteoPointPointer(row, col)->setMeteoPointValueM(getCrit3DDate(myDate), variable, value))
            return false;
    }

    return true;
}


bool Crit3DMeteoGridDbHandler::loadGridMonthlyData(QString &myError, QString meteoPoint, QDate firstDate, QDate lastDate)
{
    myError = "";
    QString table = "MonthlyData";

    // set day to 1 to better comparison
    firstDate.setDate(firstDate.year(), firstDate.month(), 1);
    lastDate.setDate(lastDate.year(), lastDate.month(), 1);

    unsigned row, col;
    if (!_meteoGrid->findMeteoPointFromId(&row, &col, meteoPoint.toStdString()) )
    {
        myError = "Missing MeteoPoint id";
        return false;
    }

    int numberOfMonths = (lastDate.year()-firstDate.year())*12 + lastDate.month() - (firstDate.month()-1);
    _meteoGrid->meteoPointPointer(row,col)->initializeObsDataM(numberOfMonths, firstDate.month(), firstDate.year());

    if (firstDate > _lastMonthlyDate || lastDate < _firstMonthlyDate)
    {
        return false;
    }

    QSqlQuery qry(_db);
    QDate date;
    int year, month, varCode;
    float value;
    QString statement = QString("SELECT * FROM `%1` WHERE `PragaYear` BETWEEN %2 AND %3 AND PointCode = '%4' ORDER BY `PragaYear`").arg(table).arg(firstDate.year()).arg(lastDate.year()).arg(meteoPoint);
    if( !qry.exec(statement) )
    {
        myError = qry.lastError().text();
        return false;
    }
    else
    {
        while (qry.next())
        {
            if (!getValue(qry.value("PragaYear"), &year))
            {
                myError = "Missing PragaYear";
                return false;
            }

            if (!getValue(qry.value("PragaMonth"), &month))
            {
                myError = "Missing PragaMonth";
                return false;
            }

            date.setDate(year,month, 1);
            if (date < firstDate || date > lastDate)
            {
                continue;
            }

            if (!getValue(qry.value("VariableCode"), &varCode))
            {
                myError = "Missing VariableCode";
                return false;
            }

            if (!getValue(qry.value("Value"), &value))
            {
                myError = "Missing Value";
            }

            meteoVariable variable = getMonthlyVarEnum(varCode);

            if (! _meteoGrid->meteoPointPointer(row,col)->setMeteoPointValueM(getCrit3DDate(date), variable, value))
                return false;
        }
    }

    return true;
}


bool Crit3DMeteoGridDbHandler::loadGridAllMonthlyData(QString &myError, QDate firstDate, QDate lastDate)
{
    myError = "";
    QString table = "MonthlyData";

    // set day to 1 to better comparison
    firstDate.setDate(firstDate.year(), firstDate.month(), 1);
    lastDate.setDate(lastDate.year(), lastDate.month(), 1);
    int numberOfMonths = (lastDate.year()-firstDate.year())*12 + lastDate.month() - (firstDate.month()-1);

    if (firstDate > _lastMonthlyDate || lastDate < _firstMonthlyDate)
    {
        return false;
    }

    // init all monthly data
    for (int row = 0; row < gridStructure().header().nrRows; row++)
    {
        for (int col = 0; col < gridStructure().header().nrCols; col++)
        {
            _meteoGrid->meteoPointPointer(row,col)->initializeObsDataM(numberOfMonths, firstDate.month(), firstDate.year());
        }
    }

    QSqlQuery qry(_db);
    QDate monthDate;
    unsigned row, col;
    int year, month, varCode;
    int lastVarCode = NODATA;
    meteoVariable variable = noMeteoVar;
    QString pointCode, lastPointCode;
    float value;

    QString statement = QString("SELECT * FROM `%1` WHERE `PragaYear` BETWEEN %2 AND %3 ORDER BY `PointCode`").arg(table).arg(firstDate.year()).arg(lastDate.year());
    if( !qry.exec(statement) )
    {
        myError = qry.lastError().text();
        return false;
    }
    else
    {
        while (qry.next())
        {
            if (! getValue(qry.value("PragaYear"), &year))
            {
                myError = "Missing PragaYear";
                return false;
            }

            if (! getValue(qry.value("PragaMonth"), &month))
            {
                myError = "Missing PragaMonth";
                return false;
            }

            monthDate.setDate(year, month, 1);
            if (monthDate < firstDate || monthDate > lastDate)
            {
                continue;
            }

            if (! getValue(qry.value("PointCode"), &pointCode))
            {
                myError = "Missing PointCode";
                return false;
            }

            if (pointCode != lastPointCode)     // new point
            {
                if (! _meteoGrid->findMeteoPointFromId(&row, &col, pointCode.toStdString()) )
                {
                    continue;
                }
                lastPointCode = pointCode;
            }

            if (! getValue(qry.value("VariableCode"), &varCode))
            {
                myError = "Missing VariableCode: " + QString::number(varCode);
                return false;
            }

            if (varCode != lastVarCode)     // new var
            {
                variable = getMonthlyVarEnum(varCode);
                lastVarCode = varCode;
            }

            if (! getValue(qry.value("Value"), &value))
            {
                myError = "Missing Value";
            }

            if (! _meteoGrid->meteoPointPointer(row, col)->setMeteoPointValueM(getCrit3DDate(monthDate), variable, value))
            {
                myError = "Error in setMeteoPointValueM()";
                return false;
            }
        }
    }

    return true;
}


std::vector<float> Crit3DMeteoGridDbHandler::loadGridDailyVar(QString *myError, QString meteoPoint,
                                    meteoVariable variable, QDate first, QDate last, QDate* firstDateDB)
{

    QSqlQuery qry(_db);
    QString tableD = _tableDaily.prefix + meteoPoint + _tableDaily.postFix;
    QDate currentDate, lastDateDB;
    std::vector<float> dailyVarList;

    int varCode = getDailyVarCode(variable);
    if (varCode == NODATA)
    {
        *myError = "Variable not existing";
        return dailyVarList;
    }

    unsigned row, col;
    if (!_meteoGrid->findMeteoPointFromId(&row, &col, meteoPoint.toStdString()) )
    {
        *myError = "Missing MeteoPoint id";
        return dailyVarList;
    }

    QString statement = QString("SELECT `%3`,`Value` FROM `%1` WHERE VariableCode = '%2' AND `%3` >= '%4' AND `%3`<= '%5' ORDER BY `%3`").arg(tableD).arg(varCode).arg(_tableDaily.fieldTime).arg(first.toString("yyyy-MM-dd")).arg(last.toString("yyyy-MM-dd"));

    if(! qry.exec(statement) )
    {
        *myError = qry.lastError().text();
        if (!_db.isOpen())
        {
            qDebug() << "qry exec: db is not open: " << *myError;
            exit(EXIT_FAILURE);
        }
        else
        {
            return dailyVarList;
        }
    }

    // read first date
    if (!qry.first())
    {
        *myError = qry.lastError().text();
        if (!_db.isOpen())
        {
            qDebug() << "qry.first: db is not open: " << *myError;
            exit(EXIT_FAILURE);
        }
        else
        {
            return dailyVarList;
        }
    }

    if (!getValue(qry.value(_tableDaily.fieldTime), firstDateDB))
    {
        *myError = "Missing first date";
        if (!_db.isOpen())
        {
            qDebug() << "qry.value: db is not open: " << *myError;
            exit(EXIT_FAILURE);
        }
        else
        {
            return dailyVarList;
        }
    }

    // read last date
    qry.last();
    if (!getValue(qry.value(_tableDaily.fieldTime), &lastDateDB))
    {
        *myError = "Missing last date";
        return dailyVarList;
    }

    // resize vector
    int nrValues = int(firstDateDB->daysTo(lastDateDB)) + 1;
    dailyVarList.resize(unsigned(nrValues));
    for (unsigned int i = 0; i < dailyVarList.size(); i++)
    {
        dailyVarList[i] = NODATA;
    }

    // assign values
    float value;
    qry.first();
    do
    {
        currentDate = qry.value(_tableDaily.fieldTime).toDate();
        int currentIndex = int(firstDateDB->daysTo(currentDate));
        if (getValue(qry.value("Value"), &value))
        {
            dailyVarList[unsigned(currentIndex)] = value;
        }
    } while (qry.next());

    return dailyVarList;
}

std::vector<float> Crit3DMeteoGridDbHandler::exportAllDataVar(QString *myError, frequencyType freq, meteoVariable variable, QString id, QDateTime myFirstTime, QDateTime myLastTime, std::vector<QString> &dateStr)
{
    QString myDateStr;
    float value;
    std::vector<float> allDataVarList;

    QSqlQuery myQuery(_db);
    QString tableName;
    QString statement;
    QString startDate;
    QString endDate;
    int idVar;

    if (freq == daily)
    {
        idVar = getDailyVarCode(variable);
        if (idVar == NODATA)
        {
            *myError = "Variable not existing";
            return allDataVarList;
        }
        tableName = _tableDaily.prefix + id + _tableDaily.postFix;
        startDate = myFirstTime.date().toString("yyyy-MM-dd");
        endDate = myLastTime.date().toString("yyyy-MM-dd");
        statement = QString( "SELECT * FROM `%1` WHERE VariableCode = '%2' AND `%3` >= '%4' AND `%3`<= '%5'")
                        .arg(tableName).arg(idVar).arg(_tableDaily.fieldTime).arg(startDate).arg(endDate);
    }
    else if (freq == hourly)
    {
        idVar = getHourlyVarCode(variable);
        if (idVar == NODATA)
        {
            *myError = "Variable not existing";
            return allDataVarList;
        }
        tableName = _tableHourly.prefix + id + _tableHourly.postFix;
        startDate = myFirstTime.date().toString("yyyy-MM-dd") + " " + myFirstTime.time().toString("hh:mm");
        endDate = myLastTime.date().toString("yyyy-MM-dd") + " " + myLastTime.time().toString("hh:mm");
        statement = QString( "SELECT * FROM `%1` WHERE VariableCode = '%2' AND `%3` >= '%4' AND `%3`<= '%5'")
                        .arg(tableName).arg(idVar).arg(_tableHourly.fieldTime).arg(startDate).arg(endDate);
    }
    else
    {
        *myError = "Frequency should be daily or hourly";
        return allDataVarList;
    }
    QDateTime dateTime;
    QDate date;
    if( !myQuery.exec(statement) )
    {
        *myError = myQuery.lastError().text();
        return allDataVarList;
    }
    else
    {
        while (myQuery.next())
        {
            if (freq == daily)
            {
                if (! getValue(myQuery.value(_tableDaily.fieldTime), &date))
                {
                    *myError = "Missing fieldTime";
                    return allDataVarList;
                }
                myDateStr = date.toString("yyyy-MM-dd");
            }
            else if (freq == hourly)
            {
                if (! getValue(myQuery.value(_tableHourly.fieldTime), &dateTime))
                {
                    *myError = "Missing fieldTime";
                    return allDataVarList;
                }
                // LC dateTime.toString direttamente ritorna una stringa vuota nelle ore di passaggio all'ora legale
                myDateStr = dateTime.date().toString("yyyy-MM-dd") + " " + dateTime.time().toString("hh:mm");
            }

            dateStr.push_back(myDateStr);
            value = myQuery.value(2).toFloat();
            allDataVarList.push_back(value);
        }
    }

    return allDataVarList;
}


std::vector<float> Crit3DMeteoGridDbHandler::loadGridDailyVarFixedFields(QString *myError, QString meteoPoint, meteoVariable variable, QDate first, QDate last, QDate* firstDateDB)
{
    QSqlQuery qry(_db);
    QString tableD = _tableDaily.prefix + meteoPoint + _tableDaily.postFix;
    QDate date, previousDate;

    std::vector<float> dailyVarList;

    float value;
    int firstRow = 1;
    QString varField;

    int varCode = getDailyVarCode(variable);

    if (varCode == NODATA)
    {
        *myError = "Variable not existing";
        return dailyVarList;
    }

    for (unsigned int i=0; i < _tableDaily.varcode.size(); i++)
    {
        if(_tableDaily.varcode[i].varCode == varCode)
        {
            varField = _tableDaily.varcode[i].varField.toLower();
            break;
        }
    }

    QString statement = QString("SELECT `%1`, `%2` FROM `%3` WHERE `%1` >= '%4' AND `%1` <= '%5' ORDER BY `%1`").arg(_tableDaily.fieldTime).arg(varField).arg(tableD).arg(first.toString("yyyy-MM-dd")).arg(last.toString("yyyy-MM-dd"));
    if( !qry.exec(statement) )
    {
        *myError = qry.lastError().text();
    }
    else
    {

        while (qry.next())
        {
            if (firstRow)
            {
                if (!getValue(qry.value(_tableDaily.fieldTime), firstDateDB))
                {
                    *myError = "Missing fieldTime";
                    return dailyVarList;
                }

                if (!getValue(qry.value(varField), &value))
                {
                    *myError = "Missing Value";
                }
                dailyVarList.push_back(value);
                previousDate = *firstDateDB;
                firstRow = 0;
            }
            else
            {
                if (!getValue(qry.value(_tableDaily.fieldTime), &date))
                {
                    *myError = "Missing fieldTime";
                    return dailyVarList;
                }

                int missingDate = previousDate.daysTo(date);
                for (int i =1; i<missingDate; i++)
                {
                    dailyVarList.push_back(NODATA);
                }

                if (!getValue(qry.value(varField), &value))
                {
                    *myError = "Missing Value";
                }
                dailyVarList.push_back(value);
                previousDate = date;
            }

        }

    }

    return dailyVarList;
}


std::vector<float> Crit3DMeteoGridDbHandler::loadGridHourlyVar(QString *myError, QString meteoPoint, meteoVariable variable, QDateTime first, QDateTime last, QDateTime* firstDateDB)
{

    QSqlQuery qry(_db);
    QString tableH = _tableHourly.prefix + meteoPoint + _tableHourly.postFix;
    QDateTime dateTime, previousDateTime;
    dateTime.setTimeSpec(Qt::UTC);
    previousDateTime.setTimeSpec(Qt::UTC);

    std::vector<float> hourlyVarList;

    float value;
    unsigned row;
    unsigned col;
    bool firstRow = true;

    int varCode = getHourlyVarCode(variable);

    if (varCode == NODATA)
    {
        *myError = "Variable not existing";
        return hourlyVarList;
    }

    if (!_meteoGrid->findMeteoPointFromId(&row, &col, meteoPoint.toStdString()) )
    {
        *myError = "Missing MeteoPoint id";
        return hourlyVarList;
    }

    QString statement = QString("SELECT * FROM `%1` WHERE VariableCode = '%2' AND `%3` >= '%4' AND `%3` <= '%5' ORDER BY `%3`").arg(tableH).arg(varCode).arg(_tableHourly.fieldTime).arg(first.toString("yyyy-MM-dd hh:mm")).arg(last.toString("yyyy-MM-dd hh:mm"));
    if( !qry.exec(statement) )
    {
        *myError = qry.lastError().text();
    }
    else
    {

        while (qry.next())
        {
            if (firstRow)
            {
                if (!getValue(qry.value(_tableHourly.fieldTime), firstDateDB))
                {
                    *myError = "Missing fieldTime";
                    return hourlyVarList;
                }

                if (!getValue(qry.value("Value"), &value))
                {
                    *myError = "Missing Value";
                }
                hourlyVarList.push_back(value);
                previousDateTime = *firstDateDB;
                firstRow = false;
            }
            else
            {
                if (!getValue(qry.value(_tableHourly.fieldTime), &dateTime))
                {
                    *myError = "Missing fieldTime";
                    return hourlyVarList;
                }

                int missingDateTime = previousDateTime.msecsTo(dateTime)/(1000*3600);
                for (int i = 1; i < missingDateTime; i++)
                {
                    hourlyVarList.push_back(NODATA);
                }

                if (!getValue(qry.value("Value"), &value))
                {
                    *myError = "Missing Value";
                }
                hourlyVarList.push_back(value);
                previousDateTime = dateTime;
            }

        }

    }

    return hourlyVarList;

}


std::vector<float> Crit3DMeteoGridDbHandler::loadGridHourlyVarFixedFields(QString *myError, QString meteoPoint, meteoVariable variable, QDateTime first, QDateTime last, QDateTime* firstDateDB)
{
    QSqlQuery qry(_db);
    QString tableH = _tableHourly.prefix + meteoPoint + _tableHourly.postFix;
    QDateTime dateTime, previousDateTime;

    std::vector<float> hourlyVarList;

    float value;
    int firstRow = 1;
    QString varField;

    int varCode = getHourlyVarCode(variable);

    if (varCode == NODATA)
    {
        *myError = "Variable not existing";
        return hourlyVarList;
    }

    for (unsigned int i=0; i < _tableHourly.varcode.size(); i++)
    {
        if(_tableHourly.varcode[i].varCode == varCode)
        {
            varField = _tableHourly.varcode[i].varField.toLower();
            break;
        }
    }
    // take also 00:00 day after
    last = last.addSecs(3600);

    QString statement = QString("SELECT `%1`, `%2` FROM `%3` WHERE `%1` >= '%4' AND `%1` <= '%5' ORDER BY `%1`").arg(_tableHourly.fieldTime).arg(varField).arg(tableH).arg(first.toString("yyyy-MM-dd hh:mm")).arg(last.toString("yyyy-MM-dd hh:mm"));
    if( !qry.exec(statement) )
    {
        *myError = qry.lastError().text();
    }
    else
    {

        while (qry.next())
        {
            if (firstRow)
            {
                if (!getValue(qry.value(_tableHourly.fieldTime), firstDateDB))
                {
                    *myError = "Missing fieldTime";
                    return hourlyVarList;
                }

                if (!getValue(qry.value(varField), &value))
                {
                    *myError = "Missing Value";
                }
                hourlyVarList.push_back(value);
                previousDateTime = *firstDateDB;
                firstRow = 0;
            }
            else
            {
                if (!getValue(qry.value(_tableHourly.fieldTime), &dateTime))
                {
                    *myError = "Missing fieldTime";
                    return hourlyVarList;
                }

                int missingDateTime = previousDateTime.msecsTo(dateTime)/(1000*3600);
                for (int i =1; i<missingDateTime; i++)
                {
                    hourlyVarList.push_back(NODATA);
                }

                if (!getValue(qry.value(varField), &value))
                {
                    *myError = "Missing Value";
                }
                hourlyVarList.push_back(value);
                previousDateTime = dateTime;
            }


        }

    }

    return hourlyVarList;
}


bool Crit3DMeteoGridDbHandler::saveCellGridDailyData(QString *myError, QString meteoPointID, int row, int col, QDate firstDate, QDate lastDate,
                                                     QList<meteoVariable> meteoVariableList, Crit3DMeteoSettings* meteoSettings)
{
    QSqlQuery qry(_db);
    QString tableD = _tableDaily.prefix + meteoPointID + _tableDaily.postFix;

    QString statement = QString("CREATE TABLE IF NOT EXISTS `%1`"
                                "(%2 date, VariableCode tinyint(3) UNSIGNED, Value float(6,1), PRIMARY KEY(%2,VariableCode))").arg(tableD).arg(_tableDaily.fieldTime);

    if( !qry.exec(statement) )
    {
        *myError = qry.lastError().text();
        return false;
    }
    else
    {
        statement =  QString(("REPLACE INTO `%1` VALUES")).arg(tableD);

        foreach (meteoVariable meteoVar, meteoVariableList)
            if (getVarFrequency(meteoVar) == daily)
            {
                int varCode = getDailyVarCode(meteoVar);
                for (QDate date = firstDate; date <= lastDate; date = date.addDays(1))
                {
                    float value = meteoGrid()->meteoPoint(row, col).getMeteoPointValueD(getCrit3DDate(date), meteoVar, meteoSettings);
                    QString valueS = QString("'%1'").arg(double(value));
                    if (isEqual(value, NODATA)) valueS = "NULL";

                    statement += QString(" ('%1','%2',%3),").arg(date.toString("yyyy-MM-dd")).arg(varCode).arg(valueS);
                }
            }

        statement = statement.left(statement.length() - 1);

        if( !qry.exec(statement) )
        {
            *myError = qry.lastError().text();
            return false;
        }
    }

    return true;
}


// warning: delete all previous data
bool Crit3DMeteoGridDbHandler::deleteAndWriteCellGridDailyData(QString& myError, QString meteoPointID, int row, int col, QDate firstDate, QDate lastDate,
                                                     QList<meteoVariable> meteoVariableList, Crit3DMeteoSettings* meteoSettings)
{
    QSqlQuery qry(_db);
    QString tableD = _tableDaily.prefix + meteoPointID + _tableDaily.postFix;

    QString statement = QString("DROP TABLE `%1`").arg(tableD);
    qry.exec(statement);

    statement = QString("CREATE TABLE `%1`(%2 date, VariableCode tinyint(3) UNSIGNED, Value float(6,1), PRIMARY KEY(%2,VariableCode))").arg(tableD).arg(_tableDaily.fieldTime);
    if( !qry.exec(statement) )
    {
        myError = qry.lastError().text();
        return false;
    }

    statement =  QString(("INSERT INTO `%1` VALUES")).arg(tableD);

    foreach (meteoVariable meteoVar, meteoVariableList)
    {
        if (getVarFrequency(meteoVar) == daily)
        {
            int varCode = getDailyVarCode(meteoVar);
            for (QDate date = firstDate; date <= lastDate; date = date.addDays(1))
            {
                float value = meteoGrid()->meteoPoint(row, col).getMeteoPointValueD(getCrit3DDate(date), meteoVar, meteoSettings);
                QString valueS = QString("'%1'").arg(double(value));
                if (isEqual(value, NODATA)) valueS = "NULL";

                statement += QString(" ('%1','%2',%3),").arg(date.toString("yyyy-MM-dd")).arg(varCode).arg(valueS);
            }
        }
    }

    statement = statement.left(statement.length() - 1);
    if( !qry.exec(statement) )
    {
        myError = qry.lastError().text();
        return false;
    }

    return true;
}


bool Crit3DMeteoGridDbHandler::saveCellGridDailyDataEnsemble(QString *myError, QString meteoPointID, int row, int col, QDate firstDate, QDate lastDate,
                                                     QList<meteoVariable> meteoVariableList, int memberNr, Crit3DMeteoSettings* meteoSettings)
{
    QSqlQuery qry(_db);
    QString tableD = _tableDaily.prefix + meteoPointID + _tableDaily.postFix;

    QString statement = QString("CREATE TABLE IF NOT EXISTS `%1`"
                                "(%2 date, VariableCode tinyint(3) UNSIGNED, Value float(6,1), MemberNr int(11), PRIMARY KEY(%2,VariableCode,MemberNr))").arg(tableD).arg(_tableDaily.fieldTime);

    if( !qry.exec(statement) )
    {
        *myError = qry.lastError().text();
        return false;
    }
    else
    {
        statement =  QString(("REPLACE INTO `%1` (%2, VariableCode, Value, MemberNr) VALUES ")).arg(tableD).arg(_tableDaily.fieldTime);

        foreach (meteoVariable meteoVar, meteoVariableList)
            if (getVarFrequency(meteoVar) == daily)
            {
                for (QDate date = firstDate; date <= lastDate; date = date.addDays(1))
                {
                    float value = meteoGrid()->meteoPoint(row, col).getMeteoPointValueD(getCrit3DDate(date), meteoVar, meteoSettings);
                    QString valueS = QString("'%1'").arg(value);
                    if (isEqual(value, NODATA)) valueS = "NULL";

                    int varCode = getDailyVarCode(meteoVar);

                    statement += QString(" ('%1','%2',%3,'%4'),").arg(date.toString("yyyy-MM-dd")).arg(varCode).arg(valueS).arg(memberNr);
                }
            }

        statement = statement.left(statement.length() - 1);

        if( !qry.exec(statement) )
        {
            *myError = qry.lastError().text();
            return false;
        }
    }

    return true;
}

bool Crit3DMeteoGridDbHandler::saveListHourlyData(QString *myError, QString meteoPointID, QDateTime firstDateTime, meteoVariable meteoVar, QList<float> values)
{
    QSqlQuery qry(_db);
    QString tableH = _tableHourly.prefix + meteoPointID + _tableHourly.postFix;
    int varCode = getHourlyVarCode(meteoVar);

    QString statement = QString("CREATE TABLE IF NOT EXISTS `%1`"
                                "(%2 datetime, VariableCode tinyint(3) UNSIGNED, Value float(6,1), PRIMARY KEY(%2,VariableCode))").arg(tableH).arg(_tableHourly.fieldTime);

    qry.exec(statement);
    int nHours = values.size();

    QDateTime last = firstDateTime.addSecs(3600*(nHours-1));
    statement = QString("DELETE FROM `%1` WHERE %2 BETWEEN CAST('%3' AS DATETIME) AND CAST('%4' AS DATETIME) AND VariableCode = '%5'")
                            .arg(tableH).arg(_tableHourly.fieldTime).arg(firstDateTime.toString("yyyy-MM-dd hh:mm:00")).arg(last.toString("yyyy-MM-dd hh:mm:00")).arg(varCode);
    if( !qry.exec(statement) )
    {
        *myError = qry.lastError().text();
        return false;
    }
    else
    {
        statement =  QString(("INSERT INTO `%1` (%2, VariableCode, Value) VALUES ")).arg(tableH).arg(_tableHourly.fieldTime);
        for (int i = 0; i<values.size(); i++)
        {
            float value = values[i];
            QString valueS = QString("'%1'").arg(value);
            QDateTime date = firstDateTime.addSecs(3600*i);
            if (isEqual(value, NODATA))
            {
                valueS = "NULL";
            }

            statement += QString(" ('%1','%2',%3),").arg(date.toString("yyyy-MM-dd hh:mm:00")).arg(varCode).arg(valueS);
        }

        statement = statement.left(statement.length() - 1);
        if( !qry.exec(statement) )
        {
            *myError = qry.lastError().text();
            return false;
        }
    }

    return true;
}

bool Crit3DMeteoGridDbHandler::saveListDailyData(QString *myError, QString meteoPointID, QDate firstDate, meteoVariable meteoVar, QList<float> values, bool reverseOrder)
{
    QSqlQuery qry(_db);
    QString tableD = _tableDaily.prefix + meteoPointID + _tableDaily.postFix;
    int varCode = getDailyVarCode(meteoVar);

    QString statement = QString("CREATE TABLE IF NOT EXISTS `%1`"
                                "(%2 date, VariableCode tinyint(3) UNSIGNED, Value float(6,1), PRIMARY KEY(%2,VariableCode))").arg(tableD).arg(_tableDaily.fieldTime);

    qry.exec(statement);
    int nDays = values.size();

    QDate lastDate = firstDate.addDays(nDays-1);
    statement = QString("DELETE FROM `%1` WHERE %2 BETWEEN CAST('%3' AS DATE) AND CAST('%4' AS DATE) AND VariableCode = '%5'")
                            .arg(tableD).arg(_tableDaily.fieldTime).arg(firstDate.toString("yyyy-MM-dd")).arg(lastDate.toString("yyyy-MM-dd")).arg(varCode);

    if( !qry.exec(statement) )
    {
        *myError = qry.lastError().text();
        return false;
    }
    else
    {
        statement =  QString(("INSERT INTO `%1` (%2, VariableCode, Value) VALUES ")).arg(tableD).arg(_tableDaily.fieldTime);
        for (int i = 0; i<values.size(); i++)
        {
            float value;
            if (reverseOrder)
            {
                value = values[values.size()-1-i];  // reverse order
            }
            else
            {
                value = values[i];
            }
            QString valueS = QString("'%1'").arg(value);
            QDate date = firstDate.addDays(i);
            if (isEqual(value, NODATA)) valueS = "NULL";
            statement += QString(" ('%1','%2',%3),").arg(date.toString("yyyy-MM-dd")).arg(varCode).arg(valueS);
        }

        statement = statement.left(statement.length() - 1);

        if( !qry.exec(statement) )
        {
            *myError = qry.lastError().text();
            return false;
        }
    }
    return true;
}

bool Crit3DMeteoGridDbHandler::saveListDailyDataEnsemble(QString *myError, QString meteoPointID, QDate date, meteoVariable meteoVar, QList<float> values)
{
    QSqlQuery qry(_db);
    QString tableD = _tableDaily.prefix + meteoPointID + _tableDaily.postFix;
    int varCode = getDailyVarCode(meteoVar);

    QString statement = QString("CREATE TABLE IF NOT EXISTS `%1`"
                                "(%2 date, VariableCode tinyint(3) UNSIGNED, Value float(6,1), MemberNr int(11), PRIMARY KEY(%2,VariableCode,MemberNr))").arg(tableD).arg(_tableDaily.fieldTime);

    qry.exec(statement);
    statement = QString("DELETE FROM `%1` WHERE %2 = DATE('%3') AND VariableCode = '%4'")
                            .arg(tableD).arg(_tableDaily.fieldTime).arg(date.toString("yyyy-MM-dd")).arg(varCode);
    if( !qry.exec(statement) )
    {
        *myError = qry.lastError().text();
        return false;
    }
    else
    {
        statement =  QString(("INSERT INTO `%1` (%2, VariableCode, Value, MemberNr) VALUES ")).arg(tableD).arg(_tableDaily.fieldTime);
        for (int i = 0; i<values.size(); i++)
        {
            float value = values[i];
            QString valueS = QString("'%1'").arg(value);
            if (isEqual(value, NODATA)) valueS = "NULL";
            int memberNr = values.size() - i;  // reverse order

            statement += QString(" ('%1','%2',%3,'%4'),").arg(date.toString("yyyy-MM-dd")).arg(varCode).arg(valueS).arg(memberNr);
        }

        statement = statement.left(statement.length() - 1);

        if( !qry.exec(statement) )
        {
            *myError = qry.lastError().text();
            return false;
        }
    }

    return true;
}

bool Crit3DMeteoGridDbHandler::cleanDailyOldData(QString *myError, QDate date)
{
    QSqlQuery qry(_db);
    QString statement = QString("SHOW TABLES LIKE '%1%%2'").arg(_tableDaily.prefix).arg(_tableDaily.postFix);
    if( !qry.exec(statement) )
    {
        *myError = qry.lastError().text();
        return false;
    }
    else
    {
        while( qry.next() )
        {
            QString tableName = qry.value(0).toString();
            statement = QString("DELETE FROM `%1` WHERE %2 < DATE('%3')")
                                        .arg(tableName).arg(_tableDaily.fieldTime).arg(date.toString("yyyy-MM-dd"));
            if( !qry.exec(statement) )
            {
                *myError = qry.lastError().text();
                return false;
            }

        }
    }
    return true;
}

bool Crit3DMeteoGridDbHandler::saveCellGridDailyDataFF(QString *myError, QString meteoPointID, int row, int col, QDate firstDate, QDate lastDate, Crit3DMeteoSettings* meteoSettings)
{
    QSqlQuery qry(_db);
    QString tableD = _tableDaily.prefix + meteoPointID + _tableDaily.postFix;
    QString tableFields;


    for (unsigned int i=0; i < _tableDaily.varcode.size(); i++)
    {
        QString var = _tableDaily.varcode[i].varPragaName;
        QString type = _mapDailyMySqlVarType[getMeteoVar(var.toStdString())];
        QString varFieldItem = _tableDaily.varcode[i].varField;
        tableFields = tableFields  + ", " + varFieldItem.toLower() + " " + type;
    }

    QString statement = QString("CREATE TABLE IF NOT EXISTS `%1`").arg(tableD) + QString("(`%1` date ").arg(_tableDaily.fieldTime) + tableFields + QString(", PRIMARY KEY(`%1`))").arg(_tableDaily.fieldTime);

    if( !qry.exec(statement) )
    {
        *myError = qry.lastError().text();
        return false;
    }
    else
    {
        statement =  QString(("REPLACE INTO `%1` VALUES")).arg(tableD);
        int nrDays = firstDate.daysTo(lastDate) + 1;
        for (int i = 0; i < nrDays; i++)
        {
            QDate date = firstDate.addDays(i);
            statement += QString(" ('%1',").arg(date.toString("yyyy-MM-dd"));
            for (unsigned int j = 0; j < _tableDaily.varcode.size(); j++)
            {
                float value = meteoGrid()->meteoPoint(row,col).getMeteoPointValueD(getCrit3DDate(date), getDailyVarFieldEnum(_tableDaily.varcode[j].varField), meteoSettings);
                QString valueS = QString("'%1'").arg(value);
                if (value == NODATA)
                    valueS = "NULL";

                statement += QString("%1,").arg(valueS);
            }
            statement = statement.left(statement.length() - 1);
            statement += QString("),");
        }
        statement = statement.left(statement.length() - 1);

        if( !qry.exec(statement) )
        {
            *myError = qry.lastError().text();
            return false;
        }
    }

    return true;
}


bool Crit3DMeteoGridDbHandler::saveCellCurrentGridDailyList(QString meteoPointID, QList<QString> listEntries, QString& errorStr)
{
    QSqlQuery qry(_db);
    QString tableD = _tableDaily.prefix + meteoPointID + _tableDaily.postFix;

    QString statement = QString("CREATE TABLE IF NOT EXISTS `%1` "
                                "(`%2` date, VariableCode tinyint(3) UNSIGNED, Value float(6,1), PRIMARY KEY(`%2`,VariableCode))")
                                .arg(tableD, _tableDaily.fieldTime);

    if( !qry.exec(statement) )
    {
        errorStr = qry.lastError().text();
        return false;
    }
    else
    {
        statement = QString("REPLACE INTO `%1` VALUES ").arg(tableD);
        statement = statement + listEntries.join(",");

        if(! qry.exec(statement) )
        {
            errorStr = qry.lastError().text();
            return false;
        }
    }

    return true;
}


bool Crit3DMeteoGridDbHandler::saveCellCurrentGridHourlyList(QString meteoPointID, QList<QString> listEntries, QString &errorStr)
{
    QSqlQuery qry(_db);
    QString tableH = _tableHourly.prefix + meteoPointID + _tableHourly.postFix;

    QString statement = QString("CREATE TABLE IF NOT EXISTS `%1` "
                                "(`%2` datetime, VariableCode tinyint(3) UNSIGNED, Value float(6,1), PRIMARY KEY(`%2`,VariableCode))")
                                .arg(tableH, _tableHourly.fieldTime);
    if(! qry.exec(statement))
    {
        errorStr = qry.lastError().text();
        return false;
    }
    else
    {
        statement = QString("REPLACE INTO `%1` VALUES ").arg(tableH);
        statement = statement + listEntries.join(",");

        if(! qry.exec(statement))
        {
            errorStr = qry.lastError().text();
            return false;
        }
    }

    return true;
}


bool Crit3DMeteoGridDbHandler::saveCellCurrentGridDaily(QString *myError, QString meteoPointID, QDate date, int varCode, float value)
{
    QSqlQuery qry(_db);
    QString tableD = _tableDaily.prefix + meteoPointID + _tableDaily.postFix;


    QString statement = QString("CREATE TABLE IF NOT EXISTS `%1` "
                                "(`%2` date, VariableCode tinyint(3) UNSIGNED, Value float(6,1), PRIMARY KEY(`%2`,VariableCode))").arg(tableD).arg(_tableDaily.fieldTime);

    if( !qry.exec(statement) )
    {
        *myError = qry.lastError().text();
        return false;
    }
    else
    {
        QString valueS = QString("'%1'").arg(value);
        if (value == NODATA)
            valueS = "NULL";

        statement = QString("REPLACE INTO `%1` VALUES ('%2','%3',%4)").arg(tableD).arg(date.toString("yyyy-MM-dd")).arg(varCode).arg(valueS);

        if( !qry.exec(statement) )
        {
            *myError = qry.lastError().text();
            return false;
        }
    }

    return true;
}


bool Crit3DMeteoGridDbHandler::saveCellCurrentGridDailyFF(QString& errorStr, QString meteoPointID, QDate date,
                                                          QString varPragaName, float value)
{
    QSqlQuery qry(_db);
    QString tableD = _tableDaily.prefix + meteoPointID + _tableDaily.postFix;
    QString tableFields;
    QString varField;

    for (unsigned int i=0; i < _tableDaily.varcode.size(); i++)
    {
        QString var = _tableDaily.varcode[i].varPragaName;
        if (var == varPragaName)
        {
            varField = _tableDaily.varcode[i].varField;
        }
        QString type = _mapDailyMySqlVarType[getMeteoVar(var.toStdString())];

        QString varFieldItem = _tableDaily.varcode[i].varField;
        tableFields = tableFields  + ", " + varFieldItem.toLower() + " " + type;
    }


    QString statement = QString("CREATE TABLE IF NOT EXISTS `%1`").arg(tableD) + QString("(%1 date ").arg(_tableDaily.fieldTime) + tableFields + QString(", PRIMARY KEY(%1))").arg(_tableDaily.fieldTime);

    if( !qry.exec(statement) )
    {
        errorStr = qry.lastError().text();
        return false;
    }
    else
    {

        QString valueS = QString("'%1'").arg(value);
        if (value == NODATA)
            valueS = "NULL";

        statement = QString("INSERT INTO `%1` (`%2`, `%3`) VALUES ('%4',%5) ON DUPLICATE KEY UPDATE `%3`= %5")
                        .arg(tableD, _tableDaily.fieldTime, varField.toLower(), date.toString("yyyy-MM-dd"), valueS);

        if( !qry.exec(statement) )
        {
            errorStr = qry.lastError().text();
            return false;
        }

    }

    return true;
}


bool Crit3DMeteoGridDbHandler::saveCellGridMonthlyData(QString *myError, QString meteoPointID, int row, int col, QDate firstDate, QDate lastDate,
                                                     QList<meteoVariable> meteoVariableList)
{
    QSqlQuery qry(_db);
    QString table = "MonthlyData";

    QString statement = QString("CREATE TABLE IF NOT EXISTS `%1`"
                                "(PragaYear smallint(4) UNSIGNED, PragaMonth tinyint(2) UNSIGNED, PointCode VARCHAR(6), "
                                "VariableCode tinyint(3) UNSIGNED, Value float(6,1), PRIMARY KEY(PragaYear,PragaMonth,PointCode,VariableCode))").arg(table);

    if( !qry.exec(statement) )
    {
        *myError = qry.lastError().text();
        return false;
    }
    else
    {
        statement =  QString(("REPLACE INTO `%1` VALUES")).arg(table);

        // set day=1 to better comparison
        firstDate.setDate(firstDate.year(),firstDate.month(),1);
        lastDate.setDate(lastDate.year(),lastDate.month(),1);

        foreach (meteoVariable meteoVar, meteoVariableList)
            if (getVarFrequency(meteoVar) == monthly)
            {
                for (QDate date = firstDate; date<=lastDate; date = date.addMonths(1))
                {
                    float value = meteoGrid()->meteoPoint(row, col).getMeteoPointValueM(getCrit3DDate(date), meteoVar);
                    QString valueS = QString("'%1'").arg(value);
                    if (isEqual(value, NODATA)) valueS = "NULL";

                    int varCode = getMonthlyVarCode(meteoVar);

                    statement += QString(" (%1,%2,'%3','%4',%5),").arg(date.year()).arg(date.month()).arg(meteoPointID).arg(varCode).arg(valueS);
                }
            }

        statement = statement.left(statement.length() - 1);

        if( ! qry.exec(statement) )
        {
            *myError = qry.lastError().text();
            return false;
        }
    }

    return true;
}

bool Crit3DMeteoGridDbHandler::saveGridData(QString *myError, QDateTime firstTime, QDateTime lastTime, QList<meteoVariable> meteoVariableList, Crit3DMeteoSettings* meteoSettings)
{
    std::string id;
    meteoVariable var;
    frequencyType freq;
    bool isHourly = false, isDaily = false;

    foreach (var, meteoVariableList)
    {
        freq = getVarFrequency(var);
        if (freq == hourly) isHourly = true;
        if (freq == daily) isDaily = true;
    }

    QDate lastDate = lastTime.date();
    if (lastTime.time().hour() == 0) lastDate = lastDate.addDays(-1);

    for (int row = 0; row < gridStructure().header().nrRows; row++)
        for (int col = 0; col < gridStructure().header().nrCols; col++)
            if (meteoGrid()->getMeteoPointActiveId(row, col, &id))
            {
                if (! gridStructure().isFixedFields())
                {
                    if (isHourly) saveCellGridHourlyData(myError, QString::fromStdString(id), row, col, firstTime, lastTime, meteoVariableList);
                    if (isDaily) saveCellGridDailyData(myError, QString::fromStdString(id), row, col, firstTime.date(), lastDate, meteoVariableList, meteoSettings);
                }
                else
                {
                    if (isHourly) saveCellGridHourlyDataFF(myError, QString::fromStdString(id), row, col, firstTime, lastTime);
                    if (isDaily) saveCellGridDailyDataFF(myError, QString::fromStdString(id), row, col, firstTime.date(), lastDate, meteoSettings);
                }
            }

    return true;
}


bool Crit3DMeteoGridDbHandler::saveGridHourlyData(QString *myError, QDateTime firstDate, QDateTime lastDate, QList<meteoVariable> meteoVariableList)
{
    std::string id;

    for (int row = 0; row < gridStructure().header().nrRows; row++)
    {
        for (int col = 0; col < gridStructure().header().nrCols; col++)
        {
            if (meteoGrid()->getMeteoPointActiveId(row, col, &id))
            {
                if (!gridStructure().isFixedFields())
                {
                    saveCellGridHourlyData(myError, QString::fromStdString(id), row, col, firstDate, lastDate, meteoVariableList);
                }
                else
                {
                    saveCellGridHourlyDataFF(myError, QString::fromStdString(id), row, col, firstDate, lastDate);
                }
            }
        }
    }

    return true;
}

bool Crit3DMeteoGridDbHandler::saveGridDailyData(QString *myError, QDateTime firstDate, QDateTime lastDate, QList<meteoVariable> meteoVariableList, Crit3DMeteoSettings* meteoSettings)
{
    std::string id;

    for (int row = 0; row < gridStructure().header().nrRows; row++)
    {
        for (int col = 0; col < gridStructure().header().nrCols; col++)
        {
            if (meteoGrid()->getMeteoPointActiveId(row, col, &id))
            {
                if (! gridStructure().isFixedFields())
                {
                    saveCellGridDailyData(myError, QString::fromStdString(id), row, col, firstDate.date(), lastDate.date(), meteoVariableList, meteoSettings);
                }
                else
                {
                    saveCellGridDailyDataFF(myError, QString::fromStdString(id), row, col, firstDate.date(), lastDate.date(), meteoSettings);
                }
            }
        }
    }

    return true;
}

bool Crit3DMeteoGridDbHandler::saveCellGridHourlyData(QString *myError, QString meteoPointID, int row, int col,
                                                      QDateTime firstTime, QDateTime lastTime, QList<meteoVariable> meteoVariableList)
{
    QSqlQuery qry(_db);
    QString tableH = _tableHourly.prefix + meteoPointID + _tableHourly.postFix;


    QString statement = QString("CREATE TABLE IF NOT EXISTS `%1` "
                                "(`%2` datetime, VariableCode tinyint(3) UNSIGNED, Value float(6,1), PRIMARY KEY(`%2`,VariableCode))").arg(tableH).arg(_tableHourly.fieldTime);

    if( !qry.exec(statement) )
    {
        *myError = qry.lastError().text();
        return false;
    }
    else
    {
        statement =  QString(("REPLACE INTO `%1` VALUES")).arg(tableH);

        foreach (meteoVariable meteoVar, meteoVariableList)
            if (getVarFrequency(meteoVar) == hourly)
            {
                for (QDateTime myTime = firstTime; myTime <= lastTime; myTime = myTime.addSecs(3600))
                {
                    float value = meteoGrid()->meteoPoint(row,col).getMeteoPointValueH(getCrit3DDate(myTime.date()), myTime.time().hour(), myTime.time().minute(), meteoVar);
                    QString valueS = QString("'%1'").arg(value);
                    if (isEqual(value, NODATA)) valueS = "NULL";

                    int varCode = getHourlyVarCode(meteoVar);
                    statement += QString(" ('%1','%2',%3),").arg(myTime.toString("yyyy-MM-dd hh:mm")).arg(varCode).arg(valueS);
                }
            }

        statement = statement.left(statement.length() - 1);

        if( !qry.exec(statement) )
        {
            *myError = qry.lastError().text();
            return false;
        }
    }

    return true;
}

bool Crit3DMeteoGridDbHandler::saveCellGridHourlyDataEnsemble(QString *myError, QString meteoPointID, int row, int col,
                                                      QDateTime firstTime, QDateTime lastTime, QList<meteoVariable> meteoVariableList, int memberNr)
{
    QSqlQuery qry(_db);
    QString tableH = _tableHourly.prefix + meteoPointID + _tableHourly.postFix;


    QString statement = QString("CREATE TABLE IF NOT EXISTS `%1` "
                                "(`%2` datetime, VariableCode tinyint(3) UNSIGNED, Value float(6,1), "
                                "MemberNr int(11), PRIMARY KEY(`%2`,VariableCode,MemberNr))").arg(tableH, _tableHourly.fieldTime);

    if( !qry.exec(statement) )
    {
        *myError = qry.lastError().text();
        return false;
    }
    else
    {
        statement =  QString(("REPLACE INTO `%1` (%2, VariableCode, Value, MemberNr) VALUES ")).arg(tableH, _tableHourly.fieldTime);

        foreach (meteoVariable meteoVar, meteoVariableList)
            if (getVarFrequency(meteoVar) == hourly)
            {
                for (QDateTime myTime = firstTime; myTime <= lastTime; myTime = myTime.addSecs(3600))
                {
                    float value = meteoGrid()->meteoPoint(row,col).getMeteoPointValueH(getCrit3DDate(myTime.date()), myTime.time().hour(), myTime.time().minute(), meteoVar);
                    QString valueS = QString("'%1'").arg(value);
                    if (isEqual(value, NODATA)) valueS = "NULL";

                    int varCode = getHourlyVarCode(meteoVar);
                    statement += QString(" ('%1','%2',%3,'%4'),").arg(myTime.toString("yyyy-MM-dd hh:mm")).arg(varCode).arg(valueS).arg(memberNr);
                }
            }

        statement = statement.left(statement.length() - 1);

        if( !qry.exec(statement) )
        {
            *myError = qry.lastError().text();
            return false;
        }
    }

    return true;
}

bool Crit3DMeteoGridDbHandler::saveCellGridHourlyDataFF(QString *myError, QString meteoPointID, int row, int col, QDateTime firstTime, QDateTime lastTime)
{
    QSqlQuery qry(_db);
    QString tableH = _tableHourly.prefix + meteoPointID + _tableHourly.postFix;
    QString tableFields;


    for (unsigned int i=0; i < _tableHourly.varcode.size(); i++)
    {
        QString var = _tableHourly.varcode[i].varPragaName;
        QString type = _mapHourlyMySqlVarType[getMeteoVar(var.toStdString())];
        QString varFieldItem = _tableHourly.varcode[i].varField;
        tableFields = tableFields  + ", " + varFieldItem.toLower() + " " + type;
    }

    QString statement = QString("CREATE TABLE IF NOT EXISTS `%1` ").arg(tableH) + QString("(`%1` datetime ").arg(_tableHourly.fieldTime) + tableFields + QString(", PRIMARY KEY(`%1`))").arg(_tableHourly.fieldTime);

    if( !qry.exec(statement) )
    {
        *myError = qry.lastError().text();
        return false;
    }
    else
    {
        statement =  QString(("REPLACE INTO `%1` VALUES")).arg(tableH);

        for (QDateTime myTime = firstTime; myTime <= lastTime; myTime = myTime.addSecs(3600))
        {
            statement += QString(" ('%1',").arg(myTime.toString("yyyy-MM-dd hh:mm"));
            for (unsigned int j = 0; j < _tableHourly.varcode.size(); j++)
            {
                float value = meteoGrid()->meteoPoint(row,col).getMeteoPointValueH(getCrit3DDate(myTime.date()),
                                            myTime.time().hour(), myTime.time().minute(),
                                            getHourlyVarFieldEnum(_tableHourly.varcode[j].varField));
                QString valueS = QString("'%1'").arg(double(value));
                if (value == NODATA)
                    valueS = "NULL";

                statement += QString("%1,").arg(valueS);
            }
            statement = statement.left(statement.length() - 1);
            statement += QString("),");
        }
        statement = statement.left(statement.length() - 1);

        if( !qry.exec(statement) )
        {
            *myError = qry.lastError().text();
            return false;
        }
    }

    return true;
}


bool Crit3DMeteoGridDbHandler::saveCellCurrentGridHourly(QString &errorStr, QString meteoPointID,
                                                         QDateTime dateTime, int varCode, float value)
{
    QSqlQuery qry(_db);
    QString tableH = _tableHourly.prefix + meteoPointID + _tableHourly.postFix;

    QString statement = QString("CREATE TABLE IF NOT EXISTS `%1` "
                                "(`%2` datetime, VariableCode tinyint(3) UNSIGNED, Value float(6,1), PRIMARY KEY(`%2`,VariableCode))")
                                .arg(tableH, _tableHourly.fieldTime);
    if( !qry.exec(statement) )
    {
        errorStr = qry.lastError().text();
        return false;
    }
    else
    {
        QString valueStr = QString("'%1'").arg(value);
        if (value == NODATA)
        {
            valueStr = "NULL";
        }

        statement = QString("REPLACE INTO `%1` VALUES ('%2','%3',%4)")
                        .arg(tableH, dateTime.toString("yyyy-MM-dd hh:mm")).arg(varCode).arg(valueStr);

        if( !qry.exec(statement) )
        {
            errorStr = qry.lastError().text();
            return false;
        }
    }

    return true;
}


bool Crit3DMeteoGridDbHandler::saveCellCurrentGridHourlyFF(QString& errorStr, QString meteoPointID, QDateTime dateTime,
                                                           QString varPragaName, float value)
{
    QSqlQuery qry(_db);
    QString tableH = _tableHourly.prefix + meteoPointID + _tableHourly.postFix;

    QString tableFields;
    QString varField;

    for (unsigned int i=0; i < _tableHourly.varcode.size(); i++)
    {
        QString var = _tableHourly.varcode[i].varPragaName;
        if (var == varPragaName)
        {
            varField = _tableHourly.varcode[i].varField;
        }
        QString varFieldItem = _tableHourly.varcode[i].varField;
        QString type = _mapHourlyMySqlVarType[getMeteoVar(var.toStdString())];
        tableFields = tableFields  + ", " + varFieldItem.toLower() + " " + type;
    }

    QString statement = QString("CREATE TABLE IF NOT EXISTS `%1` ").arg(tableH)
                        + QString("(`%1` datetime ").arg(_tableHourly.fieldTime)
                        + tableFields + QString(", PRIMARY KEY(`%1`))").arg(_tableHourly.fieldTime);

    if( !qry.exec(statement) )
    {
        errorStr = qry.lastError().text();
        return false;
    }
    else
    {
        QString valueS = QString("'%1'").arg(value);
        if (value == NODATA)
            valueS = "NULL";

        statement = QString("INSERT INTO `%1` (`%2`, `%3`) VALUES ('%4',%5) ON DUPLICATE KEY UPDATE `%3` = %5")
                        .arg(tableH, _tableHourly.fieldTime, varField.toLower(), dateTime.toString("yyyy-MM-dd hh:mm"), valueS);

        if( !qry.exec(statement) )
        {
            errorStr= qry.lastError().text();
            return false;
        }
    }

    return true;
}


bool Crit3DMeteoGridDbHandler::isDaily()
{
    if ( ! _firstDailyDate.isValid() || _firstDailyDate.year() == 1800
        || ! _lastDailyDate.isValid() || _lastDailyDate.year() == 1800 )
    {
        return false;
    }

    return true;
}


bool Crit3DMeteoGridDbHandler::isHourly()
{
    if ( ! _firstHourlyDate.isValid() || _firstHourlyDate.year() == 1800
        || ! _lastHourlyDate.isValid() || _lastHourlyDate.year() == 1800 )
    {
        return false;
    }

    return true;
}

bool Crit3DMeteoGridDbHandler::isMonthly()
{
    if ( ! _firstMonthlyDate.isValid() || _firstMonthlyDate.year() == 1800
        || ! _lastMonthlyDate.isValid() || _lastMonthlyDate.year() == 1800 )
    {
        return false;
    }

    return true;
}


QDate Crit3DMeteoGridDbHandler::getFirstDailyDate() const
{
    if (_firstDailyDate.year() == 1800)
    {
        return QDate(); // return null date
    }
    return _firstDailyDate;
}

QDate Crit3DMeteoGridDbHandler::getLastDailyDate() const
{
    if (_lastDailyDate.year() == 1800)
    {
        return QDate(); // return null date
    }
    return _lastDailyDate;
}

QDate Crit3DMeteoGridDbHandler::getFirstHourlyDate() const
{
    if (_firstHourlyDate.year() == 1800)
    {
        return QDate(); // return null date
    }
    return _firstHourlyDate;
}

QDate Crit3DMeteoGridDbHandler::getLastHourlyDate() const
{
    if (_lastHourlyDate.year() == 1800)
    {
        return QDate(); // return null date
    }
    return _lastHourlyDate;
}

QDate Crit3DMeteoGridDbHandler::getFirstMonthlytDate() const
{
    if (_firstMonthlyDate.year() == 1800)
    {
        return QDate(); // return null date
    }
    return _firstMonthlyDate;
}

QDate Crit3DMeteoGridDbHandler::getLastMonthlyDate() const
{
    if (_lastMonthlyDate.year() == 1800)
    {
        return QDate(); // return null date
    }
    return _lastMonthlyDate;
}

bool Crit3DMeteoGridDbHandler::idDailyList(QString *myError, QList<QString>* idMeteoList)
{
    QSqlQuery qry(_db);

    QString statement = QString("SHOW TABLES LIKE '%1%%2'").arg(_tableDaily.prefix, _tableDaily.postFix);
    if( !qry.exec(statement) )
    {
        *myError = qry.lastError().text();
        return false;
    }
    else
    {
        while( qry.next() )
        {
            QString tableName = qry.value(0).toString();
            if (!_tableDaily.prefix.isEmpty())
            {
                tableName.remove(0,_tableDaily.prefix.size());
            }
            if (!_tableDaily.postFix.isEmpty())
            {
                tableName.remove(tableName.size()-_tableDaily.postFix.size(),_tableDaily.postFix.size());
            }
            idMeteoList->append(tableName);
        }
    }
    return true;
}

bool Crit3DMeteoGridDbHandler::getYearList(QString *myError, QString meteoPoint, QList<QString>* yearList)
{
    QSqlQuery qry(_db);
    QString tableD = _tableDaily.prefix + meteoPoint + _tableDaily.postFix;

    QString statement = QString("SELECT DISTINCT DATE_FORMAT(`%1`,'%Y') as Year FROM `%2` ORDER BY Year").arg(_tableDaily.fieldTime, tableD);
    if( !qry.exec(statement) )
    {
        *myError = qry.lastError().text();
        return false;
    }
    else
    {
        QString year;
        while (qry.next())
        {
            getValue(qry.value("Year"), &year);
            if (year != "" && !yearList->contains(year))
            {
                yearList->append(year);
            }
        }

    }
    return true;
}

bool Crit3DMeteoGridDbHandler::saveLogProcedures(QString *myError, QString nameProc, QDate date)
{
    QSqlQuery qry(_db);
    QString table = "log_procedures";

    QString statement = QString("CREATE TABLE IF NOT EXISTS `%1` "
                                "(`nameProc` varchar(64) NOT NULL PRIMARY KEY, `lastDate` date)").arg(table);

    if( !qry.exec(statement) )
    {
        *myError = qry.lastError().text();
        return false;
    }
    else
    {
        statement = QString("REPLACE INTO `%1` VALUES ('%2','%3')").arg(table, nameProc, date.toString("yyyy-MM-dd"));

        if( !qry.exec(statement) )
        {
            *myError = qry.lastError().text();
            return false;
        }
    }

    return true;
}


/*!
 * \brief ExportDailyDataCsv
 * export gridded daily meteo data to csv files
 * \param variableList      list of meteo variables
 * \param idListFileName    text file of cells id list by columns - if idListFileName is empty save ALL cells
 * \param outputPath        path for output files
 * \return true on success, false otherwise
 */
bool Crit3DMeteoGridDbHandler::exportDailyDataCsv(QString &errorStr, QList<meteoVariable> variableList,
                                                  QDate firstDate, QDate lastDate, QString idListFileName, QString outputPath)
{
    errorStr = "";

    // check output path
    if (outputPath == "")
    {
        errorStr = "Missing output path.";
        return false;
    }

    QDir outDir(outputPath);
    // make directory
    if (! outDir.exists())
    {
        if (! outDir.mkpath(outputPath))
        {
            errorStr = "Wrong output path, unable to create directory: " + outputPath;
            return false;
        }
    }
    outputPath = outDir.absolutePath();

    bool isList = (! idListFileName.isEmpty());
    QList<QString> idList;
    if (isList)
    {
        if (! QFile::exists(idListFileName))
        {
            errorStr = "The ID list does not exist: " + idListFileName;
            return false;
        }

        idList = readListSingleColumn(idListFileName, errorStr);
        if (errorStr != "")
            return false;

        if (idList.size() == 0)
        {
            errorStr = "The ID list is empty: " + idListFileName;
            return false;
        }
    }

    for (int row = 0; row < gridStructure().header().nrRows; row++)
    {
        for (int col = 0; col < gridStructure().header().nrCols; col++)
        {
            QString id = QString::fromStdString(meteoGrid()->meteoPoints()[row][col]->id);
            if (! isList || idList.contains(id))
            {
                // read data
                bool isOk;
                if (gridStructure().isFixedFields())
                {
                    isOk = loadGridDailyDataFixedFields(errorStr, id, firstDate, lastDate);
                }
                else
                {
                    isOk = loadGridDailyData(errorStr, id, firstDate, lastDate);
                }
                if (! isOk)
                {
                    std::cout << "Error in reading cell id: " << id.toStdString() << "\n";
                    continue;
                }

                // create csv file
                QString csvFileName = outputPath + "/" + id + ".csv";
                QFile outputFile(csvFileName);
                isOk = outputFile.open(QIODevice::WriteOnly | QFile::Truncate);
                if (! isOk)
                {
                    std::cout << "Open CSV failed: " << csvFileName.toStdString() << "\n";
                    continue;
                }

                // write header
                QTextStream out(&outputFile);
                out << "Date";
                for (int i = 0; i < variableList.size(); i++)
                {
                    if (variableList[i] != noMeteoVar)
                    {
                        std::string varName = getMeteoVarName(variableList[i]);
                        std::string unit = getUnitFromVariable(variableList[i]);
                        QString VarString = QString::fromStdString(varName + " (" + unit + ")");
                        out << "," + VarString;
                    }
                }
                out << "\n";

                // write data
                QDate currentDate = firstDate;
                while (currentDate <= lastDate)
                {
                    Crit3DDate myDate = getCrit3DDate(currentDate);
                    out << currentDate.toString("yyyy-MM-dd");

                    for (int i = 0; i < variableList.size(); i++)
                    {
                        if (variableList[i] != noMeteoVar)
                        {
                            float value = _meteoGrid->meteoPointPointer(row,col)->getMeteoPointValueD(myDate, variableList[i]);
                            QString valueString = "";
                            if (value != NODATA)
                                valueString = QString::number(value);

                            out << "," << valueString;
                        }
                    }
                    out << "\n";

                    currentDate = currentDate.addDays(1);
                }

                outputFile.close();
            }
        }
    }

    return true;
}


bool Crit3DMeteoGridDbHandler::MeteoGridToRasterFlt(double cellSize, const gis::Crit3DGisSettings& gisSettings, gis::Crit3DRasterGrid& myGrid)
{
    if (! gridStructure().isUTM())
    {
        // lat/lon grid
        gis::Crit3DLatLonHeader latlonHeader = gridStructure().header();
        gis::getGeoExtentsFromLatLonHeader(gisSettings, cellSize, myGrid.header, &latlonHeader);
        if (!myGrid.initializeGrid(NODATA))
            return false;

        double utmx, utmy, lat, lon;
        int dataGridRow, dataGridCol;
        float myValue;

        for (int row = 0; row < myGrid.header->nrRows; row++)
        {
            for (int col = 0; col < myGrid.header->nrCols; col++)
            {
                myGrid.getXY(row, col, utmx, utmy);
                gis::getLatLonFromUtm(gisSettings, utmx, utmy, &lat, &lon);
                gis::getGridRowColFromXY (latlonHeader, lon, lat, &dataGridRow, &dataGridCol);
                if (dataGridRow < 0 || dataGridRow >= latlonHeader.nrRows || dataGridCol < 0 || dataGridCol >= latlonHeader.nrCols)
                {
                    myValue = NODATA;
                }
                else
                {
                    myValue = meteoGrid()->dataMeteoGrid.value[latlonHeader.nrRows-1-dataGridRow][dataGridCol];
                }
                if (myValue != NO_ACTIVE && myValue != NODATA)
                {
                    myGrid.value[row][col] = myValue;
                }
            }
        }
    }
    else
    {
        myGrid.copyGrid(meteoGrid()->dataMeteoGrid);
    }

    return true;
}


QDate Crit3DMeteoGridDbHandler::firstDate() const
{
    return _firstDate;
}

void Crit3DMeteoGridDbHandler::setFirstDate(const QDate &firstDate)
{
    _firstDate = firstDate;
}

QDate Crit3DMeteoGridDbHandler::lastDate() const
{
    return _lastDate;
}

void Crit3DMeteoGridDbHandler::setLastDate(const QDate &lastDate)
{
    _lastDate = lastDate;
}

Crit3DMeteoGrid *Crit3DMeteoGridDbHandler::meteoGrid() const
{
    return _meteoGrid;
}

void Crit3DMeteoGridDbHandler::setMeteoGrid(Crit3DMeteoGrid *meteoGrid)
{
    _meteoGrid = meteoGrid;
}

QSqlDatabase Crit3DMeteoGridDbHandler::db() const
{
    return _db;
}

void Crit3DMeteoGridDbHandler::setDb(const QSqlDatabase &db)
{
    _db = db;
}

QString Crit3DMeteoGridDbHandler::fileName() const
{
    return _fileName;
}

TXMLConnection Crit3DMeteoGridDbHandler::connection() const
{
    return _connection;
}

Crit3DMeteoGridStructure Crit3DMeteoGridDbHandler::gridStructure() const
{
    return _gridStructure;
}

TXMLTable Crit3DMeteoGridDbHandler::tableDaily() const
{
    return _tableDaily;
}

TXMLTable Crit3DMeteoGridDbHandler::tableHourly() const
{
    return _tableHourly;
}

TXMLTable Crit3DMeteoGridDbHandler::tableMonthly() const
{
    return _tableMonthly;
}

QString Crit3DMeteoGridDbHandler::tableDailyModel() const
{
    return _tableDailyModel;
}

QString Crit3DMeteoGridDbHandler::tableHourlyModel() const
{
    return _tableHourlyModel;
}
