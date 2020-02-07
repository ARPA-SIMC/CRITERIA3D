#include "dbMeteoGrid.h"
#include "basicMath.h"
#include "utilities.h"
#include "commonConstants.h"

#include <QtSql>


Crit3DMeteoGridDbHandler::Crit3DMeteoGridDbHandler()
{
    _meteoGrid = new Crit3DMeteoGrid();
}

Crit3DMeteoGridDbHandler::~Crit3DMeteoGridDbHandler()
{
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

    if (!parseXMLFile(xmlFileName, &xmlDoc, myError)) return false;

    QDomNode child;
    QDomNode secondChild;
    TXMLvar varTable;

    QDomNode ancestor = xmlDoc.documentElement().firstChild();
    QString myTag;
    QString mySecondTag;
    int nRow;
    int nCol;

    _tableDaily.exists = false;
    _tableHourly.exists = false;

    while(!ancestor.isNull())
    {
        if (ancestor.toElement().tagName().toUpper() == "CONNECTION")
        {
            child = ancestor.firstChild();
            while( !child.isNull())
            {
                myTag = child.toElement().tagName().toUpper();
                if (myTag == "PROVIDER")
                {
                    _connection.provider = child.toElement().text();
                    // remove white spaces
                    _connection.provider = _connection.provider.simplified();
                }
                else if (myTag == "SERVER")
                {
                    _connection.server = child.toElement().text();
                    // remove white spaces
                    _connection.server = _connection.server.simplified();
                }
                else if (myTag == "NAME")
                {
                    _connection.name = child.toElement().text();
                    // remove white spaces
                    _connection.server = _connection.server.simplified();
                }
                else if (myTag == "USER")
                {
                    _connection.user = child.toElement().text();
                    // remove white spaces
                    _connection.user = _connection.user.simplified();
                }
                else if (myTag == "PASSWORD")
                {
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


            child = ancestor.firstChild();
            gis::Crit3DGridHeader header;
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
                    header.llCorner.longitude = child.toElement().text().toFloat();
                }
                if (myTag == "YLL")
                {
                    header.llCorner.latitude = child.toElement().text().toFloat();
                }
                if (myTag == "NROWS")
                {
                    header.nrRows = child.toElement().text().toInt();
                    nRow = header.nrRows;
                }
                if (myTag == "NCOLS")
                {
                    header.nrCols = child.toElement().text().toInt();
                    nCol = header.nrCols;
                }
                if (myTag == "XWIDTH")
                {
                    header.dx = child.toElement().text().toFloat();
                }
                if (myTag == "YWIDTH")
                {
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


    _meteoGrid->setGridStructure(_gridStructure);

    _meteoGrid->initMeteoPoints(nRow, nCol);

    return true;
}

void Crit3DMeteoGridDbHandler::initMapMySqlVarType()
{
    _mapDailyMySqlVarType["DAILY_TMIN"] = "float(4,1)";
    _mapDailyMySqlVarType["DAILY_TMAX"] = "float(4,1)";
    _mapDailyMySqlVarType["DAILY_TAVG"] = "float(4,1)";
    _mapDailyMySqlVarType["DAILY_PREC"] = "float(4,1) UNSIGNED";
    _mapDailyMySqlVarType["DAILY_RHMIN"] = "tinyint(3) UNSIGNED";
    _mapDailyMySqlVarType["DAILY_RHMAX"] = "tinyint(3) UNSIGNED";
    _mapDailyMySqlVarType["DAILY_RHAVG"] = "tinyint(3) UNSIGNED";
    _mapDailyMySqlVarType["DAILY_RAD"] = "float(5,2) UNSIGNED";
    _mapDailyMySqlVarType["DAILY_W_INT_AVG"] = "float(3,1) UNSIGNED";
    _mapDailyMySqlVarType["DAILY_W_DIR"] = "smallint(3) UNSIGNED";
    _mapDailyMySqlVarType["DAILY_W_INT_MAX"] = "float(3,1) UNSIGNED";
    _mapDailyMySqlVarType["DAILY_ET0_HS"] = "float(3,1) UNSIGNED";
    _mapDailyMySqlVarType["DAILY_LEAFW"] = "tinyint(3) UNSIGNED";



    _mapHourlyMySqlVarType["TAVG"] = "float(4,1)";
    _mapHourlyMySqlVarType["PREC"] = "float(4,1) UNSIGNED";
    _mapHourlyMySqlVarType["RHAVG"] = "tinyint(3) UNSIGNED";
    _mapHourlyMySqlVarType["RAD"] = "float(5,1) UNSIGNED";
    _mapHourlyMySqlVarType["W_INT_AVG"] = "float(3,1) UNSIGNED";
    _mapHourlyMySqlVarType["W_DIR"] = "smallint(3) UNSIGNED";
    _mapHourlyMySqlVarType["ET0_HS"] = "float(3,1) UNSIGNED";
    _mapHourlyMySqlVarType["ET0_PM"] = "float(3,1) UNSIGNED";
    _mapHourlyMySqlVarType["LEAFW"] = "tinyint(3) UNSIGNED";

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
    // TODO problem with ssl connection
    _db.setConnectOptions();

    if (!_db.open())
    {
       *myError = "Connection with database fail.\n" + _db.lastError().text();
       return false;
    }
    else
    {
       qDebug() << "Database: connection ok";
       return true;
    }
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

    //qry.prepare( "SELECT * FROM CellsProperties ORDER BY Code" );
    QString statement = QString("SELECT * FROM `%1` ORDER BY Code").arg(tableCellsProp);

    if( !qry.exec(statement) )
    {
        *myError = qry.lastError().text();
        return false;
    }
    else
    {
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

            // facoltativa
            if (! getValue(qry.value("Height"), &height))
            {
                height = NODATA;
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


bool Crit3DMeteoGridDbHandler::updateGridDate(QString *myError)
{

    QSqlQuery qry(_db);

    int row = 0;
    int col = 0;
    int tableNotFoundError = 1146;
    std::string id;

    QDate maxDateD(QDate(1800, 1, 1));
    QDate minDateD(QDate(7800, 12, 31));

    QDate maxDateH(QDate(1800, 1, 1));
    QDate minDateH(QDate(7800, 12, 31));

    QDate temp;

    if (!_meteoGrid->findFirstActiveMeteoPoint(&id, &row, &col))
    {
        *myError = "active cell not found";
        return false;
    }

    QString tableD = _tableDaily.prefix + QString::fromStdString(id) + _tableDaily.postFix;
    QString tableH = _tableHourly.prefix + QString::fromStdString(id) + _tableHourly.postFix;

    QString statement;

    if (_tableDaily.exists)
    {
        statement = QString("SELECT MIN(`%1`) as minDate, MAX(`%1`) as maxDate FROM `%2`").arg(_tableDaily.fieldTime).arg(tableD);
        if( !qry.exec(statement) )
        {
            while( qry.lastError().number() == tableNotFoundError)
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
                    *myError = "active cell not found";
                    return false;
                }
                tableD = _tableDaily.prefix + QString::fromStdString(id) + _tableDaily.postFix;
                tableH = _tableHourly.prefix + QString::fromStdString(id) + _tableHourly.postFix;

                statement = QString("SELECT MIN(%1) as minDate, MAX(%1) as maxDate FROM `%2`").arg(_tableDaily.fieldTime).arg(tableD);
                qry.exec(statement);
            }

            if ( !qry.lastError().type() == QSqlError::NoError && qry.lastError().number() != tableNotFoundError)
            {
                *myError = qry.lastError().text();
                return false;
            }
        }
        else
        {
            if (qry.next())
            {
                if (getValue(qry.value("minDate"), &temp))
                {
                    if (temp < minDateD)
                        minDateD = temp;
                }
                else
                {
                    *myError = "Missing daily fieldTime";
                    return false;
                }

                if (getValue(qry.value("maxDate"), &temp))
                {
                    if (temp > maxDateD)
                        maxDateD = temp;
                }
                else
                {
                    *myError = "Missing daily fieldTime";
                    return false;
                }

            }
            else
            {
                *myError = "Error: fieldTime not found" ;
                return false;
            }
        }
    }

    if (_tableHourly.exists)
    {
        statement = QString("SELECT MIN(%1) as minDate, MAX(%1) as maxDate FROM `%2`").arg(_tableHourly.fieldTime).arg(tableH);
        if( !qry.exec(statement) )
        {
            while( qry.lastError().number() == tableNotFoundError)
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
                    *myError = "active cell not found";
                    return false;
                }

                tableH = _tableHourly.prefix + QString::fromStdString(id) + _tableHourly.postFix;

                statement = QString("SELECT MIN(%1) as minDate, MAX(%1) as maxDate FROM `%2`").arg(_tableHourly.fieldTime).arg(tableH);
                qry.exec(statement);
            }
            if ( !qry.lastError().type() == QSqlError::NoError && qry.lastError().number() != tableNotFoundError)
            {
                *myError = qry.lastError().text();
                return false;
            }
        }
        else
        {
            if (qry.next())
            {
                if (getValue(qry.value("minDate"), &temp))
                {
                    if (temp < minDateH)
                        minDateH = temp;
                }
                else
                {
                    *myError = "Missing hourly fieldTime";
                    return false;
                }

                if (getValue(qry.value("maxDate"), &temp))
                {
                    if (temp > maxDateH)
                        maxDateH = temp;
                }
                else
                {
                    *myError = "Missing hourly fieldTime";
                    return false;
                }

            }
            else
            {
                *myError = "Error: fieldTime not found" ;
                return false;
            }
        }
    }

    // the last hourly day is always incomplete, there is just 00.00 value
    maxDateH = maxDateH.addDays(-1);

    if (minDateD < minDateH)
        _firstDate = minDateD;
    else
        _firstDate = minDateH;

    if (maxDateD > maxDateH)
        _lastDate = maxDateD;
    else
        _lastDate = maxDateH;

    return true;

}


bool Crit3DMeteoGridDbHandler::loadGridDailyData(QString *myError, QString meteoPoint, QDate first, QDate last)
{
    QSqlQuery qry(_db);
    QString tableD = _tableDaily.prefix + meteoPoint + _tableDaily.postFix;
    QDate date;
    int varCode;
    float value;

    unsigned row;
    unsigned col;
    bool initialize = true;

    if (!_meteoGrid->findMeteoPointFromId(&row, &col, meteoPoint.toStdString()) )
    {
        *myError = "Missing MeteoPoint id";
        return false;
    }

    int numberOfDays = first.daysTo(last) + 1;
    _meteoGrid->meteoPointPointer(row,col)->initializeObsDataD(numberOfDays, getCrit3DDate(first));

    QString statement = QString("SELECT * FROM `%1` WHERE `%2`>= '%3' AND `%2`<= '%4' ORDER BY `%2`").arg(tableD).arg(_tableDaily.fieldTime).arg(first.toString("yyyy-MM-dd")).arg(last.toString("yyyy-MM-dd"));
    if( !qry.exec(statement) )
    {
        *myError = qry.lastError().text();
        return false;
    }
    else
    {
        while (qry.next())
        {
            if (!getValue(qry.value(_tableDaily.fieldTime), &date))
            {
                *myError = "Missing fieldTime";
                return false;
            }

            if (!getValue(qry.value("VariableCode"), &varCode))
            {
                *myError = "Missing VariableCode";
                return false;
            }

            if (!getValue(qry.value("Value"), &value))
            {
                *myError = "Missing Value";
            }

            meteoVariable variable = getDailyVarEnum(varCode);

            if (! _meteoGrid->meteoPointPointer(row,col)->setMeteoPointValueD(getCrit3DDate(date), variable, value))
                return false;

        }

    }

    return true;
}


bool Crit3DMeteoGridDbHandler::loadGridDailyDataFixedFields(QString *myError, QString meteoPoint, QDate first, QDate last)
{
    QSqlQuery qry(_db);
    QString tableD = _tableDaily.prefix + meteoPoint + _tableDaily.postFix;
    QDate date;
    int varCode;
    float value;

    unsigned row;
    unsigned col;

    if (!_meteoGrid->findMeteoPointFromId(&row, &col, meteoPoint.toStdString()) )
    {
        *myError = "Missing MeteoPoint id";
        return false;
    }

    int numberOfDays = first.daysTo(last) + 1;
    _meteoGrid->meteoPointPointer(row,col)->initializeObsDataD(numberOfDays, getCrit3DDate(first));

    QString statement = QString("SELECT * FROM `%1` WHERE `%2` >= '%3' AND `%2` <= '%4' ORDER BY `%2`").arg(tableD).arg(_tableDaily.fieldTime).arg(first.toString("yyyy-MM-dd")).arg(last.toString("yyyy-MM-dd"));
    if( !qry.exec(statement) )
    {
        *myError = qry.lastError().text();
        return false;
    }
    else
    {
        while (qry.next())
        {
            if (!getValue(qry.value(_tableDaily.fieldTime), &date))
            {
                *myError = "Missing fieldTime";
                return false;
            }

            for (unsigned int i=0; i < _tableDaily.varcode.size(); i++)
            {
                varCode = _tableDaily.varcode[i].varCode;
                if (!getValue(qry.value(_tableDaily.varcode[i].varField), &value))
                {
                    *myError = "Missing VarField";
                }

                meteoVariable variable = getDailyVarEnum(varCode);

                if (! _meteoGrid->meteoPointPointer(row,col)->setMeteoPointValueD(getCrit3DDate(date), variable, value))
                    return false;

            }

        }

    }

    return true;
}


bool Crit3DMeteoGridDbHandler::loadGridHourlyData(QString *myError, QString meteoPoint, QDateTime first, QDateTime last)
{

    QSqlQuery qry(_db);
    QString tableH = _tableHourly.prefix + meteoPoint + _tableHourly.postFix;
    QDateTime date;
    int varCode;
    float value;

    unsigned row;
    unsigned col;

    if (!_meteoGrid->findMeteoPointFromId(&row, &col, meteoPoint.toStdString()) )
    {
        *myError = "Missing MeteoPoint id";
        return false;
    }

    int numberOfDays = first.date().daysTo(last.date());
    _meteoGrid->meteoPointPointer(row, col)->initializeObsDataH(1, numberOfDays, getCrit3DDate(first.date()));

    QString statement = QString("SELECT * FROM `%1` WHERE `%2` >= '%3' AND `%2` <= '%4' ORDER BY `%2`")
                                .arg(tableH).arg(_tableHourly.fieldTime).arg(first.toString("yyyy-MM-dd hh:mm")).arg(last.toString("yyyy-MM-dd hh:mm"));

    if( !qry.exec(statement) )
    {
        *myError = qry.lastError().text();
    }
    else
    {
        while (qry.next())
        {
            if (!getValue(qry.value(_tableHourly.fieldTime), &date))
            {
                *myError = "Missing fieldTime";
                return false;
            }

            if (!getValue(qry.value("VariableCode"), &varCode))
            {
                *myError = "Missing VariableCode";
                return false;
            }

            if (!getValue(qry.value("Value"), &value))
            {
                *myError = "Missing Value";
            }

            meteoVariable variable = getHourlyVarEnum(varCode);

            if (! _meteoGrid->meteoPointPointer(row,col)->setMeteoPointValueH(getCrit3DDate(date.date()), date.time().hour(), date.time().minute(), variable, value))
                return false;
        }
    }

    return true;
}


bool Crit3DMeteoGridDbHandler::loadGridHourlyDataFixedFields(QString *myError, QString meteoPoint, QDateTime first, QDateTime last)
{

    QSqlQuery qry(_db);
    QString tableH = _tableHourly.prefix + meteoPoint + _tableHourly.postFix;
    QDateTime date;
    int varCode;
    float value;

    unsigned row;
    unsigned col;

    if (!_meteoGrid->findMeteoPointFromId(&row, &col, meteoPoint.toStdString()) )
    {
        *myError = "Missing MeteoPoint id";
        return false;
    }

    int numberOfDays = first.date().daysTo(last.date());
    _meteoGrid->meteoPointPointer(row, col)->initializeObsDataH(1, numberOfDays, getCrit3DDate(first.date()));

    QString statement = QString("SELECT * FROM `%1` WHERE `%2` >= '%3' AND `%2`<= '%4' ORDER BY `%2`").arg(tableH).arg(_tableHourly.fieldTime).arg(first.toString("yyyy-MM-dd hh:mm")).arg(last.toString("yyyy-MM-dd hh:mm"));
    if( !qry.exec(statement) )
    {
        *myError = qry.lastError().text();
    }
    else
    {
        while (qry.next())
        {
            if (!getValue(qry.value(_tableHourly.fieldTime), &date))
            {
                *myError = "Missing fieldTime";
                return false;
            }

            for (unsigned int i=0; i < _tableHourly.varcode.size(); i++)
            {
                varCode = _tableHourly.varcode[i].varCode;

                if (!getValue(qry.value(_tableHourly.varcode[i].varField), &value))
                {
                    *myError = "Missing fieldTime";
                }
                meteoVariable variable = getHourlyVarEnum(varCode);

                if (! _meteoGrid->meteoPointPointer(row,col)->setMeteoPointValueH(getCrit3DDate(date.date()), date.time().hour(), date.time().minute(), variable, value))
                    return false;
            }

        }

    }


    return true;
}

std::vector<float> Crit3DMeteoGridDbHandler::loadGridDailyVar(QString *myError, QString meteoPoint, meteoVariable variable, QDate first, QDate last, QDate* firstDateDB)
{
    QSqlQuery qry(_db);
    QString tableD = _tableDaily.prefix + meteoPoint + _tableDaily.postFix;
    QDate date, previousDate;

    std::vector<float> dailyVarList;

    float value;
    unsigned row;
    unsigned col;
    bool firstRow = true;

    int varCode = getDailyVarCode(variable);

    if (varCode == NODATA)
    {
        *myError = "Variable not existing";
        return dailyVarList;
    }

    if (!_meteoGrid->findMeteoPointFromId(&row, &col, meteoPoint.toStdString()) )
    {
        *myError = "Missing MeteoPoint id";
        return dailyVarList;
    }

    QString statement = QString("SELECT * FROM `%1` WHERE VariableCode = '%2' AND `%3` >= '%4' AND `%3`<= '%5' ORDER BY `%3`").arg(tableD).arg(varCode).arg(_tableDaily.fieldTime).arg(first.toString("yyyy-MM-dd")).arg(last.toString("yyyy-MM-dd"));
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

                if (!getValue(qry.value("Value"), &value))
                {
                    *myError = "Missing Value";
                }
                dailyVarList.push_back(value);
                previousDate = *firstDateDB;
                firstRow = false;
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

                if (!getValue(qry.value("Value"), &value))
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
                                                     QList<meteoVariable> meteoVariableList)
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
                for (QDate date = firstDate; date <= lastDate; date = date.addDays(1))
                {
                    float value = meteoGrid()->meteoPoint(row, col).getMeteoPointValueD(getCrit3DDate(date), meteoVar);
                    QString valueS = QString("'%1'").arg(value);
                    if (isEqual(value, NODATA)) valueS = "NULL";

                    int varCode = getDailyVarCode(meteoVar);

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

bool Crit3DMeteoGridDbHandler::saveCellGridDailyDataFF(QString *myError, QString meteoPointID, int row, int col, QDate firstDate, QDate lastDate)
{
    QSqlQuery qry(_db);
    QString tableD = _tableDaily.prefix + meteoPointID + _tableDaily.postFix;
    QString tableFields;


    for (unsigned int i=0; i < _tableDaily.varcode.size(); i++)
    {
        QString var = _tableDaily.varcode[i].varPragaName;
        QString type = _mapDailyMySqlVarType[var];
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
                float value = meteoGrid()->meteoPoint(row,col).getMeteoPointValueD(getCrit3DDate(date), getDailyVarFieldEnum(_tableDaily.varcode[j].varField));
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


bool Crit3DMeteoGridDbHandler::saveCellCurrentGridDailyFF(QString *myError, QString meteoPointID, QDate date, QString varPragaName, float value)
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
        QString type = _mapDailyMySqlVarType[var];

        QString varFieldItem = _tableDaily.varcode[i].varField;
        tableFields = tableFields  + ", " + varFieldItem.toLower() + " " + type;
    }


    QString statement = QString("CREATE TABLE IF NOT EXISTS `%1`").arg(tableD) + QString("(%1 date ").arg(_tableDaily.fieldTime) + tableFields + QString(", PRIMARY KEY(%1))").arg(_tableDaily.fieldTime);

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

        statement = QString("INSERT INTO `%1` (`%2`, `%3`) VALUES ('%4',%5) ON DUPLICATE KEY UPDATE `%3`= %5").arg(tableD).arg(_tableDaily.fieldTime).arg(varField.toLower()).arg(date.toString("yyyy-MM-dd")).arg(valueS);

        if( !qry.exec(statement) )
        {
            *myError = qry.lastError().text();
            return false;
        }

    }

    return true;
}

bool Crit3DMeteoGridDbHandler::saveGridData(QString *myError, QDateTime firstTime, QDateTime lastTime, QList<meteoVariable> meteoVariableList)
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
                    if (isDaily) saveCellGridDailyData(myError, QString::fromStdString(id), row, col, firstTime.date(), lastDate, meteoVariableList);
                }
                else
                {
                    if (isHourly) saveCellGridHourlyDataFF(myError, QString::fromStdString(id), row, col, firstTime, lastTime);
                    if (isDaily) saveCellGridDailyDataFF(myError, QString::fromStdString(id), row, col, firstTime.date(), lastDate);
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

bool Crit3DMeteoGridDbHandler::saveGridDailyData(QString *myError, QDateTime firstDate, QDateTime lastDate, QList<meteoVariable> meteoVariableList)
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
                    saveCellGridDailyData(myError, QString::fromStdString(id), row, col, firstDate.date(), lastDate.date(), meteoVariableList);
                }
                else
                {
                    saveCellGridDailyDataFF(myError, QString::fromStdString(id), row, col, firstDate.date(), lastDate.date());
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

bool Crit3DMeteoGridDbHandler::saveCellGridHourlyDataFF(QString *myError, QString meteoPointID, int row, int col, QDateTime firstTime, QDateTime lastTime)
{
    QSqlQuery qry(_db);
    QString tableH = _tableHourly.prefix + meteoPointID + _tableHourly.postFix;
    QString tableFields;


    for (unsigned int i=0; i < _tableHourly.varcode.size(); i++)
    {
        QString var = _tableHourly.varcode[i].varPragaName;
        QString type = _mapHourlyMySqlVarType[var];
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

bool Crit3DMeteoGridDbHandler::saveCellCurrentGridHourly(QString *myError, QString meteoPointID, QDateTime dateTime, int varCode, float value)
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


        QString valueS = QString("'%1'").arg(value);
        if (value == NODATA)
            valueS = "NULL";

        statement = QString("REPLACE INTO `%1` VALUES ('%2','%3',%4)").arg(tableH).arg(dateTime.toString("yyyy-MM-dd hh:mm")).arg(varCode).arg(valueS);

        if( !qry.exec(statement) )
        {
            *myError = qry.lastError().text();
            return false;
        }
    }

    return true;
}

bool Crit3DMeteoGridDbHandler::saveCellCurrentGridHourlyFF(QString *myError, QString meteoPointID, QDateTime dateTime, QString varPragaName, float value)
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
        QString type = _mapHourlyMySqlVarType[var];
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
        QString valueS = QString("'%1'").arg(value);
        if (value == NODATA)
            valueS = "NULL";

        statement = QString("INSERT INTO `%1` (`%2`, `%3`) VALUES ('%4',%5) ON DUPLICATE KEY UPDATE `%3` = %5").arg(tableH).arg(_tableHourly.fieldTime).arg(varField.toLower()).arg(dateTime.toString("yyyy-MM-dd hh:mm")).arg(valueS);

        if( !qry.exec(statement) )
        {
            *myError = qry.lastError().text();
            return false;
        }
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

QString Crit3DMeteoGridDbHandler::tableDailyModel() const
{
    return _tableDailyModel;
}

QString Crit3DMeteoGridDbHandler::tableHourlyModel() const
{
    return _tableHourlyModel;
}

