#include "commonConstants.h"
#include "download.h"
#include "dbMeteoPointsHandler.h"
#include "gis.h"

#include <QtNetwork>


const QByteArray Download::_authorization = QString("Basic " + QString("ugo:Ul1ss&").toLocal8Bit().toBase64()).toLocal8Bit();


bool Download::getPointProperties(const QList<QString> &datasetList, int utmZone, QString &errorString)
{
    bool result = true;
    QEventLoop loop;

    QNetworkAccessManager* manager = new QNetworkAccessManager(this);
    connect(manager, SIGNAL(finished(QNetworkReply*)), &loop, SLOT(quit()));

    _datasetsList = datasetList;

    QNetworkRequest request;
    request.setUrl(QUrl("http://meteozen.metarpa/simcstations/api/v1/stations"));
    request.setRawHeader("Authorization", _authorization);

    // GET
    QNetworkReply* reply = manager->get(request);

    loop.exec();

    if (reply->error() != QNetworkReply::NoError)
    {
            errorString =  "Network Error: " + reply->errorString();
            result = false;
    }
    else
    {
        QString data = (QString) reply->readAll();

        QJsonParseError *error = new QJsonParseError();
        QJsonDocument doc = QJsonDocument::fromJson(data.toUtf8(), error);

        qDebug() << "err: " << error->errorString() << " -> " << error->offset;

        // check validity of the document
        if(doc.isNull() || ! doc.isArray())
        {
            errorString = "Invalid JSON";
            result = false;
        }
        else
        {
            QJsonArray jsonArr = doc.array();

            for(int index = 0; index < jsonArr.size(); ++index)
            {
                QJsonObject obj = jsonArr[index].toObject();

                QJsonValue jsonDataset = obj.value("network");

                if (jsonDataset.isUndefined())
                    qDebug() << "jsonDataset: key id does not exist";
                else if (! jsonDataset.isString())
                    qDebug() << "jsonDataset: value is not string";
                else
                {
                    foreach(QString item, _datasetsList)
                    {
                        if (jsonDataset.toString().toUpper() == item.toUpper())
                        {
                            this->downloadMetadata(obj, utmZone);
                        }
                    }
                }
            }
        }
    }

    delete reply;
    delete manager;
    return result;
}


QMap<QString, QString> Download::getArmiketIdList(QList<QString> datasetList)
{

    QMap<QString, QString> idList;
    QEventLoop loop;

    QNetworkAccessManager* manager = new QNetworkAccessManager(this);
    connect(manager, SIGNAL(finished(QNetworkReply*)), &loop, SLOT(quit()));

    _datasetsList = datasetList;

    QNetworkRequest request;
    request.setUrl(QUrl("http://meteozen.metarpa/simcstations/api/v1/stations"));
    request.setRawHeader("Authorization", _authorization);

    // GET
    QNetworkReply* reply = manager->get(request);

    loop.exec();

    if (reply->error() != QNetworkReply::NoError)
    {
            qDebug() << "Network Error: " << reply->error();
            return idList;
    }
    else
    {
        QString data = (QString) reply->readAll();

        QJsonParseError *error = new QJsonParseError();
        QJsonDocument doc = QJsonDocument::fromJson(data.toUtf8(), error);

        qDebug() << "err: " << error->errorString() << " -> " << error->offset;

        // check validity of the document
        if(!doc.isNull() && doc.isArray())
        {
            QJsonArray jsonArr = doc.array();

            for(int index = 0; index < jsonArr.size(); ++index)
            {
                QJsonObject obj = jsonArr[index].toObject();

                QJsonValue jsonDataset = obj.value("network");

                if (jsonDataset.isUndefined())
                    qDebug() << "jsonDataset: key id does not exist";
                else if (!jsonDataset.isString())
                    qDebug() << "jsonDataset: value is not string";
                else
                    foreach(QString item, _datasetsList)
                        if (jsonDataset.toString().toUpper() == item.toUpper())
                        {
                            QString idValue;
                            QString nameValue;
                            QJsonValue jsonId = obj.value("id");
                            if (!jsonId.isNull())
                            {
                                int idInt = jsonId.toInt();
                                idValue = QString::number(idInt);
                            }
                            QJsonValue jsonName = obj.value("name");
                            if (!jsonName.isNull())
                            {
                                nameValue = jsonName.toString();
                            }
                            else
                            {
                                nameValue = " ";
                            }
                            idList.insert(idValue, nameValue);
                        }
            }
        }
         else
        {
            qDebug() << "Invalid JSON...\n";
            return idList;
        }
    }

    delete reply;
    delete manager;
    return idList;
}


void Download::downloadMetadata(const QJsonObject &obj, int utmZone)
{
    Crit3DMeteoPoint pointProp;

    QJsonValue jsonId = obj.value("id");

    if (jsonId.isNull())
    {
          qDebug() << "Id is empty\n";
          return;
    }

    int idInt = jsonId.toInt();
    pointProp.id = std::to_string(idInt);

    QJsonValue jsonName = obj.value("name");
    if (jsonName.isNull())
          qDebug() << "name is null\n";
    pointProp.name = jsonName.toString().toStdString();

    QJsonValue jsonNetwork = obj.value("network");
    pointProp.dataset = jsonNetwork.toString().toStdString();

    QJsonValue jsonGeometry = obj.value("geometry").toObject().value("coordinates");
    QJsonValue jsonLon = jsonGeometry.toArray()[0];
    if (jsonLon.isNull() || jsonLon.toInt() < -180 || jsonLon.toInt() > 180)
        qDebug() << "invalid Longitude\n";
    pointProp.longitude = jsonLon.toDouble();

    QJsonValue jsonLat = jsonGeometry.toArray()[1];
    if (jsonLat.isNull() || jsonLat.toInt() < -90 || jsonLat.toInt() > 90)
        qDebug() << "invalid Latitude\n";
    pointProp.latitude = jsonLat.toDouble();

    QJsonValue jsonLatInt = obj.value("lat");
    if (jsonLatInt.isNull())
        jsonLatInt = NODATA;
    pointProp.latInt = jsonLatInt.toInt();

    QJsonValue jsonLonInt = obj.value("lon");
    if (jsonLonInt.isNull())
        jsonLonInt = NODATA;
    pointProp.lonInt = jsonLonInt.toInt();

    QJsonValue jsonAltitude = obj.value("height");
    pointProp.point.z = jsonAltitude.toDouble();

    QJsonValue jsonState = obj.value("country").toObject().value("name");
    pointProp.state = jsonState.toString().toStdString();

    if (obj.value("region").isNull())
        pointProp.region = "";
    else
    {
        QJsonValue jsonRegion = obj.value("region").toObject().value("name");
        pointProp.region = jsonRegion.toString().toStdString();
    }

    if (obj.value("province").isNull())
        pointProp.province = "";
    else
    {
        QJsonValue jsonProvince = obj.value("province").toObject().value("name");
        pointProp.province = jsonProvince.toString().toStdString();
    }

    if (obj.value("municipality").isNull())
        pointProp.municipality = "";
    else
    {
        QJsonValue jsonMunicipality = obj.value("municipality").toObject().value("name");
        pointProp.municipality = jsonMunicipality.toString().toStdString();
    }

    double utmx, utmy;
    gis::latLonToUtmForceZone(utmZone, pointProp.latitude, pointProp.longitude, &utmx, &utmy);
    pointProp.point.utm.x = utmx;
    pointProp.point.utm.y = utmy;

    _dbMeteo->writePointProperties(pointProp);
}


bool Download::getPointPropertiesFromId(const QString &id, int utmZone, Crit3DMeteoPoint &pointProp)
{
    bool result = true;
    QEventLoop loop;

    QNetworkAccessManager* manager = new QNetworkAccessManager(this);
    connect(manager, SIGNAL(finished(QNetworkReply*)), &loop, SLOT(quit()));

    QNetworkRequest request;
    request.setUrl(QUrl("http://meteozen.metarpa/simcstations/api/v1/stations"));
    request.setRawHeader("Authorization", _authorization);

    // GET
    QNetworkReply* reply = manager->get(request);

    loop.exec();

    if (reply->error() != QNetworkReply::NoError)
    {
            qDebug() << "Network Error: " << reply->error();
            result = false;
    }
    else
    {
        QString data = (QString) reply->readAll();

        QJsonParseError *error = new QJsonParseError();
        QJsonDocument doc = QJsonDocument::fromJson(data.toUtf8(), error);

        qDebug() << "err: " << error->errorString() << " -> " << error->offset;

        // check validity of the document
        if(! doc.isNull() && doc.isArray())
        {
            QJsonArray jsonArr = doc.array();

            for(int index = 0; index < jsonArr.size(); ++index)
            {
                QJsonObject obj = jsonArr[index].toObject();

                QString jsonIdValue = QString::number(obj.value("id").toInt());

                if (obj.value("id").isUndefined() || jsonIdValue.isEmpty())
                    qDebug() << "jsonId: key id does not exist";
                else
                        if (jsonIdValue == id)
                        {
                            pointProp.id = id.toStdString();

                            QJsonValue jsonName = obj.value("name");
                            if (jsonName.isNull())
                                  qDebug() << "name is null\n";
                            pointProp.name = jsonName.toString().toStdString();

                            QJsonValue jsonNetwork = obj.value("network");
                            pointProp.dataset = jsonNetwork.toString().toStdString();

                            QJsonValue jsonGeometry = obj.value("geometry").toObject().value("coordinates");
                            QJsonValue jsonLon = jsonGeometry.toArray()[0];
                            if (jsonLon.isNull() || jsonLon.toInt() < -180 || jsonLon.toInt() > 180)
                                qDebug() << "invalid Longitude\n";
                            pointProp.longitude = jsonLon.toDouble();

                            QJsonValue jsonLat = jsonGeometry.toArray()[1];
                            if (jsonLat.isNull() || jsonLat.toInt() < -90 || jsonLat.toInt() > 90)
                                qDebug() << "invalid Latitude\n";
                            pointProp.latitude = jsonLat.toDouble();

                            QJsonValue jsonLatInt = obj.value("lat");
                            if (jsonLatInt.isNull())
                                jsonLatInt = NODATA;
                            pointProp.latInt = jsonLatInt.toInt();

                            QJsonValue jsonLonInt = obj.value("lon");
                            if (jsonLonInt.isNull())
                                jsonLonInt = NODATA;
                            pointProp.lonInt = jsonLonInt.toInt();

                            QJsonValue jsonAltitude = obj.value("height");
                            pointProp.point.z = jsonAltitude.toDouble();

                            QJsonValue jsonState = obj.value("country").toObject().value("name");
                            pointProp.state = jsonState.toString().toStdString();

                            if (obj.value("region").isNull())
                                pointProp.region = "";
                            else
                            {
                                QJsonValue jsonRegion = obj.value("region").toObject().value("name");
                                pointProp.region = jsonRegion.toString().toStdString();
                            }

                            if (obj.value("province").isNull())
                                pointProp.province = "";
                            else
                            {
                                QJsonValue jsonProvince = obj.value("province").toObject().value("name");
                                pointProp.province = jsonProvince.toString().toStdString();
                            }

                            if (obj.value("municipality").isNull())
                                pointProp.municipality = "";
                            else
                            {
                                QJsonValue jsonMunicipality = obj.value("municipality").toObject().value("name");
                                pointProp.municipality = jsonMunicipality.toString().toStdString();
                            }

                            double utmx, utmy;
                            gis::latLonToUtmForceZone(utmZone, pointProp.latitude, pointProp.longitude, &utmx, &utmy);
                            pointProp.point.utm.x = utmx;
                            pointProp.point.utm.y = utmy;

                            delete reply;
                            delete manager;
                            return result;
                        }
            }
        }
        else
        {
            qDebug() << "Invalid JSON...\n";
            result = false;
        }
    }

    delete reply;
    delete manager;
    return result;
}


bool Download::downloadDailyData(const QDate &startDate, const QDate &endDate, const QString &dataset,
                                 QList<QString> &stations, QList<int> &variables, bool prec0024, QString &errorString)
{
    // variable properties
    QList<VariablesList> variableList = _dbMeteo->getVariableProperties(variables);

    QList<QString> idVar;
    for (int i = 0; i < variableList.size(); i++)
    {
        idVar.append(QString::number(variableList[i].id()));
    }

    // create station tables
    _dbMeteo->initStationsDailyTables(startDate, endDate, stations, idVar);

    // attenzione: il reference time dei giornalieri è a fine giornata (ore 00 di day+1)
    QString refTime = QString("reftime:>%1,<=%2").arg(startDate.toString("yyyy-MM-dd"), endDate.addDays(1).toString("yyyy-MM-dd"));

    QString product = QString(";product: VM2,%1").arg(variables[0]);

    for (int i = 1; i < variables.size(); i++)
    {
        product = product % QString(" or VM2,%1").arg(variables[i]);
    }

    int maxStationSize = 100;
    int countStation = 0;
    int indexStation = 0;
    while (countStation < stations.size())
    {
        QString area;
        if (indexStation == 0)
        {
            area = QString(";area: VM2,%1").arg(stations[countStation]);
            countStation++;
            indexStation++;
        }
        while (indexStation < maxStationSize && countStation < stations.size())
        {
            area = area % QString(" or VM2,%1").arg(stations[countStation]);
            countStation++;
            indexStation++;
        }

        bool isUrlOk;
        QUrl url = QUrl(QString("%1/query").arg(_dbMeteo->getDatasetURL(dataset, isUrlOk)));
        if (! isUrlOk)
        {
            errorString = _dbMeteo->getErrorString();
            return false;
        }

        QNetworkAccessManager* manager = new QNetworkAccessManager(this);
        QEventLoop loop;
        connect(manager, SIGNAL(finished(QNetworkReply*)), &loop, SLOT(quit()));

        QNetworkRequest request;
        request.setUrl(url);
        request.setRawHeader("Authorization", _authorization);
        request.setHeader(QNetworkRequest::ContentTypeHeader, QVariant("application/x-www-form-urlencoded"));

        QUrlQuery postData;
        postData.addQueryItem("style", "postprocess");
        postData.addQueryItem("query", QString("%2%3%4").arg(refTime, area, product));

        QNetworkReply* reply = manager->post(request, postData.toString(QUrl::FullyEncoded).toUtf8());
        bool isDownloadOk = true;
        loop.exec();

        if (reply->error() != QNetworkReply::NoError)
        {
            errorString = "Network Error";
            return false;
        }

        // temporary table (clean at each cycle)
        if (! _dbMeteo->createTmpTable())
        {
            errorString = _dbMeteo->getErrorString();
            return false;
        }

        bool isFirstData = true;
        QString dateStr, idPoint, flag;
        int idArkimet, idVar;
        bool emptyLine = true;
        for (QString line = QString(reply->readLine()); !(line.isNull() || line.isEmpty());  line = QString(reply->readLine()))
        {
            emptyLine = false;
            QList<QString> fields = line.split(",");

            // warning: ref date arkimet: hour 00 of day+1
            dateStr = fields[0];
            QDate myDate = QDate::fromString(dateStr.left(8), "yyyyMMdd");
            myDate = myDate.addDays(-1);
            dateStr = myDate.toString("yyyy-MM-dd");

            idPoint = fields[1];
            flag = fields[6];

            if (idPoint != "" && flag.left(1) != "1" && flag.left(3) != "054")
            {
                double value = NODATA;
                if (flag.left(1) == "2")
                    value = fields[4].toDouble();
                else
                    value = fields[3].toDouble();

                idArkimet = fields[2].toInt();

                if (idArkimet == PREC_ID)
                    if ((prec0024 && fields[0].mid(8,2) != "00") || (!prec0024 && fields[0].mid(8,2) != "08"))
                        continue;

                // conversion from average daily radiation to integral radiation
                if (idArkimet == RAD_ID)
                {
                    value *= DAY_SECONDS / 1000000.0;
                }

                // variable
                int i = 0;
                while (i < variableList.size()
                       && variableList[i].arkId() != idArkimet) i++;

                if (i < variableList.size())
                {
                    idVar = variableList[i].id();
                    _dbMeteo->appendTmpData(dateStr, idPoint, QString::number(idVar), QString::number(value), isFirstData);
                    isFirstData = false;
                }

            }
        }

        if (! emptyLine)
        {
            isDownloadOk = _dbMeteo->saveDailyData();
        }

        delete reply;
        delete manager;

        if (! isDownloadOk)
        {
            errorString = _dbMeteo->getErrorString();
            return false;
        }

        indexStation = 0; //reset block stations counter

    } // end while

    _dbMeteo->deleteTmpTable();
    return true;
}


bool Download::downloadHourlyData(const QDate &startDate, const QDate &endDate, const QString &dataset,
                                  const QList<QString> &stationList, const QList<int> &varList, QString &errorString)
{
    QList<VariablesList> variableList = _dbMeteo->getVariableProperties(varList);
    if (variableList.size() == 0)
        return false;

    QList<QString> idVar;

    for (int i = 0; i < variableList.size(); i++)
        idVar.append(QString::number(variableList[i].id()));

    // create station tables
    _dbMeteo->initStationsHourlyTables(startDate, endDate, stationList, idVar);

    QString product = QString(";product: VM2,%1").arg(varList[0]);

    for (int i = 1; i < varList.size(); i++)
    {
        product = product % QString(" or VM2,%1").arg(varList[i]);
    }

    // start from 01:00
    QDateTime startTime(startDate, QTime(1,0,0), Qt::UTC);

    QDateTime endTime(endDate, QTime(0,0,0), Qt::UTC);
    endTime = endTime.addSecs(3600 * 24);

    // reftime
    QString refTime = QString("reftime:>=%1,<=%2").arg(startTime.toString("yyyy-MM-dd hh:mm"), endTime.toString("yyyy-MM-dd hh:mm"));

    QEventLoop loop;

    int maxStationSize = 100;
    int j = 0;
    QString area;
    QUrl url;
    QNetworkRequest request;
    int countStation = 0;

    while (countStation < stationList.size())
    {
        if (j == 0)
        {
            area = QString(";area: VM2,%1").arg(stationList[countStation]);
            j = j+1;
            countStation = countStation+1;
        }
        while (countStation < stationList.size() && j < maxStationSize)
        {
            area = area % QString(" or VM2,%1").arg(stationList[countStation]);
            countStation = countStation+1;
            j = j+1;
        }

        bool isOk;
        url = QUrl(QString("%1/query").arg(_dbMeteo->getDatasetURL(dataset, isOk)));
        if (! isOk)
        {
            errorString = _dbMeteo->getErrorString();
            return false;
        }

        QNetworkAccessManager* manager = new QNetworkAccessManager(this);
        connect(manager, SIGNAL(finished(QNetworkReply*)), &loop, SLOT(quit()));

        request.setUrl(url);
        request.setRawHeader("Authorization", _authorization);
        request.setHeader(QNetworkRequest::ContentTypeHeader, QVariant("application/x-www-form-urlencoded"));

        QUrlQuery postData;
        postData.addQueryItem("style", "postprocess");
        postData.addQueryItem("query", QString("%2%3%4").arg(refTime, area, product));

        QNetworkReply* reply = manager->post(request, postData.toString(QUrl::FullyEncoded).toUtf8());
        loop.exec();

        if (reply->error() != QNetworkReply::NoError)
        {
                errorString = "Network Error";
                delete reply;
                delete manager;
                return false;
        }
        else
        {
            _dbMeteo->queryString = "";

            QString line, dateTimeStr, idPoint, flag, varName;
            QString idVariable, value, secondValue, frequency;
            QList<QString> fields;
            int i, idVarArkimet;

            _dbMeteo->createTmpTable();
            bool isVarOk, isFirstData = true;
            bool emptyLine = true;

            for (line = QString(reply->readLine()); !(line.isNull() || line.isEmpty());  line = QString(reply->readLine()))
            {
                emptyLine = false;
                fields = line.split(",");
                dateTimeStr = QString("%1-%2-%3 %4:%5:00").arg(fields[0].left(4))
                                                           .arg(fields[0].mid(4, 2))
                                                           .arg(fields[0].mid(6, 2))
                                                           .arg(fields[0].mid(8, 2))
                                                           .arg(fields[0].mid(10, 2));
                // point
                if (fields[1] != "")
                {
                    idPoint = fields[1];

                    // variable
                    isVarOk = false;
                    idVarArkimet = fields[2].toInt();

                    for (i = 0; i < variableList.size(); i++)
                    {
                        if (variableList[i].arkId() == idVarArkimet)
                        {
                            idVariable = QString::number(variableList[i].id());
                            frequency = QString::number(variableList[i].frequency());
                            varName = variableList[i].varName();
                            isVarOk = true;
                        }
                    }

                    // value
                    if (isVarOk && fields[3] != "")
                    {
                        value = fields[3];
                        secondValue = fields[4];

                        // flag
                        flag = fields[6];
                        if (flag.left(1) != "1" && flag.left(1) != "2" && flag.left(3) != "054")
                        {
                            _dbMeteo->appendTmpData(dateTimeStr, idPoint, idVariable, value, isFirstData);
                            isFirstData = false;
                        }
                        else if(flag.left(1) == "2")
                        {
                            _dbMeteo->appendTmpData(dateTimeStr, idPoint, idVariable, secondValue, isFirstData);
                            isFirstData = false;
                        }
                    }
                }
            }

            if (_dbMeteo->queryString != "" && !emptyLine)
            {
               _dbMeteo->saveHourlyData();
            }
        }

        delete reply;
        delete manager;

        j = 0; //reset block stations counter
    }

    _dbMeteo->deleteTmpTable();
    return true;
}


