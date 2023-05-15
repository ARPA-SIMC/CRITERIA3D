#include "pragaShell.h"
#include "shell.h"
#include "utilities.h"
#include "commonConstants.h"
#include <QFile>
#include <QTextStream>


QList<QString> getPragaCommandList()
{
    QList<QString> cmdList = getSharedCommandList();

    // praga commands
    cmdList.append("List            | ListCommands");
    cmdList.append("AggrOnZones     | GridAggregationOnZones");
    cmdList.append("ComputeClimate  | ComputeClimaFromXMLSaveOnDB");
    cmdList.append("Drought         | ComputeDroughtIndexGrid");
    cmdList.append("DroughtPoint    | ComputeDroughtIndexPoint");
    cmdList.append("Download        | Download");
    cmdList.append("GridAggr        | GridAggregation");
    cmdList.append("GridDerVar      | GridDerivedVariables");
    cmdList.append("GridMonthlyInt  | GridMonthlyIntegrationVariables");
    cmdList.append("Netcdf          | ExportNetcdf");
    cmdList.append("Point           | OpenDbPoint");
    cmdList.append("Proj            | OpenProject");
    //cmdList.append("LoadForecast  | LoadForecastData");
    cmdList.append("SaveLogProc     | SaveLogProceduresGrid");
    cmdList.append("XMLToNetcdf     | ExportXMLElaborationsToNetcdf");

    return cmdList;
}


int cmdList(PragaProject* myProject)
{
    QList<QString> list = getPragaCommandList();

    myProject->logInfo("Available PRAGA Console commands:");
    myProject->logInfo("(short      | long version)");
    for (int i = 0; i < list.size(); i++)
    {
        myProject->logInfo(list[i]);
    }

    return PRAGA_OK;
}


int PragaProject::executePragaCommand(QList<QString> argumentList, bool* isCommandFound)
{
    *isCommandFound = false;
    if (argumentList.size() == 0) return PRAGA_INVALID_COMMAND;

    QString command = argumentList[0].toUpper();

    if (command == "LIST" || command == "LISTCOMMANDS")
    {
        *isCommandFound = true;
        return cmdList(this);
    }
    else if (command == "PROJ" || command == "OPENPROJECT")
    {
        *isCommandFound = true;
        return cmdOpenPragaProject(this, argumentList);
    }
    else if (command == "DOWNLOAD")
    {
        *isCommandFound = true;
        return cmdDownload(this, argumentList);
    }
    else if (command == "GRIDDING" || command == "INTERPOLATIONGRIDPERIOD")
    {
        *isCommandFound = true;
        return cmdInterpolationGridPeriod(this, argumentList);
    }
    else if (command == "GRIDAGGREGATION" || command == "GRIDAGGR")
    {
        *isCommandFound = true;
        return cmdAggregationGridPeriod(this, argumentList);
    }
    else if (command == "GRIDDERIVEDVARIABLES" || command == "GRIDDERVAR")
    {
        *isCommandFound = true;
        return cmdHourlyDerivedVariablesGrid(this, argumentList);
    }
    else if (command == "GRIDMONTHLYINTEGRATIONVARIABLES" || command == "GRIDMONTHLYINT")
    {
        *isCommandFound = true;
        return cmdMonthlyIntegrationVariablesGrid(this, argumentList);
    }
#ifdef NETCDF
    else if (command == "DROUGHTINDEX" || command == "DROUGHT")
    {
        *isCommandFound = true;
        return cmdDroughtIndexGrid(this, argumentList);
    }
    else if (command == "NETCDF" || command == "NETCDFEXPORT")
    {
        *isCommandFound = true;
        return cmdNetcdfExport(this, argumentList);
    }
    else if (command == "XMLTONETCDF" || command == "XMLNETCDFEXPORT")
    {
        *isCommandFound = true;
        return cmdExportXMLElabToNetcdf(this, argumentList);
    }
#endif
    else if (command == "AGGRONZONES" || command == "GRIDAGGREGATIONONZONES")
    {
        *isCommandFound = true;
        return cmdGridAggregationOnZones(this, argumentList);
    }
//    else if (command == "LOADFORECAST" || command == "LOADFORECASTDATA")
//    {
//        *isCommandFound = true;
//        return cmdLoadForecast(this, argumentList);
//    }
    else if (command == "CLIMATE" || command == "COMPUTECLIMATE")
    {
        *isCommandFound = true;
        return cmdComputeClimaFromXMLSaveOnDB(this, argumentList);
    }
    else if (command == "DROUGHTINDEXPOINT" || command == "DROUGHTPOINT")
    {
        *isCommandFound = true;
        return cmdDroughtIndexPoint(this, argumentList);
    }
    else if (command == "SAVELOGPROC" || command == "SAVELOGPROCEDURESGRID")
    {
        *isCommandFound = true;
        return cmdSaveLogDataProceduresGrid(this, argumentList);
    }
    else
    {
        // other specific Praga commands
        // ...
    }

    return PRAGA_INVALID_COMMAND;
}

int cmdOpenPragaProject(PragaProject* myProject, QList<QString> argumentList)
{
    if (argumentList.size() < 2)
    {
        myProject->logError("Missing project name");
        return PRAGA_INVALID_COMMAND;
    }

    // set fileName and projectFolder
    QString filename = argumentList.at(1);
    QString projectFolder = "";
    if (getFilePath(filename) == "")
    {
        if (filename.left(filename.length()-4) == ".ini")
            projectFolder = filename.left(filename.length()-4) + "/";
        else
        {
            projectFolder = filename + "/";
            filename += ".ini";
        }
    }

    myProject->projectPragaFolder = PATH_PROJECT+projectFolder;
    QString projectName = myProject->getCompleteFileName(filename, myProject->projectPragaFolder);

    if (! myProject->loadPragaProject(projectName))
    {
        myProject->logError();
        return PRAGA_ERROR;
    }

    return PRAGA_OK;
}

int cmdDownload(PragaProject* myProject, QList<QString> argumentList)
{
    if (argumentList.size() < 2)
    {
        myProject->logError("Missing parameters for download");
        return PRAGA_INVALID_COMMAND;
    }

    QDate dateIni, dateFin;
    QList<QString> varString, dailyVarString, hourlyVarString;
    QString var;
    meteoVariable meteoVar;
    bool prec0024 = true;
    bool showInfo = false;
    frequencyType myFreq;

    for (int i = 1; i < argumentList.size(); i++)
    {
        if (argumentList[i].left(3) == "-v:")
        {
            varString = argumentList[i].right(argumentList[i].length()-3).split(",");
            foreach (var,varString)
            {
                meteoVar = getMeteoVar(var.toStdString());
                if (meteoVar == noMeteoVar)
                {
                    myProject->logError("Unknown variable: " + var);
                    return PRAGA_ERROR;
                }
                else
                {
                    myFreq = getVarFrequency(meteoVar);
                    if (myFreq == noFrequency)
                    {
                        myProject->logError("Unknown frequency for variable : " + var);
                        return PRAGA_ERROR;
                    }
                    else if (myFreq == daily)
                        dailyVarString.append(var);
                    else if (myFreq == hourly)
                        hourlyVarString.append(var);
                }
            }
        }
        else if (argumentList.at(i).left(4) == "-d1:")
            dateIni = QDate::fromString(argumentList[i].right(argumentList[i].length()-4), "dd/MM/yyyy");
        else if (argumentList.at(i).left(4) == "-d2:")
            dateFin = QDate::fromString(argumentList[i].right(argumentList[i].length()-4), "dd/MM/yyyy");
        else if (argumentList.at(i).left(10) == "-yesterday")
        {
            dateIni = QDate::currentDate().addDays(-1);
            dateFin = dateIni;
        }
        else if (argumentList.at(i).left(10) == "-lastweek")
        {
            dateFin = QDate::currentDate().addDays(-1);
            dateIni = dateFin.addDays(-6);
        }
        else if (argumentList.at(i).left(3) == "-p9")
            prec0024 = false;
        else if (argumentList.at(i).left(5) == "-show")
            showInfo = true;
    }

    if (! dateIni.isValid())
    {
        myProject->logError("Wrong initial date");
        return PRAGA_INVALID_COMMAND;
    }

    if (! dateFin.isValid())
    {
        myProject->logError("Wrong final date");
        return PRAGA_INVALID_COMMAND;
    }

    if (dailyVarString.size() > 0)
        if (! myProject->downloadDailyDataArkimet(dailyVarString, prec0024, dateIni, dateFin, showInfo))
            return PRAGA_ERROR;

    if (hourlyVarString.size() > 0)
        if (! myProject->downloadHourlyDataArkimet(hourlyVarString, dateIni, dateFin, showInfo))
            return PRAGA_ERROR;

    return PRAGA_OK;
}


int cmdInterpolationGridPeriod(PragaProject* myProject, QList<QString> argumentList)
{
    if (argumentList.size() < 2)
    {
        myProject->logError("Missing parameters for gridding");
        return PRAGA_INVALID_COMMAND;
    }

    QDate dateIni, dateFin;
    bool saveRasters = false;
    QList <QString> varString, aggrVarString;
    QList <meteoVariable> variables, aggrVariables;
    QString var;
    meteoVariable meteoVar;
    int saveInterval = 1;
    int loadInterval = NODATA;
    bool parseSaveInterval = true;
    bool parseLoadInterval = true;

    for (int i = 1; i < argumentList.size(); i++)
    {
        if (argumentList[i].left(3) == "-v:")
        {
            varString = argumentList[i].right(argumentList[i].length()-3).split(",");
            foreach (var,varString)
            {
                meteoVar = getMeteoVar(var.toStdString());
                if (meteoVar == noMeteoVar) {
                    myProject->logError("Unknown variable: " + var);
                    return PRAGA_INVALID_COMMAND;
                }

                variables << meteoVar;
            }
        }
        else if (argumentList[i].left(3) == "-a:")
        {
            varString = argumentList[i].right(argumentList[i].length()-3).split(",");
            foreach (var,varString)
            {
                meteoVar = getMeteoVar(var.toStdString());

                if (meteoVar == noMeteoVar) {
                    myProject->logError("Unknown variable: " + var);
                    return PRAGA_INVALID_COMMAND;
                }

                aggrVariables << meteoVar;
            }
        }
        else if (argumentList.at(i).left(4) == "-d1:")
            dateIni = QDate::fromString(argumentList[i].right(argumentList[i].length()-4), "dd/MM/yyyy");
        else if (argumentList.at(i).left(4) == "-d2:")
            dateFin = QDate::fromString(argumentList[i].right(argumentList[i].length()-4), "dd/MM/yyyy");
        else if (argumentList.at(i).left(10) == "-yesterday")
        {
            dateIni = QDate::currentDate().addDays(-1);
            dateFin = dateIni;
        }
        else if (argumentList.at(i).left(10) == "-lastweek")
        {
            dateFin = QDate::currentDate().addDays(-1);
            dateIni = dateFin.addDays(-6);
        }
        else if (argumentList.at(i).left(2) == "-r")
            saveRasters = true;
        else if (argumentList.at(i).left(3) == "-s:")
            saveInterval = argumentList[i].right(argumentList[i].length()-3).toInt(&parseSaveInterval);
        else if (argumentList.at(i).left(3) == "-l:")
            loadInterval = argumentList[i].right(argumentList[i].length()-3).toInt(&parseLoadInterval);

    }

    if (! dateIni.isValid())
    {
        myProject->logError("Wrong initial date");
        return PRAGA_INVALID_COMMAND;
    }

    if (! dateFin.isValid())
    {
        myProject->logError("Wrong final date");
        return PRAGA_INVALID_COMMAND;
    }

    if (saveInterval == NODATA || ! parseSaveInterval)
    {
        myProject->logError("Wrong saving interval number");
        return PRAGA_INVALID_COMMAND;
    }

    if (! parseLoadInterval)
    {
        myProject->logError("Wrong loading interval number");
        return PRAGA_INVALID_COMMAND;
    }

    if (! myProject->interpolationMeteoGridPeriod(dateIni, dateFin, variables, aggrVariables, saveRasters, loadInterval, saveInterval))
        return PRAGA_ERROR;

    return PRAGA_OK;
}

int cmdAggregationGridPeriod(PragaProject* myProject, QList<QString> argumentList)
{
    if (argumentList.size() < 2)
    {
        myProject->logError("Missing parameters for aggregation");
        return PRAGA_INVALID_COMMAND;
    }

    // default date
    QDate dateIni = QDate::currentDate();
    QDate dateFin = dateIni.addDays(9);

    QList <meteoVariable> variables;
    QList <QString> varString;
    QString var;
    meteoVariable meteoVar;

    for (int i = 1; i < argumentList.size(); i++)
    {
        if (argumentList[i].left(3) == "-v:")
        {
            varString = argumentList[i].right(argumentList[i].length()-3).split(",");
            foreach (var,varString)
            {
                meteoVar = getMeteoVar(var.toStdString());
                if (meteoVar != noMeteoVar) variables << meteoVar;
            }
        }
        else if (argumentList.at(i).left(4) == "-d1:")
        {
            QString dateIniStr = argumentList[i].right(argumentList[i].length()-4);
            dateIni = QDate::fromString(dateIniStr, "dd/MM/yyyy");
        }
        else if (argumentList.at(i).left(4) == "-d2:")
        {
            QString dateFinStr = argumentList[i].right(argumentList[i].length()-4);
            dateFin = QDate::fromString(dateFinStr, "dd/MM/yyyy");
        }

    }

    if (! dateIni.isValid())
    {
        myProject->logError("Wrong initial date");
        return PRAGA_INVALID_COMMAND;
    }

    if (! dateFin.isValid())
    {
        myProject->logError("Wrong final date");
        return PRAGA_INVALID_COMMAND;
    }

    if (! myProject->timeAggregateGrid(dateIni, dateFin, variables, true, true))
        return PRAGA_ERROR;

    return PRAGA_OK;
}

int cmdHourlyDerivedVariablesGrid(PragaProject* myProject, QList<QString> argumentList)
{

    // default date
    QDate first = QDate::currentDate();
    QDate last = first.addDays(9);

    for (int i = 1; i < argumentList.size(); i++)
    {
        if (argumentList.at(i).left(4) == "-d1:")
        {
            QString dateIniStr = argumentList[i].right(argumentList[i].length()-4);
            first = QDate::fromString(dateIniStr, "dd/MM/yyyy");
        }
        else if (argumentList.at(i).left(4) == "-d2:")
        {
            QString dateFinStr = argumentList[i].right(argumentList[i].length()-4);
            last = QDate::fromString(dateFinStr, "dd/MM/yyyy");
        }

    }

    if (! first.isValid())
    {
        myProject->logError("Wrong initial date");
        return PRAGA_INVALID_COMMAND;
    }

    if (! last.isValid())
    {
        myProject->logError("Wrong final date");
        return PRAGA_INVALID_COMMAND;
    }

    if (! myProject->hourlyDerivedVariablesGrid(first, last, true, true))
        return PRAGA_ERROR;

    return PRAGA_OK;
}

int cmdMonthlyIntegrationVariablesGrid(PragaProject* myProject, QList<QString> argumentList)
{

    // default date
    QDate first = QDate::currentDate();
    QDate last = first.addDays(9);
    QList <QString> varString;
    QList <meteoVariable> variables;
    QString var;
    meteoVariable meteoVar;

    for (int i = 1; i < argumentList.size(); i++)
    {
        if (argumentList.at(i).left(3) == "-v:")
        {
            varString = argumentList[i].right(argumentList[i].length()-3).split(",");
            foreach (var,varString)
            {
                meteoVar = getMeteoVar(var.toStdString());
                if (meteoVar != noMeteoVar) variables << meteoVar;
            }
        }
        else if (argumentList.at(i).left(4) == "-d1:")
        {
            QString dateIniStr = argumentList[i].right(argumentList[i].length()-4);
            first = QDate::fromString(dateIniStr, "dd/MM/yyyy");
        }
        else if (argumentList.at(i).left(4) == "-d2:")
        {
            QString dateFinStr = argumentList[i].right(argumentList[i].length()-4);
            last = QDate::fromString(dateFinStr, "dd/MM/yyyy");
        }

    }

    if (! first.isValid())
    {
        myProject->logError("Wrong initial date");
        return PRAGA_INVALID_COMMAND;
    }

    if (variables.isEmpty())
    {
        myProject->logError("Wrong variable");
        return PRAGA_INVALID_COMMAND;
    }

    if (! last.isValid())
    {
        myProject->logError("Wrong final date");
        return PRAGA_INVALID_COMMAND;
    }

    if (! myProject->monthlyVariablesGrid(first, last, variables))
        return PRAGA_ERROR;

    return PRAGA_OK;
}

#ifdef NETCDF
    int cmdDroughtIndexGrid(PragaProject* myProject, QList<QString> argumentList)
    {
        if (argumentList.size() < 2)
        {
            myProject->logError("Missing xml name");
            return PRAGA_INVALID_COMMAND;
        }

        QString xmlName = myProject->getCompleteFileName(argumentList.at(1), PATH_PROJECT);
        if (!myProject->exportXMLElabGridToNetcdf(xmlName))
        {
            return PRAGA_ERROR;
        }

        return PRAGA_OK;
    }
#endif

int cmdGridAggregationOnZones(PragaProject* myProject, QList<QString> argumentList)
{
    if (argumentList.size() < 4)
    {
        myProject->logError("Missing parameters for aggregation on zones");
        return PRAGA_INVALID_COMMAND;
    }

    QDate first, last;
    QList <meteoVariable> variables;
    QList <QString> varString;
    QList <QString> aggregationList;
    QString var, aggregation;
    meteoVariable meteoVar;

    for (int i = 1; i < argumentList.size(); i++)
    {
        // variables
        if (argumentList.at(i).left(3) == "-v:")
        {
            varString = argumentList[i].right(argumentList[i].length()-3).split(",");
            foreach (var,varString)
            {
                meteoVar = getMeteoVar(var.toStdString());
                if (meteoVar != noMeteoVar) variables << meteoVar;
            }
        }
        // aggregation: STDDEV, MEDIAN, AVG or PERC95
        else if (argumentList.at(i).left(3) == "-a:")
        {
            aggregationList = argumentList[i].right(argumentList[i].length()-3).toUpper().split(",");
            foreach (aggregation, aggregationList)
            {
                if (aggregation != "STDDEV" && aggregation != "MEDIAN" && aggregation != "AVG" && aggregation != "PERC95")
                {
                    myProject->logError("Valid aggregation: STDDEV, MEDIAN, AVG, PERC95)");
                    return PRAGA_INVALID_COMMAND;
                }
            }
        }
        else if (argumentList.at(i).left(4) == "-d1:")
        {
            QString dateIniStr = argumentList[i].right(argumentList[i].length()-4);
            first = QDate::fromString(dateIniStr, "dd/MM/yyyy");
        }
        else if (argumentList.at(i).left(4) == "-d2:")
        {
            QString dateFinStr = argumentList[i].right(argumentList[i].length()-4);
            last = QDate::fromString(dateFinStr, "dd/MM/yyyy");
        }
        else if (argumentList.at(i).left(10) == "-yesterday")
        {
            first = QDate::currentDate().addDays(-1);
            last = first;
        }
        else if (argumentList.at(i).left(6) == "-today")
        {
            first = QDate::currentDate();
            last = first;
        }
        else if (argumentList.at(i).left(11) == "-next10days")
        {
            first = QDate::currentDate();
            last = QDate::currentDate().addDays(9);
        }

    }
    if (variables.isEmpty())
    {
        myProject->logError("Wrong variable");
        return PRAGA_INVALID_COMMAND;
    }

    if (aggregationList.isEmpty())
    {
        myProject->logError("Wrong aggregation");
        return PRAGA_INVALID_COMMAND;
    }

    if (! first.isValid())
    {
        myProject->logError("Wrong initial date");
        return PRAGA_INVALID_COMMAND;
    }

    if (! last.isValid())
    {
        myProject->logError("Wrong final date");
        return PRAGA_INVALID_COMMAND;
    }

    std::vector<float> outputValues;
    float threshold = NODATA;
    meteoComputation elab1MeteoComp = noMeteoComp;
    QString periodType = "D";

    QString rasterName;
    if (!myProject->aggregationDbHandler->getRasterName(&rasterName))
    {
        myProject->logError("Missing Raster Name inside aggregation db.");
        return PRAGA_ERROR;
    }

    // open raster
    gis::Crit3DRasterGrid* myRaster = new gis::Crit3DRasterGrid();
    QString fnWithoutExt = myProject->projectPragaFolder+"/"+rasterName;
    std::string myError = "";
    if (! gis::readEsriGrid(fnWithoutExt.toStdString(), myRaster, myError))
    {
        myProject->logError("Load raster failed: " + QString::fromStdString(myError));
        delete myRaster;
        return PRAGA_ERROR;
    }

    for (int i = 0; i<variables.size(); i++)
    {
        for (int j = 0; j < aggregationList.size(); j++)
        {
            myProject->logInfo("Computing variable number: "+QString::number(i) + ", aggregation number: "+QString::number(j));
            if (!myProject->averageSeriesOnZonesMeteoGrid(variables[i], elab1MeteoComp, aggregationList[j], threshold, myRaster, first, last, periodType, outputValues, false))
            {
                delete myRaster;
                return PRAGA_ERROR;
            }
        }
    }
    delete myRaster;

    return PRAGA_OK;
}

int executeCommand(QList<QString> argumentList, PragaProject* myProject)
{
    if (argumentList.size() == 0) return PRAGA_INVALID_COMMAND;
    bool isCommandFound;
    int isExecuted;

    myProject->logInfo(getTimeStamp(argumentList));

    isExecuted = executeSharedCommand(myProject, argumentList, &isCommandFound);
    if (isCommandFound) return isExecuted;

    isExecuted = myProject->executePragaCommand(argumentList, &isCommandFound);
    if (isCommandFound) return isExecuted;

    myProject->logError("This is not a valid PRAGA command.");
    return PRAGA_INVALID_COMMAND;
}


int pragaBatch(PragaProject* myProject, QString scriptFileName)
{
    #ifdef _WIN32
        attachOutputToConsole();
    #endif

    myProject->logInfo("\nPRAGA v1");
    myProject->logInfo("Execute script: " + scriptFileName);

    if (scriptFileName == "")
    {
        myProject->logError("No script file provided");
        return PRAGA_MISSING_FILE;
    }

    QFile scriptFile(scriptFileName);
    if(! scriptFile.open (QIODevice::ReadOnly))
    {
        myProject->logError(scriptFile.errorString());
        return PRAGA_ERROR;
    }

    QTextStream myStream (&scriptFile);
    QString cmdLine;

    int result;
    while (! scriptFile.atEnd())
    {
        cmdLine = scriptFile.readLine();
        QList<QString> argumentList = getArgumentList(cmdLine);
        result = executeCommand(argumentList, myProject) ;
        if (result != 0)
        {
            myProject->logError("Praga batch error code: "+QString::number(result));
            return result;
        }
    }

    myProject->logInfo("Batch finished at: " + QDateTime::currentDateTime().toString("yyyy-MM-dd hh:mm:ss"));

    scriptFile.close();

    #ifdef _WIN32
        // Send "enter" to release application from the console
        // This is a hack, but if not used the console doesn't know the application has
        // returned. The "enter" key only sent if the console window is in focus.
        if (isConsoleForeground()) sendEnterKey();
    #endif

    return PRAGA_OK;
}


int pragaShell(PragaProject* myProject)
{
    #ifdef _WIN32
        openNewConsole();
    #endif
    int result;
    while (! myProject->requestedExit)
    {
        QString commandLine = getCommandLine("PRAGA");
        if (commandLine != "")
        {
            QList<QString> argumentList = getArgumentList(commandLine);
            result = executeCommand(argumentList, myProject);
            if (result != 0)
            {
                myProject->logError("Praga shell error code: "+QString::number(result));
                //return result;
            }
        }
    }

    return PRAGA_OK;
}

#ifdef NETCDF

    int cmdNetcdfExport(PragaProject* myProject, QList<QString> argumentList)
    {
        if (argumentList.size() < 2)
        {
            myProject->logError("Missing netcdf name");
            return PRAGA_INVALID_COMMAND;
        }

        QString netcdfName = myProject->getCompleteFileName(argumentList.at(1), PATH_PROJECT);
        if (! myProject->checkMeteoGridForExport())
        {
            return PRAGA_ERROR;
        }

        if (! myProject->exportMeteoGridToNetCDF(netcdfName, "MeteoGrid", "Variable", "unit", NO_DATE, 0, 0, 0))
        {
            return PRAGA_ERROR;
        }
        return PRAGA_OK;
    }

    int cmdExportXMLElabToNetcdf(PragaProject* myProject, QList<QString> argumentList)
    {
        if (argumentList.size() < 2)
        {
            myProject->logError("Missing xml name");
            return PRAGA_INVALID_COMMAND;
        }

        QString xmlName = myProject->getCompleteFileName(argumentList.at(1), PATH_PROJECT);
        if (!myProject->exportXMLElabGridToNetcdf(xmlName))
        {
            return PRAGA_ERROR;
        }

        return PRAGA_OK;
    }

#endif
    /*
    bool cmdLoadForecast(PragaProject* myProject, QList<QString> argumentList)
    {
        if (argumentList.size() < 2)
        {
            myProject->logError("Missing file name");
            return false;
        }

        bool overWrite = false;
        bool checkTables = false;

        for (int i = 2; i < argumentList.size(); i++)
        {
            if (argumentList[i] == "-o")
            {
                overWrite = true;
            }
            else if (argumentList[i] == "-c")
            {
                checkTables = true;
            }
            else
            {
                myProject->logError("Unknow option: loadforecast file -o -c ");
                return false;
            }
        }

        QString fileName = myProject->getCompleteFileName(argumentList.at(1), PATH_PROJECT);
        if (!myProject->loadForecastToGrid(fileName, overWrite, checkTables))
        {
            return false;
        }

        return true;
    }
    */
    int cmdComputeClimaFromXMLSaveOnDB(PragaProject* myProject, QList<QString> argumentList)
    {
        if (argumentList.size() < 2)
        {
            myProject->logError("Missing xml name");
            return PRAGA_INVALID_COMMAND;
        }

        QString xmlName = myProject->getCompleteFileName(argumentList.at(1), PATH_PROJECT);
        if (!myProject->computeClimaFromXMLSaveOnDB(xmlName))
        {
            return PRAGA_ERROR;
        }

        return PRAGA_OK;
    }

    int cmdDroughtIndexPoint(PragaProject* myProject, QList<QString> argumentList)
    {

        if (argumentList.size() < 5)
        {
            myProject->logError("Missing parameters for computing drought index point");
            return PRAGA_INVALID_COMMAND;
        }

        bool ok = false;
        int timescale;
        int ry1, ry2;
        droughtIndex index;

        for (int i = 1; i < argumentList.size(); i++)
        {
            if (argumentList.at(i).left(3) == "-i:")
            {
                QString indexStr = argumentList[i].right(argumentList[i].length()-3).toUpper();

                if (indexStr == "SPI" || indexStr == "INDEX_SPI")
                {
                    index = INDEX_SPI;
                }
                else if (indexStr == "SPEI" || indexStr == "INDEX_SPEI")
                {
                    index = INDEX_SPEI;
                }
                else if (indexStr == "DECILES" || indexStr == "INDEX_DECILES")
                {
                    index = INDEX_DECILES;
                }
                else
                {
                    myProject->logError("Wrong index: -i:<SPI/SPEI/DECILES>");
                    return PRAGA_INVALID_COMMAND;
                }
            }
            if (argumentList.at(i).left(3) == "-t:")
            {
                timescale = argumentList[i].right(argumentList[i].length()-3).toInt(&ok);
                if (!ok)
                {
                    myProject->logError("Wrong timescale: -t:<integer number>");
                    return PRAGA_INVALID_COMMAND;
                }
            }
            if (argumentList.at(i).left(5) == "-ry1:")
            {
                ry1 = argumentList[i].right(argumentList[i].length()-5).toInt(&ok);
                if (!ok)
                {
                    myProject->logError("Wrong reference start year: -ry1:<integer number>");
                    return PRAGA_INVALID_COMMAND;
                }
            }
            else if (argumentList.at(i).left(5) == "-ry2:")
            {
                ry2 = argumentList[i].right(argumentList[i].length()-5).toInt(&ok);
                if (!ok)
                {
                    myProject->logError("Wrong reference end year: -ry2:<integer number>");
                    return PRAGA_INVALID_COMMAND;
                }
            }
        }
        if (! myProject->computeDroughtIndexPoint(index, timescale, ry1, ry2))
        {
            return PRAGA_ERROR;
        }

        return PRAGA_OK;
    }

    int cmdSaveLogDataProceduresGrid(PragaProject* myProject, QList<QString> argumentList)
    {
        if (argumentList.size() < 3)
        {
            myProject->logError("Missing procedure name or date to save");
            return PRAGA_INVALID_COMMAND;
        }

        QString nameProc = argumentList.at(1);
        QString dateStr = argumentList.at(2);
        QDate date = QDate::fromString(dateStr, "dd/MM/yyyy");

        if (!myProject->saveLogProceduresGrid(nameProc, date))
        {
            return PRAGA_ERROR;
        }

        return PRAGA_OK;
    }
