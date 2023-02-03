#ifndef PRAGASHELL_H
#define PRAGASHELL_H

    #ifndef PRAGAPROJECT_H
        #include "pragaProject.h"
    #endif

    QList<QString> getPragaCommandList();
    int cmdList(PragaProject* myProject);

    int executeCommand(QList<QString> argumentList, PragaProject* myProject);
    int pragaShell(PragaProject* myProject);
    int pragaBatch(PragaProject* myProject, QString batchFileName);

    int cmdOpenPragaProject(PragaProject* myProject, QList<QString> argumentList);
    int cmdDownload(PragaProject* myProject, QList<QString> argumentList);
    int cmdInterpolationGridPeriod(PragaProject* myProject, QList<QString> argumentList);
    int cmdAggregationGridPeriod(PragaProject* myProject, QList<QString> argumentList);
    int cmdHourlyDerivedVariablesGrid(PragaProject* myProject, QList<QString> argumentList);
    int cmdGridAggregationOnZones(PragaProject* myProject, QList<QString> argumentList);
    int cmdMonthlyIntegrationVariablesGrid(PragaProject* myProject, QList<QString> argumentList);
    //bool cmdLoadForecast(PragaProject* myProject, QList<QString> argumentList);

    #ifdef NETCDF
        int cmdDroughtIndexGrid(PragaProject* myProject, QList<QString> argumentList);
        int cmdNetcdfExport(PragaProject* myProject, QList<QString> argumentList);
        int cmdExportXMLElabToNetcdf(PragaProject* myProject, QList<QString> argumentList);
    #endif

#endif // PRAGASHELL_H
