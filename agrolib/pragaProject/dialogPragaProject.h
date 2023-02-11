#ifndef DIALOGPRAGAPROJECT_H
#define DIALOGPRAGAPROJECT_H

#include <QWidget>

#ifndef PRAGAPROJECT_H
    #include "pragaProject.h"
#endif

#ifndef DIALOGPROJECT_H
    #include "dialogProject.h"
#endif


class DialogPragaProject : public DialogProject
{
    Q_OBJECT

    public:
        explicit DialogPragaProject(PragaProject* myProject);

        void accept();

    protected:
        PragaProject* project_;

    private:

};


#ifdef NETCDF
    bool netCDF_ExportDataSeries(NetCDFHandler* netCDF, int& varId, QDateTime& firstDate, QDateTime& lastDate);
    bool netCDF_ChooseVariable(NetCDFHandler *netCDF, int& varId, frequencyType currentFrequency);
#endif


#endif // DIALOGPRAGAPROJECT_H
