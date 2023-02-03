#ifndef NETCDFHANDLER_H
#define NETCDFHANDLER_H

    #ifndef GIS_H
        #include "gis.h"
    #endif

    #include <sstream>

    class NetCDFVariable
    {
    public:
        std::string name;
        std::string longName;
        std::string unit;
        int id;
        int type;

        NetCDFVariable();
        NetCDFVariable(char *_name, int _id, int _type);

        std::string getVarName();
    };

    class NetCDFHandler
    {
    private:
        int utmZone;
        long nrX, nrY, nrLat, nrLon, nrTime;
        int idTime, idTimeBnds, idX, idY, idLat, idLon;

        float *x, *y;
        float *lat, *lon;
        double *time;
        bool isLatDecreasing;

        Crit3DDate firstDate;
        Crit3DDate lastDate;

        std::stringstream metadata;
        std::string timeUnit;
        std::vector<NetCDFVariable> dimensions;
        std::vector<NetCDFVariable> variables;

    public:
        int ncId;
        bool isUTM;
        bool isLatLon;
        bool isRotatedLatLon;

        bool isStandardTime;
        bool isHourly;
        bool isDaily;

        gis::Crit3DRasterGrid dataGrid;
        gis::Crit3DGridHeader latLonHeader;

        NetCDFHandler();

        void close();
        void clear();

        inline void initialize(int _utmZone) { this->close(); utmZone = _utmZone; }

        inline bool isLoaded() { return (variables.size() > 0); }

        inline bool isTimeReadable() { return (getFirstTime() != NO_DATETIME); }

        inline unsigned int getNrVariables() { return unsigned(variables.size()); }

        //inline std::string getVarName(int idVar) { return getVariableFromId(idVar).getVarName(); }

        inline std::string getMetadata() { return metadata.str(); }

        inline std::string getTimeUnit() { return timeUnit; }

        inline Crit3DTime getFirstTime() { return getTime(0); }

        inline Crit3DTime getLastTime() { return getTime(nrTime-1); }

        bool isPointInside(gis::Crit3DGeoPoint geoPoint);

        bool setVarLongName(const std::string &varName, const std::string& varLongName);
        bool setVarUnit(const std::string& varName, const std::string &varUnit);

        int getDimensionIndex(char* dimName);
        std::string getDateTimeStr(int timeIndex);

        Crit3DTime getTime(int timeIndex);

        NetCDFVariable getVariableFromId(int idVar);
        NetCDFVariable getVariableFromIndex(int index);

        bool readProperties(std::string fileName);
        bool exportDataSeries(int idVar, gis::Crit3DGeoPoint geoPoint, Crit3DTime firstTime, Crit3DTime lastTime, std::stringstream *buffer);
        bool extractVariableMap(int idVar, const Crit3DTime &myTime, std::string &error);

        bool createNewFile(std::string fileName);

        bool writeMetadata(const gis::Crit3DGridHeader& latLonHeader, const std::string &title,
                           const std::string &variableName, const std::string &variableUnit,
                           const Crit3DDate &myDate, int nDays, int refYearStart, int refYearEnd);

        bool writeData_NoTime(const gis::Crit3DRasterGrid& myDataGrid);
    };


#endif // NETCDFHANDLER_H
