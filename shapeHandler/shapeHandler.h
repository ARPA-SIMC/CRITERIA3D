#ifndef SHAPEHANDLER_H
#define SHAPEHANDLER_H

    #include <string>
    #include <vector>
    #include <shapelib/shapefil.h>
    #include "shapeObject.h"

    class Crit3DShapeHandler
    {
    protected:
        SHPHandle	m_handle;
        DBFHandle   m_dbf;
        int         m_count;
        int			m_type;
        int         m_fields;
        std::string m_filepath;
        std::vector<std::string> m_fieldsList;
        std::vector<DBFFieldType> m_fieldsTypeList;
        std::vector< std::vector<std::vector<unsigned int>>> holes;
        bool        m_isWGS84;
        bool        m_isNorth;
        int         m_utmZone;
        int         m_parts;
        int         m_holes;

    public:
        Crit3DShapeHandler();
        ~Crit3DShapeHandler();

        bool open(std::string filename);
        bool openDBF(std::string filename);
        bool openSHP(std::string filename);
        void newShapeFile(std::string filename, int nShapeType);
        bool isWGS84Proj(std::string prjFileName);
        bool setUTMzone(std::string prjFileName);
        void close();
        void closeDBF();
        void closeSHP();
        bool getShape(int index, ShapeObject &shape);
        int	getShapeCount();
        int	getDBFRecordCount();
        int	getDBFFieldIndex(const char *pszFieldName);
        int	isDBFRecordDeleted(int record);
        int	getType();
        int getFieldNumbers();
        std::string	getTypeString();

        std::string	getFieldName(int fieldPos);
        DBFFieldType getFieldType(int fieldPos);

        int getFieldPos(std::string fieldName);
        bool existField(std::string fieldName);

        DBFFieldType getFieldType(std::string fieldName);

        int readIntAttribute(int shapeNumber, int fieldPos);
        bool writeIntAttribute(int shapeNumber, int fieldPos, int nFieldValue);

        double readDoubleAttribute(int shapeNumber, int fieldPos);
        bool writeDoubleAttribute(int shapeNumber, int fieldPos, double dFieldValue);

        std::string readStringAttribute(int shapeNumber, int fieldPos);
        bool writeStringAttribute(int shapeNumber, int fieldPos, const char* pszFieldValue);

        bool deleteRecord(int shapeNumber);
        //bool addRecord(std::vector<std::string> fields);

        bool addShape(std::string type, std::vector<double> coordinates);
        bool addField(const char * fieldName, int type, int nWidth, int nDecimals );
        bool removeField(int iField);

        void setFilepath(std::string filename);
        std::string getFilepath() const;

        bool getIsWGS84() const;
        bool getIsNorth() const;
        int getUtmZone() const;

        void packDBF(std::string newFile);
        void packSHP(std::string newFile);
        bool existRecordDeleted();

        int nWidthField(int fieldIndex);
        int nDecimalsField(int fieldIndex);

        double getNumericValue(int shapeNumber, std::string fieldName);
        double getNumericValue(int shapeNumber, int fieldPos);
        std::string getStringValue(int shapeNumber, std::string fieldName);

        std::vector<unsigned int> getHoles(int shapeNumber, int partNumber);
        int getNrParts() const;
        int getNrHoles() const;
    };


#endif // SHAPEHANDLER_H
