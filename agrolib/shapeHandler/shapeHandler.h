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

        bool getShape(int index, ShapeObject &shape) const;
        int	getDBFFieldIndex(const char *pszFieldName) const;
        int	isDBFRecordDeleted(int record) const;

        int getFieldPos(std::string fieldName) const;
        bool existField(std::string fieldName) const;
        DBFFieldType getFieldType(std::string fieldName) const;

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

        void packDBF(std::string newFile);
        void packSHP(std::string newFile);
        bool existRecordDeleted();

        double getNumericValue(int shapeNumber, std::string fieldName);
        double getNumericValue(int shapeNumber, int fieldPos);
        std::string getStringValue(int shapeNumber, std::string fieldName);

        std::vector<unsigned int> getHoles(int shapeNumber, int partNumber);

        int getShapeIndexfromPoint(double utmX, double utmY);
        std::string getAttributesList(int index);

        std::string getTypeString() const
        { return getShapeTypeAsString(m_type); }

        std::string	getFieldName(int fieldPos) const
        { return m_fieldsList.at(unsigned(fieldPos)); }

        DBFFieldType getFieldType(int fieldPos) const
        { return m_fieldsTypeList.at(unsigned(fieldPos)); }

        int	getType() const { return m_type; }
        int getFieldNumbers() const { return m_fields; }
        int getShapeCount() const { return m_count; }
        int	getDBFRecordCount() const { return m_dbf->nRecords; }

        bool getIsWGS84() const { return m_isWGS84; }
        bool getIsNorth() const { return m_isNorth; }
        int getUtmZone() const { return m_utmZone; }

        std::string getFilepath() const { return m_filepath; }
        void setFilepath(std::string filename) { m_filepath = filename; }

        int nWidthField(int fieldIndex) { return m_dbf->panFieldSize[fieldIndex]; }
        int nDecimalsField(int fieldIndex) { return m_dbf->panFieldDecimals[fieldIndex]; }

        int getNrParts() const { return m_parts; }
        int getNrHoles() const { return m_holes; }
    };


#endif // SHAPEHANDLER_H
