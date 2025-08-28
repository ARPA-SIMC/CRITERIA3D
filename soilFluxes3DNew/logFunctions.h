#ifndef SOILFLUXES3D_LOGFUNCTIONS_H
#define SOILFLUXES3D_LOGFUNCTIONS_H

#include <array>
#include <utility>
#include <sstream>
#include <iomanip>
#include "mat.h"
#include "matrix.h"

#include "types_cpu.h"

using namespace soilFluxes3D::New;

namespace soilFluxes3D::Log
{
    enum class approxFieldsKeys : uint8_t {matA, vecB, vecX, stepRes, numFields};
    enum class matrixFieldsKeys : uint8_t {rowIdx, colIdx, values, numFields};

    constexpr std::array<std::pair<approxFieldsKeys, const char*>, static_cast<std::size_t>(approxFieldsKeys::numFields)> approxFieldsMap =
        {{
            {approxFieldsKeys::matA, "matrixA"},
            {approxFieldsKeys::vecB, "vectorB"},
            {approxFieldsKeys::vecX, "vectorX"},
            {approxFieldsKeys::stepRes, "stepResult"}
        }};

    constexpr std::array<std::pair<matrixFieldsKeys, const char*>, static_cast<std::size_t>(matrixFieldsKeys::numFields)> matrixFieldsMap =
        {{
            {matrixFieldsKeys::rowIdx, "rowIndeces"},
            {matrixFieldsKeys::colIdx, "colIndexes"},
            {matrixFieldsKeys::values, "elemValues"},
        }};

    template<typename EnumType>
    inline const char** getFieldsNameArray(const std::array<std::pair<EnumType, const char*>, static_cast<std::size_t>(EnumType::numFields)>& map)
    {
        constexpr int numElements = static_cast<int>(EnumType::numFields);
        static const char* fieldsNameArray[numElements] = {};

        for(const auto& [key, value] : map)
            fieldsNameArray[static_cast<uint8_t>(key)] = value;

        return fieldsNameArray;
    }

    template<typename EnumType>
    inline const char* getFieldName(const std::array<std::pair<EnumType, const char*>, static_cast<std::size_t>(EnumType::numFields)>& map, EnumType selectedKey)
    {
        for(const auto& [key, value] : map)
            if(key == selectedKey)
                return value;

        return "";
    }

    struct logData_t
    {
        std::string projectName;
        std::string basePath;
        std::string timeData;
        uint16_t fileCounter = 0;
        uint16_t numStepDone = 0;

        const uint16_t numApprox = 30;        //TO DO: legare alla memoria?
        mxArray* mainStruct = nullptr;

        const char* masterFieldsNames[2] = {"projectName", "approxDataArray"};
        const char** approxFieldsNames = getFieldsNameArray(approxFieldsMap);
        const char** matrixFieldsNames = getFieldsNameArray(matrixFieldsMap);
    };


    SF3Derror_t inizializeLogData(const std::string &logPath, const std::string &projectName);
    void inizializeLogStructure();
    void createCurrStepLog(const MatrixCPU& matrix, const VectorCPU& vectorB, const VectorCPU& vectorX, bool stepResult);
    void createBinData(uint16_t stepNum, const MatrixCPU& matrix, const VectorCPU& vectorB, const VectorCPU& vectorX);
    SF3Derror_t writeLogFile();

    inline std::string customPadInteger(int number, std::streamsize totalDigits = 5)
    {
        std::ostringstream oss;
        oss << std::setw(totalDigits) << std::setfill('0') << number;
        return oss.str();
    }

}


#endif // SOILFLUXES3D_LOGFUNCTIONS_H
