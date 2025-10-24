#pragma once

#include <array>
#include <utility>
#include <sstream>
#include <iomanip>
#include <type_traits>

#include "mat.h"
#include "matrix.h"

#ifdef CUDA_ENABLED
    #include <cuda.h>
    #include <cuda_runtime.h>
#endif

#include "types_cpu.h"

using namespace soilFluxes3D::v2;

namespace soilFluxes3D::v2::Log
{
    //Log system
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
            fieldsNameArray[static_cast<u8_t>(key)] = value;

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
        u16_t fileCounter = 0;
        u16_t numStepDone = 0;

        const u16_t numApprox = 30;        //TO DO: legare alla memoria?
        mxArray* mainStruct = nullptr;

        const char* masterFieldsNames[2] = {"projectName", "approxDataArray"};
        const char** approxFieldsNames = getFieldsNameArray(approxFieldsMap);
        const char** matrixFieldsNames = getFieldsNameArray(matrixFieldsMap);
    };


    SF3Derror_t initializeLogData(const std::string &logPath, const std::string &projectName);
    void initializeLogStructure();
    void createCurrStepLog(const MatrixCPU& matrix, const VectorCPU& vectorB, const VectorCPU& vectorX, bool stepResult);
    void createBinData(u16_t stepNum, const MatrixCPU& matrix, const VectorCPU& vectorB, const VectorCPU& vectorX);
    SF3Derror_t writeLogFile();

    inline std::string customPadInteger(int number, std::streamsize totalDigits = 5)
    {
        std::ostringstream oss;
        oss << std::setw(totalDigits) << std::setfill('0') << number;
        return oss.str();
    }

    template<typename T>
    constexpr mxClassID getTypeClassID()
    {
        if constexpr (std::is_enum_v<T>) return getTypeClassID<std::underlying_type_t<T>>();
        else if constexpr (std::is_same_v<T, bool>) return mxLOGICAL_CLASS;
        else if constexpr (std::is_same_v<T, double>) return mxDOUBLE_CLASS;
        else if constexpr (std::is_same_v<T, int>) return mxINT32_CLASS;
        else if constexpr (std::is_same_v<T, SF3Duint_t>) return mxUINT64_CLASS;
        else if constexpr (std::is_same_v<T, u16_t>) return mxUINT16_CLASS;
        else if constexpr (std::is_same_v<T, u8_t>) return mxUINT8_CLASS;
        else return mxUNKNOWN_CLASS;
    }

    template<typename T>
    inline void logVectorData(MATFile* file, const T* ptr, const std::size_t size, const char* name, const solverType deviceUsed)
    {
        constexpr mxClassID classT = getTypeClassID<T>();
        static_assert(classT != mxUNKNOWN_CLASS, "Type non encoded.");
        static_assert(classT != mxLOGICAL_CLASS, "TO DO: bool needs to be handled with explicit casting.");

        mxArray* vec = nullptr;
        if(classT == mxLOGICAL_CLASS)
            vec = mxCreateLogicalMatrix(size, 1);
        else
            vec = mxCreateNumericMatrix(size, 1, classT, mxREAL);

        switch(deviceUsed)
        {
            case solverType::CPU:
                std::memcpy(mxGetData(vec), ptr, size * sizeof(T));
                break;
            case solverType::GPU:
                #ifdef CUDA_ENABLED
                    cudaMemcpy(mxGetData(vec), ptr, size * sizeof(T), cudaMemcpyDeviceToHost);
                #endif
                break;
            default:
                std::exit(EXIT_FAILURE);
        }
        auto tmp = cudaGetLastError();
        matPutVariable(file, name, vec);
        mxDestroyArray(vec);
    }

    #define logVector(ptr, size, name) logVectorData(binFile, ptr, size, name, deviceUsed)
    #define formatName(name) (std::string(name) + customPadInteger(logsCounter, 3)).c_str()

    #define numLogsInFile 5
    #define maxLogFileNum 20

    inline void logNodeGridStruct(const nodesData_t& nodeGrid, const solverType deviceUsed)
    {
        static int logsCounter = 0;
        static int fileCounter = 0;

        if(logsCounter % numLogsInFile == 0)
            fileCounter++;

        if(fileCounter > maxLogFileNum)
            std::exit(EXIT_SUCCESS);    //Maybe move to a simple return

        const char* mode = (logsCounter % numLogsInFile == 0) ? "w" : "u";
        std::string fileName;
        switch(deviceUsed)
        {
            case solverType::CPU:
                fileName = "../../../../Codice/MATLAB script/DebugGPU/TempFileCPU" + customPadInteger(fileCounter, 3) + ".mat";
                break;
            case solverType::GPU:
                fileName = "../../../../Codice/MATLAB script/DebugGPU/TempFileGPU" + customPadInteger(fileCounter, 3) + ".mat";
                break;
            default:
                std::exit(EXIT_FAILURE);
                break;
        }

        MATFile *binFile = matOpen(fileName.c_str(), mode);
        if(binFile == nullptr)
            std::exit(EXIT_FAILURE);

        //Log index
        logVector(&(++logsCounter), 1, formatName("logIndex"));

        //Topology Data
        logVector(nodeGrid.size, nodeGrid.numNodes, formatName("size"));
        logVector(nodeGrid.x, nodeGrid.numNodes, formatName("x"));
        logVector(nodeGrid.y, nodeGrid.numNodes, formatName("y"));
        logVector(nodeGrid.z, nodeGrid.numNodes, formatName("z"));

        //Soil/surface properties pointers
        // logVectDoubleGPU(nodeGrid.soilSurfacePointers, soilSurface_ptr, nodeGrid.numNodes);

        //Boundary data
        // logVectDoubleGPU(nodeGrid.boundaryData.boundaryType, boundaryType_t, nodeGrid.numNodes);
        logVector(nodeGrid.boundaryData.boundarySlope, nodeGrid.numNodes, formatName("bSlope"));
        logVector(nodeGrid.boundaryData.boundarySize, nodeGrid.numNodes, formatName("bSize"));
        logVector(nodeGrid.boundaryData.waterFlowRate, nodeGrid.numNodes, formatName("bWFR"));
        logVector(nodeGrid.boundaryData.waterFlowSum, nodeGrid.numNodes, formatName("bWFS"));
        logVector(nodeGrid.boundaryData.prescribedWaterPotential, nodeGrid.numNodes, formatName("bPWP"));

        //Link data
        // logVectDoubleGPU(nodeGrid.numLateralLink, uint8_t, nodeGrid.numNodes);
        for(u8_t idx = 0; idx < 10; ++idx)
        {
            std::string str = std::to_string(idx);
            // logVectDoubleGPU(nodeGrid.linkData[idx].linktype, linkType_t, nodeGrid.numNodes);
            // logVectDoubleGPU(nodeGrid.linkData[idx].linkIndex, uint64_t, nodeGrid.numNodes);
            logVector(nodeGrid.linkData[idx].interfaceArea, nodeGrid.numNodes, formatName(("l" + str + "IA").c_str()));
            logVector(nodeGrid.linkData[idx].waterFlowSum, nodeGrid.numNodes, formatName(("l" + str + "WFS").c_str()));
        }

        //Water data
        logVector(nodeGrid.waterData.saturationDegree, nodeGrid.numNodes, formatName("wdSA"));
        logVector(nodeGrid.waterData.waterConductivity, nodeGrid.numNodes, formatName("wdWC"));
        logVector(nodeGrid.waterData.waterFlow, nodeGrid.numNodes, formatName("wdWF"));
        logVector(nodeGrid.waterData.pressureHead, nodeGrid.numNodes, formatName("wdPH"));
        logVector(nodeGrid.waterData.waterSinkSource, nodeGrid.numNodes, formatName("wdWSS"));
        logVector(nodeGrid.waterData.pond, nodeGrid.numNodes, formatName("wdP"));
        logVector(nodeGrid.waterData.invariantFluxes, nodeGrid.numNodes, formatName("wdIF"));
        logVector(nodeGrid.waterData.oldPressureHead, nodeGrid.numNodes, formatName("wdOPH"));
        logVector(nodeGrid.waterData.bestPressureHead, nodeGrid.numNodes, formatName("wdBPH"));
        logVector(nodeGrid.waterData.partialCourantWaterLevels, nodeGrid.numNodes, formatName("wdPCWLs"));
        logVector(&(nodeGrid.waterData.CourantWaterLevel), 1, formatName("wdCWL"));

        // logVector(vectBdata, nodeGrid.numNodes, formatName("vectB"));
        // logVector(vectXdata, nodeGrid.numNodes, formatName("vectX"));
        // logVector(vectCdata, nodeGrid.numNodes, formatName("vectC"));

        // logVector(matrixStruct.d_values, matrixStruct.totValuesSize, formatName("matA_values"));
        // logVector(matrixStruct.d_diagonalValues, nodeGrid.numNodes, formatName("matA_diagValues"));

        matClose(binFile);
    }
}
