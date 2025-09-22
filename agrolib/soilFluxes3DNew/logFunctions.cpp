#include "logFunctions.h"

namespace soilFluxes3D::Log
{
    logData_t logData;

    SF3Derror_t inizializeLogData(const std::string& logPath, const std::string& projectName)
    {
        if((logPath == "") || (projectName == "default"))
            return SF3Derror_t::SF3Dok;

        logData.basePath = logPath;
        logData.projectName = projectName;
        logData.fileCounter = 0;
        logData.numStepDone = 0;

        inizializeLogStructure();
        return SF3Derror_t::SF3Dok;
    }

    void inizializeLogStructure()
    {
        if(logData.mainStruct != nullptr)
            mxDestroyArray(logData.mainStruct);

        logData.mainStruct = mxCreateStructMatrix(1, 1, 2, logData.masterFieldsNames);
        mxSetField(logData.mainStruct, 0, logData.masterFieldsNames[0], mxCreateString(logData.projectName.c_str()));
        mxSetField(logData.mainStruct, 0, logData.masterFieldsNames[1], mxCreateStructMatrix(logData.numApprox, 1, static_cast<int>(approxFieldsKeys::numFields), logData.approxFieldsNames));
    }

    void createCurrStepLog(const MatrixCPU& matrix, const VectorCPU& vectorB, const VectorCPU& vectorX, bool stepResult)
    {
        if(logData.numStepDone >= logData.numApprox)
            writeLogFile();

        if(logData.mainStruct == nullptr)
            inizializeLogStructure();

        mxSetField(mxGetField(logData.mainStruct, 0, logData.masterFieldsNames[1]), logData.numStepDone, getFieldName(approxFieldsMap, approxFieldsKeys::stepRes), mxCreateLogicalScalar(stepResult));
        createBinData(logData.numStepDone, matrix, vectorB, vectorX);

        logData.numStepDone++;
    }

    void createBinData(uint16_t stepNum, const MatrixCPU& matrix, const VectorCPU& vectorB, const VectorCPU& vectorX)
    {
        size_t size = static_cast<size_t>(matrix.numRows);

        mxArray* matA = mxCreateStructMatrix(1, 1, 3, logData.matrixFieldsNames);

        const uint64_t nnz = size * matrix.maxColumns;
        mxArray *rowIdx = mxCreateNumericMatrix(nnz, 1, mxUINT64_CLASS, mxREAL);
        mxArray *colIdx = mxCreateNumericMatrix(nnz, 1, mxUINT64_CLASS, mxREAL);
        mxArray *values = mxCreateNumericMatrix(nnz, 1, mxDOUBLE_CLASS, mxREAL);

        uint64_t* rowPtr = static_cast<uint64_t*>(mxGetData(rowIdx));
        uint64_t* colPtr = static_cast<uint64_t*>(mxGetData(colIdx));
        double* valPtr = static_cast<double*>(mxGetData(values));

        uint64_t cnz = 0;
        for(uint64_t rIdx = 0; rIdx < matrix.numRows; ++rIdx)
            for(uint64_t cIdx = 0; cIdx < matrix.numColumns[rIdx]; ++cIdx)
            {
                rowPtr[cnz] = rIdx;
                colPtr[cnz] = matrix.colIndeces[rIdx][cIdx];
                valPtr[cnz] = matrix.values[rIdx][cIdx];
                cnz++;
            }

        mxSetField(matA, 0, getFieldName(matrixFieldsMap, matrixFieldsKeys::rowIdx), rowIdx);
        mxSetField(matA, 0, getFieldName(matrixFieldsMap, matrixFieldsKeys::colIdx), colIdx);
        mxSetField(matA, 0, getFieldName(matrixFieldsMap, matrixFieldsKeys::values), values);

        mxSetField(mxGetField(logData.mainStruct, 0, logData.masterFieldsNames[1]), stepNum, getFieldName(approxFieldsMap, approxFieldsKeys::matA), matA);

        mxArray* vecB = mxCreateDoubleMatrix(size, 1, mxREAL);
        std::memcpy(mxGetPr(vecB), vectorB.values, size * sizeof(double));
        mxSetField(mxGetField(logData.mainStruct, 0, logData.masterFieldsNames[1]), stepNum, getFieldName(approxFieldsMap, approxFieldsKeys::vecB), vecB);

        mxArray* vecX = mxCreateDoubleMatrix(size, 1, mxREAL);
        std::memcpy(mxGetPr(vecX), vectorX.values, size * sizeof(double));
        mxSetField(mxGetField(logData.mainStruct, 0, logData.masterFieldsNames[1]), stepNum, getFieldName(approxFieldsMap, approxFieldsKeys::vecX), vecX);
    }

    SF3Derror_t writeLogFile()
    {
        if(logData.mainStruct == nullptr)
            return SF3Derror_t::SF3Dok;

        std::string fileName = logData.basePath + "binLogData_" + logData.projectName + customPadInteger(++logData.fileCounter) + ".mat";    //TO DO: implement date in fileName
        MATFile *binFile = matOpen(fileName.c_str(), "w");
        if(binFile == nullptr)
            return SF3Derror_t::FileError;

        matPutVariable(binFile, "mainStruct", logData.mainStruct);  //TO DO: decouple the struct? + Add check
        matClose(binFile);

        mxDestroyArray(logData.mainStruct);
        logData.mainStruct = nullptr;
        logData.numStepDone = 0;
        return SF3Derror_t::SF3Dok;
    }
}
