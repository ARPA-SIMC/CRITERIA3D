#ifndef ZONALSTATISTIC_H
#define ZONALSTATISTIC_H

    #ifndef SHAPEHANDLER_H
        #include "shapeHandler.h"
    #endif
    #ifndef GIS_H
        #include "gis.h"
    #endif

    std::vector <std::vector<int> > computeMatrixAnalysis(Crit3DShapeHandler &shapeRef, Crit3DShapeHandler &shapeVal,
                              gis::Crit3DRasterGrid &rasterRef, gis::Crit3DRasterGrid &rasterVal, std::vector<int> &vectorNull);

    bool zonalStatisticsShape(Crit3DShapeHandler &shapeRef, Crit3DShapeHandler &shapeVal,
                              std::vector<std::vector<int> > &matrix, std::vector<int>& vectorNull,
                              std::string valField, std::string valFieldOutput, std::string aggregationType,
                              double threshold, std::string &error);

    bool zonalStatisticsShapeMajority(Crit3DShapeHandler &shapeRef, Crit3DShapeHandler &shapeVal,
                              std::vector<std::vector<int> > &matrix, std::vector<int> &vectorNull,
                              std::string valField, std::string valFieldOutput,
                              double threshold, std::string &error);

#endif // ZONALSTATISTIC_H
