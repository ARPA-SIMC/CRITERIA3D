#ifndef LINEALIALIB_H
#define LINEALIALIB_H

#pragma once

#include <QLibrary>
#include "linealia.hpp"

class LinealiaLib {
public:
    static LinealiaLib& instance();

    bool load();
    bool isLoaded() const;

    // API wrapper
    LinealiaIterativeResult solveSOR(LinealiaMatrix, LinealiaVector, LinealiaVector,
                                     LinealExecutionParams,
                                     LinealiaIterativeSolverParams,
                                     LinealiaRelaxedParams);

    LinealiaIterativeResult solveSSOR(LinealiaMatrix, LinealiaVector, LinealiaVector,
                                      LinealExecutionParams,
                                      LinealiaIterativeSolverParams,
                                      LinealiaRelaxedParams);

    LinealiaIterativeResult solveCG(LinealiaMatrix, LinealiaVector, LinealiaVector,
                                    LinealExecutionParams,
                                    LinealiaIterativeSolverParams);

    LinealiaIterativeResult solvePCG_SOR(LinealiaMatrix, LinealiaVector, LinealiaVector,
                                         LinealExecutionParams,
                                         LinealiaIterativeSolverParams,
                                         LinealiaRelaxedPreconditionerParams);

    LinealiaIterativeResult solvePCG_SSOR(LinealiaMatrix, LinealiaVector, LinealiaVector,
                                          LinealExecutionParams,
                                          LinealiaIterativeSolverParams,
                                          LinealiaRelaxedPreconditionerParams);

    LinealiaIterativeResult solvePCG_AMG_SOR(LinealiaMatrix, LinealiaVector, LinealiaVector,
                                             LinealExecutionParams,
                                             LinealiaIterativeSolverParams,
                                             LinealiaPcgAmgParams);

    LinealiaIterativeResult solvePCG_AMG_SSOR(LinealiaMatrix, LinealiaVector, LinealiaVector,
                                              LinealExecutionParams,
                                              LinealiaIterativeSolverParams,
                                              LinealiaPcgAmgParams);

private:
    LinealiaLib();
    QLibrary lib;

    typedef LinealiaIterativeResult (*solve_sor_t)(LinealiaMatrix, LinealiaVector, LinealiaVector,
                                                   LinealExecutionParams,
                                                   LinealiaIterativeSolverParams,
                                                   LinealiaRelaxedParams);

    typedef LinealiaIterativeResult (*solve_ssor_t)(LinealiaMatrix, LinealiaVector, LinealiaVector,
                                                    LinealExecutionParams,
                                                    LinealiaIterativeSolverParams,
                                                    LinealiaRelaxedParams);

    typedef LinealiaIterativeResult (*solve_cg_t)(LinealiaMatrix, LinealiaVector, LinealiaVector,
                                                  LinealExecutionParams,
                                                  LinealiaIterativeSolverParams);

    typedef LinealiaIterativeResult (*solve_pcg_sor_t)(LinealiaMatrix, LinealiaVector, LinealiaVector,
                                                       LinealExecutionParams,
                                                       LinealiaIterativeSolverParams,
                                                       LinealiaRelaxedPreconditionerParams);

    typedef LinealiaIterativeResult (*solve_pcg_ssor_t)(LinealiaMatrix, LinealiaVector, LinealiaVector,
                                                        LinealExecutionParams,
                                                        LinealiaIterativeSolverParams,
                                                        LinealiaRelaxedPreconditionerParams);

    typedef LinealiaIterativeResult (*solve_pcg_amg_sor_t)(LinealiaMatrix, LinealiaVector, LinealiaVector,
                                                           LinealExecutionParams,
                                                           LinealiaIterativeSolverParams,
                                                           LinealiaPcgAmgParams);

    typedef LinealiaIterativeResult (*solve_pcg_amg_ssor_t)(LinealiaMatrix, LinealiaVector, LinealiaVector,
                                                            LinealExecutionParams,
                                                            LinealiaIterativeSolverParams,
                                                            LinealiaPcgAmgParams);

    // pointers
    solve_sor_t p_solve_sor = nullptr;
    solve_ssor_t p_solve_ssor = nullptr;
    solve_cg_t p_solve_cg = nullptr;
    solve_pcg_sor_t p_solve_pcg_sor = nullptr;
    solve_pcg_ssor_t p_solve_pcg_ssor = nullptr;
    solve_pcg_amg_sor_t p_solve_pcg_amg_sor = nullptr;
    solve_pcg_amg_ssor_t p_solve_pcg_amg_ssor = nullptr;
};



#endif // LINEALIALIB_H
