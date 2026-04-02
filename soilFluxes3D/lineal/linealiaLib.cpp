#include "linealiaLib.h"
#include <QDebug>
#include <iostream>

LinealiaLib& LinealiaLib::instance() {
    static LinealiaLib instance;
    return instance;
}

LinealiaLib::LinealiaLib()
    : lib("liblinealia") {}

bool LinealiaLib::load() {
    if (lib.isLoaded())
        return true;

    if (!lib.load())
    {
        //std::cout << "Error in loading Lineal dll: " << lib.errorString().toStdString() << std::endl;
        return false;
    }

    p_solve_sor = (solve_sor_t) lib.resolve("linealia_solve_sor");
    p_solve_ssor = (solve_ssor_t) lib.resolve("linealia_solve_ssor");
    p_solve_cg = (solve_cg_t) lib.resolve("linealia_solve_cg");
    p_solve_pcg_sor = (solve_pcg_sor_t) lib.resolve("linealia_solve_pcg_sor");
    p_solve_pcg_ssor = (solve_pcg_ssor_t) lib.resolve("linealia_solve_pcg_ssor");
    p_solve_pcg_amg_sor = (solve_pcg_amg_sor_t) lib.resolve("linealia_solve_pcg_amg_sor");
    p_solve_pcg_amg_ssor = (solve_pcg_amg_ssor_t) lib.resolve("linealia_solve_pcg_amg_ssor");

    // global check
    if (!p_solve_sor || !p_solve_ssor || !p_solve_cg ||
        !p_solve_pcg_sor || !p_solve_pcg_ssor ||
        !p_solve_pcg_amg_sor || !p_solve_pcg_amg_ssor) {

        qDebug() << "Error in resolve functions.";
        return false;
    }

    return true;
}

bool LinealiaLib::isLoaded() const {
    return lib.isLoaded();
}

LinealiaIterativeResult LinealiaLib::solveSOR(
    LinealiaMatrix a, LinealiaVector x, LinealiaVector b,
    LinealExecutionParams e, LinealiaIterativeSolverParams i,
    LinealiaRelaxedParams s)
    {
        return p_solve_sor(a, x, b, e, i, s);
    }

LinealiaIterativeResult LinealiaLib::solveSSOR(
    LinealiaMatrix a, LinealiaVector x, LinealiaVector b,
    LinealExecutionParams e, LinealiaIterativeSolverParams i,
    LinealiaRelaxedParams s)
    {
        return p_solve_ssor(a, x, b, e, i, s);
    }

LinealiaIterativeResult LinealiaLib::solveCG(
    LinealiaMatrix a, LinealiaVector x, LinealiaVector b,
    LinealExecutionParams e, LinealiaIterativeSolverParams i)
    {
        return p_solve_cg(a, x, b, e, i);
    }

LinealiaIterativeResult LinealiaLib::solvePCG_SOR(
    LinealiaMatrix a, LinealiaVector x, LinealiaVector b,
    LinealExecutionParams e, LinealiaIterativeSolverParams i,
    LinealiaRelaxedPreconditionerParams p)
    {
        return p_solve_pcg_sor(a, x, b, e, i, p);
    }

LinealiaIterativeResult LinealiaLib::solvePCG_SSOR(
    LinealiaMatrix a, LinealiaVector x, LinealiaVector b,
    LinealExecutionParams e, LinealiaIterativeSolverParams i,
    LinealiaRelaxedPreconditionerParams p)
    {
        return p_solve_pcg_ssor(a, x, b, e, i, p);
    }

LinealiaIterativeResult LinealiaLib::solvePCG_AMG_SOR(
    LinealiaMatrix a, LinealiaVector x, LinealiaVector b,
    LinealExecutionParams e, LinealiaIterativeSolverParams i,
    LinealiaPcgAmgParams p)
    {
        return p_solve_pcg_amg_sor(a, x, b, e, i, p);
    }

LinealiaIterativeResult LinealiaLib::solvePCG_AMG_SSOR(
    LinealiaMatrix a, LinealiaVector x, LinealiaVector b,
    LinealExecutionParams e, LinealiaIterativeSolverParams i,
    LinealiaPcgAmgParams p)
    {
        return p_solve_pcg_amg_ssor(a, x, b, e, i, p);
    }
