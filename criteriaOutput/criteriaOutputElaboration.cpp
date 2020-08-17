#include "criteriaOutputElaboration.h"
#include "criteriaOutputProject.h"
#include "commonConstants.h"
#include "utilities.h"
#include <QtSql>

int selectSimpleVar(QSqlDatabase db, QString idCase, QString varName, QString computation, QDate firstDate, QDate lastDate, float irriRatio, QVector<float>* resVector, QString* projectError)
{

    QSqlQuery qry(db);
    int count = 0;
    QString statement;
    float result = NODATA;
    statement = QString("SELECT %1(`%2`) FROM `%3` WHERE DATE >= '%4' AND DATE <= '%5'").arg(computation).arg(varName).arg(idCase).arg(firstDate.toString("yyyy-MM-dd")).arg(lastDate.toString("yyyy-MM-dd"));
    if( !qry.exec(statement) )
    {
        *projectError = "Wrong computation: " + computation + "\n" + qry.lastError().text();
        return ERROR_OUTPUT_VARIABLES ;
    }
    qry.first();
    if (!qry.isValid())
    {
        *projectError = "Missing data: " + statement;
        return ERROR_MISSING_DATA ;
    }
    do
    {
        getValue(qry.value(0), &result);
        count = count+1;
        if (varName == "IRRIGATION")
        {
            result = result * irriRatio;
        }
        resVector->push_back(result);

    }
    while(qry.next());


    if (count < firstDate.daysTo(lastDate)+1)
    {
        *projectError = "Incomplete data: " + statement;
        return ERROR_INCOMPLETE_DATA;
    }

    return CRIT3D_OK;

}

int computeDTX(QSqlDatabase db, QString idCase, int period, QString computation, QDate firstDate, QDate lastDate, QVector<float>* resVector, QString *projectError)
{

    QSqlQuery qry(db);
    QString statement;
    float res = NODATA;
    QVector<float> dtx;
    int count = 0;
    int count2 = 0;
    float var1, var2;
    QDate end = firstDate;
    QDate start;
    while (end <= lastDate)
    {
        start = end.addDays(-period+1);
        statement = QString("SELECT COUNT(TRANSP_MAX),COUNT(TRANSP) FROM `%1` WHERE DATE >= '%2' AND DATE <= '%3'").arg(idCase).arg(start.toString("yyyy-MM-dd")).arg(end.toString("yyyy-MM-dd"));
        if( !qry.exec(statement) )
        {
            *projectError = qry.lastError().text();
            return ERROR_OUTPUT_VARIABLES;
        }
        qry.first();
        if (!qry.isValid())
        {
            *projectError = qry.lastError().text();
            return ERROR_OUTPUT_VARIABLES ;
        }
        getValue(qry.value(0), &count);
        getValue(qry.value(1), &count2);
        if (count+count2 < period*2)
        {
            dtx.push_back(NODATA);
            return ERROR_INCOMPLETE_DATA;
        }
        statement = QString("SELECT SUM(TRANSP_MAX),SUM(TRANSP) FROM `%1` WHERE DATE >= '%2' AND DATE <= '%3'").arg(idCase).arg(start.toString("yyyy-MM-dd")).arg(end.toString("yyyy-MM-dd"));
        if( !qry.exec(statement) )
        {
            *projectError = qry.lastError().text();
            return ERROR_OUTPUT_VARIABLES ;
        }
        qry.first();
        if (!qry.isValid())
        {
            *projectError = qry.lastError().text();
            return ERROR_OUTPUT_VARIABLES ;
        }
        getValue(qry.value("SUM(TRANSP_MAX)"), &var1);
        getValue(qry.value("SUM(TRANSP)"), &var2);
        dtx.push_back((var1 - var2));
        end = end.addDays(1);
    }
    if (computation.isEmpty())
    {
        resVector->append(dtx);
        return CRIT3D_OK;
    }
    else if (computation == "SUM")
    {
        res = 0;
        for(int i=0; i<dtx.size();i++)
        {
            res = res + dtx[i];
        }
    }
    else if (computation == "AVG")
    {
        res = 0;
        for(int i=0; i<dtx.size();i++)
        {
            res = res + dtx[i];
        }
        res = res/dtx.size();
    }
    else if (computation == "MAX")
    {
        res = *std::max_element(dtx.begin(), dtx.end());
    }
    else if (computation == "MIN")
    {
        res = *std::min_element(dtx.begin(), dtx.end());
    }

    resVector->push_back(res);
    return CRIT3D_OK;
}
