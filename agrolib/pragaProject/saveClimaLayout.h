#ifndef SAVECLIMALAYOUT_H
#define SAVECLIMALAYOUT_H

#include <QString>
#include <QGridLayout>
#include <QtWidgets>

bool compareClimateElab(const QString &el1, const QString &el2);

class SaveClimaLayout : public QDialog
{

    Q_OBJECT

    public:
        SaveClimaLayout();

        QString getFirstYear() const;
        void setFirstYear(const QString &valueStr);
        QString getLastYear() const;
        void setLastYear(const QString &valueStr);
        QString getVariable() const;
        void setVariable(const QString &valueStr);
        QString getPeriod() const;
        void setPeriod(const QString &valueStr);
        QString getGenericPeriodEnd() const;
        void setGenericPeriodEnd(const QString &valueStr);
        QString getGenericNYear() const;
        void setGenericNYear(const QString &valueStr);
        QString getSecondElab() const;
        void setSecondElab(const QString &valueStr);
        QString getElab2Param() const;
        void setElab2Param(const QString &valueStr);
        QString getElab() const;
        void setElab(const QString &valueStr);
        QString getElab1Param() const;
        void setElab1Param(const QString &valueStr);
        QString getGenericPeriodStartDay() const;
        void setGenericPeriodStartDay(const QString &value);
        QString getGenericPeriodStartMonth() const;
        void setGenericPeriodStartMonth(const QString &valueStr);
        QString getGenericPeriodEndDay() const;
        void setGenericPeriodEndDay(const QString &valueStr);
        QString getGenericPeriodEndMonth() const;
        void setGenericPeriodEndMonth(const QString &valueStr);

        void addElab();
        void deleteRaw();
        void deleteAll();

        void saveElabList();
        void loadElabList();

        QList<QString> getList() const;
        void setList(const QList<QString> &valueList);

        QString getElab1ParamFromdB() const;
        void setElab1ParamFromdB(const QString &valueStr);

private:

        QVBoxLayout mainLayout;
        QVBoxLayout listLayout;
        QHBoxLayout saveButtonLayout;

        QListWidget listView;
        QList<QString> list;

        QPushButton saveList;
        QPushButton loadList;

        QString firstYear;
        QString lastYear;
        QString variable;
        QString period;

        QString genericPeriodStartDay;
        QString genericPeriodStartMonth;
        QString genericPeriodEndDay;
        QString genericPeriodEndMonth;
        QString genericNYear;

        QString secondElab;
        QString elab2Param;

        QString elab;
        QString elab1Param;
        QString elab1ParamFromdB;


};


#endif // SAVECLIMALAYOUT_H
