#ifndef VARDATASET_H
#define VARDATASET_H

#include <QVector>

class VarDataset
{
private:
    QVector<double> hourlyValueList;
    QString myVar;
public:
    VarDataset();
    VarDataset(QString var);
    QString getVar() const;
    int getHourlyValueIndex(double value);
    double getHourlyValue(int index);
    void setHourlyValue(int index, double value);
};

#endif // VARDATASET_H
