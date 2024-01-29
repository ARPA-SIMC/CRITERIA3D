#include <QFileDialog>
#include <QCheckBox>
#include <QDialogButtonBox>
#include <QPushButton>
#include <QListWidget>
#include <QRadioButton>
#include <QMessageBox>
#include <QLineEdit>
#include <QLabel>
#include <QDateEdit>
#include <QDoubleValidator>
#include <QSettings>
#include <QGridLayout>
#include <QComboBox>
#include <QtWidgets>

#include "commonConstants.h"
#include "dialogSelection.h"
#include "color.h"
#include "utilities.h"


QString editValue(QString windowsTitle, QString defaultValue)
{
    QDialog myDialog;
    QVBoxLayout mainLayout;
    QHBoxLayout layoutValue;
    QHBoxLayout layoutOk;

    myDialog.setWindowTitle(windowsTitle);
    myDialog.setFixedWidth(300);

    QLabel *valueLabel = new QLabel("Value: ");
    QLineEdit valueEdit(defaultValue);
    valueEdit.setFixedWidth(150);

    layoutValue.addWidget(valueLabel);
    layoutValue.addWidget(&valueEdit);

    QDialogButtonBox buttonBox(QDialogButtonBox::Ok | QDialogButtonBox::Cancel);

    myDialog.connect(&buttonBox, SIGNAL(accepted()), &myDialog, SLOT(accept()));
    myDialog.connect(&buttonBox, SIGNAL(rejected()), &myDialog, SLOT(reject()));

    layoutOk.addWidget(&buttonBox);

    mainLayout.addLayout(&layoutValue);
    mainLayout.addLayout(&layoutOk);
    myDialog.setLayout(&mainLayout);
    myDialog.exec();

    if (myDialog.result() != QDialog::Accepted)
        return "";
    else
        return valueEdit.text();
}


meteoVariable chooseColorScale()
{
    QDialog myDialog;
    QVBoxLayout mainLayout;
    QVBoxLayout layoutVariable;
    QHBoxLayout layoutOk;

    myDialog.setWindowTitle("Choose color scale");
    myDialog.setFixedWidth(400);

    QRadioButton Dem("Elevation");
    QRadioButton Temp("Air temperature");
    QRadioButton Prec("Precipitation");
    QRadioButton RH("Air relative humidity");
    QRadioButton Rad("Solar radiation");
    QRadioButton Wind("Wind intensity");
    QRadioButton Anomaly("Anomaly");

    layoutVariable.addWidget(&Dem);
    layoutVariable.addWidget(&Temp);
    layoutVariable.addWidget(&Prec);
    layoutVariable.addWidget(&RH);
    layoutVariable.addWidget(&Rad);
    layoutVariable.addWidget(&Wind);
    layoutVariable.addWidget(&Anomaly);

    QDialogButtonBox buttonBox(QDialogButtonBox::Ok | QDialogButtonBox::Cancel);

    myDialog.connect(&buttonBox, SIGNAL(accepted()), &myDialog, SLOT(accept()));
    myDialog.connect(&buttonBox, SIGNAL(rejected()), &myDialog, SLOT(reject()));

    layoutOk.addWidget(&buttonBox);

    mainLayout.addLayout(&layoutVariable);
    mainLayout.addLayout(&layoutOk);
    myDialog.setLayout(&mainLayout);
    myDialog.exec();

    if (myDialog.result() != QDialog::Accepted)
        return noMeteoVar;

    if (Dem.isChecked())
        return noMeteoTerrain;
    else if (Temp.isChecked())
        return airTemperature;
    else if (Prec.isChecked())
        return precipitation;
    else if (RH.isChecked())
        return airRelHumidity;
    else if (Rad.isChecked())
        return globalIrradiance;
    else if (Wind.isChecked())
        return windScalarIntensity;
    else if (Anomaly.isChecked())
        return anomaly;
    else
        return noMeteoTerrain;
}


frequencyType chooseFrequency(const Project &myProject)
{
    QDialog myDialog;
    QVBoxLayout mainLayout;
    QVBoxLayout layoutFrequency;
    QHBoxLayout layoutOk;

    myDialog.setWindowTitle("Choose frequency");
    myDialog.setFixedWidth(300);

    QRadioButton Daily("Daily");
    QRadioButton Hourly("Hourly");
    QRadioButton Monthly("Monthly");

    layoutFrequency.addWidget(&Daily);
    layoutFrequency.addWidget(&Hourly);
    layoutFrequency.addWidget(&Monthly);

    frequencyType myFreq = myProject.getCurrentFrequency();

    if (myFreq == daily)
        Daily.setChecked(true);
    else if (myFreq == hourly)
        Hourly.setChecked(true);
    else if (myFreq == monthly)
        Monthly.setChecked(true);

    QDialogButtonBox buttonBox(QDialogButtonBox::Ok | QDialogButtonBox::Cancel);

    myDialog.connect(&buttonBox, SIGNAL(accepted()), &myDialog, SLOT(accept()));
    myDialog.connect(&buttonBox, SIGNAL(rejected()), &myDialog, SLOT(reject()));

    layoutOk.addWidget(&buttonBox);

    mainLayout.addLayout(&layoutFrequency);
    mainLayout.addLayout(&layoutOk);
    myDialog.setLayout(&mainLayout);
    myDialog.exec();

    if (myDialog.result() != QDialog::Accepted)
        return noFrequency;

   if (Daily.isChecked())
       return daily;
   else if (Hourly.isChecked())
       return hourly;
   else if (Monthly.isChecked())
       return monthly;
   else
       return noFrequency;

}


meteoVariable chooseMeteoVariable(const Project &myProject)
{
    if (myProject.getCurrentFrequency() == noFrequency)
    {
        QMessageBox::information(nullptr, "No frequency", "Choose frequency before");
        return noMeteoVar;
    }

    QDialog myDialog;
    QVBoxLayout mainLayout;
    QVBoxLayout layoutVariable;
    QHBoxLayout layoutOk;

    myDialog.setWindowTitle("Choose variable");
    myDialog.setFixedWidth(300);

    meteoVariable myCurrentVar = myProject.getCurrentVariable();

    QRadioButton Tavg("Average air temperature");
    QRadioButton Tmin("Minimum air temperature");
    QRadioButton Tmax("Maximum air temperature");
    QRadioButton Prec("Precipitation");
    QRadioButton RHavg("Average air relative humidity");
    QRadioButton RHmin("Minimum air relative humidity");
    QRadioButton RHmax("Maximum air relative humidity");
    QRadioButton Rad("Solar radiation");
    QRadioButton ET0HS("Reference evapotranspiration (Hargreaves-Samani)");
    QRadioButton ET0PM("Reference evapotranspiration (Penman-Monteith)");
    QRadioButton BIC("Hydroclimatic balance");
    QRadioButton WindVAvg("Average wind vector intensity");
    QRadioButton WindVMax("Maximum wind vector intensity");
    QRadioButton WindVDir("Prevailing wind vector direction");
    QRadioButton WindSAvg("Average wind scalar intensity");
    QRadioButton WindSMax("Maximum wind scalar intensity");
    QRadioButton LeafWD("Leaf wetness");

    QRadioButton T("Air temperature");
    QRadioButton P("Precipitation");
    QRadioButton RH("Air relative humidity");
    QRadioButton DewT("Air dew temperature (°C)");
    QRadioButton Irr("Solar irradiance");
    QRadioButton IrrNet("Net irradiance");
    QRadioButton ET0PMh("Reference evapotranspiration (Penman-Monteith)");
    QRadioButton WSInt("Wind scalar intensity");
    QRadioButton WVInt("Wind vector intensity");
    QRadioButton WVDir("Wind vector direction");
    QRadioButton WX("Wind vector component X");
    QRadioButton WY("Wind vector component Y");
    QRadioButton LW("Leaf wetness");

    QRadioButton MTavg("Average air temperature");
    QRadioButton MTmin("Minimum air temperature");
    QRadioButton MTmax("Maximum air temperature");
    QRadioButton MP("Precipitation");
    QRadioButton MET0HS("Reference evapotranspiration (H-S)");
    QRadioButton MRad("Solar radiation");
    QRadioButton MBIC("Hydroclimatic balance");

    if (myProject.getCurrentFrequency() == daily)
    {
        layoutVariable.addWidget(&Tmin);
        layoutVariable.addWidget(&Tavg);
        layoutVariable.addWidget(&Tmax);
        layoutVariable.addWidget(&Prec);
        layoutVariable.addWidget(&RHmin);
        layoutVariable.addWidget(&RHavg);
        layoutVariable.addWidget(&RHmax);
        layoutVariable.addWidget(&Rad);
        layoutVariable.addWidget(&ET0HS);
        layoutVariable.addWidget(&ET0PM);
        layoutVariable.addWidget(&BIC);
        layoutVariable.addWidget(&WindSAvg);
        layoutVariable.addWidget(&WindSMax);
        layoutVariable.addWidget(&WindVAvg);
        layoutVariable.addWidget(&WindVMax);
        layoutVariable.addWidget(&WindVDir);
        layoutVariable.addWidget(&LeafWD);

        if (myCurrentVar == dailyAirTemperatureMin)
            Tmin.setChecked(true);
        else if (myCurrentVar == dailyAirTemperatureMax)
            Tmax.setChecked(true);
        else if (myCurrentVar == dailyAirTemperatureAvg)
            Tavg.setChecked(true);
        else if (myCurrentVar == dailyPrecipitation)
            Prec.setChecked(true);
        else if (myCurrentVar == dailyAirRelHumidityMin)
            RHmin.setChecked(true);
        else if (myCurrentVar == dailyAirRelHumidityMax)
            RHmax.setChecked(true);
        else if (myCurrentVar == dailyAirRelHumidityAvg)
            RHavg.setChecked(true);
        else if (myCurrentVar == dailyGlobalRadiation)
            Rad.setChecked(true);
        else if (myCurrentVar == dailyReferenceEvapotranspirationHS)
            ET0HS.setChecked(true);
        else if (myCurrentVar == dailyReferenceEvapotranspirationPM)
            ET0PM.setChecked(true);
        else if (myCurrentVar == dailyBIC)
            BIC.setChecked(true);
        else if (myCurrentVar == dailyWindScalarIntensityAvg)
            WindSAvg.setChecked(true);
        else if (myCurrentVar == dailyWindScalarIntensityMax)
            WindSMax.setChecked(true);
        else if (myCurrentVar == dailyWindVectorIntensityAvg)
            WindVAvg.setChecked(true);
        else if (myCurrentVar == dailyWindVectorIntensityMax)
            WindVMax.setChecked(true);
        else if (myCurrentVar == dailyWindVectorDirectionPrevailing)
            WindVDir.setChecked(true);
        else if (myCurrentVar == dailyLeafWetness)
            LeafWD.setChecked(true);
    }
    else if (myProject.getCurrentFrequency() == hourly)
    {
        layoutVariable.addWidget(&T);
        layoutVariable.addWidget(&P);
        layoutVariable.addWidget(&RH);
        layoutVariable.addWidget(&DewT);
        layoutVariable.addWidget(&Irr);
        layoutVariable.addWidget(&IrrNet);
        layoutVariable.addWidget(&ET0PMh);
        layoutVariable.addWidget(&WSInt);
        layoutVariable.addWidget(&WVInt);
        layoutVariable.addWidget(&WVDir);
        layoutVariable.addWidget(&WX);
        layoutVariable.addWidget(&WY);
        layoutVariable.addWidget(&LW);

        if (myCurrentVar == airTemperature)
            T.setChecked(true);
        else if (myCurrentVar == precipitation)
            P.setChecked(true);
        else if (myCurrentVar == airRelHumidity)
            RH.setChecked(true);
        else if (myCurrentVar == airDewTemperature)
            DewT.setChecked(true);
        else if (myCurrentVar == globalIrradiance)
            Irr.setChecked(true);
        else if (myCurrentVar == netIrradiance)
            IrrNet.setChecked(true);
        else if (myCurrentVar == referenceEvapotranspiration)
            ET0PMh.setChecked(true);
        else if (myCurrentVar == windScalarIntensity)
            WSInt.setChecked(true);
        else if (myCurrentVar == windVectorIntensity)
            WVInt.setChecked(true);
        else if (myCurrentVar == windVectorDirection)
            WVDir.setChecked(true);
        else if (myCurrentVar == windVectorX)
            WX.setChecked(true);
        else if (myCurrentVar == windVectorY)
            WY.setChecked(true);
        else if (myCurrentVar == leafWetness)
            LW.setChecked(true);
    }
    else if (myProject.getCurrentFrequency() == monthly)
    {
        layoutVariable.addWidget(&MTavg);
        layoutVariable.addWidget(&MTmin);
        layoutVariable.addWidget(&MTmax);
        layoutVariable.addWidget(&MP);
        layoutVariable.addWidget(&MET0HS);
        layoutVariable.addWidget(&MRad);
        layoutVariable.addWidget(&MBIC);

        if (myCurrentVar == monthlyAirTemperatureMin)
            MTmin.setChecked(true);
        else if (myCurrentVar == monthlyAirTemperatureMax)
            MTmax.setChecked(true);
        else if (myCurrentVar == monthlyAirTemperatureAvg)
            MTavg.setChecked(true);
        else if (myCurrentVar == monthlyPrecipitation)
            MP.setChecked(true);
        else if (myCurrentVar == monthlyReferenceEvapotranspirationHS)
            MET0HS.setChecked(true);
        else if (myCurrentVar == monthlyGlobalRadiation)
            MRad.setChecked(true);
        else if (myCurrentVar == monthlyPrecipitation)
            MBIC.setChecked(true);
    }
    else return noMeteoVar;

    QDialogButtonBox buttonBox(QDialogButtonBox::Ok | QDialogButtonBox::Cancel);

    myDialog.connect(&buttonBox, SIGNAL(accepted()), &myDialog, SLOT(accept()));
    myDialog.connect(&buttonBox, SIGNAL(rejected()), &myDialog, SLOT(reject()));

    layoutOk.addWidget(&buttonBox);

    mainLayout.addLayout(&layoutVariable);
    mainLayout.addLayout(&layoutOk);
    myDialog.setLayout(&mainLayout);
    myDialog.exec();

    if (myDialog.result() != QDialog::Accepted)
    {
        return noMeteoVar;
    }

   if (myProject.getCurrentFrequency() == daily)
   {
       if (Tmin.isChecked())
           return (dailyAirTemperatureMin);
       else if (Tmax.isChecked())
           return (dailyAirTemperatureMax);
       else if (Tavg.isChecked())
           return (dailyAirTemperatureAvg);
       else if (Prec.isChecked())
           return (dailyPrecipitation);
       else if (Rad.isChecked())
           return (dailyGlobalRadiation);
       else if (RHmin.isChecked())
           return (dailyAirRelHumidityMin);
       else if (RHmax.isChecked())
           return (dailyAirRelHumidityMax);
       else if (RHavg.isChecked())
           return (dailyAirRelHumidityAvg);
       else if (ET0HS.isChecked())
           return (dailyReferenceEvapotranspirationHS);
       else if (ET0PM.isChecked())
           return (dailyReferenceEvapotranspirationPM);
       else if (BIC.isChecked())
           return (dailyBIC);
       else if (WindSAvg.isChecked())
           return (dailyWindScalarIntensityAvg);
       else if (WindSMax.isChecked())
           return (dailyWindScalarIntensityMax);
       else if (WindVAvg.isChecked())
           return (dailyWindVectorIntensityAvg);
       else if (WindVMax.isChecked())
           return (dailyWindVectorIntensityMax);
       else if (WindVDir.isChecked())
           return (dailyWindVectorDirectionPrevailing);
       else if (LeafWD.isChecked())
           return (dailyLeafWetness);
   }

   if (myProject.getCurrentFrequency() == hourly)
   {
       if (T.isChecked())
           return (airTemperature);
       else if (RH.isChecked())
           return (airRelHumidity);
       else if (P.isChecked())
           return (precipitation);
       else if (Irr.isChecked())
           return (globalIrradiance);
       else if (IrrNet.isChecked())
           return (netIrradiance);
       else if (WSInt.isChecked())
           return (windScalarIntensity);
       else if (WVInt.isChecked())
           return windVectorIntensity;
       else if (WVDir.isChecked())
           return windVectorDirection;
       else if (WX.isChecked())
           return windVectorX;
       else if (WY.isChecked())
           return windVectorY;
       else if (DewT.isChecked())
           return (airDewTemperature);
       else if (LW.isChecked())
           return (leafWetness);
       else if (ET0PMh.isChecked())
           return (referenceEvapotranspiration);
   }

   if (myProject.getCurrentFrequency() == monthly)
   {
       if (MTmin.isChecked())
           return (monthlyAirTemperatureMin);
       else if (MTmax.isChecked())
           return (monthlyAirTemperatureMax);
       else if (MTavg.isChecked())
           return (monthlyAirTemperatureAvg);
       else if (MP.isChecked())
           return (monthlyPrecipitation);
       else if (MRad.isChecked())
           return (monthlyGlobalRadiation);
       else if (MET0HS.isChecked())
           return (monthlyReferenceEvapotranspirationHS);
       else if (MBIC.isChecked())
           return (monthlyBIC);
   }

   return noMeteoVar;
}
