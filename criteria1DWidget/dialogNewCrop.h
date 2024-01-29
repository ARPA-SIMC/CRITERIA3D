#ifndef DIALOGNEWCROP_H
#define DIALOGNEWCROP_H

    #include <QtWidgets>
    class Crit3DCrop;
    class QSqlDatabase;

    class DialogNewCrop : public QDialog
    {
        Q_OBJECT
    public:
        DialogNewCrop(QSqlDatabase* dbCrop, Crit3DCrop* newCrop);

        void on_actionChooseType(QString cropType);
        void done(int res);
        bool checkData();
        QString getNameCrop();

    private:
        QSqlDatabase* dbCrop;
        Crit3DCrop* newCrop;
        QLineEdit* idCropValue;
        QLineEdit* nameCropValue;
        QLineEdit* typeCropValue;
        QLineEdit* templateCropValue;
        QLabel *sowingDoY;
        QSpinBox* sowingDoYValue;
        QLabel *cycleMaxDuration;
        QSpinBox* cycleMaxDurationValue;
        QComboBox* templateCropComboBox;
    };

#endif // DIALOGNEWCROP_H
