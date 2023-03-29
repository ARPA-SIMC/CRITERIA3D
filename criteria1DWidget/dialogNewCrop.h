#ifndef DIALOGNEWCROP_H
#define DIALOGNEWCROP_H

    #include <QtWidgets>
    class Crit3DCrop;

    class DialogNewCrop : public QDialog
    {
        Q_OBJECT
    public:
        DialogNewCrop(Crit3DCrop* newCrop);
        void on_actionChooseType(QString type);
        void done(int res);
        bool checkData();
        QString getNameCrop();

    private:
        Crit3DCrop* newCrop;
        QLineEdit* idCropValue;
        QLineEdit* nameCropValue;
        QLineEdit* typeCropValue;
        QLabel *sowingDoY;
        QSpinBox* sowingDoYValue;
        QLabel *cycleMaxDuration;
        QSpinBox* cycleMaxDurationValue;
    };

#endif // DIALOGNEWCROP_H
