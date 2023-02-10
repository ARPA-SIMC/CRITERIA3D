#ifndef DIALOGNEWPROJECT_H
#define DIALOGNEWPROJECT_H

    #include <QObject>
    #include <QtWidgets>

    #define NEW_DB 0
    #define DEFAULT_DB 1
    #define CHOOSE_DB 2

    class DialogNewProject : public QDialog
    {
        Q_OBJECT
    public:
        DialogNewProject();
        QGroupBox *createSoilGroup();
        QGroupBox *createMeteoGroup();
        QGroupBox *createCropGroup();
        void chooseSoilDb();
        void chooseMeteoDb();
        void chooseCropDb();
        void hideSoilName();
        void hideMeteoName();
        void hideCropName();

        QString getDbSoilCompletePath() const;
        QString getDbMeteoCompletePath() const;
        QString getDbCropCompletePath() const;

        int getSoilDbOption();
        int getMeteoDbOption();
        int getCropDbOption();

        void done(int res);

        QString getProjectName() const;

    private:
        QLineEdit* projectName;
        QGroupBox *soilGroup();
        QGroupBox *meteoGroup();
        QGroupBox *cropGroup();

        QRadioButton *newSoil;
        QRadioButton *defaultSoil;
        QRadioButton *chooseSoil;

        QRadioButton *newMeteo;
        QRadioButton *defaultMeteo;
        QRadioButton *chooseMeteo;

        QRadioButton *defaultCrop;
        QRadioButton *chooseCrop;

        QLineEdit* dbSoilName;
        QLineEdit* dbMeteoName;
        QLineEdit* dbCropName;

        QString dbSoilCompletePath;
        QString dbMeteoCompletePath;
        QString dbCropCompletePath;
    };

#endif // DIALOGNEWPROJECT_H
