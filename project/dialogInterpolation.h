#ifndef INTERPOLATIONDIALOG_H
#define INTERPOLATIONDIALOG_H

    #ifndef PROJECT_H
        #include "project.h"
    #endif

    #include <QSettings>
    #include <QDialog>
    #include <QtWidgets>
    #include <QPushButton>
    #include <QVBoxLayout>

    class DialogInterpolation : public QDialog
    {
        Q_OBJECT

        public:
            explicit DialogInterpolation(Project *myProject);

            QComboBox algorithmEdit;
            QLineEdit minRegressionR2Edit;
            QLineEdit maxTdMultiplierEdit;
            QCheckBox* lapseRateCodeEdit;
            QCheckBox* thermalInversionEdit;
            QCheckBox* optimalDetrendingEdit;
            QCheckBox* multipleDetrendingEdit;
            QCheckBox* topographicDistanceEdit;
            QCheckBox* localDetrendingEdit;
            QCheckBox* upscaleFromDemEdit;
            QCheckBox* useDewPointEdit;
            QCheckBox* useInterpolTForRH;
            QComboBox gridAggregationMethodEdit;
            QVBoxLayout *layoutProxyList;
            QListWidget *proxyListCheck;

            void redrawProxies();
            void accept();

        private:
            Project* _project;
            QSettings* _paramSettings;
            Crit3DInterpolationSettings* _interpolationSettings;
            Crit3DInterpolationSettings* _qualityInterpolationSettings;

        private slots:
            void editProxies();
            void upscaleFromDemChanged(int active);
            void multipleDetrendingChanged(int active);
            void localDetrendingChanged(int active);
    };

    class ProxyDialog : public QDialog
    {
        Q_OBJECT

        public:
            explicit ProxyDialog(Project *myProject);

            int proxyIndex;

            QComboBox _proxyCombo;
            QComboBox _field;
            QComboBox _table;
            QLineEdit _proxyGridName;
            QCheckBox _forQuality;

            void changedProxy(bool savePrevious);
            void changedTable();
            void selectGridFile();
            void listProxies();
            void addProxy();
            void deleteProxy();
            void saveProxies();
            void saveProxy(Crit3DProxy *myProxy);
            bool checkProxies(QString *error);
            void accept();

        private:
            Project *_project;
            std::vector <Crit3DProxy> _proxy;
    };


#endif // INTERPOLATIONDIALOG_H
