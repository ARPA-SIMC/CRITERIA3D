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
            QComboBox elevationFunctionEdit;
            QLineEdit minRegressionR2Edit;
            QLineEdit maxTdMultiplierEdit;
            QLineEdit minPointsLocalDetrendingEdit;
            QCheckBox* lapseRateCodeEdit;
            QCheckBox* thermalInversionEdit;
            QCheckBox* optimalDetrendingEdit;
            QCheckBox* multipleDetrendingEdit;
            QCheckBox* topographicDistanceEdit;
            QCheckBox* localDetrendingEdit;
			QCheckBox* glocalDetrendingEdit;
            QCheckBox* doNotRetrendEdit;
            QCheckBox* retrendOnlyEdit;
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
            void glocalDetrendingChanged(int active);
            void optimalDetrendingChanged(int active);
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
            QTextEdit _param0;
            QTextEdit _param1;
            QTextEdit _param2;
            QTextEdit _param3;
            QTextEdit _param4;
            QTextEdit _param5;


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
