#ifndef DIALOGPROJECT_H
#define DIALOGPROJECT_H

    #include <QDialog>
    #include <QLineEdit>

    class Project;

    class DialogProject : public QDialog
    {
        Q_OBJECT

        public:
            explicit DialogProject(Project* myProject);

            QLineEdit* lineEditProjectName;
            QLineEdit* lineEditProjectDescription;
            QLineEdit* lineEditProjectPath;

            void getPath();
            void accept();

        private:
            Project* project_;
    };

#endif // DIALOGPROJECT_H
