/****************************************************************************
**
** Copyright (C) 2016 The Qt Company Ltd.
** Contact: https://www.qt.io/licensing/
**
** This file is part of the examples of the Qt Toolkit.
**
** $QT_BEGIN_LICENSE:BSD$
** Commercial License Usage
** Licensees holding valid commercial Qt licenses may use this file in
** accordance with the commercial license agreement provided with the
** Software or, alternatively, in accordance with the terms contained in
** a written agreement between you and The Qt Company. For licensing terms
** and conditions see https://www.qt.io/terms-conditions. For further
** information use the contact form at https://www.qt.io/contact-us.
**
** BSD License Usage
** Alternatively, you may use this file under the terms of the BSD license
** as follows:
**
** "Redistribution and use in source and binary forms, with or without
** modification, are permitted provided that the following conditions are
** met:
**   * Redistributions of source code must retain the above copyright
**     notice, this list of conditions and the following disclaimer.
**   * Redistributions in binary form must reproduce the above copyright
**     notice, this list of conditions and the following disclaimer in
**     the documentation and/or other materials provided with the
**     distribution.
**   * Neither the name of The Qt Company Ltd nor the names of its
**     contributors may be used to endorse or promote products derived
**     from this software without specific prior written permission.
**
**
** THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
** "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
** LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR
** A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT
** OWNER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL,
** SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT
** LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE,
** DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY
** THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
** (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
** OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE."
**
** $QT_END_LICENSE$
**
****************************************************************************/

#include "glWidget.h"
#include "basicMath.h"
#include <QMouseEvent>
#include <QOpenGLShaderProgram>
#include <math.h>


Crit3DOpenGLWidget::Crit3DOpenGLWidget(Crit3DGeometry *geometry, QWidget *parent)
    : QOpenGLWidget(parent),
      m_xRotation(0),
      m_zRotation(0),
      m_xTraslation(0),
      m_yTraslation(0),
      m_zoom(1.f),
      m_program(nullptr),
      m_geometry (geometry)
{ }

Crit3DOpenGLWidget::~Crit3DOpenGLWidget()
{
    clear();
}


void Crit3DOpenGLWidget::clear()
{
    if (m_program == nullptr)
        return;
    makeCurrent();
    m_bufferObject.destroy();
    delete m_program;
    m_program = nullptr;
    m_geometry->clear();
    doneCurrent();
}


QSize Crit3DOpenGLWidget::minimumSizeHint() const
{
    return QSize(100, 100);
}

QSize Crit3DOpenGLWidget::sizeHint() const
{
    return QSize(1000, 600);
}

void Crit3DOpenGLWidget::setXRotation(int angle)
{
    angle = std::max(angle, 0);
    angle = std::min(angle, 360 * DEGREE_MULTIPLY);
    if (angle != m_xRotation)
    {
        m_xRotation = angle;
        emit xRotationChanged(angle);
        update();
    }
}

void Crit3DOpenGLWidget::setXTraslation(float traslation)
{
    if (! isEqual(traslation, m_xTraslation))
    {
        m_xTraslation = traslation;
        update();
    }
}

void Crit3DOpenGLWidget::setYTraslation(float traslation)
{
    if (! isEqual(traslation, m_yTraslation))
    {
        m_yTraslation = traslation;
        update();
    }
}

static void qNormalizeAngle(int &angle)
{
    while (angle < 0)
        angle += 360 * DEGREE_MULTIPLY;
    while (angle > 360 * DEGREE_MULTIPLY)
        angle -= 360 * DEGREE_MULTIPLY;
}


void Crit3DOpenGLWidget::setZRotation(int angle)
{
    qNormalizeAngle(angle);
    if (angle != m_zRotation)
    {
        m_zRotation = angle;
        emit zRotationChanged(angle);
        update();
    }
}


void Crit3DOpenGLWidget::setZoom(float zoom)
{
    zoom = std::max(0.5f, zoom);
    if (! isEqual(zoom, m_zoom))
    {
        m_zoom = zoom;
        update();
    }
}


static const char *vertexShaderSource =
    "attribute vec4 vertex;\n"
    "attribute vec3 normal;\n"
    "attribute vec4 colAttr;\n"
    "varying vec3 vert;\n"
    "varying vec3 vertNormal;\n"
    "varying vec3 color;\n"
    "uniform mat4 projMatrix;\n"
    "uniform mat4 mvMatrix;\n"
    "uniform mat3 normalMatrix;\n"
    "void main() {\n"
    "   vert = vertex.xyz;\n"
    "   vertNormal = normalMatrix * normal;\n"
    "   color = colAttr.xyz;\n"
    "   gl_Position = projMatrix * mvMatrix * vertex;\n"
    "}\n";

static const char *fragmentShaderSource =
    "varying vec3 vert;\n"
    "varying vec3 vertNormal;\n"
    "varying vec3 color;\n"
    "uniform vec3 lightPos;\n"
    "void main() {\n"
    "   vec3 L = normalize(lightPos - vert);\n"
    "   float NL = max(dot(normalize(vertNormal), L), 0.0);\n"
    "   vec3 col = clamp(color * 0.5 + color * 0.5 * NL, 0.0, 1.0);\n"
    "   gl_FragColor = vec4(col, 1.0);\n"
    "}\n";


void Crit3DOpenGLWidget::initializeGL()
{
    initializeOpenGLFunctions();
    glClearColor(1, 1, 1, 0);

    m_program = new QOpenGLShaderProgram;
    m_program->addShaderFromSourceCode(QOpenGLShader::Vertex, vertexShaderSource);
    m_program->addShaderFromSourceCode(QOpenGLShader::Fragment, fragmentShaderSource);
    m_program->bindAttributeLocation("vertex", 0);
    m_program->bindAttributeLocation("normal", 1);
    m_program->bindAttributeLocation("colAttr", 2);
    m_program->link();

    m_program->bind();
    m_projMatrixLoc = m_program->uniformLocation("projMatrix");
    m_mvMatrixLoc = m_program->uniformLocation("mvMatrix");
    m_normalMatrixLoc = m_program->uniformLocation("normalMatrix");
    m_lightPosLoc = m_program->uniformLocation("lightPos");

    // Setup our vertex buffer object
    m_bufferObject.create();
    m_bufferObject.bind();
    m_bufferObject.allocate(m_geometry->getData(), m_geometry->count() * sizeof(GLfloat));

    // Store the vertex attribute bindings for the program
    setupVertexAttribs();

    // set vertex colors
    glEnableVertexAttribArray(2);
    glVertexAttribPointer(2, 3, GL_FLOAT, GL_FALSE, 3 * sizeof(GLfloat), m_geometry->getColors());

    // Light position is fixed
    m_program->setUniformValue(m_lightPosLoc, QVector3D(0, m_geometry->defaultDistance()*10, m_geometry->defaultDistance() * 100));
    m_program->release();

    // set default zoom
    setZoom(m_geometry->defaultDistance());

    glEnable(GL_DEPTH_TEST);
    if (m_geometry->showHiddenFace())
        glEnable(GL_CULL_FACE);
}


void Crit3DOpenGLWidget::setupVertexAttribs()
{
    m_bufferObject.bind();
    QOpenGLFunctions *f = QOpenGLContext::currentContext()->functions();
    f->glEnableVertexAttribArray(0);
    f->glEnableVertexAttribArray(1);
    f->glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 6 * sizeof(GLfloat), nullptr);
    f->glVertexAttribPointer(1, 3, GL_FLOAT, GL_FALSE, 6 * sizeof(GLfloat), reinterpret_cast<void *>(3 * sizeof(GLfloat)));
    m_bufferObject.release();
}

void Crit3DOpenGLWidget::paintGL()
{
    //glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

    m_camera.setToIdentity();
    m_camera.translate(m_xTraslation, m_yTraslation, -m_zoom);
    m_camera.rotate(- float(m_xRotation / DEGREE_MULTIPLY), 1, 0, 0);
    m_camera.rotate(float(m_zRotation / DEGREE_MULTIPLY), 0, 0, 1);

    m_program->bind();
    m_program->setUniformValue(m_projMatrixLoc, m_proj);
    m_program->setUniformValue(m_mvMatrixLoc, m_camera * m_world);
    m_program->setUniformValue(m_normalMatrixLoc, m_camera.normalMatrix());

    glDrawArrays(GL_TRIANGLES, 0, m_geometry->vertexCount());

    m_program->release();
}

void Crit3DOpenGLWidget::resizeGL(int w, int h)
{
    m_proj.setToIdentity();
    m_proj.perspective(45.0f, GLfloat(w) / GLfloat(h), 0.1f, 1000000.0f);
}

void Crit3DOpenGLWidget::mousePressEvent(QMouseEvent *event)
{
    m_lastPos = event->pos();
}

void Crit3DOpenGLWidget::mouseMoveEvent(QMouseEvent *event)
{
    int dx = event->pos().x() - m_lastPos.x();
    int dy = event->pos().y() - m_lastPos.y();

    if (event->buttons() & Qt::LeftButton)
    {
        setXTraslation(m_xTraslation + m_zoom * float(dx) * 0.002f);
        setYTraslation(m_yTraslation - m_zoom * float(dy) * 0.002f);
    }
    else if (event->buttons() & Qt::RightButton)
    {
        setXRotation(m_xRotation - dy * DEGREE_MULTIPLY * 0.5f);
        setZRotation(m_zRotation + dx * DEGREE_MULTIPLY * 0.5f);
    }
    m_lastPos = event->pos();
}


void Crit3DOpenGLWidget::wheelEvent(QWheelEvent *event)
{
    int dy = event->angleDelta().y();
    setZoom(m_zoom * (1 - float(dy) * 0.001f));
}

