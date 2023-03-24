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
#include "commonConstants.h"
#include "basicMath.h"
#include <math.h>
#include <QMouseEvent>
#include <QOpenGLShaderProgram>
#include <QOpenGLFunctions_4_0_Core>


Crit3DOpenGLWidget::Crit3DOpenGLWidget(Crit3DGeometry *geometry, QWidget *parent)
    : QOpenGLWidget(parent),
      m_xRotation(0),
      m_zRotation(0),
      m_xTraslation(0),
      m_yTraslation(0),
      m_zoom(1.f),
      m_program(nullptr),
      m_geometry(geometry)
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
    glDisableVertexAttribArray(0);
    glDisableVertexAttribArray(1);
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


void Crit3DOpenGLWidget::setMagnify(float magnify)
{
    if (! isEqual(magnify * 0.1f, m_geometry->magnify()))
    {
        m_geometry->setMagnify(magnify * 0.1f);

        m_bufferObject.bind();
        m_bufferObject.allocate(m_geometry->getVertices(), m_geometry->dataCount() * long(sizeof(GLfloat)));
        m_bufferObject.release();

        update();
    }
}


static const char *vertexShaderSource =
    "#version 330 core\n"
    "attribute lowp vec4 vertex;\n"
    "attribute lowp vec4 color;\n"
    "varying lowp vec4 myCol;\n"
    "uniform lowp mat4 projMatrix;\n"
    "uniform lowp mat4 mvMatrix;\n"
    "void main() {\n"
    "   myCol = color;\n"
    "   gl_Position = projMatrix * mvMatrix * vertex;\n"
    "}\n";

static const char *fragmentShaderSource =
    "#version 330 core\n"
    "varying lowp vec4 myCol;\n"
    "void main() {\n"
    "   gl_FragColor = myCol;\n"
    "}\n";


void Crit3DOpenGLWidget::initializeGL()
{
    initializeOpenGLFunctions();
    // blue sky
    glClearColor(0.52f, 0.81f, 0.92f, 0.f);

    m_program = new QOpenGLShaderProgram;
    m_program->addShaderFromSourceCode(QOpenGLShader::Vertex, vertexShaderSource);
    m_program->addShaderFromSourceCode(QOpenGLShader::Fragment, fragmentShaderSource);
    m_program->bindAttributeLocation("vertex", 0);
    m_program->bindAttributeLocation("color", 1);
    m_program->link();

    m_program->bind();
    m_projMatrixLoc = m_program->uniformLocation("projMatrix");
    m_mvMatrixLoc = m_program->uniformLocation("mvMatrix");
    m_program->release();

    // setup vertex buffer object
    m_bufferObject.create();
    m_bufferObject.bind();
    m_bufferObject.allocate(m_geometry->getVertices(), m_geometry->dataCount() * long(sizeof(GLfloat)));
    glEnableVertexAttribArray(0);
    glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 3 * sizeof(GLfloat), nullptr);
    m_bufferObject.release();

    // set vertex colors
    glEnableVertexAttribArray(1);
    glVertexAttribPointer(1, 3, GL_UNSIGNED_BYTE, GL_TRUE, 3 * sizeof(GLubyte), m_geometry->getColors());

    // set default zoom
    setZoom(m_geometry->defaultDistance());

    glEnable(GL_DEPTH_TEST);
    // hidden face
    //glEnable(GL_CULL_FACE);
    // wireframe
    //glPolygonMode(GL_FRONT_AND_BACK, GL_LINE);
}


void Crit3DOpenGLWidget::paintGL()
{
    //glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

    m_camera.setToIdentity();
    m_camera.translate(0, 0, -m_zoom);

    m_world.setToIdentity();

    m_world.rotate(float(-m_xRotation / DEGREE_MULTIPLY), 1, 0, 0);
    m_world.rotate(float(m_zRotation / DEGREE_MULTIPLY), 0, 0, 1);
    m_world.translate(m_xTraslation, m_yTraslation, 0);

    m_program->bind();
    m_program->setUniformValue(m_projMatrixLoc, m_proj);
    m_program->setUniformValue(m_mvMatrixLoc, m_camera * m_world);
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
        float cosAngle = cos(m_zRotation / DEGREE_MULTIPLY * DEG_TO_RAD);
        float sinAngle = sin(m_zRotation / DEGREE_MULTIPLY * DEG_TO_RAD);
        setXTraslation(m_xTraslation + m_zoom * (dx * cosAngle - dy * sinAngle) * 0.002f);
        setYTraslation(m_yTraslation - m_zoom * (dy * cosAngle + dx * sinAngle) * 0.002f);
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

