CONFIG(OMP_CONFIG) {
    win32:{
        QMAKE_CXXFLAGS += -openmp:llvm
    }
    unix:{
        QMAKE_CXXFLAGS += -fopenmp
        QMAKE_LFLAGS += -fopenmp
    }
    macx:{
        QMAKE_CXXFLAGS += -fopenmp
        QMAKE_LFLAGS += -fopenmp
    }
}

CONFIG(LTO_CONFIG) {
    win32:{
        QMAKE_CXXFLAGS += -GL
        QMAKE_LFLAGS += -LTCG
    }
    unix:{
        QMAKE_CXXFLAGS += #-flto
        QMAKE_LFLAGS += #-flto
    }
    macx:{
        QMAKE_CXXFLAGS += #-flto
        QMAKE_LFLAGS += #-flto
    }
}


CONFIG(CUDA_CONFIG) {
    # CUDA settings
    CUDA_DIR = $$(CUDA_PATH)

    isEmpty(CUDA_DIR) {
        message("CUDA_PATH non definita, salto configurazione CUDA")
    } else {
        DEFINES += CUDA_ENABLED
        INCLUDEPATH  += $$CUDA_DIR/include
        QMAKE_LIBDIR += $$CUDA_DIR/lib/x64
        LIBS += -lcudart -lcuda -lcusparse

        QMAKE_CXXFLAGS -= -GL -flto
        QMAKE_LFLAGS -= -LTCG -flto
        HOST_C_FLAGS = $$join(QMAKE_CXXFLAGS,',','"','"')
        HOST_L_FLAGS = $$join(QMAKE_LFLAGS,',','"','"')

        HOST_DEFINES = $$join(DEFINES,' -D','-D')

        cudaC_FLAGS = $$HOST_DEFINES -m64 -std=c++17
        cudaL_FLAGS = -Wno-deprecated-gpu-targets -arch=$$CUDA_ARCH

        CONFIG(debug, debug|release) {
            MSVCRT_LINK_FLAG = "/MDd"
            SUBPATH = debug
            cudaC_FLAGS += -g -G
        } else {
            MSVCRT_LINK_FLAG = "/MD"
            SUBPATH = release
        }

        CUDA_INC = $$join(INCLUDEPATH,'" -I"','-I"','"')
    }
}


CONFIG(MCR_CONFIG) {
    DEFINES += MCR_ENABLED

    LIBS += -L$$MCR_PATH/lib/win64/microsoft libmx.lib libmat.lib
    LIBS += -L$$MCR_PATH/bin/win64

    INCLUDEPATH += $$MCR_PATH/include
}
