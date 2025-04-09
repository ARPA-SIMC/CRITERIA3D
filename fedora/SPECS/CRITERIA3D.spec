%global releaseno 1

# Note: define srcarchivename in CI build only.
%{!?srcarchivename: %global srcarchivename %{name}-%{version}-%{releaseno}}

Name:           CRITRIA3D
Version:        1.0.5
Release:        %{releaseno}%{?dist}
Summary:        three-dimensional agro-hydrological model

URL:            https://github.com/ARPA-SIMC/CRITERIA3D
Source0:        https://github.com/ARPA-SIMC/CRITERIA3D/archive/v%{version}-%{releaseno}.tar.gz#/%{srcarchivename}.tar.gz
License:        GPL

BuildRequires:  qt5-qtbase
BuildRequires:  qt5-qtbase-devel
BuildRequires:  qt5-qtcharts
BuildRequires:  qt5-qtcharts-devel

Requires:       qt5-qtbase-mysql

%description
CRITERIA-3D is a three-dimensional agro-hydrological model for small catchments.
It includes a numerical solution for three-dimensional water and heat flow in the soil, 
coupled surface and subsurface flow, meteorological data interpolation, radiation budget, 
crop development and crop water uptake. It needs hourly meteo data as input 
(air temperature, precipitation, solar irradiance, air relative humidity, wind speed).


%prep
%autosetup -n %{srcarchivename}

%build
pushd mapGraphics
qmake-qt5 MapGraphics.pro -spec linux-g++-64 CONFIG+=release CONFIG+=force_debug_info CONFIG+=c++11 CONFIG+=qtquickcompiler
make
popd

pushd bin/Makeall_CRITERIA3D
qmake-qt5 Makeall_CRITERIA3D.pro -spec linux-g++-64 CONFIG+=release CONFIG+=force_debug_info CONFIG+=c++11 CONFIG+=qtquickcompiler
make qmake_all
make
popd

%install
rm -rf $RPM_BUILD_ROOT
mkdir -p %{buildroot}/%{_bindir}/
cp -a bin/CRITERIA3D %{buildroot}/%{_bindir}/

%files
%{_bindir}/CRITERIA3D


%changelog
* Tue Apr 08 2025 Fausto Tomei <ftomei@arpae.it> - 1.0.5
- Release 1.0.5


