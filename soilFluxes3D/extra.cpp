/*!
    \name extra.cpp
    \copyright (C) 2011 Fausto Tomei, Gabriele Antolini, Antonio Volta,
                        Alberto Pistocchi, Marco Bittelli

    This file is part of CRITERIA3D.
    CRITERIA3D has been developed under contract issued by A.R.P.A. Emilia-Romagna

    CRITERIA3D is free software: you can redistribute it and/or modify
    it under the terms of the GNU Lesser General Public License as published by
    the Free Software Foundation, either version 3 of the License, or
    (at your option) any later version.

    CRITERIA3D is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
    GNU Lesser General Public License for more details.

    You should have received a copy of the GNU Lesser General Public License
    along with CRITERIA3D.  If not, see <http://www.gnu.org/licenses/>.

    contacts:
    ftomei@arpae.it
    gantolini@arpae.it
*/


#include <math.h>
#include <stdio.h>

#include "commonConstants.h"
#include "header/extra.h"
#include "header/types.h"


void initializeExtraHeat(TCrit3DNodeHeat* myNodeExtraHeat)
{
    (*myNodeExtraHeat).T = ZEROCELSIUS + 20;
    (*myNodeExtraHeat).oldT = ZEROCELSIUS + 20;
    (*myNodeExtraHeat).Qh = 0;
    (*myNodeExtraHeat).sinkSource = 0.;
}

void initializeExtra(TCrit3DnodeExtra *myNodeExtra, bool computeHeat, bool computeSolutes)
{
    if (computeHeat)
    {
        (*myNodeExtra).Heat = new(TCrit3DNodeHeat);
        initializeExtraHeat(myNodeExtra->Heat);
    }
    else (*myNodeExtra).Heat = nullptr;

    if (computeSolutes)
    {
        // TODO
    }
}

void initializeNodeHeatFlux(TCrit3DLinkedNodeExtra* myLinkExtra, bool initHeat, bool initWater)
{
    if (myLinkExtra == nullptr) return;
    if (myLinkExtra->heatFlux == nullptr) return;
    if (! myStructure.computeHeat) return;

    if (myStructure.saveHeatFluxesType == SAVE_HEATFLUXES_TOTAL && initHeat)
        myLinkExtra->heatFlux->fluxes[HEATFLUX_TOTAL] = NODATA;
    else if (myStructure.saveHeatFluxesType == SAVE_HEATFLUXES_ALL)
    {
        if (initHeat)
        {
            myLinkExtra->heatFlux->fluxes[HEATFLUX_TOTAL] = NODATA;
            myLinkExtra->heatFlux->fluxes[HEATFLUX_DIFFUSIVE] = NODATA;
            myLinkExtra->heatFlux->fluxes[HEATFLUX_LATENT_ISOTHERMAL] = NODATA;
            myLinkExtra->heatFlux->fluxes[HEATFLUX_LATENT_THERMAL] = NODATA;
            myLinkExtra->heatFlux->fluxes[HEATFLUX_ADVECTIVE] = NODATA;
        }

        if (initWater)
        {
            myLinkExtra->heatFlux->fluxes[WATERFLUX_LIQUID_ISOTHERMAL] = NODATA;
            myLinkExtra->heatFlux->fluxes[WATERFLUX_LIQUID_THERMAL] = NODATA;
            myLinkExtra->heatFlux->fluxes[WATERFLUX_VAPOR_ISOTHERMAL] = NODATA;
            myLinkExtra->heatFlux->fluxes[WATERFLUX_VAPOR_THERMAL] = NODATA;
        }
    }

}

void initializeLinkExtra(TCrit3DLinkedNodeExtra* myLinkedNodeExtra, bool computeHeat, bool computeSolutes)
{
    if (computeHeat)
    {
        myLinkedNodeExtra->heatFlux = new(THeatFlux);

        (*myLinkedNodeExtra).heatFlux->waterFlux = 0.;
        (*myLinkedNodeExtra).heatFlux->vaporFlux = 0.;

        if (myStructure.saveHeatFluxesType == SAVE_HEATFLUXES_ALL)
            (*myLinkedNodeExtra).heatFlux->fluxes = new float[9];
        else if (myStructure.saveHeatFluxesType == SAVE_HEATFLUXES_TOTAL)
            (*myLinkedNodeExtra).heatFlux->fluxes = new float[1];
        else
            (*myLinkedNodeExtra).heatFlux->fluxes = nullptr;

        initializeNodeHeatFlux(myLinkedNodeExtra, true, true);

    }
    else (*myLinkedNodeExtra).heatFlux = nullptr;

    if (computeSolutes)
    {
        // TODO
    }
}


