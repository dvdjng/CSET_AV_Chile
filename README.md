# CSET_AV_Chile



# AV_Chile_Analisis
Aim: Import geodata on agricultural areas und supplemental information (climate (change), infrastructure, social, etc...) to compute potential for AV implementation.

Content:
1. Create Working Geodataframe
2. Get Climate data (TMYs)
3. Technical and Economic modelling
  - Set Parameters of AV Systems
  - Shadow Simulation with optional optimization of row distance (bifacialvf)
  - PV yield simulation (pvlib)
  - EV simulation (pyet)
  - LCOE simulation
4. Multi Criteria Deciscion Making

-----
1. Create Working Geodataframe

Based on input geodata for agricultural areas and additional information, a working geodataframe (gdf) with points in a user-defined-distance is created (raster like). The information on agricultural areas and additional data is discretized and stored for every point.
  
2. Get Climate data (TMYs)

TMYs are downloaded from PVGIS based on coordinates and saved locally as csv-files. Before starting the dowload, existing files are checked with the working gdf: not matching csv-files are deleted and missing TMY data is dowloaded. Information from the TMYs such as GHI (kWh/a) altitude (m) and information on the download ("ok", "missing climate data", "missing solar data") is saved in the working gdf.

3. Technical and Economic modelling
- Set Parameters of AV Systems

Different AV systems can be defined based on the following parameters.
Common parameters: "length_module", "width_module", "cap_module"

Individual parameters for each AV system: "tracking", "pvrow_azimuth", "pvrow_tilt", "pvrow_distance","pvrow_width", "pvrow_height", "bifaciality", "capex", "opex"

- Shadow Simulation with optional optimization of row distance (bifacialvf)

Based on NRELs bifacial viewfactors libary (https://github.com/NREL/bifacialvf), the *simulate* function is modified to only perform actions that are needed to calculate irradiation on the ground (and not on the PV modules). Input variables for AV geometries have to be normalized by the pv module width (=1). The irradiation on the ground is computed for 100 points between two rows of PV modules. The irradiation values are further processed to effective PAR:
![image](https://github.com/user-attachments/assets/a55234b5-6da8-4028-8cc2-11058de8f3e9)

Optional optimization of the pv row pitch based on average DLI target value is possible (with sipy.minimize_scalar).

  - PV yield simulation (pvlib)

Based on *pvlib* libary (https://pvlib-python.readthedocs.io/en/stable/) using the *pvfactors_timeseries* and *pvwatts* modules. Hourly AC generation is computed.

  - EV simulation (pyet)

Based on *pyet* libary (https://github.com/pyet-org/pyet), computes the reference evapotranspiration according to FAO 56 standards.

  - LCOE simulation

LCOE computation based on inhouse developed code.
    
4. Multi Criteria Deciscion Making

The Multi-Criteria Decision-Making (MCDM) method is used in the present study to evaluate and prioritize potential areas for feasibility. Criteria like solar radiation levels, land suitability, infrastructure proximity, environmental impact, regulations, and economic factors. Each criterion is then weighted based on its importance.

With the criteria in place, potential sites are identified and scored. Techniques like Analytic Hierarchy Process (AHP) or Technique for Order of Preference by Similarity to Ideal Solution (TOPSIS) are used to calculate aggregate scores. The sites are ranked based on these scores to identify the most suitable areas for agrivoltaic development. MCDM provides a transparent and objective way to determine the feasibility of potential agrivoltaic sites, allowing for a balanced approach that considers multiple factors in the decision-making process. Respective method is a common approach applied in geospatial AV potential studies. 

