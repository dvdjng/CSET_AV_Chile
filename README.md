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
2. Get Climate data (TMYs)


4. Technical and Economic modelling
  - Set Parameters of AV Systems

Common parameters: "length_module", "width_module", "cap_module"
Individual parameters for each AV system: "tracking", "pvrow_azimuth", "pvrow_tilt", "pvrow_distance","pvrow_width", "pvrow_height", "bifaciality", "capex", "opex"
  - Shadow Simulation with optional optimization of row distance (bifacialvf)
Based on NRELs bifacial viewfactors libary (https://github.com/NREL/bifacialvf), the *simulate* function is modified to only perform actions that are needed to calculate irradiation on the ground (and not on the PV modules). Input variables for AV geometries have to be normalized by the pv module width (=1). The irradiation on the ground is computed for 100 points between two rows of PV modules. The irradiation values are further processed to effective PAR:
![image](https://github.com/user-attachments/assets/a55234b5-6da8-4028-8cc2-11058de8f3e9)

Optional optimization of the pv row pitch based on average DLI target value is possible. Wit

  - PV yield simulation (pvlib)
  - EV simulation (pyet)
  - LCOE simulation
    
4. Multi Criteria Deciscion Making

The Multi-Criteria Decision-Making (MCDM) method is used in the present study to evaluate and prioritize potential areas for feasibility. Criteria like solar radiation levels, land suitability, infrastructure proximity, environmental impact, regulations, and economic factors. Each criterion is then weighted based on its importance.

With the criteria in place, potential sites are identified and scored. Techniques like Analytic Hierarchy Process (AHP) or Technique for Order of Preference by Similarity to Ideal Solution (TOPSIS) are used to calculate aggregate scores. The sites are ranked based on these scores to identify the most suitable areas for agrivoltaic development. MCDM provides a transparent and objective way to determine the feasibility of potential agrivoltaic sites, allowing for a balanced approach that considers multiple factors in the decision-making process. Respective method is a common approach applied in geospatial AV potential studies. 

