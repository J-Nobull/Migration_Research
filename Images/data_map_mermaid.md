# Migration Data Processing Map


```mermaid
flowchart TB
    %% Raw Data Sources
    BEA[BEA API<br/>Per Capita Income<br/>GDP<br/>RPP<br/>2011-2021]
    BLS[BLS API<br/>Unemployment<br/>2011-2021]
    CENSUS[Census API<br/>Demographics<br/>Housing<br/>2011-2021]
    IRS[IRS CSV Files<br/>County Migration Flows<br/>countyinflow2011-2021.csv]
    USDA[USDA Files<br/>RUCC, Amenities<br/>Typology]
    INCENTIVE[Manual Collection<br/>Incentive Programs<br/>2016-2021]

    %% Cleaned Individual Files
    BEA --> BEA_CLEAN[BEA_clean.csv<br/>3,118 FIPS × 11 years]
    BLS --> BLS_CLEAN[BLS_clean.csv<br/>3,135 FIPS × 11 years]
    CENSUS --> CENSUS_CLEAN[Census_clean.csv<br/>3,146 FIPS × 11 years]
    IRS --> IRS_CLEAN[IRS_clean.csv<br/>3,081 FIPS × 11 years]
    USDA --> USDA_CLEAN[USDA_clean.csv<br/>3,142 FIPS × 11 years]
    INCENTIVE --> INCENTIVE_CLEAN[Incentive_clean.csv<br/>656 FIPS, 4,417 records]

    %% IRS Processing
    IRS --> IRS_PANEL[IRS_panel.csv<br/>County-Year Aggregates<br/>net_movers, net_agi]
    IRS --> IRS_DYADIC[IRS_dyadic.csv<br/>Origin-Destination-Year<br/>Directional Flows]
    IRS_DYADIC --> IRS_RECIPROCAL[IRS_dyadic_reciprocal.csv<br/>Bidirectional Pairs<br/>gross_flow, net_flow]

    %% Geographic Standardization
    BEA_CLEAN --> GEO_STANDARD[Geographic Standardization<br/>- FIPS zero-padding<br/>- Drop state totals xx000<br/>- Remap 15901→15009<br/>- Remap 46102→46113]
    BLS_CLEAN --> GEO_STANDARD
    CENSUS_CLEAN --> GEO_STANDARD
    USDA_CLEAN --> GEO_STANDARD
    IRS_PANEL --> GEO_STANDARD

    %% Geographic Consolidation
    GEO_STANDARD --> VA_CITIES[Virginia Cities Aggregation<br/>22 cities → counties<br/>Population variables]
    GEO_STANDARD --> BEDFORD[Bedford City Special<br/>51515→51019<br/>2011-2012 only]
    GEO_STANDARD --> ALASKA[Alaska Consolidation<br/>36 counties → 02001<br/>Sum counts, weighted avg rates]

    %% Merge Process
    VA_CITIES --> MERGE[Outer Join on FIPS + YEAR]
    BEDFORD --> MERGE
    ALASKA --> MERGE
    INCENTIVE_CLEAN --> MERGE

    %% Post-Merge Cleaning
    MERGE --> POST_CLEAN[Post-Merge Cleaning<br/>- Drop VA cities 51510-51958<br/>- Drop CT regions 09110-09190<br/>- Drop 97000<br/>- Consolidate 46102→46113<br/>- Remove duplicates]

    %% Missing Value Imputation
    POST_CLEAN --> IMPUTE[Missing Value Imputation<br/>Strategy 1: Within-county mean<br/>Strategy 2: Adjacent year avg<br/>Strategy 3: Zero-fill<br/>Strategy 4: National averages]

    %% Final Panel
    IMPUTE --> MIGRATION_PANEL[migration_panel.csv<br/>34,287 rows<br/>3,117 FIPS × 11 years<br/>~80 variables]

    %% Create Composites
    MIGRATION_PANEL --> COMPOSITES[Create Composite Variables<br/>- college_degree<br/>- hs_some_college<br/>- pct_white_collar<br/>- pct_blue_collar]

    %% Final Output
    COMPOSITES --> FINAL[migration_panel_composites.csv<br/>34,287 rows<br/>3,117 FIPS × 11 years<br/>84 variables<br/>READY FOR MODELING]

    %% Dyadic Analysis Path
    IRS_RECIPROCAL --> DYADIC_MERGE[Merge Panel Characteristics<br/>Origin + Destination]
    MIGRATION_PANEL --> DYADIC_MERGE
    DYADIC_MERGE --> MODEL_3[Model 3: Gravity Analysis<br/>Bilateral flows<br/>Push-pull factors]

    %% Modeling Outputs
    FINAL --> MODEL_1[Model 1: Panel FE<br/>County + Year FE]
    FINAL --> MODEL_2[Model 2: DiD<br/>Incentive Programs]
    FINAL --> MODEL_4[Model 4: Spatial<br/>2019 cross-section]
    FINAL --> MODEL_5[Model 5: Dynamic<br/>Lagged DV]

    %% Style Definitions
    classDef rawData fill:#e1f5ff,stroke:#01579b,stroke-width:2px
    classDef cleaned fill:#fff9c4,stroke:#f57f17,stroke-width:2px
    classDef process fill:#f3e5f5,stroke:#4a148c,stroke-width:2px
    classDef final fill:#c8e6c9,stroke:#1b5e20,stroke-width:3px
    classDef model fill:#ffccbc,stroke:#bf360c,stroke-width:2px

    class BEA,BLS,CENSUS,IRS,USDA,INCENTIVE rawData
    class BEA_CLEAN,BLS_CLEAN,CENSUS_CLEAN,IRS_CLEAN,USDA_CLEAN,INCENTIVE_CLEAN,IRS_PANEL,IRS_DYADIC,IRS_RECIPROCAL cleaned
    class GEO_STANDARD,VA_CITIES,BEDFORD,ALASKA,MERGE,POST_CLEAN,IMPUTE,COMPOSITES,DYADIC_MERGE process
    class MIGRATION_PANEL,FINAL final
    class MODEL_1,MODEL_2,MODEL_3,MODEL_4,MODEL_5 model
```


## Key
- **Blue boxes**: Raw data sources (APIs, CSVs)
- **Yellow boxes**: Cleaned individual datasets
- **Purple boxes**: Processing steps (joins, aggregations)
- **Green boxes**: Final analysis-ready datasets
- **Orange boxes**: Statistical models

## Viewing Options
1. GitHub: Will auto-render the diagram
2. VS Code: Install 'Markdown Preview Mermaid Support' extension
3. Online: Copy code to https://mermaid.live/
4. Jupyter: Create markdown cell and paste the mermaid block
