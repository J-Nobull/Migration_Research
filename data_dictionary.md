# Data Dictionary - County Migration Analysis (2011-2021)

## Identifiers
| Variable | Description | Source | Type |
|----------|-------------|--------|------|
| FIPS | 5-digit county FIPS code | All sources | String |
| Year | Year of observation (2011-2021) | All sources | Integer |

## IRS Migration Variables (Source: A_2122inpublicmigration.pdf)
| Variable | Description | Source | Type |
|----------|-------------|--------|------|
| move_in | Number of returns moving into county | IRS County Inflow | Integer |
| move_out | Number of returns moving out of county | IRS County Outflow | Integer |
| move_net | Net migration (move_in - move_out) | Derived | Integer |
| agi_in | Adjusted Gross Income of in-migrants (thousands) | IRS County Inflow | Integer |
| agi_out | Adjusted Gross Income of out-migrants (thousands) | IRS County Outflow | Integer |
| agi_net | Net AGI (agi_in - agi_out) | Derived | Integer |
| movers | Number of individuals migrating | IRS Bilateral | Integer |
| movers_agi | AGI of bilateral migrants (thousands) | IRS Bilateral | Integer |

## BEA Economic Variables (Source: B_MethodologyforRegionalPriceParities.pdf)
| Variable | Description | Source | Type |
|----------|-------------|--------|------|
| RPP | Regional Price Parity index (US=100) | BEA Regional | Float |
| BEA_PCI | Per Capita Income (dollars) | BEA CAINC1 | Integer |
| BEA_GDP | Real GDP by county (thousands, 2017$) | BEA CAGDP1 | Integer |

## Census ACS-5 Variables (Section 4.2)

### Population & Age
| Variable | Description | ACS Table | Type |
|----------|-------------|-----------|------|
| total_population | Total population | B01003_001E | Integer |
| median_age | Median age (years) | B01002_001E | Float |

### Housing
| Variable | Description | ACS Table | Type |
|----------|-------------|-----------|------|
| housing_total | Total housing units | B25003_001E | Integer |
| owner_occupied | Owner-occupied units | B25003_002E | Integer |
| renter_occupied | Renter-occupied units | B25003_003E | Integer |
| median_home_value | Median home value (dollars) | B25077_001E | Integer |
| median_property_taxes | Median property taxes (dollars) | B25103_001E | Integer |
| %owner_occupied | Owner-occupied rate | Derived (Section 7) | Float |

### Income
| Variable | Description | ACS Table | Type |
|----------|-------------|-----------|------|
| median_hh_income | Median household income (dollars) | B19013_001E | Integer |

### Marital Status
| Variable | Description | ACS Table | Type |
|----------|-------------|-----------|------|
| marital_total | Total marital universe | B12001_001E | Integer |
| never_married_male | Never married males | B12001_003E | Integer |
| now_married_male | Married males | B12001_004E | Integer |
| widowed_male | Widowed males | B12001_009E | Integer |
| divorced_male | Divorced males | B12001_010E | Integer |
| never_married_female | Never married females | B12001_012E | Integer |
| now_married_female | Married females | B12001_013E | Integer |
| widowed_female | Widowed females | B12001_018E | Integer |
| divorced_female | Divorced females | B12001_019E | Integer |
| %never_married_male | Pct never married males | Derived (Section 7) | Float |
| %now_married_male | Pct married males | Derived (Section 7) | Float |
| %divorced_male | Pct divorced males | Derived (Section 7) | Float |
| %never_married_female | Pct never married females | Derived (Section 7) | Float |
| %now_married_female | Pct married females | Derived (Section 7) | Float |
| %divorced_female | Pct divorced females | Derived (Section 7) | Float |
| %widowed_female | Pct widowed females | Derived (Section 7) | Float |

### Household Composition
| Variable | Description | ACS Table | Type |
|----------|-------------|-----------|------|
| family_households | Family households | B11001_002E | Integer |
| under_18_in_hh | Children under 18 in households | B09001_002E | Integer |

### Race & Ethnicity
| Variable | Description | ACS Table | Type |
|----------|-------------|-----------|------|
| white | White alone population | B03002_003E | Integer |
| black | Black alone population | B03002_004E | Integer |
| native | Native American population | B03002_005E | Integer |
| asian | Asian population | B03002_006E | Integer |
| pacific_islander | Pacific Islander population | B03002_007E | Integer |
| other_race | Other race population | B03002_008E | Integer |
| mixed_non_h | Mixed race non-Hispanic | B03002_009E | Integer |
| hispanic | Hispanic population | B03002_012E | Integer |
| %white | Pct white | Derived (Section 7) | Float |
| %black | Pct black | Derived (Section 7) | Float |
| %native | Pct Native American | Derived (Section 7) | Float |
| %asian | Pct Asian | Derived (Section 7) | Float |
| %pacific_islander | Pct Pacific Islander | Derived (Section 7) | Float |
| %other_race | Pct other race | Derived (Section 7) | Float |
| %hispanic | Pct Hispanic | Derived (Section 7) | Float |

### Education
| Variable | Description | ACS Table | Type |
|----------|-------------|-----------|------|
| education_total_sex | Total education universe (25+) | B15002_001E | Integer |
| male_complete_hs | Males HS graduate | B15002_011E | Integer |
| male_less1yr_college | Males <1yr college | B15002_012E | Integer |
| male_more1yr_college | Males >1yr college | B15002_013E | Integer |
| male_associates | Males associate's degree | B15002_014E | Integer |
| male_bachelors | Males bachelor's degree | B15002_015E | Integer |
| male_masters | Males master's degree | B15002_016E | Integer |
| male_professional | Males professional degree | B15002_017E | Integer |
| male_doctorate | Males doctorate | B15002_018E | Integer |
| female_complete_hs | Females HS graduate | B15002_028E | Integer |
| female_less1yr_college | Females <1yr college | B15002_029E | Integer |
| female_more1yr_college | Females >1yr college | B15002_030E | Integer |
| female_associates | Females associate's degree | B15002_031E | Integer |
| female_bachelors | Females bachelor's degree | B15002_032E | Integer |
| female_masters | Females master's degree | B15002_033E | Integer |
| female_professional | Females professional degree | B15002_034E | Integer |
| female_doctorate | Females doctorate | B15002_035E | Integer |
| %college_degree | Pct with college degree (25+) | Derived (Section 7) | Float |
| %HSorCollege_NOdegree | Pct HS/some college, no degree | Derived (Section 7) | Float |

### Commute Time
| Variable | Description | ACS Table | Type |
|----------|-------------|-----------|------|
| commute_less_5min | Commute <5 min | B08303_002E | Integer |
| commute_5_9min | Commute 5-9 min | B08303_003E | Integer |
| commute_10_14min | Commute 10-14 min | B08303_004E | Integer |
| commute_15_19min | Commute 15-19 min | B08303_005E | Integer |
| commute_20_24min | Commute 20-24 min | B08303_006E | Integer |
| commute_25_29min | Commute 25-29 min | B08303_007E | Integer |
| commute_30_34min | Commute 30-34 min | B08303_008E | Integer |
| commute_35_39min | Commute 35-39 min | B08303_009E | Integer |
| commute_40_44min | Commute 40-44 min | B08303_010E | Integer |
| commute_45_59min | Commute 45-59 min | B08303_011E | Integer |
| commute_60_89min | Commute 60-89 min | B08303_012E | Integer |
| commute_90_plus_min | Commute 90+ min | B08303_013E | Integer |
| work_in_owned_home | Work from owned home | B08137_020E | Integer |
| work_in_rental | Work from rental | B08137_021E | Integer |

### Occupation
| Variable | Description | ACS Table | Type |
|----------|-------------|-----------|------|
| occupation_total | Total occupation universe | C24060_001E | Integer |
| Mgmt_Biz_Sci_Arts | Management/business/science/arts | C24060_002E | Integer |
| Services | Service occupations | C24060_003E | Integer |
| Sales_Admin | Sales and administrative | C24060_004E | Integer |
| Nat-rsrc_Constr_Maint | Natural resources/construction | C24060_005E | Integer |
| Prod_Transp_Mvng | Production/transportation | C24060_006E | Integer |
| %white_collar | Pct white collar jobs | Derived (Section 7) | Float |

## BLS Variables
| Variable | Description | Source | Type |
|----------|-------------|--------|------|
| unemploy_rate | Unemployment rate (percent) | BLS LAUS | Float |

## USDA Variables

### Rural-Urban Continuum Codes (RUCC)
| Variable | Description | Source | Type |
|----------|-------------|--------|------|
| RUC_code | Rural-Urban Continuum Code (1-9) | USDA ERS | Integer |

**RUC_code Values:**
- 1-3: Metro counties (1=largest, 3=smallest)
- 4-9: Nonmetro counties (4=adjacent to metro, 9=remote rural)

### County Typology (Section 4.4)
| Variable | Description | Source | Type |
|----------|-------------|--------|------|
| Farming | Farming-dependent county | USDA Typology 2015 | Binary |
| Mining | Mining-dependent county | USDA Typology 2015 | Binary |
| Mfging | Manufacturing-dependent county | USDA Typology 2015 | Binary |
| Govt | Government-dependent county | USDA Typology 2015 | Binary |
| Rec | Recreation-dependent county | USDA Typology 2015 | Binary |
| Nonspec | Nonspecialized county | USDA Typology 2015 | Binary |
| Low_Ed_cnty | Low education county | USDA Typology 2015 | Binary |
| Low_employ_cnty | Low employment county (2008-2012) | USDA Typology 2015 | Binary |
| Retire_dest_cnty | Retirement destination county | USDA Typology 2015 | Binary |
| Persistent_Pov_cnty | Persistent poverty county | USDA Typology 2015 | Binary |
| Pers_chld_pov_cnty | Persistent child poverty county | USDA Typology 2015 | Binary |

### Natural Amenities (Source: C_Nat_amenities.pdf)
| Variable | Description | Source | Type |
|----------|-------------|--------|------|
| Amenity_scale | Natural amenity scale (-2 to +7 std dev) | USDA ERS | Float |

**Amenity Components (not in final panel):**
- Warm winter (January temperature)
- Winter sun (sunny January days)
- Temperate summer (low winter-summer gap)
- Low summer humidity (July humidity)
- Water area (percent surface water)
- Topographic variation (terrain scale)

## Housing Incentive Variables
| Variable | Description | Source | Type |
|----------|-------------|--------|------|
| has_incentive | Binary indicator for incentive program | Compiled | Binary |
| Incentive_CAT | Category of incentive program | Compiled | String |
| COVID_program | COVID-era program indicator | Compiled | Binary |
| PULL | Treatment indicator (synonym for has_incentive) | Compiled | Binary |

## Notes
- All percentage variables are rounded to 2 decimal places
- FIPS codes are standardized to 5 digits with leading zeros
- Years range from 2011-2021 (11 years)
- Missing values coded as NaN or 0 depending on context
- Alaska boroughs consolidated to FIPS 02001
- Connecticut planning regions mapped to counties
- Kalawao County HI (15005) and Bedford City VA (51515) excluded
- Puerto Rico and territories (FIPS > 56999) excluded
