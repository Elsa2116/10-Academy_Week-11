# Data Enrichment Log
**Project:** Ethiopia Financial Inclusion Forecasting  
**Analyst:** Selam Analytics  
**Date:** 2024-01-01  

---

## Summary of Additions

### New Observations Added (beyond starter dataset)

| ID | Indicator | Date | Source | Reason |
|----|-----------|------|--------|--------|
| OBS010 | ACC_OWNERSHIP_FEMALE (36%) | 2021 | Findex 2021 Microdata | Gender gap analysis |
| OBS011 | ACC_OWNERSHIP_FEMALE (44%) | 2024 | Findex 2024 | Gender trend tracking |
| OBS012 | ACC_OWNERSHIP_MALE (56%) | 2021 | Findex 2021 Microdata | Gender gap calculation |
| OBS013 | ACC_OWNERSHIP_MALE (54%) | 2024 | Findex 2024 | Interesting: male ownership slightly fell |
| OBS014 | INFRA_MOBILE_PENETRATION (47%) | 2021 | GSMA Intelligence | Key enabler for digital finance |
| OBS015 | INFRA_MOBILE_PENETRATION (65%) | 2023 | GSMA Intelligence | Rapid growth in mobile connectivity |
| OBS016 | INFRA_4G_COVERAGE (52%) | 2022 | GSMA Intelligence | 4G is prerequisite for digital payments |
| OBS017 | INFRA_4G_COVERAGE (68%) | 2024 | GSMA Intelligence | Tracks infrastructure expansion |
| OBS018 | ACC_TELEBIRR_USERS (10M) | 2022 | Ethio Telecom | Operator-side data for validation |
| OBS019 | ACC_TELEBIRR_USERS (40M) | 2023 | Ethio Telecom | Rapid growth trajectory |
| OBS020 | ACC_TELEBIRR_USERS (54M) | 2024 | Ethio Telecom | Latest figure widely cited |
| OBS021 | ACC_MPESA_USERS (10M) | 2024 | Safaricom Ethiopia | New entrant tracking |
| OBS022 | USG_P2P_TRANSFER (180B ETB) | 2023 | NBE Annual Report | Transaction value as usage proxy |
| OBS023 | USG_ATM_WITHDRAWAL (160B ETB) | 2023 | NBE Annual Report | P2P surpassed ATM – key milestone |
| OBS024 | INFRA_BANK_BRANCH (5.2/100k) | 2022 | IMF FAS | Physical infrastructure baseline |
| OBS025 | INFRA_ATM_DENSITY (3.8/100k) | 2022 | IMF FAS | ATM as proxy for formal banking access |
| OBS026 | INFRA_AGENT_DENSITY (85/100k) | 2023 | NBE Annual Report | Agent networks critical for rural access |
| OBS027 | ACC_OWNERSHIP_URBAN (62%) | 2024 | Findex 2024 | Urban-rural gap analysis |
| OBS028 | ACC_OWNERSHIP_RURAL (41%) | 2024 | Findex 2024 | Rural access remains challenge |
| OBS029 | USG_DIGITAL_PAYMENT_FEMALE (28%) | 2024 | Findex 2024 | Gender usage gap |
| OBS030 | USG_DIGITAL_PAYMENT_MALE (42%) | 2024 | Findex 2024 | Male vs female payment behaviour |

### New Events Added

| ID | Event | Date | Rationale |
|----|-------|------|-----------|
| EVT006 | Fayda Digital ID Launch | Sep 2021 | Digital ID reduces KYC friction; critical enabler |
| EVT007 | NBE Agent Banking Directive | Jul 2020 | Physical access expansion; predates Telebirr |
| EVT008 | Foreign Bank Entry Reform | Mar 2023 | Structural reform for competition |
| EVT009 | Ethio Telecom Rural 4G Expansion | Jun 2022 | Connectivity prerequisite for mobile money |
| EVT010 | P2P Surpasses ATM (milestone) | Jan 2023 | Behavioral landmark; evidences deepening usage |

### New Impact Links Added

| ID | Event → Indicator | Direction | Magnitude | Rationale |
|----|------------------|-----------|-----------|-----------|
| IMP007 | Fayda → ACC_OWNERSHIP | positive | small | Digital ID reduces KYC friction (India Aadhaar precedent) |
| IMP008 | Agent Banking → ACC_OWNERSHIP | positive | medium | Physical access points are key adoption driver |
| IMP009 | 4G Expansion → ACC_OWNERSHIP | positive | small | Infrastructure enabler for rural mobile money |
| IMP010 | 4G Expansion → USG_DIGITAL_PAYMENT | positive | small | Connectivity enables digital payment behaviour |
| IMP011 | NFIS-II → ACC_OWNERSHIP | positive | medium | Policy coordination creates institutional accountability |
| IMP012 | Safaricom Entry → ACC_OWNERSHIP | positive | small | Competition drives coverage and price reduction |
| IMP013 | Safaricom Entry → USG_DIGITAL_PAYMENT | positive | medium | Data affordability drives digital payment use |
| IMP014 | Foreign Bank Reform → ACC_OWNERSHIP | positive | small | Long-term structural effect on product diversity |

---

## Key Observations on Schema Design

### Why Events Don't Have Pillars
Events (policies, product launches) are NOT pre-assigned to pillars (access/usage) because a single event can affect both pillars simultaneously. For example:
- Telebirr launch affects **Access** (new accounts) AND **Usage** (digital payments)
- 4G expansion affects **Access** (mobile money adoption) AND **Usage** (digital payment frequency)

Assigning a pillar would create artificial bias. Impact_link records capture the specific effects on each indicator.

### Confidence Assessment
- **High confidence**: Findex survey data (primary source, rigorous methodology)
- **Medium confidence**: Operator reports and NBE data (may have definitional differences from Findex)
- **Low confidence**: Estimates derived from comparable countries

---

## Data Gaps Identified

1. **No annual Findex data**: Survey conducted only every 3 years (2011, 2014, 2017, 2021, 2024) — creates interpolation uncertainty
2. **Active vs registered users**: All operator figures are registered users; Findex measures active usage — major discrepancy (54M Telebirr registered vs 9.45% mobile money ownership in Findex)
3. **Regional disaggregation**: No region-level Findex data available for Ethiopia
4. **Historical gender data**: Female ownership available only from 2021 onwards in this dataset
5. **Transaction frequency**: No data on how often users transact (active user definition varies)
6. **Credit data**: Almost no credit penetration data for Ethiopia (~0.5% as per market nuances)
