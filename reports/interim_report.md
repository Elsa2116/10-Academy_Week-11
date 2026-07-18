# Interim Report: Financial Inclusion Forecasting in Ethiopia
**Selam Analytics** | Interim Submission — July 19, 2026

---

## Data Enrichment Summary

### What We Added and Why

We enriched the starter dataset with 21 new observation records, 5 new events, and 8 new impact links.

**New observations** focus on four categories:
1. **Gender disaggregation** (4 records) — Female/Male ownership and payment data to quantify the gender gap
2. **Infrastructure indicators** (6 records) — Mobile penetration, 4G coverage, ATM density, bank branches, agent density as leading indicators
3. **Operator data** (3 records) — Telebirr and M-Pesa registered user counts to cross-validate Findex trends
4. **Transaction data** (2 records) — P2P transfer and ATM withdrawal values to measure behavioural depth
5. **Urban-rural split** (2 records) — Geographic disaggregation of account ownership
6. **Gender-payment disaggregation** (2 records) — Usage by gender

**New events** capture important drivers missing from the starter dataset:
- **Fayda Digital ID** (Sep 2021) — Reduces KYC friction; India Aadhaar precedent shows 12–18 month lag to account opening
- **NBE Agent Banking Directive** (Jul 2020) — Expanded physical access points before Telebirr; critical precondition
- **Foreign Bank Entry Reform** (Mar 2023) — Structural supply-side change; long-term effect
- **Ethio Telecom Rural 4G Expansion** (Jun 2022) — Connectivity enabler for rural mobile money
- **P2P/ATM Crossover Milestone** (Jan 2023) — Behavioural landmark rather than a policy event, but important for impact modeling

---

## Key Insights from EDA (5 Required Insights)

### Insight 1: The Registration-Activity Gap Explains the 2021–2024 Slowdown
Telebirr grew to 54M registered users, yet Findex mobile money account ownership moved from 4.7% to 9.45% — representing only ~6.1M adults. An estimated 10:1 registered-to-active ratio reflects that most Telebirr users already had bank accounts, never transacted, or used multiple SIMs. The Findex measures *active accounts in past 12 months*; operators measure *any registration*. These are fundamentally different metrics.

### Insight 2: Gender Gap Halved in Access, Persists in Usage
Female account ownership rose from 36% (2021) to 44% (2024) while male ownership fell from 56% to 54% — a dramatic convergence. However, the digital payment gender gap (Male: 42%, Female: 28%) remains large at 14pp. Women are entering the formal financial system through Telebirr/agent banking, but aren't translating that access into active digital payment behavior.

### Insight 3: Urban-Rural Gap Is the Primary Structural Challenge
The 21pp urban-rural gap (Urban: 62%, Rural: 41% in 2024) is structurally important: ~80% of Ethiopians live in rural areas. National headline figures are disproportionately influenced by rural performance. Agent networks (85/100k) are already the de facto rural infrastructure vs bank branches (5.2/100k).

### Insight 4: P2P Surpassing ATM Signals Ecosystem Maturity
In 2023, P2P transfers (ETB 180B) exceeded ATM withdrawals (ETB 160B) for the first time. Unlike registration statistics, this is a real transaction metric. It signals that digital payments have crossed a critical adoption threshold — users are actively choosing digital over cash.

### Insight 5: Infrastructure Is Both Enabler and Constraint
Mobile penetration grew from 47% (2021) to 65% (2023) and 4G coverage from 52% (2022) to 68% (2024). Correlations between mobile connectivity and account ownership are strongly positive. The remaining 32% without 4G access is disproportionately rural — the same population with lowest financial inclusion. Infrastructure expansion is necessary (though not sufficient) for sustained access growth.

### Insight 6: Market Nuances Make Ethiopia Unique
Ethiopia-specific dynamics differ from other SSA markets: mobile-money-only users are extremely rare (~0.5%); bank accounts are relatively easy to open (high branch density in urban areas); P2P is used as a commerce substitute (not just person-to-person transfers); and credit penetration is near zero. These dynamics mean the inclusion ladder looks different here than in Kenya or Tanzania.

### Insight 7: Events Build Cumulatively, Not Instantly
The Telebirr launch (May 2021) → EthSwitch interoperability (Mar 2022) → Safaricom entry (Aug 2022) → M-Pesa launch (Aug 2023) represents a cascade of reinforcing events. No single event explains the trajectory; it's the compounding of supply-side improvements, regulatory coordination, and competitive pressure over 2021–2024.

---

## Preliminary Event-Indicator Relationships

| Event | Primary Indicator | Lag | Magnitude | Confidence |
|-------|------------------|-----|-----------|-----------|
| Telebirr Launch | ACC_MM_ACCOUNT | 6–12 mo | Large | High |
| EthSwitch Interop | USG_DIGITAL_PAYMENT | 3–6 mo | Medium | Medium |
| M-Pesa Launch | USG_DIGITAL_PAYMENT | 6–12 mo | Medium | Medium |
| Agent Banking Directive | ACC_OWNERSHIP | 12 mo | Medium | High |
| Fayda Digital ID | ACC_OWNERSHIP | 18 mo | Small | Low–Medium |
| 4G Rural Expansion | ACC_OWNERSHIP, USG_DIGITAL_PAYMENT | 18 mo | Small | Medium |

---

## Data Limitations Identified

1. **Sparse Findex time series**: Only 5 data points over 13 years (2011, 2014, 2017, 2021, 2024) — severely limits statistical model power
2. **Active vs registered gap**: Cannot reconcile operator and Findex measures without raw transaction data
3. **No regional disaggregation**: Findex Ethiopia data is national-level only
4. **Pre-2021 gender data**: Female/male breakdown not available before 2021 in accessible public data
5. **Transaction frequency**: No data on how often users transact (active user threshold varies by definition)
6. **Credit data absence**: ~0.5% credit penetration but no time series available
