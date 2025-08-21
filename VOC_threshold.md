# VOC Vapor Pressure Threshold in German Regulations (TA Luft Context)

**Goal**: Identify the *exact* vapor-pressure threshold used in German regulation for VOC compliance and explain how to use it in this project.

---

## TL;DR (What you will implement)

- **Regulatory VOC definition (EU / Germany)**: A substance is a **VOC** if its vapour pressure is **≥ 0.01 kPa at 20–21 °C (293.15 K)** or has a corresponding volatility under use conditions.  
  – This comes from the EU Industrial Emissions Directive (IED) and is transposed in Germany via the **31. BImSchV (Lösemittelverordnung)**, which TA Luft refers to for solvent‑related topics.

- **TA Luft operational threshold** (when extra controls kick in): For *liquid organic substances*, TA Luft applies specific requirements when the vapour pressure is **≥ 1.3 kPa at 293.15 K** (≈20 °C) — e.g., in **Nr. 5.2.6** and related sections for storage/handling and abatement.  

- **Project rule of thumb** (from Task 13): We will still **flag > 5.0 kPa** as *non‑compliant* for our simplified checker, but document that this is **stricter than** the legal trigger levels and is a design choice for conservative screening.

---

## Why two numbers appear in German practice

1) **VOC definition threshold (0.01 kPa @ 20 °C)**  
   This is the *classification* threshold for whether a compound is considered a *VOC*. It originates in EU law and is used across German regulations for consistency.

2) **TA Luft operational threshold (1.3 kPa @ 293.15 K)**  
   TA Luft (2021) contains **emission and handling requirements** that become applicable for *liquid organic substances* when their vapour pressure is **≥ 1.3 kPa** at ~20 °C. This is **not** redefining what a VOC is; it is a **trigger for additional technical requirements** (e.g., capture and abatement) under TA Luft.

In short: **0.01 kPa** decides if a substance is a VOC; **1.3 kPa** decides if certain TA Luft provisions apply due to volatility.

---

## Citations (primary/official sources)

- **EU / Germany – VOC definition (≥ 0.01 kPa @ 20–21 °C)**  
  *Industrial Emissions Directive* definition as used in Germany (see BAuA explanatory note summarizing IED vs. Decopaint definitions):  
  https://www.baua.de/dok/3998892  

  (Quote in German: *„VOC Definition … Dampfdruck ≥ 0,01 kPa bei 21 °C oder vergleichbare Flüchtigkeit …“*)

- **TA Luft 2021 (operational trigger at ≥ 1.3 kPa @ 293.15 K)**  
  *LAI Vollzugsfragen zur TA Luft* (Bund/Länder working group guidance), clarifying Nr. 5.2.6 applicability for liquids with **vapour pressure ≥ 1.3 kPa @ 293.15 K**:  
  https://www.lai-immissionsschutz.de/documents/auslegungsfragen-ta-luft-stand-04-2025_2_1744279307.pdf

  Supporting passages in drafts/summaries reflecting the same cutoff in storage/handling sections:  
  https://www.bmuv.de/fileadmin/Daten_BMU/Download_PDF/Glaeserne_Gesetze/19._Lp/ta_luft_neu/Entwurf/ta_luft_neu_refe_bf.pdf  
  https://www.ihk.de/blueprint/servlet/resource/blob/6510710/de3b7c6e7baf8f5e607e62fad1010f75/zusammenfassende-synopse-ta-luft-data.pdf

> **Note on §‑numbering**: TA Luft is a *Verwaltungsvorschrift* and uses **numbered sections (e.g., Nr. 5.2.6)** rather than „§“. The **VOC definition itself is not re‑stated inside TA Luft**; it follows the EU/31. BImSchV definition. If you cite a „§ 11“ VOC definition, that is typically from **31. BImSchV § 2 / § 11 context** (Solvent Ordinance), not TA Luft. Our documentation therefore cites the correct TA Luft section numbers and the EU/German definition source separately.

---

## How to use these thresholds in our project

We will implement three tiers in code and docs:

1. **VOC Classification (Regulatory)**  
   ```text
   is_voc = (vapour_pressure_kpa_at_20C >= 0.01)
   ```

2. **TA Luft High‑volatility Trigger (Operational)**  
   ```text
   ta_luft_trigger = (vapour_pressure_kpa_at_20C >= 1.3)
   ```

3. **Project Compliance Rule (Conservative)** — *per Task 13 design*  
   ```text
   non_compliant = (vapour_pressure_kpa_at_20C > 5.0)
   ```

…and we will clearly annotate in the README that **(3)** is a **project-specific conservative screening** above the legal trigger levels.

---

## README (BASF/Siemens context)

> **VOC threshold used in Germany** — In line with the EU IED and German 31. BImSchV, a compound is a VOC if its vapour pressure is **≥ 0.01 kPa at 20–21 °C**. TA Luft (2021) applies specific control requirements to **liquid organic substances** when vapour pressure is **≥ 1.3 kPa at 293.15 K** (≈20 °C). For conservative industrial screening, this project flags **> 5.0 kPa** as *non‑compliant* and also reports whether the **1.3 kPa** TA Luft trigger is met. (Sources: BAuA on IED VOC definition; LAI “Vollzugsfragen zur TA Luft”.)

---

## QA checklist

- [x] **Primary source for VOC definition** collected and cited.  
- [x] **Primary/official TA Luft guidance** confirming 1.3 kPa threshold collected and cited.  
- [x] **Clarified §‑numbering** (TA Luft uses “Nr.”, VOC definition from IED/31. BImSchV).  
- [x] **Implementation mapping** (0.01 kPa → VOC; 1.3 kPa → TA Luft trigger; 5.0 kPa → project rule).

---

*Prepared: 21 Aug 2025 *
