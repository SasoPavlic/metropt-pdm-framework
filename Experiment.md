## MetroPT-IForest Experiments (opis workflowa)

Ta dokument v razumljivem jeziku razlozi, kako program bere podatke, kako gradi znacilke, kako trenira modele in kako izracuna dve vrsti anomalij (point‑wise in collective). Opis je 1:1 skladen z aktualno kodo.

---

## 1) Branje in priprava podatkov

### Kakšni so podatki v CSV
MetroPT3 vsebuje casovne vrste iz tramvaja (APU). V stolpcih se pojavljajo:
- analogni senzorji (npr. tlaki, temperature, tok motorja),
- binarni/kvazi‑binarni signali (npr. stikala ali stanja ventilov 0/1),
- `timestamp` (casovni zig).

Program CSV prebere, `timestamp` uredi in nastavi kot indeks.

### PRE_DOWNSAMPLE_RULE (opcijsko)
`PRE_DOWNSAMPLE_RULE` je pravilo resamplinga (npr. `"60s"`). Ce je nastavljen:
- podatke agregiramo na enakomeren casovni korak,
- v vsakem intervalu vzamemo median.

**Zakaj to pomaga:**
- odpravi neenakomerne casovne razmake,
- zmanjsa sum,
- omogoci stabilne rolling znacilke.

### Izbor znacilk (feature engineering)
Program vzame samo numericne stolpce. Nato:
- lahko odstrani kvazi‑binarne stolpce (ce so prevec 0/1),
- izmed preostalih izbere `MAX_BASE_FEATURES` z najvecjo varianco.

**Zakaj to delamo:**
- zmanjsamo dimenzionalnost,
- omejimo sum,
- ohranimo najbolj informativne signale.

### Rolling znacilke
Na izbranih signalih naredimo rolling statistike v oknu `ROLLING_WINDOW` (npr. mean, std, min, max, skew). Rezultat je matrika `X` (cas × znacilke), ki jo dobi model.

---

## 2) Maintenance okna in faze (operation_phase)

Maintenance okna so vzeta **iz clanka Davari et al. (2021)**. Na tej osnovi vsak timestamp dobi fazo:
- `operation_phase = 0` → normalno,
- `operation_phase = 1` → pre‑maintenance (npr. 2 uri pred servisom; `PRE_MAINTENANCE_HOURS`),
- `operation_phase = 2` → maintenance (servis).

Faze **niso vhod modela**, uporabljajo se izkljucno za evaluacijo.

---

# Dve vrsti detekcije anomalij

## A) Point‑wise anomalije (po vrstici)
Model vsaki vrstici dodeli `is_anomaly` (0/1).
- `is_anomaly = 1`, ce je `anom_score >= threshold`.
- `threshold` = `Q3 + 3*IQR` iz **ucnih** score‑ov.

### Point‑wise evalvacija (TP/FP/FN/TN)
V evaluacijo gredo le faze 0 in 1:
- **TP**: `phase=1` in `is_anomaly=1`
- **FP**: `phase=0` in `is_anomaly=1`
- **FN**: `phase=1` in `is_anomaly=0`
- **TN**: `phase=0` in `is_anomaly=0`
- `phase=2` je **ignorirana**.

Iz point‑wise evaluacije so dodatno izloceni:
- zacetni ucni interval (`TRAIN_FRAC` minut),
- vsi post‑maintenance ucni intervali,
da je primerjava med rezimoma fer.

---

## B) Collective anomalije (event‑level)
Collective anomalija ni nova oznaka v podatkih, ampak agregacija point‑wise anomalij skozi casovno okno.

**Koraki:**
1. `exceedance`  
   - `exceedance = 1`, kjer je `anom_score >= threshold`.  
   - to je binarni signal "presega prag".
2. `maintenance_risk`  
   - rolling povprecje `exceedance` v oknu `RISK_WINDOW_MINUTES`.  
   - predstavlja **delez anomalij v zadnjem casovnem oknu**.
3. Alarm interval  
   - vsak neprekinjen segment, kjer `maintenance_risk >= θ`.

**Mini primer (poenostavljeno):**
```
timestamp           exceedance  maintenance_risk  alarm?
10:00               0           0.02              no
10:05               1           0.05              no
10:10               1           0.12              yes
10:15               1           0.18              yes
10:20               0           0.09              no
```
Alarm interval je 10:10–10:15.

### Event‑level TP/FP/FN (primer iz MetroPT)
Recimo servis #7:  
`2020-05-18 05:00 → 05:30` in `EARLY_WARNING_MINUTES = 120`.
Potem je veljavno okno za TP:
`03:00 → 05:30`.

- **TP**: zacetek alarma pade v to okno (npr. 03:10).
- **FN**: noben alarm se ne zacne v tem oknu.
- **FP**: alarm interval, ki ni povezan z nobenim servisom.
- **TN** na event nivoju ni definiran (ni negativnih “eventov”).

---

# Rezim A: Single model (`EXPERIMENT_MODE="single"`)

## Ucenje
- Model se uci **na prvih `TRAIN_FRAC` minutah** (trenutno 1440 min = 24h).
- Iz ucnih vrstic so izlocene `operation_phase == 2`.
  - **Zakaj?** Ne zelimo, da se model "nauci" vzorec servisa kot normalno stanje.

## Scoring
- Model oceni **vse** vrstice.
- `anom_score` je osnovni IF score; `anom_score_lpf` je opcijsko glajenje (LPF), ki zmanjsa sum in kratke spike.

## Evaluacija
Point‑wise in event‑level logika je enaka kot zgoraj.  
Za fer primerjavo se izlocijo tudi post‑maintenance ucni intervali.

---

# Rezim B: Per‑maintenance model (`EXPERIMENT_MODE="per_maint"`)

## Globalni baseline
- **Globalni baseline** = isti zacetni TRAIN_FRAC interval (brez phase 2).
- Ta baseline je **vedno** del ucnih podatkov vsakega per‑maint modela.
- To **ni fine‑tuning**; vsak model se uci znova na (baseline + lokalni interval).

## Oznake (da bo zapis jasno sledljiv)
- `Wj` = j‑ti servisni interval  
- `start_j`, `end_j` = zacetek in konec servisa  
- `gap` = interval `(end_j, start_{j+1})`  
- `POST_MAINT_TRAIN_MINUTES` = dolzina lokalnega ucnega intervala po servisu

## Ucenje in test po servisih
Za vsak servis `Wj`:
- definira se `gap = (end_j, start_{j+1})`
- ce je `gap <= POST_MAINT_TRAIN_MINUTES`:
  - **ne treniramo** novega modela, uporabimo prejsnjega
- ce je gap daljsi:
  - lokalni ucni interval = prvih `POST_MAINT_TRAIN_MINUTES` po `end_j`
  - ucenje = **baseline + lokalni interval** (brez phase 2)
  - test = preostanek gap‑a do naslednjega servisa

## Scoring
- Vsak segment dobi svoj IF model (ali ponovno uporabi prejsnjega).
- `anom_score`, `is_anomaly` in `threshold` so segment‑specificni.
- **"Zlepljen exceedance"** pomeni:
  - za vsak timestamp vzamemo `exceedance` iz modela, ki pokriva ta segment,
  - vse segmente zdruzimo v en casovni niz,
  - iz tega nastane enoten `maintenance_risk`.

## Point‑wise evaluacija
Da, vsak model oznacuje **samo** vrstice v svojem test intervalu.  
Nato se vse oznake zdruzijo in evalvirajo enako kot pri single:
- phase 1 = pozitivno, phase 0 = negativno, phase 2 ignorirano.
- izlocimo baseline + post‑maintenance ucne intervale (fer primerjava).

## Event‑level evaluacija
Event‑level logika je **enaka** kot pri single, ker:
- na koncu dobimo **enoten `maintenance_risk`** niz,
- alarm intervali se racunajo na celotnem nizu,
- TP/FN/FP definicije so iste.

---

# Ključne razlike (kratko in pregledno)

| Lastnost | Single | Per‑maintenance |
|---|---|---|
| Stevilo modelov | 1 | vec (po servisih) |
| Ucni podatki | samo zacetni TRAIN_FRAC interval | baseline + lokalni post‑maint interval |
| Score/threshold | globalen | segment‑specificen |
| Point‑wise logika | ista | ista |
| Collective logika | ista | ista |

---

# Rezultati (zadnji tek)

### Event‑level (Best θ)
| Regime | Tag | θ | TP | FP | FN | Precision | Recall | F1 |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| Single | `[RISK]` | 0.60 | 7 | 32 | 14 | 0.1795 | 0.3333 | 0.2333 |
| Per‑maint | `[RISK-PERMAINT]` | 0.27 | 2 | 85 | 19 | 0.0230 | 0.0952 | 0.0370 |

### Point‑wise
| Regime | Tag | TP | FP | FN | TN | Precision | Recall | F1 | Accuracy |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Single | `[METRIC] Single-model` | 2261 | 240374 | 6718 | 1111512 | 0.0093 | 0.2518 | 0.0180 | 0.8184 |
| Per‑maint | `[METRIC] Per-maint-model` | 2456 | 201039 | 6523 | 1150846 | 0.0121 | 0.2735 | 0.0231 | 0.8475 |

---

# Vizualizacija (mentalni model)

Single:
```
TRAIN_FRAC --------> [model A] ----------------------------------> scoring vseh vrstic
```

Per‑maint:
```
baseline + post‑maint_1 --> model B --> scoring gap_1
baseline + post‑maint_2 --> model C --> scoring gap_2
...
```

Rezultat v obeh rezimih je:
- `is_anomaly` za vsako vrstico (point‑wise)
- `maintenance_risk` za casovno okno (collective)
- TP/FP/FN/TN (point‑wise) in TP/FP/FN (event‑level)
