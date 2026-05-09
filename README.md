# MetaDrive ADAS Solution

Modularno rešenje za MetaDrive Autonomous Driving Challenge sa potpunim ADAS slojem.

## Struktura

```
solution.py    — Glavni Solution: Autopilot + DriverArbiter + ADAS pipeline
adas.py        — Modularna ADAS arhitektura (LKA, ACC, AEB, SHM, SmoothSteering)
dashboard.py   — (Opciono) HCI dashboard za prikaz stanja sistema vozaču
```

## Arhitektura

```
                                            ┌─ ACC (Adaptive Cruise Control)
   raw observation                          ├─ LKA (Lane Keeping Assist)
        │                                   ├─ AEB (Autonomous Emergency Braking)
        ▼                                   ├─ SHM (Sensor Health Monitor)
  Perception ─► Autopilot ─┐                └─ SSC (Smooth Steering Controller)
                            │                          │
                            ├─► DriverArbiter ──► AdasStack ──► [steer, throttle]
                            │
                       user_input
```

### Filozofija kombinovanja korisnickog ulaza

1. **Autopilot kao baza**: uvek se računa "default" akcija na osnovu Perception-a.
2. **Driver Arbiter**: blendje korisnički ulaz sa autopilotom koristeći asimetrično 
   smoothing — *brz attack, spor release*. Vozač odmah preuzima kad pritisne taster, 
   ali kad pusti, autopilot postepeno preuzima kontrolu.
3. **ADAS kao bezbednosna mreža**: čak i kad vozač upravlja, ADAS slojevi mogu 
   trenutno preuzeti kontrolu u opasnim situacijama (AEB, izlazak iz trake).

### Modularnost

Svaki ADAS sloj se može uključiti/isključiti nezavisno:

```python
sol = Solution(game)
sol.adas.set_enabled("LKA", False)        # samo iskljuci LKA
sol.adas.set_enabled("AEB", True)         # ostavi AEB
sol.use_autopilot = False                  # samo manuelna voznja sa ADAS-om
sol.use_adas = False                       # full manual passthrough
```

## ADAS slojevi

### 1. LKA (Lane Keeping Assist) — `adas.LaneKeepingAssist`
- Drži vozilo u centru trake
- PD kontroler na lateral grešku + heading + feedforward iz krivine puta
- Smanjuje uticaj kad vozač aktivno skreće (ne ometa nameru)

### 2. ACC (Adaptive Cruise Control) — `adas.AdaptiveCruiseControl`
- Održava ciljnu brzinu (default 45 km/h)
- Smanjuje target u zavojima (curvature_factor)
- Sledi vozila ispred sa sigurnim rastojanjem

### 3. AEB (Autonomous Emergency Braking) — `adas.CollisionAvoidance`
- Hitna kočnica kad je prepreka unutar ~7.5m (najviši prioritet)
- Pokušaj evazije skretanjem ka strani sa više prostora
- Bypass-uje rate-limit smoothera u kritičnim situacijama

### 4. SHM (Sensor Health Monitor) — `adas.SensorHealthMonitor`
- Detektuje degradaciju lidara: NaN-ove, dropout, šum
- Izračunava `degradation_score` ∈ [0, 1] preko vremena
- Pri "DEGRADED" stanju ograničava brzinu na ~20 km/h
- Pri "LIDAR LOST" forsira gotovo zaustavljanje

### 5. SSC (Smooth Steering Controller) — `adas.SmoothSteeringController`
- Slew-rate limiter na steering i throttle (smanjuje jerk)
- Allow-list za emergency: u kritičnim situacijama kočnica prolazi instantno

## Robusnost

- Sve ulazne vrednosti su klipovane u [-1, 1]
- NaN/inf u lidaru zamenjuju se sa 1.0 ("prazno polje") za safe processing,
  ali se zadržava nečista verzija za SHM detekciju
- None observation, prazne dictove, malformirane vektore — sve hendlamo bez exception-a
- Auto-detekcija veličine state vektora (radi sa različitim verzijama MetaDrive-a)

## Test scenariji koje smo proverili

- ✓ Čist put — autopilot ubrzava do target_speed i drži
- ✓ Prepreka 7m — AEB instantno koči (-1.0)
- ✓ Prepreka 12m — ACC postepeno smanjuje gas
- ✓ Lidar sa NaN-ovima — SHM detektuje, ograničava brzinu
- ✓ Lidar sa Gauss šumom — SHM detektuje, oprezna vožnja
- ✓ Vozač aktivan — Arbiter brzo prepušta kontrolu
- ✓ Vozač pušta — autopilot postepeno preuzima
- ✓ Zavoj — autopilot skreće u pravom smeru, smanjuje brzinu
- ✓ Sve invalid observation-ke — bez exception-a, valid output

## Tačke za prezentaciju

**Zašto ovaj pristup?**
- Ne pravimo monolitnu logiku — ADAS slojevi su realni asistivni sistemi koji se 
  mogu testirati nezavisno (LKA bez ACC, AEB bez ostalog, itd.)
- Vozač uvek može preuzeti kontrolu — nikad nije potpuno isključen
- Bezbednosni sloj (AEB + SHM) štiti i autopilota i vozača od grešaka

**Kompromisi:**
- SmoothSteeringController namerno usporava reakciju na pedale za ~70ms da bi 
  smanjio jerk. Emergency case ima ekplicitan bypass.
- Konzervativni target_speed (45 km/h) — bolje proći nego sleteti.

**HCI principi u dashboardu:**
- Hijerarhija: jedna kritična poruka u datom trenutku, ne 5 istovremenih
- Konzistentnost boja (crveno = odmah reaguj, žuto = pažnja, zeleno = OK)
- Brzina kao najveći prikaz (ono što vozač najčešće gleda)
