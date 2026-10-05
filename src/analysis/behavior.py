"""Behavior Discovery and Classification.

Analyzes organism movement, combat, social and foraging statistics.
Automatically classifies behavior into emergent behavioral roles without
hard-coding rules into organism brains.

Roles:
- Explorer: High distance traveled, high max displacement, wanders across world
- Grazer: High plant foraging time, low combat, steady energy consumption
- Predator: High pursuit time, attacks, meat consumption, damage dealt
- Scavenger: High carrion intake, opportunistic feeding
- Ambusher: High still streak before strikes, ambush attacks
- Territorial: High home time (stays in local area), attacks intruders
- Social: High neighbor proximity time, follows herd/pack
- Erratic: High turning variance, unpredictable changes of direction
- Energy Efficient: High ratio of energy gained to energy spent
"""
from __future__ import annotations

from typing import Dict, List, Tuple
import numpy as np

BEHAVIOR_NAMES = [
    "Grazer",
    "Predator",
    "Explorer",
    "Scavenger",
    "Ambusher",
    "Territorial",
    "Social",
    "Erratic",
    "Energy Efficient",
]

B_GRAZER, B_PREDATOR, B_EXPLORER, B_SCAVENGER, B_AMBUSHER, B_TERRITORIAL, B_SOCIAL, B_ERRATIC, B_EFFICIENT = range(len(BEHAVIOR_NAMES))

BEHAVIOR_COLORS = [
    (80, 220, 100),   # Grazer: vibrant emerald
    (240, 70, 70),    # Predator: crimson
    (70, 180, 240),   # Explorer: cyan/sky
    (180, 140, 70),   # Scavenger: ochre
    (160, 60, 200),   # Ambusher: purple
    (240, 170, 50),   # Territorial: amber/orange
    (70, 230, 200),   # Social: turquoise
    (220, 120, 200),  # Erratic: pink/magenta
    (120, 240, 150),  # Energy Efficient: light green
]


def classify_organism_stats(
    age: float,
    distance: float,
    max_disp: float,
    home_t: float,
    forage_t: float,
    pursuit_t: float,
    flee_t: float,
    social_t: float,
    still_t: float,
    turn_abs: float,
    turn_sq: float,
    ambush: float,
    food_plant: float,
    food_meat: float,
    food_carrion: float,
    dmg_dealt: float,
    attacks: int,
    gained: float,
    spent: float,
) -> Tuple[int, float, List[str]]:
    """Compute behavioral classification, confidence score, and explainable reasons."""
    safe_age = max(age, 1.0)
    forage_rate = forage_t / safe_age
    pursuit_rate = pursuit_t / safe_age
    social_rate = social_t / safe_age
    still_rate = still_t / safe_age
    home_rate = home_t / safe_age
    turn_rate = turn_abs / safe_age
    turn_var = (turn_sq / safe_age) - (turn_rate ** 2)
    turn_var = max(0.0, turn_var)
    speed_avg = distance / safe_age
    tot_food = food_plant + food_meat + food_carrion
    meat_ratio = food_meat / max(tot_food, 1.0)
    carrion_ratio = food_carrion / max(tot_food, 1.0)
    efficiency = gained / max(spent, 1.0)
    attack_rate = attacks / safe_age

    scores = np.zeros(len(BEHAVIOR_NAMES), dtype=np.float32)
    reasons_map: Dict[int, List[str]] = {i: [] for i in range(len(BEHAVIOR_NAMES))}

    # 1. Predator
    pred_score = 0.0
    if attacks > 0 or pursuit_rate > 0.08 or meat_ratio > 0.3:
        if attacks > 0:
            pred_score += min(1.0, attack_rate * 4.0) * 0.35
            reasons_map[B_PREDATOR].append("Frequent attack actions")
        if pursuit_rate > 0.05:
            pred_score += min(1.0, pursuit_rate * 3.5) * 0.30
            reasons_map[B_PREDATOR].append("Persistent target pursuit")
        if meat_ratio > 0.25:
            pred_score += meat_ratio * 0.25
            reasons_map[B_PREDATOR].append("Primary diet is other organisms (meat)")
        if dmg_dealt > 10.0:
            pred_score += 0.10
            reasons_map[B_PREDATOR].append("Inflicted significant combat damage")
    scores[B_PREDATOR] = pred_score

    # 2. Ambusher
    amb_score = 0.0
    if ambush > 0 or (still_rate > 0.25 and attacks > 0):
        if ambush > 0:
            amb_score += min(1.0, (ambush / max(attacks, 1)) * 1.2) * 0.45
            reasons_map[B_AMBUSHER].append(f"Recorded {int(ambush)} surprise ambush strike(s)")
        if still_rate > 0.25:
            amb_score += min(1.0, still_rate * 2.0) * 0.35
            reasons_map[B_AMBUSHER].append("Spends long stationary waiting periods")
        if attacks > 0 and speed_avg < 25.0:
            amb_score += 0.20
            reasons_map[B_AMBUSHER].append("Low cruising speed between strikes")
    scores[B_AMBUSHER] = amb_score

    # 3. Scavenger
    scav_score = 0.0
    if carrion_ratio > 0.2 or food_carrion > 5.0:
        scav_score += min(1.0, carrion_ratio * 1.5) * 0.6
        reasons_map[B_SCAVENGER].append("High preference for carrion / corpses")
        if attacks == 0:
            scav_score += 0.25
            reasons_map[B_SCAVENGER].append("Zero predatory combat initiates")
        if forage_rate > 0.1:
            scav_score += 0.15
            reasons_map[B_SCAVENGER].append("Active opportunist searching")
    scores[B_SCAVENGER] = scav_score

    # 4. Grazer
    graz_score = 0.0
    if food_plant > 0 or forage_rate > 0.08:
        if tot_food > 0 and (food_plant / tot_food) > 0.6:
            graz_score += 0.45
            reasons_map[B_GRAZER].append("Strict herbivore plant consumption")
        if forage_rate > 0.08:
            graz_score += min(1.0, forage_rate * 2.5) * 0.35
            reasons_map[B_GRAZER].append("High sustained foraging activity")
        if attacks == 0 and dmg_dealt == 0:
            graz_score += 0.20
            reasons_map[B_GRAZER].append("Completely non-violent behavior")
    scores[B_GRAZER] = graz_score

    # 5. Explorer
    exp_score = 0.0
    if max_disp > 200.0 or distance > 1000.0:
        if max_disp > 300.0:
            exp_score += min(1.0, max_disp / 800.0) * 0.45
            reasons_map[B_EXPLORER].append(f"High world displacement ({max_disp:.0f} units from origin)")
        if speed_avg > 30.0:
            exp_score += min(1.0, speed_avg / 60.0) * 0.35
            reasons_map[B_EXPLORER].append("Sustained high roaming velocity")
        if home_rate < 0.25:
            exp_score += 0.20
            reasons_map[B_EXPLORER].append("Rarely returns to home spawn region")
    scores[B_EXPLORER] = exp_score

    # 6. Territorial
    terr_score = 0.0
    if home_rate > 0.55 and safe_age > 10.0:
        terr_score += min(1.0, home_rate) * 0.50
        reasons_map[B_TERRITORIAL].append(f"Stationed {home_rate*100:.0f}% of life in home perimeter")
        if max_disp < 150.0:
            terr_score += 0.30
            reasons_map[B_TERRITORIAL].append("Confined movement radius (< 150 units)")
        if attacks > 0:
            terr_score += 0.20
            reasons_map[B_TERRITORIAL].append("Defends local space from encroaching organisms")
    scores[B_TERRITORIAL] = terr_score

    # 7. Social
    soc_score = 0.0
    if social_rate > 0.4:
        soc_score += min(1.0, social_rate) * 0.55
        reasons_map[B_SOCIAL].append(f"Spends {social_rate*100:.0f}% of time clustered near conspecifics")
        if attacks == 0:
            soc_score += 0.25
            reasons_map[B_SOCIAL].append("Peaceful proximity without aggression")
        if speed_avg > 15.0:
            soc_score += 0.20
            reasons_map[B_SOCIAL].append("Coordinates collective movement with peers")
    scores[B_SOCIAL] = soc_score

    # 8. Erratic
    err_score = 0.0
    if turn_var > 0.6 or (turn_rate > 1.8 and speed_avg > 15.0):
        if turn_var > 0.5:
            err_score += min(1.0, turn_var / 2.0) * 0.55
            reasons_map[B_ERRATIC].append("High angular variance and abrupt heading changes")
        if turn_rate > 1.5:
            err_score += min(1.0, turn_rate / 3.0) * 0.45
            reasons_map[B_ERRATIC].append("Continuous rapid course corrections")
    scores[B_ERRATIC] = err_score

    # 9. Energy Efficient
    eff_score = 0.0
    if efficiency > 1.4 and safe_age > 8.0:
        eff_score += min(1.0, (efficiency - 1.0) / 1.5) * 0.60
        reasons_map[B_EFFICIENT].append(f"Optimal energy balance: {efficiency:.2f}x gain/expenditure")
        if still_rate > 0.3:
            eff_score += 0.25
            reasons_map[B_EFFICIENT].append("Conserves calories when food is not directly adjacent")
        if forage_rate > 0.15:
            eff_score += 0.15
            reasons_map[B_EFFICIENT].append("Targeted high-yield nutrient acquisition")
    scores[B_EFFICIENT] = eff_score

    # Determine winning role
    best_idx = int(np.argmax(scores))
    best_score = float(scores[best_idx])

    if best_score < 0.20:
        # Default baseline if still very young or ambiguous
        if forage_rate >= pursuit_rate:
            best_idx = B_GRAZER
            best_score = 0.40
            reasons = ["Early stage foraging baseline", "Balanced exploration and sustenance intake"]
        else:
            best_idx = B_EXPLORER
            best_score = 0.35
            reasons = ["Initial exploratory wander", "Assessing surrounding territory"]
    else:
        reasons = reasons_map[best_idx]
        if not reasons:
            reasons = ["Consistent behavioral profile across sampled metrics"]

    confidence = min(0.99, max(0.30, best_score))
    return best_idx, confidence, reasons


def classify_population(eco) -> Tuple[np.ndarray, np.ndarray]:
    """Vectorized / batch classification of all living organisms."""
    s = eco.org
    n = s.n
    if n == 0:
        eco.behavior_labels = np.zeros(0, dtype=np.int16)
        eco.behavior_conf = np.zeros(0, dtype=np.float32)
        return eco.behavior_labels, eco.behavior_conf

    age = np.maximum(s.age[:n], 1.0)
    food_p = s.food_plant[:n]
    food_m = s.food_meat[:n]
    food_c = s.food_carrion[:n]
    tot_f = food_p + food_m + food_c + 1e-6
    meat_r = food_m / tot_f
    carr_r = food_c / tot_f
    atks = s.attacks[:n]
    pursuit = s.pursuit_t[:n] / age
    forage = s.forage_t[:n] / age
    disp = s.max_disp[:n]
    home_r = s.home_t[:n] / age
    social_r = s.social_t[:n] / age
    turn_r = s.turn_abs[:n] / age
    eff = s.gained[:n] / np.maximum(s.spent[:n], 1.0)
    ambush = s.ambush[:n]
    still_r = s.still_t[:n] / age

    # Default to Grazer
    labels = np.full(n, B_GRAZER, dtype=np.int16)
    conf = np.full(n, 0.50, dtype=np.float32)

    # Explorer
    is_exp = (disp > 280.0) & (home_r < 0.3)
    labels[is_exp] = B_EXPLORER
    conf[is_exp] = np.clip(disp[is_exp] / 600.0, 0.5, 0.95)

    # Social
    is_soc = social_r > 0.45
    labels[is_soc] = B_SOCIAL
    conf[is_soc] = np.clip(social_r[is_soc], 0.5, 0.95)

    # Territorial
    is_terr = (home_r > 0.6) & (disp < 160.0) & (age > 10.0)
    labels[is_terr] = B_TERRITORIAL
    conf[is_terr] = np.clip(home_r[is_terr], 0.5, 0.95)

    # Scavenger
    is_scav = (carr_r > 0.3) | ((food_c > 10.0) & (atks == 0))
    labels[is_scav] = B_SCAVENGER
    conf[is_scav] = np.clip(carr_r[is_scav] * 1.5, 0.5, 0.95)

    # Erratic
    is_err = turn_r > 2.2
    labels[is_err] = B_ERRATIC
    conf[is_err] = np.clip(turn_r[is_err] / 3.0, 0.5, 0.95)

    # Ambusher
    is_amb = (ambush > 0) | ((still_r > 0.3) & (atks > 0))
    labels[is_amb] = B_AMBUSHER
    conf[is_amb] = 0.85

    # Predator (higher priority than general wanderers)
    is_pred = (atks > 0) | (pursuit > 0.1) | (meat_r > 0.35)
    labels[is_pred] = B_PREDATOR
    conf[is_pred] = np.clip(np.maximum(meat_r[is_pred], pursuit[is_pred] * 3.0), 0.55, 0.98)

    # Energy Efficient (if exceptional efficiency and not in active predatory frenzy)
    is_eff = (eff > 1.8) & (age > 12.0) & (~is_pred)
    labels[is_eff] = B_EFFICIENT
    conf[is_eff] = np.clip((eff[is_eff] - 1.0) / 1.5, 0.6, 0.95)

    eco.behavior_labels = labels
    eco.behavior_conf = conf
    return labels, conf


def explain_organism(eco, idx: int) -> dict:
    """Full explainable behavior report for a single organism inspector."""
    s = eco.org
    if idx < 0 or idx >= s.n:
        return {"name": "Unknown", "confidence": 0.0, "reasons": ["Organism not found"]}

    role_id, confidence, reasons = classify_organism_stats(
        age=float(s.age[idx]),
        distance=float(s.distance[idx]),
        max_disp=float(s.max_disp[idx]),
        home_t=float(s.home_t[idx]),
        forage_t=float(s.forage_t[idx]),
        pursuit_t=float(s.pursuit_t[idx]),
        flee_t=float(s.flee_t[idx]),
        social_t=float(s.social_t[idx]),
        still_t=float(s.still_t[idx]),
        turn_abs=float(s.turn_abs[idx]),
        turn_sq=float(s.turn_sq[idx]),
        ambush=float(s.ambush[idx]),
        food_plant=float(s.food_plant[idx]),
        food_meat=float(s.food_meat[idx]),
        food_carrion=float(s.food_carrion[idx]),
        dmg_dealt=float(s.dmg_dealt[idx]),
        attacks=int(s.attacks[idx]),
        gained=float(s.gained[idx]),
        spent=float(s.spent[idx]),
    )

    return {
        "id": role_id,
        "name": BEHAVIOR_NAMES[role_id],
        "confidence": confidence,
        "reasons": reasons,
        "color": BEHAVIOR_COLORS[role_id],
    }
