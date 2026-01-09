---
title: "Un pipeline Deep Learning × Reinforcement Learning sur MNG.MA"
date: 2026-01-09 12:00:00 +0100
categories: [Project, Finance]
tags: [reinforcement learning, deep learning, time series, trading, xlstm, lstm]
math: true

---

# De LSTM à xLSTM, puis au trading : un pipeline DL×RL sur MNG.MA

## Résumé
Cet article propose (i) une mise en perspective scientifique **LSTM vs xLSTM** à partir des motivations architecturales de la mémoire récurrente moderne, puis (ii) une étude expérimentale reproductible sur **MNG.MA** (Bourse de Casablanca).

Nous y explorons un pipeline où un backbone séquentiel (**LSTM** ou **xLSTM**) est **pré-entraîné en supervisé** pour produire une représentation latente $h_t$ de l’état du marché, ensuite exploitée par un agent d’**apprentissage par renforcement** (**PPO, A2C, DQN**) dans un environnement **Long/Flat**.

L’évaluation se concentre sur l’**out-of-sample (TEST)** avec deux métriques : **ROI (%)** et **MDD (%)**.

---

# Partie I — Fondations : LSTM et xLSTM

## 1. LSTM : une mémoire scalaire contrôlée par portes
Le LSTM (Hochreiter & Schmidhuber, 1997) stabilise l’apprentissage des dépendances longues via une **mémoire explicite** $c_t$ et des **portes** (gates). Dans sa forme canonique :

$$
\begin{aligned}
i_t &= \sigma(W_i x_t + U_i h_{t-1} + b_i) \\
f_t &= \sigma(W_f x_t + U_f h_{t-1} + b_f) \\
o_t &= \sigma(W_o x_t + U_o h_{t-1} + b_o)
\end{aligned}
$$

La mise à jour de la mémoire se fait ainsi :

$$
\tilde{c}_t = \tanh(W_c x_t + U_c h_{t-1} + b_c)
$$

$$
c_t = f_t \odot c_{t-1} + i_t \odot \tilde{c}_t, \quad h_t = o_t \odot \tanh(c_t)
$$

**Interprétation** :  
- $c_t$ sert de mémoire longue.
- Les portes sigmoïdes $\sigma(\cdot) \in (0,1)$ modulent finement *ce qu’on conserve* et *ce qu’on écrit*.

### Limites structurantes
Dans un contexte financier, la série temporelle est **non-stationnaire**. Deux difficultés apparaissent :
1. **Révision de mémoire :** Mettre à jour $c_t$ lors d'un changement de régime est difficile en présence de bruit.
2. **Échelle :** La récurrence séquentielle limite la parallélisation GPU.

---

## 2. xLSTM : Gating exponentielle et blocs résiduels
xLSTM (*Extended LSTM*, 2024) conserve l’intuition “mémoire + gating” mais modifie deux aspects majeurs :

### 2.1 Gating exponentielle
Les portes utilisent une activation exponentielle normalisée (au lieu de sigmoïdes). Cela améliore la **dynamique de mise à jour** (révision plus franche de la mémoire) et le **flux de gradient**.

### 2.2 sLSTM et mLSTM
xLSTM introduit deux variantes de cellules :
- **sLSTM** : Mémoire scalaire avec gating exponentielle.
- **mLSTM** : Mémoire **matricielle** (covariance), hautement parallélisable et à capacité plus riche.

---

## 3. Pourquoi cette discussion compte pour le RL ?
La question n’est pas "xLSTM prédit-il mieux ?", mais :

> *xLSTM produit-il une représentation latente plus informative pour une politique RL ?*

En RL, la qualité de la représentation $h_t$ conditionne l’exploitabilité du signal et la stabilité des gradients.

---

# Partie II — Étude expérimentale sur MNG.MA

## 4. Données et Protocole
Nous utilisons l'historique de **MNG.MA** (Bourse de Casablanca).
- **Train** : Apprentissage (Supervisé + RL)
- **Validation** : Sélection de modèle
- **Test (OOS)** : Évaluation finale (**185 pas**)

## 5. Feature Engineering
L'état $X_t$ est construit sur une fenêtre de longueur $L$ :
- Rendements (Log-returns)
- Indicateurs de tendance et volatilité
- Signaux de régime

$$
X_t \in \mathbb{R}^{L \times F}
$$

## 6. Pré-entraînement & Feature Extractor
Un backbone $f_\theta$ (LSTM ou xLSTM) est pré-entraîné pour produire un embedding :

$$
h_t = f_\theta(X_t) \in \mathbb{R}^{d}
$$

Ensuite, cet embedding est projeté pour l'agent RL :

$$
z_t = \mathrm{Norm}(h_{t,L}) \in \mathbb{R}^d
$$

L’agent RL apprend la politique $\pi(a_t \mid z_t)$.

## 7. Environnement : Long/Flat
L'action est discrète :
- $a_t = 1$ (Long)
- $a_t = 0$ (Flat / Cash)

La dynamique de l'equity (simplifiée) :

$$
\text{equity}_{t+1} = \text{equity}_t \cdot (1 + a_t r_t) - \text{coûts}
$$

---

# Partie III — Résultats (TEST Out-of-Sample)

Voici les performances sur l'ensemble de TEST (inconnu durant l'entraînement).

## 9. ROI (%)
| Modèle | ROI (Test) |
| :--- | :--- |
| **DQN (FAST) + xLSTM (Gelé)** | **+36.43%** |
| DQN + LSTM (Gelé) | +27.32% |
| PPO + xLSTM (Gelé) | +14.53% |
| A2C + xLSTM (Gelé) | -26.80% |

> **Constat :** DQN est l’agent qui exploite le mieux la représentation xLSTM.

## 10. MDD (%) (Risque)
| Modèle | MDD (Test) |
| :--- | :--- |
| **DQN (FAST) + xLSTM (Gelé)** | **4.98%** |
| DQN + LSTM (Gelé) | 8.09% |
| PPO + xLSTM (Gelé) | 11.02% |

Le couple **DQN + xLSTM** offre le meilleur ratio rendement/risque.

---

# Partie IV — Analyse & Conclusion

## 11. Le couplage Représentation-Agent
La performance suit cette règle :
$$
\text{Perf} \approx \text{Qualité}(h_t) \times \text{Capacité}(\text{Agent})
$$
xLSTM fournit une représentation riche, et DQN (Value-based) s'avère plus robuste que les méthodes Policy-Gradient (PPO/A2C) sur ce dataset bruité.

## 12. Frozen vs Fine-tuning
Geler le backbone (Frozen) donne de meilleurs résultats OOS. Le fine-tuning tend à "overfitter" les régimes vus en entraînement, réduisant la robustesse face aux nouveaux régimes de marché.

---

# Références
1. S. Hochreiter, J. Schmidhuber (1997). *Long Short-Term Memory*.
2. M. Beck et al. (2024). *xLSTM: Extended Long Short-Term Memory*.
3. Sarlakifar et al. (2025). *RL Approach to Automated Stock Trading using xLSTM*.
