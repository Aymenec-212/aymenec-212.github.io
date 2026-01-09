---
title: "De LSTM à xLSTM, puis au trading : un pipeline scientifique Deep Learning × Reinforcement Learning sur MNG.MA"
date: 2026-01-09 12:00:00 +0100
categories: [Project, Finance]
tags: [reinforcement learning, deep learning, time series, trading, xlstm, lstm, stable-baselines3]
# image:
#   path: /assets/img/mng/banner.png
#   alt: "MNG.MA — Deep Learning × RL"
---

# De LSTM à xLSTM, puis au trading : un pipeline scientifique Deep Learning × Reinforcement Learning sur MNG.MA

## Résumé
Cet article propose (i) une mise en perspective scientifique **LSTM vs xLSTM** à partir des motivations architecturales de la mémoire récurrente moderne, puis (ii) une étude expérimentale reproductible sur **MNG.MA** (Bourse de Casablanca), où un backbone séquentiel (**LSTM** ou **xLSTM**) est **pré-entraîné en supervisé** afin de produire une représentation latente \(h_t\) de l’état du marché, ensuite exploitée par un agent d’**apprentissage par renforcement** (**PPO, A2C, DQN**) dans un environnement **Long/Flat**.

L’évaluation présentée ici se concentre sur l’**out-of-sample (TEST)** et deux métriques financières : **ROI (%)** et **MDD (%)**.

---

# Partie I — Fondations : LSTM et xLSTM (rappels et implications)

## 1. LSTM : une mémoire scalaire contrôlée par portes
Le LSTM (Hochreiter & Schmidhuber, 1997) a été conçu pour stabiliser l’apprentissage des dépendances longues via une **mémoire explicite** \(c_t\) et des **portes** (gates) qui contrôlent l’écriture, l’oubli et la lecture. Dans la forme canonique :

\[
i_t = \sigma(W_i x_t + U_i h_{t-1} + b_i), \quad
f_t = \sigma(W_f x_t + U_f h_{t-1} + b_f), \quad
o_t = \sigma(W_o x_t + U_o h_{t-1} + b_o)
\]

\[
\tilde{c}_t = \tanh(W_c x_t + U_c h_{t-1} + b_c)
\]

\[
c_t = f_t \odot c_{t-1} + i_t \odot \tilde{c}_t, \quad
h_t = o_t \odot \tanh(c_t)
\]

**Interprétation** :  
- \(c_t\) sert de mémoire longue,  
- les portes sigmoïdes \(\sigma(\cdot)\in(0,1)\) modulent finement *ce qu’on conserve* et *ce qu’on écrit*.

### Limites structurantes (dans des signaux non-stationnaires)
Dans un contexte financier, la série temporelle est **non-stationnaire** (régimes) et bruitée. Deux difficultés récurrentes apparaissent :
1) **Révision de mémoire** : “mettre à jour” une information stockée dans \(c_t\) lorsque le régime change peut exiger des signaux d’oubli/écriture très cohérents ; en présence de bruit, cela devient difficile.  
2) **Échelle et parallélisation** : la récurrence impose un déroulement séquentiel, limitant le débit GPU quand les séquences deviennent longues ou quand le modèle est profond.

Ces limites motivent des extensions modernes de la mémoire récurrente.

---

## 2. xLSTM : deux cellules, un gating revisité, et des blocs résiduels empilables
xLSTM (*Extended LSTM*, 2024) conserve l’intuition fondamentale “mémoire + gating”, mais modifie deux aspects majeurs :

### 2.1 Gating exponentielle (principe)
Les portes sont basées sur des transformations de type **exponentiel normalisé** (au lieu de sigmoïdes simples). L’objectif est d’améliorer :
- la **dynamique de mise à jour** (capacité à réviser ce qui est stocké),
- le **flux de gradient** (éviter des saturations trop rigides),
- la flexibilité de l’échelle d’importance attribuée aux nouvelles informations.

Sans entrer dans l’implémentation exacte (qui varie selon sLSTM/mLSTM), l’idée est que le modèle dispose d’un mécanisme plus expressif pour pondérer et réviser.

### 2.2 Deux types de mémoire : sLSTM et mLSTM
xLSTM introduit deux cellules complémentaires :

- **sLSTM** : mémoire “scalaire” proche de l’esprit LSTM, mais avec gating exponentielle et mécanismes de mixing plus flexibles.  
- **mLSTM** : mémoire **matricielle**, mise à jour via des opérations structurées (type outer-product / covariance), conçue pour être **hautement parallélisable** et à capacité plus riche.

### 2.3 Macro-architecture : blocs résiduels empilables
Au-delà de la cellule, xLSTM est pensé comme **backbone moderne** : des blocs empilables avec normalisation, projections, connexions résiduelles. Cette conception rapproche l’entraînement de recettes “scalables” (stabilité, profondeur, accélération GPU), tout en conservant un inductif bias récurrent.

---

## 3. Pourquoi cette discussion compte pour le trading (et pour RL)
Dans notre pipeline, la question n’est pas “xLSTM prédit-il mieux un prix ?”, mais plutôt :

> *xLSTM produit-il une représentation latente plus informative et plus stable pour une politique RL ?*

En RL, la qualité de la représentation \(h_t\) conditionne directement :
- l’exploitabilité du signal (si \(h_t\) encode bien tendance/régime/risque),
- la stabilité d’apprentissage (réduction de bruit, gradients moins chaotiques),
- la robustesse out-of-sample.

C’est précisément l’hypothèse testée empiriquement dans la Partie II.

---

# Partie II — Notre étude : pipeline DL×RL sur MNG.MA (Bourse de Casablanca)

## 4. Données, splits, objectif expérimental
### 4.1 Source et cadre
Nous utilisons un historique de marché de **MNG.MA** issu de la **Bourse de Casablanca**, sur une période couvrant différents régimes (tendance haussière marquée, chocs de volatilité).

### 4.2 Découpage temporel
Le protocole suit une logique de généralisation temporelle :
- **Train** : apprentissage (supervisé + RL)
- **Validation** : sélection / early stopping
- **Test (out-of-sample)** : évaluation finale

Dans notre expérimentation, l’ensemble **TEST** contient **185 pas** et constitue la référence principale.

---

## 5. Feature engineering : transformer un prix en état exploitable
Les prix bruts sont non-stationnaires ; une politique RL sur prix bruts est rarement stable. Nous construisons donc un état \(X_t\) basé sur :
- **rendements** (log-returns / returns simples),
- indicateurs de **tendance** (moyennes glissantes, écarts normalisés),
- estimations de **volatilité** (volatilité glissante, amplitudes),
- signaux de **régime** (proxy de drawdown / stress, si utilisé).

Chaque observation correspond à une fenêtre temporelle de longueur \(L\) (lookback) :

\[
X_t \in \mathbb{R}^{L \times F}
\]

où \(F\) est le nombre de features.

**Rôle de cette étape** : fournir une entrée plus informative et mieux conditionnée numériquement, afin que le backbone et le RL apprennent des invariants plutôt que le niveau absolu des prix.

---

## 6. Pré-entraînement supervisé : apprendre un état latent \(h_t\)
Nous entraînons un backbone séquentiel \(f_\theta\) (LSTM ou xLSTM) sur une tâche supervisée (direction / rendement, éventuellement multi-tâche) afin d’apprendre une représentation latente :

\[
h_t = f_\theta(X_t) \in \mathbb{R}^{d}
\]

Ce vecteur \(h_t\) (souvent appelé **hidden state** ou **embedding**) a une fonction centrale :

- il résume le contexte récent,
- il encode les motifs utiles (tendance, retournement, volatilité),
- il sert d’interface entre **DL** et **RL**.

---

## 7. xLSTM comme extracteur de caractéristiques pour RL (pont DL → SB3)
Dans notre implémentation RL, le backbone pré-entraîné est réutilisé comme **feature extractor** au sens SB3 : l’observation brute \(X_t\) est projetée et encodée par xLSTM, puis on récupère la représentation finale \(z_t\) (typiquement le dernier état du lookback) :

\[
z_t = \mathrm{Norm}(\,h_{t,L}\,) \in \mathbb{R}^d
\]

Cette représentation \(z_t\) est ensuite consommée par la politique RL \(\pi(a_t\mid z_t)\).

Deux points méthodologiques importants :
1) **Warm start** : les poids du backbone sont initialisés par pré-entraînement supervisé.  
2) **Backbone gelé vs affiné** : on teste (Stage 1) un backbone gelé, puis (Stage 2) un affinage contrôlé.

Cette conception clarifie la séparation des rôles :
- le backbone apprend une **représentation de marché**,
- l’agent RL apprend une **politique de décision**.

---

## 8. Environnement RL : décision Long/Flat et fonction de récompense
Nous travaillons dans un environnement discret **Long/Flat** :
- \(a_t = 1\) (Long) : exposition au rendement du marché
- \(a_t = 0\) (Flat) : exposition nulle (cash)

La dynamique de capital suit, de façon simplifiée :

\[
\text{equity}_{t+1} = \text{equity}_t \cdot (1 + a_t r_t) - \text{costs}(\Delta a_t)
\]

avec :
- \(r_t\) : rendement à l’instant \(t\),
- \(\text{costs}(\Delta a_t)\) : pénalités de turnover / coûts de transaction.

Pour stabiliser l’apprentissage, l’entraînement RL est réalisé avec vectorisation (plusieurs environnements en parallèle), normalisation de récompense, et échelle de récompense durant l’entraînement (sans altérer l’évaluation financière finale).

---

# Partie III — Résultats (TEST uniquement) : ROI et MDD

Cette section présente uniquement les résultats **out-of-sample (TEST)**, sans référence à un benchmark. Deux métriques sont rapportées : **ROI** et **MDD**.

## 9. ROI (%) sur TEST (185 pas)
- **DQN (FAST) + xLSTM gelé** : **ROI = +36,43%**
- **DQN + LSTM gelé** : **ROI = +27,32%**
- **PPO + xLSTM gelé (Stage 1)** : **ROI = +14,53%**
- **PPO + xLSTM affiné (Stage 2)** : **ROI = +9,49%**
- **PPO + LSTM gelé (Stage 1)** : **ROI = +8,28%**
- **PPO + LSTM affiné (Stage 2)** : **ROI = +6,46%**
- **A2C + LSTM gelé** : **ROI = +5,20%**
- **A2C + xLSTM gelé** : **ROI = -26,80%**

**Lecture** : sur l’out-of-sample, **DQN** est l’agent qui obtient les rendements les plus élevés, en particulier lorsqu’il exploite une représentation **xLSTM**.

## 10. MDD (%) sur TEST
- **DQN (FAST) + xLSTM gelé** : **MDD = 4,98%**
- **DQN + LSTM gelé** : **MDD = 8,09%**
- **A2C + LSTM gelé** : **MDD = 9,06%**
- **PPO + xLSTM gelé** : **MDD = 11,02%**
- **PPO + xLSTM affiné (Stage 2)** : **MDD = 12,10%**
- **PPO + LSTM gelé (Stage 1)** : **MDD = 14,45%**
- **PPO + LSTM affiné (Stage 2)** : **MDD = 16,08%**
- **A2C + xLSTM gelé** : **MDD = 28,93%**

**Lecture** : la combinaison **DQN (FAST) + xLSTM** obtient simultanément :
- le **plus haut ROI**,
- le **plus faible MDD**,
ce qui indique un compromis rendement/risque particulièrement favorable sur TEST.

---

# Partie IV — Analyse critique et implications

## 11. Couplage représentation–agent :
Un enseignement méthodologique majeur est que la performance ne dépend pas seulement du backbone (LSTM vs xLSTM), mais du **couplage** :

\[
\text{Performance} \approx \text{Qualité de } h_t \ \times \ \text{Capacité de l’agent à l’exploiter}
\]

Dans notre cas :
- xLSTM fournit une représentation riche,
- **DQN** (value-based, actions discrètes) s’est montré particulièrement capable de l’exploiter dans un cadre Long/Flat.

## 12. Frozen vs fine-tuning :
Le fine-tuning peut améliorer l’adéquation in-sample, mais il expose au risque de spécialisation au régime de validation. Sur TEST, l’affinage (Stage 2) ne s’est pas traduit par une amélioration systématique des métriques, ce qui renforce l’idée que des backbones **gelés** peuvent offrir une représentation plus robuste pour la généralisation temporelle.

## 13. Pourquoi A2C peut échouer ici
A2C est on-policy et peut présenter une variance plus forte dans l’estimation de l’avantage, ce qui, combiné à :
- une représentation haute dimension,
- un signal de récompense bruité,
peut entraîner une instabilité d’apprentissage et une politique défavorable out-of-sample.

---

# Limites et perspectives
- **Non-stationnarité** : une seule découpe temporelle ne suffit pas à caractériser la robustesse ; une validation walk-forward renforcerait la conclusion.
- **Espace d’action** : Long/Flat simplifie la microstructure ; extensions possibles : short, position sizing continu, gestion du levier.
- **Généralisation** : réplication sur plusieurs titres/secteurs marocains pour tester la stabilité des conclusions.

---

# Références
- S. Hochreiter, J. Schmidhuber (1997). *Long Short-Term Memory*. Neural Computation, 9(8), 1735–1780.  
- M. Beck et al. (2024). *xLSTM: Extended Long Short-Term Memory*. arXiv:2405.04517.
- T. Kabbani and E. Duman, ‘Deep Reinforcement Learning Approach for Trading Automation in the Stock Market,’ IEEE Access, vol. 10,pp. 93564–93574, 2022.
- Sarlakifar, A., Asl, M. P., & Ghorbani, A. A. (2025). A Deep Reinforcement Learning Approach to Automated Stock Trading, using xLSTM Networks. arXiv preprint arXiv:2503.09655.

---

# Liens (sera publié très prochainement)
- Notebook Kaggle (reproductibilité) : …
- Dépôt GitHub : …
