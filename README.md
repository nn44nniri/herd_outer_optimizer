# Herd Optimizer for LiGAPS-Beef

## Summary

This project builds an outer-loop optimizer around **LiGAPS-Beef**, a mechanistic livestock model for beef production systems. LiGAPS-Beef was developed to simulate **potential** and **feed-limited** beef production, and to identify the **defining** and **limiting** biophysical factors that affect growth and production in cattle. The model integrates three connected sub-models: **thermoregulation**, **feed intake and digestion**, and **energy and protein utilisation**. Its main inputs include breed-specific parameters, generic cattle parameters, physical and chemical parameters, daily weather data, feed characteristics, diet composition, and feed availability. Its main outputs include feed intake, beef production, total body weight, and feed efficiency, along with the factors that define or limit growth. :contentReference[oaicite:0]{index=0} :contentReference[oaicite:1]{index=1}

The purpose of this repository is to turn LiGAPS-Beef into a decision-support and optimization workflow in which a Bayesian optimizer repeatedly calls the simulator, learns the response surface of livestock production outcomes, and identifies climate-management regimes that best balance **high meat production** and **low resource consumption**. The optimizer also supports an **operation phase**, in which the trained model compares the **realized past seasonal regime** with an **optimal full-season target**, and estimates the **required future cumulative regime** needed to remain close to the desired optimum. This framing is consistent with modern multi-objective Bayesian optimization under partial information. :contentReference[oaicite:2]{index=2} :contentReference[oaicite:3]{index=3}

---

## Project Contents

This repository contains two conceptual layers:

1. **LiGAPS-Beef simulator layer**
   - Mechanistic herd/animal simulation code derived from `LiGAPSBeef20180301_herd_worked.py`
   - Climate-history input data such as `FRACHA19982012.csv`
   - Case definitions for genotype, feeding strategy, housing, and slaughter-weight conditions. :contentReference[oaicite:4]{index=4} :contentReference[oaicite:5]{index=5}

2. **Outer-loop optimizer layer**
   - Synchronization code connecting the optimizer to LiGAPS-Beef
   - Training workflow using multi-objective Bayesian optimization
   - Operation workflow for time-conditioned recommendation of remaining cumulative climate-management targets
   - JSON export for archives and operation reports
   - PNG graphical report generation from training and operation JSON outputs

The repository is intended for research use in livestock systems analysis, yield-gap analysis, and climate-management optimization for beef production systems. LiGAPS-Beef itself was designed for researchers, with the expectation that results can then be made accessible to practitioners and policy makers. :contentReference[oaicite:6]{index=6}

---

## Background

LiGAPS-Beef stands for **Livestock simulator for Generic analysis of Animal Production Systems – Beef cattle**. It is based on concepts of production ecology and was developed to simulate interactions among **cattle genotype, climate, feed quality, and feed quantity**. The model was illustrated for Charolais and ¾ Brahman × ¼ Shorthorn cattle in France and Australia. A major innovation of the model is that it does not only simulate production outcomes, but also identifies **when** and **which** biophysical factors define or limit growth, such as genotype, heat stress, cold stress, digestion-capacity limitation, energy deficiency, and protein deficiency. :contentReference[oaicite:7]{index=7} :contentReference[oaicite:8]{index=8}

This project extends that capability by embedding LiGAPS-Beef inside a Bayesian optimization loop. The optimizer does not replace the mechanistic simulator. Instead, it repeatedly evaluates LiGAPS-Beef, builds a surrogate model of its response, and uses acquisition strategies from multi-objective Bayesian optimization to search for high-quality trade-offs across a feasible state space. :contentReference[oaicite:9]{index=9}

---

## Core Problem

The core problem addressed here is:

> How can we identify climate-management regimes for a beef production system that maximize meat production while minimizing resource consumption, and then translate those optimal full-season regimes into actionable future cumulative targets during operation?

This broad problem naturally splits into two subproblems:

### 1. Training problem
During training, the optimizer repeatedly proposes candidate decision vectors, runs LiGAPS-Beef, and observes resulting production outcomes. The goal is to learn a surrogate model and estimate a **Pareto frontier** of trade-offs between competing objectives such as beef production and resource-efficiency measures. This is a standard **multi-objective optimization** setting, where there is generally no single best design, but rather a set of non-dominated trade-off solutions. :contentReference[oaicite:10]{index=10}

### 2. Operation problem
During operation, the optimizer is asked a more specific question:

> Given the current day in the season, the climatic regime realized so far, and the archive of trained optimal solutions, what cumulative climate-management regime is still needed from now until the season end to remain close to the selected optimum?

This is a **time-conditioned**, **partial-information** decision problem, because only part of the season has already been observed and the future must be inferred relative to an end-of-season target. The Hypervolume Knowledge Gradient literature explicitly motivates this kind of Bayesian optimization with incomplete information. :contentReference[oaicite:11]{index=11} :contentReference[oaicite:12]{index=12}

---

## Research Questions

This repository is designed around the following practical and scientific questions:

### Q1. Can LiGAPS-Beef be used as the simulation engine for outer-loop optimization?
**Answer:** Yes. LiGAPS-Beef already simulates the essential interactions among genotype, climate, feed quality, and feed quantity, and produces outcomes such as feed intake, beef production, body weight, and feed efficiency. These characteristics make it suitable as a mechanistic black-box simulator inside a Bayesian optimization workflow. :contentReference[oaicite:13]{index=13}

### Q2. What should be optimized?
**Answer:** At minimum, the optimizer should search for trade-offs between **beef production** and **resource consumption** or a resource-efficiency proxy. In this repository, the optimization target is framed as a multi-objective problem in which high meat production is desirable while climate-management and feed-related resource use should be kept as low as feasible. This is aligned with Pareto-based multi-objective optimization. :contentReference[oaicite:14]{index=14}

### Q3. Why is a multi-objective optimizer needed?
**Answer:** Because maximizing meat production and minimizing resource use are competing objectives. Multi-objective optimization searches for a set of non-dominated solutions rather than a single scalar optimum, allowing the decision-maker to choose among trade-offs. Hypervolume-based quality measures and Pareto-front inference are standard in this setting. :contentReference[oaicite:15]{index=15} :contentReference[oaicite:16]{index=16}

### Q4. Why use Bayesian optimization instead of exhaustive simulation?
**Answer:** Because LiGAPS-Beef is an expensive mechanistic simulator, and the search space includes multiple interacting variables. Bayesian optimization is designed for expensive black-box problems in which evaluations are limited and surrogate-based candidate selection is more efficient than brute-force search. :contentReference[oaicite:17]{index=17}

### Q5. Why is operation mode different from training mode?
**Answer:** In operation mode, the system no longer searches from an empty state. It receives the **current position in the season**, the **realized climatic history so far**, and must infer the **remaining cumulative target** required until the end of the season. This is a partial-information decision-support problem rather than a pure offline design problem. :contentReference[oaicite:18]{index=18}

---

## LiGAPS-Beef Inputs and Outputs

### Main Inputs
According to the LiGAPS-Beef model description, the main inputs are:
- breed-specific parameters
- generic cattle parameters
- physical and chemical parameters
- daily weather data
- feed characteristics
- diet composition
- feed availability. :contentReference[oaicite:19]{index=19}

In the provided code version, cases also include:
- genotype
- location / weather file
- housing regime
- diet number
- feed quantities
- slaughter-weight setting. :contentReference[oaicite:20]{index=20} :contentReference[oaicite:21]{index=21}

### Main Outputs
The main outputs of LiGAPS-Beef are:
- feed intake
- beef production
- total body weight
- feed efficiency
- defining and limiting factors for growth. :contentReference[oaicite:22]{index=22}

The model was explicitly designed to identify the biophysical factors that define or limit growth during different periods. This is central for yield-gap analysis and for suggesting improvement options. :contentReference[oaicite:23]{index=23}

---

## Optimization Formalisms

### 1. Mechanistic simulation formalism
LiGAPS-Beef is deterministic and uses a daily time step. It integrates:
- a thermoregulation sub-model
- a feed intake and digestion sub-model
- an energy and protein utilisation sub-model. :contentReference[oaicite:24]{index=24} :contentReference[oaicite:25]{index=25}

These sub-models exchange energy and protein flows and respond to genotype, climate, feed quality, and feed quantity. :contentReference[oaicite:26]{index=26}

### 2. Multi-objective optimization formalism
The optimizer treats the simulator as a vector-valued function:

\[
f(x) = \left(f^{(1)}(x), \dots, f^{(M)}(x)\right)
\]

where \(x\) is a climate-management decision vector and the outputs are competing objectives such as meat production and resource-efficiency metrics. In multi-objective optimization, the goal is not usually a single best solution, but the **Pareto set** of non-dominated designs and its image, the **Pareto frontier**. :contentReference[oaicite:27]{index=27}

A solution is Pareto-optimal if no other solution is at least as good in all objectives and strictly better in at least one. The quality of the Pareto frontier is often measured with the **hypervolume indicator**. :contentReference[oaicite:28]{index=28}

### 3. Hypervolume-based Bayesian optimization formalism
The Hypervolume Knowledge Gradient paper formulates one-step look-ahead optimization of hypervolume and proposes **HV-KG** as an acquisition strategy that can explicitly condition on incomplete information. The idea is to select the next candidate by maximizing the expected increase in hypervolume of the inferred Pareto frontier after receiving new data. :contentReference[oaicite:29]{index=29}

Conceptually:

$$
\alpha_{\mathrm{HV-KG}}(x) = \mathbb{E}\left[ \max_{X' \subseteq \mathcal{X}} HV\big(\mu(X' \mid \mathcal{D}_x) \big) - \psi^* \right] $$

where $(\psi^*)$ is the current best hypervolume under the current data $(\mathcal{D}\)$, and $(\mathcal{D}_x)$ denotes the data after a new observation at candidate $(x)$. :contentReference[oaicite:30]{index=30}

### 4. Partial-information / operation-phase formalism
In this repository, the operation phase compares:
- the **optimal full-season cumulative regime**
- the **realized past regime**
- the **required future regime**

The remaining target is interpreted as the gap:

$required future regime = $optimal full-season target - realized past regime$

subject to feasibility, season length remaining, and the trained optimizer archive. This makes the operation phase a time-conditioned recommendation problem under partial information. That framing is consistent with the HV-KG motivation for MOBO under incomplete information. :contentReference[oaicite:31]{index=31}

---

## Approach

### Approach A. Use LiGAPS-Beef as the black-box simulator
LiGAPS-Beef provides the physically grounded response of the livestock system to climate, feed, and genotype inputs. This preserves the biological structure and allows optimization without replacing the mechanistic basis of the model. :contentReference[oaicite:32]{index=32}

### Approach B. Build an outer-loop Bayesian optimizer
The optimizer repeatedly:
1. proposes a candidate decision vector,
2. synchronizes that candidate with LiGAPS-Beef inputs,
3. runs the simulator,
4. records the resulting objectives,
5. updates a surrogate model,
6. proposes the next candidate using a multi-objective acquisition strategy.

This is the standard outer-loop expensive black-box optimization pattern. :contentReference[oaicite:33]{index=33}

### Approach C. Maintain an optimization archive
All evaluated design points and their objective values are stored in an archive. This archive is then used in both:
- training diagnostics and plotting,
- operation-mode target selection.

### Approach D. Provide operation-mode gap analysis
In the operation phase, the optimizer:
1. loads the trained archive,
2. loads the climate history,
3. identifies the current time position in the season,
4. computes the realized past cumulative regime,
5. selects a target from the archive,
6. estimates the required future cumulative regime,
7. exports the result as JSON and visual reports.

### Approach E. Provide report generation from JSON
This repository generates graphical reports from the JSON outputs of both training and operation phases. Comparative plots are used to visualize:
- optimal full-season regime,
- realized past regime,
- required future regime,
- remaining gap,
- progress toward optimum.

---

## Results from the Source Literature

### Model illustration results
The original LiGAPS-Beef paper reported that:
- Charolais bulls had the highest feed efficiency in France
- ¾ Brahman × ¼ Shorthorn bulls had the highest feed efficiency in Australia
- cattle breeds adapted to regional climate conditions achieved higher feed efficiency in those conditions
- the model successfully identified expected defining and limiting factors such as cold stress in France winters, heat stress in France summers and Australia, and energy deficiency under feed-limited cases. :contentReference[oaicite:34]{index=34}

The herd-level illustration also showed large differences in herd feed efficiency, herd beef production, and slaughter weight across cases and climates. :contentReference[oaicite:35]{index=35}

### Sensitivity-analysis results
The sensitivity-analysis paper found that:
- several influential parameters are related to net-energy maintenance requirements and heat release
- thermoregulation becomes more important under feed-quality-limited production
- the one-at-a-time sensitivity design was informative but limited, and joint effects and global sensitivity remain important future directions. :contentReference[oaicite:36]{index=36}

### Model-evaluation results
The evaluation paper concluded that LiGAPS-Beef performed reasonably well across systems in Australia, Uruguay, and the Netherlands. Reported aggregate evaluation metrics included:
- **MAE = 137 g live weight/day**
- **RMSE = 170 g live weight/day**
which corresponded to about **20.1%** and **25.0%** of mean measured average daily gain, respectively. The defining and limiting factors identified by the model, especially heat stress, energy deficiency, and protein deficiency, were broadly consistent with experiments. :contentReference[oaicite:37]{index=37} :contentReference[oaicite:38]{index=38}

### Optimizer-method results from HV-KG literature
The HV-KG paper reported substantial performance gains over several state-of-the-art MOBO baselines on synthetic and real-world multi-fidelity and decoupled problems, and positioned HV-KG as particularly useful when optimization must proceed with incomplete information. :contentReference[oaicite:39]{index=39} :contentReference[oaicite:40]{index=40}

---

## Conclusions

LiGAPS-Beef provides a strong mechanistic basis for simulating beef production systems and identifying the defining and limiting biophysical factors that shape yield gaps. The source literature concluded that the model meets the aim it was developed for and can be used as a tool to assess potential and feed-limited production after appropriate sensitivity analysis and evaluation. :contentReference[oaicite:41]{index=41} :contentReference[oaicite:42]{index=42}

The optimizer built in this repository extends LiGAPS-Beef from a simulation and analysis tool into a **decision-support optimization framework**. Its main contribution is to:
- search for Pareto-efficient full-season climate-management targets during training,
- then translate those targets into time-conditioned future cumulative recommendations during operation.

This makes the repository suitable for research on livestock yield-gap analysis, sustainable intensification, and climate-aware herd-management planning. The approach is especially relevant when the future must be managed under partial seasonal information and limited simulation budget. :contentReference[oaicite:43]{index=43} :contentReference[oaicite:44]{index=44}

---

## Current Scope and Limitations

- The quality of optimization depends on the validity domain of LiGAPS-Beef. The model-evaluation paper notes that further evaluation across additional breeds, climates, and feeding strategies is still needed to better delineate that validity domain. :contentReference[oaicite:45]{index=45}
- The sensitivity-analysis paper notes that one-at-a-time sensitivity analysis does not capture all interactions, and that global sensitivity analysis would be a valuable next step. :contentReference[oaicite:46]{index=46}
- Operation-mode recommendations are only as strong as the trained archive, surrogate quality, and chosen objective definitions.
- Practical use for management decisions should eventually be combined with economics, environmental legislation, animal welfare, and social constraints, as highlighted by the LiGAPS-Beef evaluation paper. :contentReference[oaicite:47]{index=47}

---

## References

- Bellocchi G, Rivington M, Donatelli M and Matthews K. 2010. Validation of biophysical models: issues and methodologies. A review. *Agronomy for Sustainable Development* 30, 109–130. :contentReference[oaicite:48]{index=48}
- Daulton S, Balandat M and Bakshy E. qNEHVI and related hypervolume-based MOBO methods are discussed as baselines in the HV-KG paper. :contentReference[oaicite:49]{index=49}
- McGovern REB and Bruce JM. 2000. Thermal balance and livestock thermoregulation foundations used by LiGAPS-Beef. :contentReference[oaicite:50]{index=50}
- Pianosi F et al. 2016. Sensitivity-analysis methodology cited by the LiGAPS-Beef sensitivity paper. :contentReference[oaicite:51]{index=51}
- Thornley JHM and France J. 2007. *Mathematical models in agriculture: quantitative methods for the plant, animal and ecological sciences.* CABI. :contentReference[oaicite:52]{index=52}
- Turnpenny JR, McArthur AJ, Clark JA and Wathes CM. 2000. Thermal balance of livestock. *Agricultural and Forest Meteorology* 101, 15–27. :contentReference[oaicite:53]{index=53}
- Turnpenny JR, Wathes CM, Clark JA and McArthur AJ. 2000. Thermal balance of livestock 2. Applications of a parsimonious model. *Agricultural and Forest Meteorology* 101, 29–52. :contentReference[oaicite:54]{index=54}
- Van de Ven GWJ, de Ridder N, van Keulen H and van Ittersum MK. 2003. Concepts in production ecology for analysis and design of animal and plant-animal systems. *Agricultural Systems* 76, 507–525. :contentReference[oaicite:55]{index=55}
- Van der Linden A, Oosting SJ, Van de Ven GWJ, De Boer IJM and Van Ittersum MK. 2015. A framework for quantitative analysis of livestock systems using theoretical concepts of production ecology. *Agricultural Systems* 139, 100–109. :contentReference[oaicite:56]{index=56}
- Van der Linden A, Van de Ven GWJ, Oosting SJ, Van Ittersum MK and De Boer IJM. 2018a. *LiGAPS-Beef, a mechanistic model to explore potential and feed-limited beef production 1. Model description and illustration.* :contentReference[oaicite:57]{index=57}
- Van der Linden A, Van de Ven GWJ, Oosting SJ, Van Ittersum MK and De Boer IJM. 2018b. *LiGAPS-Beef, a mechanistic model to explore potential and feed-limited beef production 2. Sensitivity analysis and evaluation of sub-models.* :contentReference[oaicite:58]{index=58}
- Van der Linden A, Van de Ven GWJ, Oosting SJ, Van Ittersum MK and De Boer IJM. 2018c / companion evaluation paper. *LiGAPS-Beef, a mechanistic model to explore potential and feed-limited beef production 3. Model evaluation.* :contentReference[oaicite:59]{index=59}
- Van Ittersum MK, Cassman KG, Grassini P, Wolf J, Tittonell P and Hochman Z. 2013. Yield gap analysis with local to global relevance – a review. *Field Crops Research* 143, 4–17. :contentReference[oaicite:60]{index=60}
- Wu, Ament, Eriksson, Balandat, et al. *Hypervolume Knowledge Gradient: A Lookahead Approach for Multi-Objective Bayesian Optimization with Partial Information.* The paper motivates HV-KG for MOBO with incomplete information, decoupling, and multi-fidelity structure. :contentReference[oaicite:61]{index=61} :contentReference[oaicite:62]{index=62}