# Bayesian Network for Car Diagnostics

Probabilistic reasoning system using Bayesian Networks to diagnose automotive failures through causal inference and exact enumeration algorithms.

## 🎯 Project Overview

This project implements a Bayesian Network (BN) for automotive fault diagnosis using directed acyclic graphs (DAG) to model causal relationships between vehicle components. The system performs exact probabilistic inference to determine root causes of car failures based on observed symptoms.

## 🧠 Problem Domain

### Diagnostic Scenario
A car won't start, and the mechanic needs to identify the root cause among multiple potential failures:
- Battery problems (age, dead battery, battery flat)
- Charging system issues (alternator, fan belt, no charging)
- Fuel system problems (no gas)
- Observable symptoms (lights, gas gauge, car won't start)

### Bayesian Network Structure

```
                    Battery Age (ba)
                         ↓
                    Battery Dead (bd) ←─────┐
                         ↓                   │
    Alternator ──→ No Charging (nc) ────→ Battery Flat (bf)
    Broken (ab)         ↑                    ↓   ↓   ↓
                        │                    │   │   │
    Fan Belt ───────────┘                    ↓   ↓   ↓
    Broken (fb)                          Lights Gas  Car Won't
                                          (l)  Gauge Start (cws)
                                                (gg)    ↑
    No Gas (ng) ────────────────────────────────┴───────┘
```

### Nodes (10 Variables)
1. **ab** - Alternator Broken (root)
2. **fb** - Fan Belt Broken (root)
3. **ba** - Battery Age (root)
4. **ng** - No Gas (root)
5. **bd** - Battery Dead (depends on ba)
6. **nc** - No Charging (depends on ab, fb)
7. **bf** - Battery Flat (depends on bd, nc)
8. **l** - Lights (depends on bf)
9. **gg** - Gas Gauge (depends on bf, ng)
10. **cws** - Car Won't Start (depends on bf, ng)

## 📊 Probability Tables

### Prior Probabilities (Root Nodes)
| Node | P(Yes) | P(No) |
|------|--------|-------|
| Alternator Broken (ab) | 0.10 | 0.90 |
| Fan Belt Broken (fb) | 0.30 | 0.70 |
| Battery Age (ba) | 0.20 | 0.80 |
| No Gas (ng) | 0.20 | 0.80 |

### Conditional Probability Tables (CPTs)

#### Battery Dead | Battery Age
| ba | P(bd=yes) | P(bd=no) |
|----|-----------|----------|
| yes | 0.70 | 0.30 |
| no | 0.40 | 0.60 |

#### No Charging | Alternator, Fan Belt
| ab | fb | P(nc=yes) | P(nc=no) |
|----|----|-----------| ---------|
| yes | yes | 0.75 | 0.25 |
| yes | no | 0.40 | 0.60 |
| no | yes | 0.60 | 0.40 |
| no | no | 0.10 | 0.90 |

#### Battery Flat | Battery Dead, No Charging
| bd | nc | P(bf=yes) | P(bf=no) |
|----|----|-----------|----------|
| yes | yes | 0.95 | 0.05 |
| yes | no | 0.85 | 0.15 |
| no | yes | 0.80 | 0.20 |
| no | no | 0.10 | 0.90 |

## 🔬 Inference Results

### R2: Car Won't Start Given Battery Age and No Lights
```
Query: P(cws = yes | ba = yes, l = no)
Result: 0.15514 (15.5%)
```
**Interpretation**: If the battery is old and lights don't work, there's a 15.5% chance the car won't start.

### R3: Comparative Diagnosis
```
Query: P(cws = yes | ab = yes, l = no)
Result: Lower than R2
```
**Interpretation**: Battery age (ba=yes) is a more significant predictor than alternator failure (ab=yes) when lights don't work.

### R4: Prior Probability of Battery Flat
```
Query: P(bf = yes)
Result: 0.56181 (56.2%)
```
**Interpretation**: Without any observations, there's a 56% probability of battery being flat.

### R5: Diagnostic Significance Analysis

**Key Finding**: When comparing P(cws | ba, ¬l) vs P(cws | ab, ¬l):
- R2 probability (battery age) > R3 probability (alternator broken)
- **Conclusion**: Battery difficulties are more likely than alternator issues when lights fail

**Real-World Application**: 
- If car won't start AND lights don't work → Check battery first
- Battery age is the stronger predictor than alternator failure
- This guides efficient diagnostic workflows for mechanics

## 🛠️ Technologies Used

- **Python 3.x**
- **NetworkX** - Graph structure and DAG representation
- **NumPy** - Numerical computations
- **Matplotlib** - Network visualization
- **itertools** - Combinatorial enumeration

## 🏗️ Implementation Details

### 1. Network Construction
```python
import networkx as nx

G = nx.DiGraph()
G.add_nodes_from(["ab", "fb", "ba", "bd", "nc", "bf", "ng", "l", "gg", "cws"])
G.add_edges_from([
    ("ab", "nc"), ("fb", "nc"),
    ("ba", "bd"), ("bd", "bf"),
    ("nc", "bf"), ("bf", "l"),
    ("bf", "gg"), ("bf", "cws"),
    ("ng", "gg"), ("ng", "cws")
])
```

### 2. Enumeration Algorithm
```python
def enumerate_ask(query_var, evidence, hidden_vars):
    """
    Exact inference using full enumeration
    Sums over all possible assignments to hidden variables
    """
    for values in itertools.product([True, False], repeat=len(hidden_vars)):
        # Calculate joint probability
        joint_prob = calculate_joint(values, evidence)
        total += joint_prob
    
    return normalize(probabilities)
```

### 3. Probability Retrieval
```python
def get_prob(G, var, evidence):
    """
    Retrieve probability from CPT based on parent values
    Handles both prior probabilities and conditional probabilities
    """
    if var in root_nodes:
        return prior_probability(var, evidence)
    else:
        return conditional_probability(var, parents, evidence)
```

## 📦 Installation

```bash
# Clone the repository
git clone https://github.com/Devotion25/Bayesian-Network-Car-Diagnostics.git

# Install dependencies
pip install -r requirements.txt
```

## 🚀 Usage

```python
# Open Jupyter notebook
jupyter notebook bayesian_network.ipynb

# Run inference queries
result = enumerate_ask_cws_given_ba_l(G)
print(f"P(cws=yes|ba=yes,l=no) = {result:.5f}")
```

## 📊 Visualizations

The project includes:
1. **Network Graph**: Visual representation of causal relationships
2. **Probability Tables**: Comprehensive CPT documentation
3. **Inference Results**: Query results with interpretations
4. **Diagnostic Flow**: Decision tree for fault diagnosis

## 🔍 Key Insights

1. **Causal Reasoning**: BNs model cause-effect relationships, not just correlations
2. **Efficient Diagnosis**: Narrow down problem space using evidence
3. **Uncertainty Quantification**: Probabilities guide repair priorities
4. **Multi-Evidence Integration**: Combine multiple symptoms for better diagnosis
5. **Explainable AI**: Transparent reasoning process for diagnostic decisions

## 💡 Real-World Applications

This Bayesian Network approach can be applied to:
- **Automotive Diagnostics**: Fault finding in vehicles
- **Medical Diagnosis**: Disease identification from symptoms
- **Network Troubleshooting**: IT infrastructure fault detection
- **Industrial Maintenance**: Equipment failure prediction
- **Risk Assessment**: Financial and insurance risk modeling
- **Expert Systems**: Knowledge-based decision support

## ⚙️ Algorithm Complexity

### Enumeration Inference
- **Time Complexity**: O(2^n) where n = number of hidden variables
- **Space Complexity**: O(n) for recursion stack
- **Scalability**: Feasible for small networks (<20 nodes)

### Optimization Opportunities
1. **Variable Elimination**: Reduce exponential blowup
2. **Junction Tree**: Exact inference with better complexity
3. **Approximate Inference**: Sampling methods (MCMC, Gibbs)
4. **Compiled Inference**: Arithmetic circuits

## 🎓 Theoretical Foundations

### Bayes' Rule
```
P(A|B) = P(B|A) * P(A) / P(B)
```

### Chain Rule for Bayesian Networks
```
P(X₁, ..., Xₙ) = ∏ P(Xᵢ | Parents(Xᵢ))
```

### Conditional Independence
```
X ⊥ Y | Z ⟺ P(X|Y,Z) = P(X|Z)
```

## 🎓 Course Information

- **Course**: CSCE 5215 - Machine Learning
- **Institution**: University of North Texas
- **Semester**: Spring 2025
- **Topic**: Probabilistic Reasoning and Graphical Models

## 👨‍💻 Author

**Devotion Ekueku**
- GitHub: [@Devotion25](https://github.com/Devotion25)
- LinkedIn: [devotionekueku](https://www.linkedin.com/in/devotionekueku/)

## 📄 License

This project is for educational purposes as part of coursework at the University of North Texas.

## 🙏 Acknowledgments

- Russell & Norvig's "Artificial Intelligence: A Modern Approach"
- NetworkX library developers
- Course instructors at UNT
- Tutorial 4 documentation

## 📚 References

1. Pearl, J. (1988). Probabilistic Reasoning in Intelligent Systems. Morgan Kaufmann.
2. Koller, D., & Friedman, N. (2009). Probabilistic Graphical Models. MIT Press.
3. Russell, S., & Norvig, P. (2020). Artificial Intelligence: A Modern Approach (4th ed.).
4. Murphy, K. P. (2012). Machine Learning: A Probabilistic Perspective. MIT Press.
5. NetworkX Documentation: https://networkx.org/

## 🔗 Related Resources

- [Bayesian Networks Tutorial](https://www.bayesia.com/bayesian-networks)
- [pgmpy Library](https://pgmpy.org/) - Python library for Probabilistic Graphical Models
- [Probabilistic Reasoning Course Notes](http://www.cs.cmu.edu/~arielpro/15381f16/)
