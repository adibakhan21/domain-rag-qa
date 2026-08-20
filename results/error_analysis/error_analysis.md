# Error analysis — `rag_hybrid_lora`

155/437 questions answered with F1 = 1.0 (35.5%); 282 failures categorised below.

## Failure distribution

| category | count | share of failures |
|---|---:|---:|
| `partial_match` | 193 | 68.4% |
| `retrieval_miss` | 19 | 6.7% |
| `chunk_boundary` | 1 | 0.4% |
| `reranker_demotion` | 37 | 13.1% |
| `context_truncated` | 12 | 4.3% |
| `reader_miss` | 20 | 7.1% |

## Mean F1 by question type

| type | n | mean F1 |
|---|---:|---:|
| what | 290 | 57.6 |
| how | 55 | 56.8 |
| other | 29 | 68.6 |
| which | 20 | 68.1 |
| when | 12 | 73.3 |
| why | 12 | 62.4 |
| where | 12 | 57.1 |
| is | 3 | 53.8 |
| who | 3 | 80.0 |
| are | 1 | 100.0 |

## Mean F1 by gold answer length

| length | n | mean F1 |
|---|---:|---:|
| 1-20 chars | 77 | 71.3 |
| 21-60 | 122 | 67.4 |
| 61-150 | 142 | 57.6 |
| 150+ | 96 | 42.6 |

## Representative failures

### `partial_match`

**Q:** What is the percentage reduction in pneumonia cases due to  vaccination?

- **gold:** Haemophilus influenzae type B conjugate vaccination in high-burden communities, the vaccination was associated with an 18% decrease in radiologic pneumonia [13] . Introduction of pneumococcal conjugate vaccination was associated with a 26% decrease in radiologic pneumonia in California between 1995 
- **predicted:** 15% to 65%
- F1 0.03 · answer_in_context 1 · groundedness 1.00
- gold chunks `['1571::8']` · retrieved `['1571::13', '1571::8', '1571::22', '1571::3', '1571::21']`

**Q:** What are examples of  viral vectors for delivering vaccines?

- **gold:** recombinant vaccines are based on both DNA viruses (such as fowlpox virus-based vaccines which target avian influenza virus and fowlpox virus, or vaccinia virusbased vectors against the rabies virus in wildlife) and RNA viruses [such as Newcastle disease virus-based vaccines to be used in poultry or
- **predicted:** poxviruses, adenoviruses baculovirus, paramyxovirus, rhabdovirus, and others
- F1 0.03 · answer_in_context 1 · groundedness 1.00
- gold chunks `['1645::7']` · retrieved `['1719::55', '1645::6', '1645::0', '1645::7', '1645::37']`

**Q:** What is the reduction in bacterial pneumonia under the revised WHO  definition of bacterial pneumonia?

- **gold:** Using the revised definition, the 10-valent pneumococcal conjugate vaccine (pneumococcal conjugate vaccination-10), had a vaccine efficacy of 22% in preventing presumed bacterial pneumonia in young children in South America [22] , and pneumococcal conjugate vaccination-13 had a vaccine efficacy of 3
- **predicted:** pneumococcal infection was considered more likely
- F1 0.03 · answer_in_context 1 · groundedness 1.00
- gold chunks `['1571::10']` · retrieved `['1571::10', '1571::28', '1571::9', '1571::14', '1664::8']`

### `retrieval_miss`

**Q:** What is the case fatality rate of SARS and MERS?

- **gold:** ranged from 14% to 45%
- **predicted:** 9.6%)
- F1 0.00 · answer_in_context 0 · groundedness 1.00
- gold chunks `['1545::1']` · retrieved `['2551::1', '2463::8', '187::3', '2551::35', '187::4']`

**Q:** How many samples were obtained?

- **gold:** 11,399
- **predicted:** 629
- F1 0.00 · answer_in_context 0 · groundedness 1.00
- gold chunks `['1573::1']` · retrieved `['1663::36', '1566::5', '1564::25', '2675::9', '1691::12']`

**Q:** What were the outcomes of the test?

- **gold:** showed good performances
- **predicted:** admission to hospital, ED length of stay (LOS), test turnaround time, and the patient receiving their test result before leaving the ED
- F1 0.00 · answer_in_context 0 · groundedness 1.00
- gold chunks `['2440::2']` · retrieved `['1556::9', '1556::8', '1556::10', '1556::7', '1556::13']`

### `chunk_boundary`

**Q:** What is the summary of this report?

- **gold:** 2019-nCoV is a novel lineage B Betacoronavirus closely related to bat SARS-related coronaviruses. It also has unique genomic features which deserves further investigation to ascertain their roles in viral replication cycle and pathogenesis. More animal sampling to determine its natural animal reserv
- **predicted:** we find that the presence of viral RNA in the blood and anal swab is positively correlated with the severe disease stage and that early monitoring of virus RNA in blood and the digestive tract on top of the respiratory tract might benefit the disease prediction
- F1 0.16 · answer_in_context 0 · groundedness 1.00
- gold chunks `['2634::18', '2634::19']` · retrieved `['2519::17', '185::0', '1663::35', '2461::28', '1599::11']`

### `reranker_demotion`

**Q:** What causes the outbreak of SARS and MERS.

- **gold:** Highly virulent species of HCoV
- **predicted:** novel coronaviruses such as severe acute respiratory syndrome (SARS) and Middle East respiratory syndrome (MERS) have evolved and infected humans, causing severe illness, epidemics and pandemics
- F1 0.00 · answer_in_context 0 · groundedness 1.00
- gold chunks `['1545::1']` · retrieved `['1568::3', '2463::11', '2653::2', '1671::8', '1671::2']`

**Q:** What is the estimated R0 of COVID-19?

- **gold:** 2.2
- **predicted:** 0.97
- F1 0.00 · answer_in_context 1 · groundedness 1.00
- gold chunks `['187::9']` · retrieved `['2440::23', '2555::0', '2555::8', '2440::0', '2592::4']`

**Q:** Which is the best method to identify pneumonia in a person?

- **gold:** Chest radiologic changes have been considered the gold standard for defining a pneumonia event
- **predicted:** IP-10
- F1 0.00 · answer_in_context 0 · groundedness 1.00
- gold chunks `['1571::6']` · retrieved `['1584::17', '1604::9', '1604::8', '1584::34', '1575::7']`

### `context_truncated`

**Q:** What is SI?

- **gold:** swarm intelligence
- **predicted:** the time period between the onset of symptoms of the last case in initial outbreak (N) and the onset of symptoms of the first case in the following outbreak (N+1).
- F1 0.00 · answer_in_context 0 · groundedness 1.00
- gold chunks `['2440::5']` · retrieved `['1634::19', '2620::11', '2620::13', '2683::57', '2440::5']`

**Q:** What is considered the mechanism of human-to-human transmission of MERS-COV?

- **gold:** Droplet spread between humans
- **predicted:** infected respiratory secretions
- F1 0.00 · answer_in_context 0 · groundedness 1.00
- gold chunks `['1741::58']` · retrieved `['1741::61', '1741::1', '2551::34', '1741::84', '1741::58']`

**Q:** What trials have been done to demonstrate the potential of phage in applications for nanomaterials?

- **gold:** Taking advantage of the ability to display substrate-specific peptides at known locations on the phage filament Hess et al., 2012) , this pioneering FIGURE 3 | Chemically addressable groups of the filamentous bacteriophage major coat protein lattice.
- **predicted:** above a certain concentration threshold, phage can form ordered crystalline suspensions
- F1 0.05 · answer_in_context 0 · groundedness 1.00
- gold chunks `['1674::44']` · retrieved `['1674::57', '1674::1', '1674::46', '1674::43', '1674::44']`

### `reader_miss`

**Q:** What was the clinical attack rate in the 2009 H1N1 pandemic?

- **gold:** 20%
- **predicted:** 8.26%.
- F1 0.00 · answer_in_context 1 · groundedness 1.00
- gold chunks `['187::7']` · retrieved `['1601::36', '187::7', '1601::35', '1601::1', '776::6']`

**Q:** How much  is the reduction in the childhood pneumonia deaths?

- **gold:** Pneumonia deaths decreased from 1.8 million in 2000 to 900,000 in 2013
- **predicted:** 27%
- F1 0.00 · answer_in_context 1 · groundedness 1.00
- gold chunks `['1571::3']` · retrieved `['1571::29', '1571::2', '1571::3', '1571::26', '1571::13']`

**Q:** What percentage of patients tested positive for HBoV1?

- **gold:** 2.2%
- **predicted:** 45.2%
- F1 0.00 · answer_in_context 1 · groundedness 1.00
- gold chunks `['1573::1']` · retrieved `['1573::1', '1573::10', '1573::15', '1573::16', '1573::3']`
