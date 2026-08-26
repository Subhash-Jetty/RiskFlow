# Macro Scenario Stress Testing Report

## Scenario: base

### Portfolio-Level Impacts
| Metric | Stressed Rate |
|--------|---------------|
| delinquency_3m_rate | 0.7901 |
| delinquency_6m_rate | 0.7988 |
| default_12m_rate | 0.6441 |
| prepayment_12m_rate | 0.0935 |


### Top 10 Most Impacted Segments
| Segment Type | Segment Value | Metric | Base Rate | Stressed Rate | Delta |
|--------------|---------------|--------|-----------|---------------|-------|
| credit_score_band | 620-659 | next_3m_delinquency_flag | 0.8788 | 0.8788 | 0.0000 |
| credit_score_band | 620-659 | next_6m_delinquency_flag | 0.8921 | 0.8921 | 0.0000 |
| credit_score_band | 620-659 | next_12m_default_flag | 0.7765 | 0.7765 | 0.0000 |
| credit_score_band | 620-659 | next_12m_prepayment_flag | 0.0503 | 0.0503 | 0.0000 |
| credit_score_band | 660-699 | next_3m_delinquency_flag | 0.7903 | 0.7903 | 0.0000 |
| credit_score_band | 660-699 | next_6m_delinquency_flag | 0.7934 | 0.7934 | 0.0000 |
| credit_score_band | 660-699 | next_12m_default_flag | 0.6011 | 0.6011 | 0.0000 |
| credit_score_band | 660-699 | next_12m_prepayment_flag | 0.0946 | 0.0946 | 0.0000 |
| credit_score_band | 700-739 | next_3m_delinquency_flag | 0.7008 | 0.7008 | 0.0000 |
| credit_score_band | 700-739 | next_6m_delinquency_flag | 0.7084 | 0.7084 | 0.0000 |


## Scenario: adverse_credit

### Portfolio-Level Impacts
| Metric | Stressed Rate |
|--------|---------------|
| delinquency_3m_rate | 0.8541 |
| delinquency_6m_rate | 0.8560 |
| default_12m_rate | 0.6618 |
| prepayment_12m_rate | 0.0244 |


### Top 10 Most Impacted Segments
| Segment Type | Segment Value | Metric | Base Rate | Stressed Rate | Delta |
|--------------|---------------|--------|-----------|---------------|-------|
| credit_score_band | 700-739 | next_12m_prepayment_flag | 0.1306 | 0.0259 | -0.1046 |
| state | MA | next_12m_prepayment_flag | 0.1285 | 0.0286 | -0.0999 |
| credit_score_band | 740-779 | next_12m_prepayment_flag | 0.1192 | 0.0269 | -0.0923 |
| credit_score_band | 700-739 | next_3m_delinquency_flag | 0.7008 | 0.7925 | 0.0917 |
| credit_score_band | 700-739 | next_6m_delinquency_flag | 0.7084 | 0.7983 | 0.0899 |
| state | FL | next_3m_delinquency_flag | 0.7379 | 0.8275 | 0.0896 |
| state | NC | next_12m_prepayment_flag | 0.1257 | 0.0361 | -0.0896 |
| state | MA | next_3m_delinquency_flag | 0.7433 | 0.8321 | 0.0888 |
| state | FL | next_12m_prepayment_flag | 0.1075 | 0.0196 | -0.0879 |
| vintage_year | 2022 | next_3m_delinquency_flag | 0.7269 | 0.8134 | 0.0865 |


## Scenario: high_prepayment

### Portfolio-Level Impacts
| Metric | Stressed Rate |
|--------|---------------|
| delinquency_3m_rate | 0.7898 |
| delinquency_6m_rate | 0.7974 |
| default_12m_rate | 0.6424 |
| prepayment_12m_rate | 0.0857 |


### Top 10 Most Impacted Segments
| Segment Type | Segment Value | Metric | Base Rate | Stressed Rate | Delta |
|--------------|---------------|--------|-----------|---------------|-------|
| state | TX | next_12m_prepayment_flag | 0.1070 | 0.0837 | -0.0233 |
| state | WI | next_12m_prepayment_flag | 0.0995 | 0.0778 | -0.0217 |
| state | WA | next_12m_prepayment_flag | 0.0818 | 0.0640 | -0.0178 |
| state | NC | next_12m_prepayment_flag | 0.1257 | 0.1103 | -0.0154 |
| state | PA | next_12m_prepayment_flag | 0.1041 | 0.0907 | -0.0134 |
| state | NY | next_12m_prepayment_flag | 0.1296 | 0.1164 | -0.0131 |
| credit_score_band | 700-739 | next_12m_prepayment_flag | 0.1306 | 0.1177 | -0.0129 |
| state | TN | next_12m_prepayment_flag | 0.0823 | 0.0703 | -0.0120 |
| state | MI | next_12m_prepayment_flag | 0.0684 | 0.0564 | -0.0120 |
| credit_score_band | 740-779 | next_12m_prepayment_flag | 0.1192 | 0.1078 | -0.0114 |

