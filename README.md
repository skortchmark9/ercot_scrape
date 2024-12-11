
## Optimization ##

The optimization.py file contains a program to determine optimal battery placements on the ERCOT grid.
The rest of the files are used to scrape relevant data for the optimizer.

Tasks:
- [X] get list of settlement points - from that get list of busses
- [X] filter down list of busses to list of settlement points
- [X] rearrange code so it works one node at a time / alternatively set up constraints so it picks top N locations
- [X] figure out how to properly apply efficiency metric to maximum charge/discharge rates.
- [X] run script across all 5 years of data (~15m /yr)
- [X] incorporate real-time data. LMPs are forecasts, so evaluate strategy against actual data and compare profits
- [ ] plot results on map of texas: https://www.ercot.com/content/cdr/contours/rtmLmp.html
- [ ] mess around with different battery size characteristics
- [ ] different size / single-trip efficiency modifiers?



##

Across all years
```
[('CHEYENNE_8', 36492249.607171066),
 ('DOLRHIDE_8', 35408276.215197355),
 ('UNOCALDH_8', 35221385.22467102),
 ('WOO_WOODWRD2', 34240317.70848683),
 ('TNHACKBERY1', 31086921.45592106),
 ('EA_EAGLE_HY1', 30371896.23703946),
 ('TNKEYSTONE0', 30299284.149276324),
 ('ELMAR_8', 30130605.81901314),
 ('WOLFCAMP_9', 28401913.51677631),
 ('NNATURAL_9', 28358901.70177631)]
```
