
## Optimization ##

The optimization.py file contains a program to determine optimal battery placements on the ERCOT grid.
The rest of the files are used to scrape relevant data for the optimizer.

Tasks:
- [X] get list of settlement points - from that get list of busses
- [X] filter down list of busses to list of settlement points
- [X] rearrange code so it works one node at a time / alternatively set up constraints so it picks top N locations
- [ ] run script across all 5 years of data (~15m /yr)
- [ ] incorporate real-time data. LMPs are forecasts, so evaluate strategy against actual data and compare profits
- [ ] plot results on map of texas: https://www.ercot.com/content/cdr/contours/rtmLmp.html
- [ ] mess around with different battery size characteristics
- [ ] different size / single-trip efficiency modifiers?
