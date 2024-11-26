
## Optimization ##

Stuff to do:
X get list of settlement points - from that get list of busses
X filter down list of busses to list of settlement points
* rearrange code so it works one node at a time / alternatively set up constraints so it picks top N locations

* incorporate real-time data. LMPs are forecasts, so evaluate strategy against actual data and compare profits
* plot results on map of texas: https://www.ercot.com/content/cdr/contours/rtmLmp.html
* mess around with different battery size characteristics
* different size / single-trip efficiency modifiers?




Settlement Points (SPs) are the buses that have distinct prices (ercot settle prices at these buses), while the other buses prices are based on the nearest SP.
the system has ~17k buses but only around 900 SPs