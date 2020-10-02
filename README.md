# INTRODUCTION

Nowadays, electricity networks of advanced countries rely more and more in non-operable renewable
energy sources, mainly wind and solar. However, in order to integrate energy sources in the electricity
network, it is required that the amount of energy to be generated to be forecasted 24 hours in advance,
so that energy plants connected to the electricity network can be planned and prepared to meet supply
and demand during the next day.

This is not an issue for traditional energy sources (gas, oil, hydropower, …) because they can be generated
at will (by burning more gas, for example). But solar and wind energies are not under the control of the
energy operator (i.e. they are non-operable), because they depend on the weather.

Therefore, they must be forecasted with high accuracy. This can be achieved to some extent by accurate weather forecasts. The
Global Forecast System (GFS, USA) and the European Centre for Medium-Range Weather Forecasts
(ECMWF) are two of the most important Numerical Weather Prediction models (NWP) for this purpose.
Yet, although NWP’s are very good at predicting accurately variables like “Downward long-wave radiative
flux average at the surface”, related to solar radiation, the relation between those variables and the
electricity actually produced is not straightforward. Machine Learning models can be used for this task.

In particular, meteorological variables forecasted by GFS are used as input attributes to a
machine learning model that is able to estimate how much solar energy is going to be produced at one of
the solar plants in Oklahoma. See the figure below, where the red points are the solar plants and the blue
points are locations for which meteorological predictions are provided (by GFS).

![](https://github.com/DanielLapido/Solar_energy_production/blob/main/grid.jpg)
