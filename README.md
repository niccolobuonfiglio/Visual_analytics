# Projection Fidelity Toolkit

A 2-D scatter plot of high-dimensional data gets read literally: points that look
close are assumed to be similar. Often they aren't. This toolkit measures, **for
every individual point**, how badly a 2-D projections misrepresent that
point's real neighbourhood, and puts the answer on the plot next to the point.

It then tells you which of the two projections to trust for your data, and says
so with a confidence interval rather than a bare number.
