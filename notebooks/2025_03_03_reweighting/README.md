# Simulation reweighting

https://arxiv.org/abs/2012.08567

> We choose three quantities on which to re-weight: the magni-
tude (calculated from the mean of the flux in the 𝑟, 𝑖 and 𝑧 bands),
the distance to the nearest neighbor, and the magnitude of the near-
est neighbor (again based on the 𝑟, 𝑖 and 𝑧 bands). The neighbor can
be any detected object with non-negative flux (i.e. we do not restrict
the neighbor candidates to objects passing shape catalog cuts). We
then aim to produce a set of weights to apply to the simulation re-
sults that will improve the match of the joint distributions of these
quantities between simulations and data.
>
> To accomplish this, we use k-means clustering to define clus-
ters of these three quantities based on 200,000 randomly selected
objects in each photo-𝑧 bin. We then assign all objects in both the
simulations and DES Y3 data to these clusters. Weights are pro-
duced for each photo-𝑧 bin by taking the ratio of the number of
the number of objects assigned to each data cluster to the number
assigned to each simulation cluster.

See
- https://github.com/des-science/y3-wl_image_sims/blob/master/catalog_stuff/sim_reweighting.py
- https://github.com/des-science/y3-wl_image_sims/blob/master/catalog_stuff/sim_reweight_bl.py

Notes from Y3:
- K means with 200 clusters
- Clusters fit to random subset of sim catalog (for faster runtime)
- Y3 and sim data mapped onto clusters and saved