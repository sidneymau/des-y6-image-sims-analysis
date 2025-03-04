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

See https://github.com/des-science/y3-wl_image_sims/blob/master/catalog_stuff/sim_reweight_bl.py

```
#Calculate kmeans regions
k[b] = KMeans(n_clusters=150,precompute_distances=True,tol=1e-2)#,init=np.loadtxt('centers.txt'))
k[b].fit(np.vstack((mag,np.log10(nbrdist),nbrmag)).T,sample_weight=w)
if b is None:
    bin_mask = np.arange(len(data_bins))
else:
    bin_mask = data_bins==b
# Calculate data region and save
prd = k[b].predict(np.vstack((data_mag[bin_mask],np.log10(data_nbrdist[bin_mask]),data_nbrmag[bin_mask])).T,sample_weight=data_w[bin_mask])
```

Notes from Y3:
- K means with 150 clusters
- Clusters fit to random subset of sim catalog (for faster runtime)
- Y3 and sim data mapped onto clusters and saved