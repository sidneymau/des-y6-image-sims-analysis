import numpy as np
import twopoint
from scipy import interpolate, signal

# def pileup(hists, zs, zmeans, z_pileup, dz, weight, nbins):
#     """Cuts off z, zmean and Nz at a pileup-z, and stacks tail on pileup-z and renormalises"""
#     ## Pile up very high z in last bin
#     import copy
#     # print(hists)
#     hists_piled = copy.copy(hists)
#     zbegin = int(z_pileup / dz)
#     print("Dz, new-end-z,weight: ", dz, z_pileup, weight)
#     for b in range(nbins):
#         s = np.sum(hists[b, zbegin:])
#         hists_piled[b, zbegin - 1] += s * weight
#         hists_piled[b, zbegin:] = 0.

#     # print(hists_piled)
#     zs = zs[:zbegin + 1]
#     zmeans_piled = zmeans[:zbegin]
#     hists_piled = hists_piled[:, :zbegin]
#     # print(hists_piled)

#     for b in range(nbins):
#         hists_piled[b, :] = hists_piled[b, :] / np.sum(hists_piled[b, :] * dz)

#     # print(hists_piled)
#     return zs, zmeans_piled, hists_piled

# def smooth(outfilesmooth, twoptfile, nzsmoothfile, runname, label, data_dir, oldnz):
#     # Troxel's smoothing adapted
#     nosmooth = twopoint.TwoPointFile.from_fits(twoptfile)
#     z = nosmooth.kernels[0].z
#     for i in range(4):
#         b = signal.savgol_filter(nosmooth.kernels[0].nzs[i], 25, 2)
#         f = interp.interp1d(nosmooth.kernels[0].z, b, bounds_error=False, fill_value=0.)
#         nosmooth.kernels[0].nzs[i] = f(z)
#     nosmooth.to_fits(outfilesmooth, clobber=True, overwrite=True)
#     np.savetxt(nzsmoothfile, np.vstack((nosmooth.kernels[0].zlow, nosmooth.kernels[0].nzs[0],
#                                         nosmooth.kernels[0].nzs[1], nosmooth.kernels[0].nzs[2],
#                                         nosmooth.kernels[0].nzs[3])).T)

#     oldnz = twopoint.TwoPointFile.from_fits(twoptfile)
#     means_smooth, sigmas_smooth = get_mean_sigma(nosmooth.kernels[0].z, nosmooth.kernels[0].nzs)
#     means_bc_piled, sigmas_bc_piled = get_mean_sigma(oldnz.kernels[0].z, oldnz.kernels[0].nzs)

#     plt.figure(figsize=(12., 8.))
#     colors = ['blue', 'orange', 'green', 'red']
#     for i in range(4):
#         plt.fill_between(oldnz.kernels[0].z, oldnz.kernels[0].nzs[i], color=colors[i], alpha=0.3)  # ,label="fiducial")
#         plt.axvline(means_smooth[i], linestyle='-.', color=colors[i], label=str(i) + 'smooth: %.3f' % (means_smooth[i]))
#         plt.plot(nosmooth.kernels[0].z, nosmooth.kernels[0].nzs[i], color=colors[i])  # ,label="smooth")
#         plt.axvline(means_bc_piled[i], linestyle='-', color=colors[i],
#                     label=str(i) + ' %.3f' % (means_bc_piled[i]))
#     plt.xlabel(r'$z$', fontsize=16)
#     plt.ylabel(r'$p(z)$', fontsize=16)
#     #plt.xlim(0, 3)
#     plt.xlim(0, 4)
#     plt.ylim(-0.5, 4)
#     plt.legend(loc='upper right', fontsize=16)
#     # plt.title('Wide n(z) for Y3 SOM', fontsize=16)
#     plt.savefig(data_dir + 'Y3_smooth_wide_nz_faint.png')
    

class Tz:
    def __init__(self, dz, nz, z0=None):
        '''Class representing sawtooth n(z) kernels (bins) in z.
        First kernel is centered at z0, which defaults to dz if not
        given.  If z0<dz, then its triangle is adjusted to go to zero at
        0, then peak at z0, down to zero at z0+dz/2.
        Arguments:
        `dz`: the step between kernel centers
        `nz`: the number of kernels.
        `z0`: peak of first bin'''
        self.dz = dz
        self.nz = nz
        if z0 is None:
            self.z0 = dz
        else:
            self.z0 = z0
        # Set a flag if we need to treat kernel 0 differently
        self.cut0 = self.z0<dz

    def __call__(self,k,z):
        '''Evaluate dn/dz for the kernel with index k at an array of
        z values.'''
        # Doing duplicative calculations to make this compatible
        # with JAX arithmetic.

        if self.cut0 and k==0:
            # Lowest bin is unusual:
            out = np.where(z>self.z0, 1-(z-self.z0)/self.dz, z/self.z0)
            out = np.maximum(0., out) / ((self.z0+self.dz)/2.)
        else:
            out = np.maximum(0., 1 - np.abs((z-self.z0)/self.dz-k)) / self.dz
        return out
    
    def zbounds(self):
        '''Return lower, upper bounds in z of all the bins in (nz,2) array'''
        zmax = np.arange(1,1+self.nz)*self.dz + self.z1
        zmin = zmax = 2*self.dz
        if self.cut0:
            zmin[0] = 0.
        return np.stack( (zmin, zmax), axis=1)
    
    def dndz(self,coeffs, z):
        '''Calculate dn/dz at an array of z values given set(s) of
        coefficients for the kernels/bins.  The coefficients will
        be normalized to sum to unity, i.e. they will represent the
        fractions within each kernel.
        Arguments:
        `coeffs`:  Array of kernel fractions of shape [...,nz]
        `z`:       Array of redshifts of arbitrary length
        Returns:
        Array of shape [...,len(z)] giving dn/dz at each z for
        each set of coefficients.'''
        
        # Make the kernel coefficients at the z's
        kk = np.array([self(k,z) for k in range(self.nz)])
        return np.einsum('...i,ij->...j',coeffs,kk) / np.sum(coeffs, axis=-1)

def rebin(nz):
    redshift_original = np.append(nz, np.array([0, 0]))
    zbinsc_laigle = np.arange(0,3.02,0.01)
    zbinsc_integrate = np.arange(0.015 ,3.015,0.01)

    interp_func = interpolate.interp1d(zbinsc_laigle, redshift_original, kind='linear', axis=0, bounds_error=False, fill_value=0)
    values = interp_func(zbinsc_integrate)


    values = values.reshape((60, 5))
    redshift_integrated = np.sum(values, axis=1)
    
    return redshift_integrated


# t = Tz(0.05,60,z0=0.035)


# for i in range(4):
#     bin_cell = tomo_bins[i]
#     bhat_msk = np.where(np.isin(cell_wide, bin_cell) == True)[0]

#     Z = Z_all[bhat_msk]
#     response_weight = response_weight_all[bhat_msk]
#     zbins = np.arange(0,6,0.01)
#     nz = np.histogram(Z, zbins, weights=response_weight, density=True)[0]

#     #Pileup at z=3
#     nz[299] = np.sum(nz[299:])
#     nz = nz[:300]

#     #Convert to Tz and dz=0.05
#     z = np.arange(0.035,3,0.05)
#     z = np.concatenate((np.array([0]), z))
#     y = rebin(nz)
#     dndz_true = t.dndz(y, z)
#     full_h5.create_dataset('sompz/pzdata_weighted_true_dz005/'+bin_list[i], data = dndz_true)