read -r -p "username: " -s username
printf "\n"
read -r -p "password: " -s pw
printf "\n"

outdir="${SCRATCH}/Y6A1_COADD"
mkdir -p $outdir

while read tile; do
	curl -u ${username}:${pw} https://www.cosmo.bnl.gov/Private/gpfs/workarea/desdata/jpg/Y6A1_COADD/${tile}/${tile}-gri.jpg -o ${outdir}/${tile}-gri.jpg
done < tiles-y6.txt

