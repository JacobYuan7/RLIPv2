m=0;
for i in /mnt/data-nas/peizhi/data/open-imagev6/OI_VG_images/images/*; 
do
    mv "$i" /mnt/data-nas/peizhi/data/open-imagev6/OI_VG_images/;
    m=$((m+1));
    echo $m;
done