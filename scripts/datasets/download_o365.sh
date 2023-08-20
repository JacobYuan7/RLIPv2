### Download training set
train_path="/Path/To/data/Objects365/train/";
for ((i=0; i<=50; i++))  
do  
download_train="wget -P $train_path dorc.ks3-cn-beijing.ksyun.com/data-set/2020Objects365数据集/train/patch$i.tar.gz"
eval "$download_train";
echo "Finish '$download_train'.";  
done

download_train_anno="wget -P $train_path dorc.ks3-cn-beijing.ksyun.com/data-set/2020Objects365数据集/train/zhiyuan_objv2_train.tar.gz"
eval "$download_train_anno";
echo "Finish '$download_train_anno'."; 


### Download val set
val_path="/Path/To/data/Objects365/val/";
for ((i=0; i<=15; i++))
do  
download_val="wget -P $val_path dorc.ks3-cn-beijing.ksyun.com/data-set/2020Objects365数据集/val/images/v1/patch$i.tar.gz"
eval "$download_val";
echo "Finish '$download_val'.";  
done

for ((i=16; i<=43; i++))
do  
download_val="wget -P $val_path dorc.ks3-cn-beijing.ksyun.com/data-set/2020Objects365数据集/val/images/v2/patch$i.tar.gz"
eval "$download_val";
echo "Finish '$download_val'.";  
done

download_val_anno="wget -P $val_path dorc.ks3-cn-beijing.ksyun.com/data-set/2020Objects365数据集/val/zhiyuan_objv2_val.json"
eval "$download_val_anno";
echo "Finish '$download_val_anno'."; 
download_val_sample="wget -P $val_path dorc.ks3-cn-beijing.ksyun.com/data-set/2020Objects365数据集/val/sample_2020.json.tar.gz"
eval "$download_val_sample";
echo "Finish '$download_val_sample'."; 


### Download test set
test_path="/Path/To/data/Objects365/test/";
for ((i=0; i<=15; i++))
do  
download_test="wget -P $test_path dorc.ks3-cn-beijing.ksyun.com/data-set/2020Objects365数据集/test/images/v1/patch$i.tar.gz"
eval "$download_test";
echo "Finish '$download_test'.";  
done

for ((i=16; i<=50; i++))
do  
download_test="wget -P $test_path dorc.ks3-cn-beijing.ksyun.com/data-set/2020Objects365数据集/test/images/v2/patch$i.tar.gz"
eval "$download_test";
echo "Finish '$download_test'.";  
done


### Download license
license_path="/Path/To/data/Objects365/license/";
download_license="wget -P $license_path dorc.ks3-cn-beijing.ksyun.com/data-set/2020Objects365数据集/license/license.txt.tar.gz"
eval "$download_license";
echo "Finish '$download_license'.";