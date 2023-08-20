# Training set
train_path="/Path/To/data/open-imagev7/train/";
for ((i=0; i<=9; i++))  
do  
file_name="train_$i.tar.gz";
unzip_train="tar -zxvf $train_path$file_name -C /Path/To/data/open-imagev7/train/"
eval "$unzip_train";
echo "Finish '$unzip_train'.";  
done

idices=("a" "b" "c" "d" "e" "f")
for i in ${idices[*]}
do
file_name="train_$i.tar.gz";
unzip_train="tar -zxvf $train_path$file_name -C /Path/To/data/open-imagev7/train/"
eval "$unzip_train";
echo "Finish '$unzip_train'.";  
done

unzip_val="tar -zxvf /Path/To/data/open-imagev7/val/validation.tar.gz -C /Path/To/data/open-imagev7/val/"
eval "$unzip_val"
echo "Finish '$unzip_val'"

unzip_test="tar -zxvf /Path/To/data/open-imagev7/test/test.tar.gz -C /Path/To/data/open-imagev7/test/"
eval "$unzip_test"
echo "Finish '$unzip_test'"
