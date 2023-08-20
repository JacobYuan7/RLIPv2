# Training set
train_path="/Path/To/data/Objects365/train/";
for ((i=0; i<=50; i++))  
do  
file_name="patch$i.tar.gz";
unzip_train="tar -zxvf $train_path$file_name"
eval "$unzip_train";
echo "Finish '$unzip_train'.";  
done

# Val set
train_path="/Path/To/data/Objects365/val/";
for ((i=0; i<=43; i++))  
do  
file_name="patch$i.tar.gz";
unzip_train="tar -zxvf $train_path$file_name"
eval "$unzip_train";
echo "Finish '$unzip_train'.";  
done
