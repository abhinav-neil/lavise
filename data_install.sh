cd data

cd coco

wget http://images.cocodataset.org/zips/train2017.zip
wget http://images.cocodataset.org/zips/val2017.zip

unzip train2017.zip
unzip val2017.zip

rm train2017.zip
rm val2017.zip

wget http://images.cocodataset.org/annotations/annotations_trainval2017.zip
wget http://images.cocodataset.org/annotations/stuff_annotations_trainval2017.zip

unzip annotations_trainval2017.zip

rm annotations_trainval2017.zip

cd ../

cd vg

mkdir VG_100K

wget https://cs.stanford.edu/people/rak248/VG_100K_2/images.zip
wget https://cs.stanford.edu/people/rak248/VG_100K_2/images2.zip

unzip images.zip -d VG_100K
unzip images2.zip -d VG_100K

rm images.zip
rm images2.zip
