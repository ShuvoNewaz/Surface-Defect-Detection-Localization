kaggle datasets download -d kaustubhdikshit/neu-surface-defect-database
unzip neu-surface-defect-database.zip -d data
rm neu-surface-defect-database.zip
mv data/NEU-DET/train data
mv data/NEU-DET/validation data
mv data/validation/annotations/crazing_240.xml data/train/annotations
rmdir data/NEU-DET