kaggle datasets download -d kaustubhdikshit/neu-surface-defect-database
unzip neu-surface-defect-database.zip -d dataset
rm neu-surface-defect-database.zip
mv dataset/NEU-DET/train dataset
mv dataset/NEU-DET/validation dataset
mv dataset/validation/annotations/crazing_240.xml dataset/train/annotations
rmdir dataset/NEU-DET