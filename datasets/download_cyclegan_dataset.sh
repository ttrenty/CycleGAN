if [ -d "./datasets/apple2orange64" ]; then
    echo "Dataset already exists. If you want to redownload, please delete the existing dataset."
    exit 0
fi

FILE=apple2orange
URL="https://drive.google.com/uc?export=download&id=1GSpQs_9hqCcXimYfVcbNDnkJ6K09Sywj" # apple2orange 64x64
ZIP_FILE=./datasets/$FILE.zip
TARGET_DIR=./datasets/$FILE/
wget -N $URL -O $ZIP_FILE
mkdir $TARGET_DIR
unzip $ZIP_FILE -d ./datasets/
rm $ZIP_FILE