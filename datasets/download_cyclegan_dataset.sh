
FILE=$1

if [[ $FILE != "apple2orange64" && $FILE != "male2female64" ]]; then
    echo "Available datasets are: apple2orange64, male2female64"
    exit 1
fi

if [ -d "./datasets/$FILE" ]; then
    echo "Dataset already exists. If you want to redownload, please delete the existing dataset."
    exit 0
fi

if [ $FILE == "apple2orange64" ]; then
    URL="https://drive.google.com/uc?export=download&id=1GSpQs_9hqCcXimYfVcbNDnkJ6K09Sywj"
fi

if [ $FILE == "male2female64" ]; then
    URL="https://drive.google.com/uc?export=download&id=15-FeMAd1nKEtga2GxcjgtTmD2lV0IepN"
fi

ZIP_FILE=./datasets/$FILE.zip
TARGET_DIR=./datasets/$FILE/
wget -N $URL -O $ZIP_FILE
mkdir $TARGET_DIR
unzip $ZIP_FILE -d ./datasets/
rm $ZIP_FILE