#Absolute path to predictions file
IN_FILE=$1
#Absolute directory to testing set
TEST_DIR=$2
#Output file
OUT_FILE=$3
 
patients="$(cat $IN_FILE | awk '{if (NR !=1) {print}}' | awk -F '_' '{print $1}' | awk '!patients[$0]++')"

for patient in ${patients};
do
  episodes="$(cat $IN_FILE | grep "^${patient}_" | awk -F '_' '{print $2}')"
  
  for episode in ${episodes};
  do
    icu_stay_id="$(cat "${TEST_DIR}\\${patient}\\$episode.csv" | awk -F ',' 'FNR ==2 {print $1}')"
    echo "${patient}_${episode}_timeseries.csv,${icu_stay_id}" >> $OUT_FILE
  done
done



