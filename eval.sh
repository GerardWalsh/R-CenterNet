# clean dirs
rm -r /home/gexegetic/R-CenterNet/validation/detections/*
rm -r /home/gexegetic/mAP/input/detections/*

# run eval script
python evaluation.py
mv /home/gexegetic/R-CenterNet/validation/detections/* /home/gexegetic/mAP/input/detections/

# run mAP calculation
python /home/gexegetic/mAP/main.py -na -q -np --rotated True --overlap 0.5