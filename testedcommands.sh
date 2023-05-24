

python3 demo.py -g -1 -s val -y 17 -D ../data.nosync/DAVIS2017  -p results/checkpoints/downloaded/STM_FinalCheckpoint.pth -backbone resnet50


python3 eval.py -g -1 -s val -y 17 -D ../data.nosync/DAVIS2017  -p results/checkpoints/downloaded/STM_FinalCheckpoint.pth -backbone resnet50


python3 evaluateAndSave.py -g -1 -s val -y 17 -D ../data.nosync/DAVIS2017  -p results/checkpoints/downloaded/STM_FinalCheckpoint.pth -backbone resnet50



python3 train_coco.py -Ddavis ../data.nosync/DAVIS2017 -Dcoco ../data.nosync/Ms-COCO/ -backbone resnet50 -save results/checkpoints/trained/pretrained/



python3 train_davis.py -Ddavis .../data.nosync/DAVIS2017 -Dyoutube ../data.nosync/YoutubeDataset/ -backbone resnet50
