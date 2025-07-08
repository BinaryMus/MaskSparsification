model="resnet18"
dataset="cifar100"
num_class=100
device="cuda:1"

cutlayers=(1 5 9)

comp_params=(
  "ms --r 0.01 --b 2 --positive 1"
  "vq --b 3"
  "qq --b 3"
  "fpq --ebit 2 --mbit 1"
  "vs --r 0.04125"
  "rts --r 0.04125"
)

for cutlayer in "${cutlayers[@]}"; do
  for params in "${comp_params[@]}"; do
    comp=$(echo $params | awk '{print $1}')
    other_params=$(echo $params | cut -d' ' -f2-)
    
    echo "python vision_main.py --model "$model" --dataset "$dataset" --num_class "$num_class" --cutlayer "$cutlayer" --comp "$comp" --device "$device" $other_params"
    python vision_main.py --model "$model" --dataset "$dataset" --num_class "$num_class" --cutlayer "$cutlayer" --comp "$comp" --device "$device" $other_params
  done
done