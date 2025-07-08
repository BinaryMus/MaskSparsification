model="qwen"
device="cuda:1"

cutlayers=(1 14 27)

comp_params=(
  "ms --r 0.01 --b 3 --positive 0"
  "vq --b 3"
  "qq --b 3"
  "fpq --ebit 2 --mbit 1"
  "vs --r 0.0725"
  "rts --r 0.0725"
)

for cutlayer in "${cutlayers[@]}"; do
  for params in "${comp_params[@]}"; do
    comp=$(echo $params | awk '{print $1}')
    other_params=$(echo $params | cut -d' ' -f2-)
    
    echo "python nlp_main.py --model "$model" --cutlayer "$cutlayer" --comp "$comp" --device "$device" $other_params"
    python nlp_main.py --model "$model" --cutlayer "$cutlayer" --comp "$comp" --device "$device" $other_params
  done
done