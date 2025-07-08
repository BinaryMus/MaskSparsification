configs=(
  "resnet18 cifar100 1"
  "resnet18 cifar100 5"
  "resnet18 cifar100 9"
  "vgg16 cifar10 1"
  "vgg16 cifar10 7"
  "vgg16 cifar10 13"
)

for config in "${configs[@]}"; do
  read -r model data cutlayer <<< "$config"
  
  echo "Plotting for model=$model, data=$data, cutlayer=$cutlayer"
  python plot.py \
    --model "$model" \
    --data "$data" \
    --cutlayer "$cutlayer"
done