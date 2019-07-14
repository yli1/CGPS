## Requirements
### Dependencies
The code needs TensorFlow and other requirements.
```
pip install -r requirements.txt
sudo apt-get install python-tk
```

### Dataset
```
git clone https://github.com/brendenlake/SCAN.git
```

## Main Experiments
Please fix CUDA_VISIBLE_DEVICES in the scrips.

SCAN Primitive Tasks
```
sh experiments/jump_main/jump_main_A.sh
sh experiments/turn_main/turn_main_A.sh
```

SCAN Template-matching
```
sh experiments/template_jar/template_jar_A.sh # jump around right
sh experiments/template_r/template_r_A.sh     # primitive right
sh experiments/template_or/template_or_A.sh   # primitive opposite right
sh experiments/template_ar/template_ar_A.sh   # primitive around right
```

Primitive and Functional Information exist in One Word
```
sh experiments/adj_main/adj_main_A.sh
sh experiments/adj_compare/adj_compare_A.sh
```

Few-shot Learning Task
```
sh experiments/fewshot_main/fewshot_main_A.sh
```

Compositionality in Machine Translation
```
sh experiments/translation_main/translation_main_A.sh
```

## Discussion Experiments
Ablation Study
```
sh experiments/jump_ablation_onerep/jump_ablation_onerep_A.sh   # Jump A
sh experiments/jump_ablation_prim/jump_ablation_prim_A.sh       # Jump B
sh experiments/jump_ablation_func/jump_ablation_func_A.sh       # Jump C
sh experiments/jump_ablation_both/jump_ablation_both_A.sh       # Jump D
sh experiments/jump_ablation_decoder/jump_ablation_decoder_A.sh # Jump E
sh experiments/turn_ablation_onerep/turn_ablation_onerep_A.sh   # TurnLeft A
sh experiments/turn_ablation_prim/turn_ablation_prim_A.sh       # TurnLeft B
sh experiments/turn_ablation_func/turn_ablation_func_A.sh       # TurnLeft C
sh experiments/turn_ablation_both/turn_ablation_both_A.sh       # TurnLeft D
sh experiments/turn_ablation_decoder/turn_ablation_decoder_A.sh # TurnLeft E
```

Influence on Other Tasks
```
sh experiments/simple_main/simple_main_A.sh
sh experiments/length_main/length_main_A.sh
```