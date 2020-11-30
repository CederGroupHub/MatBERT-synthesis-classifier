## MatBERT synthesis classifier
A synthesis paragraph classifier built by fine-tuning MatBERT. 
Built by the text-mining team at [CEDER group](https://ceder.berkeley.edu). 

## Usage

```
from synthesis_classifier import get_model, get_tokenizer, run_batch

model = get_model()
tokenizer = get_tokenizer()

paragraphs = [...]

for batch in batches:
    result = run_batch(batch, model, tokenizer)
    print(result)

# Output:
# [{
# 'text': '10.1063/1.3676216: The raw materials were BaCO3, ZnO, Nb2O5, 
# and Ta2O5 powders with purity of more than 99.5%. Ba[Zn1/3 (Nb1−xTax)2/3]O3 
# (BZNT, x\u2009=\u20090.0, 0.2, 0.4, 0.6, 0.8, 1.0) solid solutions were 
# synthesized by conventional solid-state sintering technique. Oxide compounds 
# were mixed for 12\u2009h in polyethylene jars with zirconia balls and then 
# dried and calcined at 1100 °C for 2\u2009h. After remilling, the powders were 
# dried and pressed into discs of 15\u2009mm\u2009×\u20091\u2009mm and next 
# sintered at 1500 °C for 3\u2009h.', 
# 
# 'tokens': ['10', '.', '106', '##3', '/', '1', '.', '367', '##62', '##16', ':', 
# 'The', 'raw', 'materials', 'were', 'Ba', '##CO3', ',', 'ZnO', ',', 'Nb2O5', ',', 
# 'and', 'Ta2O5', 'powders', 'with', 'purity', 'of', 'more', 'than', '99', '.', 
# '5', '%', '.', 'Ba', '[', 'Zn', '##1', '/', '3', '(', 'Nb', '##1−x', '##Ta', 
# '##x', ')', '2', '/', '3', ']', 'O3', '(', 'BZ', '##NT', ',', 'x', '=', '0', 
# '.', '0', ',', '0', '.', '2', ',', '0', '.', '4', ',', '0', '.', '6', ',', '0', 
# '.', '8', ',', '1', '.', '0', ')', 'solid', 'solutions', 'were', 'synthesized', 
# 'by', 'conventional', 'solid', '-', 'state', 'sintering', 'technique', '.', 
# 'Oxid', '##e', 'compounds', 'were', 'mixed', 'for', '12', 'h', 'in', 'polyethylene', 
# 'j', '##ars', 'with', 'zirconia', 'balls', 'and', 'then', 'dried', 'and', 'calcined', 
# 'at', '1100', '°C', 'for', '2', 'h', '.', 'After', 'rem', '##illing', ',', 'the', 
# 'powders', 'were', 'dried', 'and', 'pressed', 'into', 'discs', 'of', '15', 'mm', 
# '×', '1', 'mm', 'and', 'next', 'sintered', 'at', '1500', '°C', 'for', '3', 'h', '.'], 
#
# 'scores': {
# 'solid_state_ceramic_synthesis': 0.9992626309394836, 
# 'sol_gel_ceramic_synthesis': 0.00024707740521989763, 
# 'hydrothermal_ceramic_synthesis': 8.356467151315883e-05, 
# 'precipitation_ceramic_synthesis': 8.224111661547795e-05, 
# 'something_else': 0.00032462377566844225}}
```

## Citing

While we are working on a new paper, it's always nice to cite our previous paper:

```
@article{huo2019semi,
  title={Semi-supervised machine-learning classification of materials synthesis procedures},
  author={Huo, Haoyan and Rong, Ziqin and Kononova, Olga and Sun, Wenhao and Botari, Tiago and He, Tanjin and Tshitoyan, Vahe and Ceder, Gerbrand},
  journal={npj Computational Materials},
  volume={5},
  number={1},
  pages={1--7},
  year={2019},
  publisher={Nature Publishing Group}
}
```