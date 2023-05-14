# Het-PR
Heteroscedastic Personalized Regression

## Contributors
[Zhirong Chen](https://rong-hash.github.io)

[Haohan Wang](https://haohanwang.github.io)

[Caleb N. Ellington](https://cnellington.github.io)


## Set up
```
pip install -r requirements.txt
```

## Run the code

```
python run.py X.npy label.npy C.npy <base_model>
```
Base model can be chosen among lmm, lr, lasso.

C.npy can be replaced by X.npy if you don't have covariate data.