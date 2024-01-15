# autoregressive-ml
Experiments with autoregressive ML

Run on atmlxgpu1 via:

```
nohup python -m scripts.autoregression --output-dir ~/nobackups/predictions --num-steps 360 --year 2016 --month 1 --day 1 &> output.log &
```