# Commands for the data, the networks and the runs

All commands run from the repository root with the virtual environment active. Trained models go to
`results/models/<name>/` (`best.pt`, `model_int.npz`, `train.json`), system-model results to `results/system/`.

## Data

Tonic downloads the recordings to `data/raw/`; compact event arrays go to `data/prepared/`.

```bash
python -m snn.data prepare --dataset nmnist        # N-MNIST
python -m snn.data prepare --dataset nmnist32      # derived from it: the central 32 x 32 x 2 pixels
python -m snn.data prepare --dataset dvsgesture    # DVS128 Gesture, downsampled to 32 x 32 x 2, 1.5 s slices
python -m snn.data prepare --dataset ntidigits     # N-TIDIGITS18, single digits
python -m snn.data prepare --dataset ssc           # Spiking Speech Commands, 35 classes
```

## Networks

Every network is trained with surrogate gradients in the time-stepped, pipelined form that the time-stepped
schedules execute (one step of delay per layer), with int8 weights, thresholds and leak shifts frozen at integers,
and progressive pruning to the core graph. A spike-count output (`--spiking_out`) is trained in a second stage from
a network with an integrating output (`--init_from`). The networks fine-tuned for step-free execution use
`--mode async` (micro-bins within the step); the 1 ms networks are fine-tuned from the T-step networks. Input memory
is `--in_delays_ms 32,96` with the threshold floor `--theta_min 128`.

The commands below are reconstructed from every model's stored configuration (`train.json`) and checked: parsed by
`snn.train`, each gives that configuration back. Options not shown are at their defaults. The accuracy is the test
accuracy of the training model at the end of the sample window.

### N-MNIST, T = 5 ms

`nmnist32_pipelined1_dt5000_T66_h4096-4096-4096-4096-4096_w8_th256_s0tn4096t5` (test accuracy 98.1 % in the training model)

```bash
python -m snn.train \
  --dataset nmnist32 \
  --hidden 4096,4096,4096,4096,4096 \
  --mode pipelined \
  --dt_us 5000 \
  --t_end_ms 330.0 \
  --batch 96 \
  --lr 0.1 \
  --wbits_first 8 \
  --train_theta \
  --train_leak \
  --refractory_every 1 \
  --k_leak 5 \
  --k_out 5 \
  --spike_penalty 10.0 \
  --jitter_us 500.0 \
  --drop 0.1 \
  --core_prune 256 \
  --log_every 200 \
  --tag tn4096t5 \
  --out results/models/nmnist32_pipelined1_dt5000_T66_h4096-4096-4096-4096-4096_w8_th256_s0tn4096t5
```

`nmnist32_pipelined1_dt5000_T66_h4096-4096-4096-4096-4096_w8_th256_s0tn4096t5so` (test accuracy 97.9 % in the training model)

```bash
python -m snn.train \
  --dataset nmnist32 \
  --hidden 4096,4096,4096,4096,4096 \
  --mode pipelined \
  --dt_us 5000 \
  --t_end_ms 330.0 \
  --epochs 5 \
  --batch 96 \
  --lr 0.02 \
  --wbits_first 8 \
  --refractory_every 1 \
  --k_out 5 \
  --spike_penalty 10.0 \
  --jitter_us 500.0 \
  --drop 0.1 \
  --spiking_out \
  --train_theta_out \
  --init_from results/models/nmnist32_pipelined1_dt5000_T66_h4096-4096-4096-4096-4096_w8_th256_s0tn4096t5/best.pt \
  --log_every 100 \
  --tag tn4096t5so \
  --out results/models/nmnist32_pipelined1_dt5000_T66_h4096-4096-4096-4096-4096_w8_th256_s0tn4096t5so
```

### N-MNIST, 1 ms

`nmnist32_pipelined1_dt1000_T330_h4096-4096-4096-4096-4096_w8_th256_s0tn4096t5to1ms` (test accuracy 98.0 % in the training model)

```bash
python -m snn.train \
  --dataset nmnist32 \
  --hidden 4096,4096,4096,4096,4096 \
  --mode pipelined \
  --t_end_ms 330.0 \
  --epochs 4 \
  --batch 16 \
  --lr 0.02 \
  --wbits_first 8 \
  --train_theta \
  --train_leak \
  --refractory_every 1 \
  --snap_at 0.5 \
  --k_out 7 \
  --spike_penalty 10.0 \
  --jitter_us 500.0 \
  --drop 0.1 \
  --spiking_out \
  --train_theta_out \
  --init_from results/models/nmnist32_pipelined1_dt5000_T66_h4096-4096-4096-4096-4096_w8_th256_s0tn4096t5so/best.pt \
  --log_every 100 \
  --tag tn4096t5to1ms \
  --out results/models/nmnist32_pipelined1_dt1000_T330_h4096-4096-4096-4096-4096_w8_th256_s0tn4096t5to1ms
```

### N-MNIST, M2

`nmnist32_async_dt1250_T264_h4096-4096-4096-4096-4096_w8_th256_s0tn4096t5m2` (test accuracy 97.9 % in the training model)

```bash
python -m snn.train \
  --dataset nmnist32 \
  --hidden 4096,4096,4096,4096,4096 \
  --dt_us 1250 \
  --t_end_ms 330.0 \
  --epochs 4 \
  --batch 24 \
  --lr 0.02 \
  --wbits_first 8 \
  --leak_every 4 \
  --train_theta \
  --train_leak \
  --refractory_every 0 \
  --snap_at 0.5 \
  --k_out 5 \
  --spike_penalty 10.0 \
  --jitter_us 500.0 \
  --drop 0.1 \
  --spiking_out \
  --train_theta_out \
  --init_from results/models/nmnist32_pipelined1_dt5000_T66_h4096-4096-4096-4096-4096_w8_th256_s0tn4096t5so/best.pt \
  --log_every 100 \
  --tag tn4096t5m2 \
  --out results/models/nmnist32_async_dt1250_T264_h4096-4096-4096-4096-4096_w8_th256_s0tn4096t5m2
```

### N-MNIST, M2r

`nmnist32_async_dt1250_T264_h4096-4096-4096-4096-4096_w8_th256_s0tn4096t5m2r` (test accuracy 97.8 % in the training model)

```bash
python -m snn.train \
  --dataset nmnist32 \
  --hidden 4096,4096,4096,4096,4096 \
  --dt_us 1250 \
  --t_end_ms 330.0 \
  --epochs 2 \
  --batch 24 \
  --lr 0.02 \
  --wbits_first 8 \
  --leak_every 4 \
  --train_theta \
  --train_leak \
  --refractory_every 4 \
  --snap_at 0.5 \
  --k_out 5 \
  --spike_penalty 10.0 \
  --jitter_us 500.0 \
  --drop 0.1 \
  --spiking_out \
  --train_theta_out \
  --init_from results/models/nmnist32_pipelined1_dt5000_T66_h4096-4096-4096-4096-4096_w8_th256_s0tn4096t5so/best.pt \
  --tag tn4096t5m2r \
  --out results/models/nmnist32_async_dt1250_T264_h4096-4096-4096-4096-4096_w8_th256_s0tn4096t5m2r
```

### N-MNIST, T = 20 ms

`nmnist32_pipelined1_dt20000_T21_h4096-4096-4096-4096-4096_w8_th256_s0tn4096pr8` (test accuracy 98.0 % in the training model)

```bash
python -m snn.train \
  --dataset nmnist32 \
  --hidden 4096,4096,4096,4096,4096 \
  --mode pipelined \
  --dt_us 20000 \
  --t_end_ms 420.0 \
  --wbits_first 8 \
  --train_theta \
  --train_leak \
  --refractory_every 1 \
  --k_leak 3 \
  --k_out 3 \
  --jitter_us 500.0 \
  --drop 0.1 \
  --core_prune 256 \
  --log_every 200 \
  --tag tn4096pr8 \
  --out results/models/nmnist32_pipelined1_dt20000_T21_h4096-4096-4096-4096-4096_w8_th256_s0tn4096pr8
```

`nmnist32_pipelined1_dt20000_T21_h4096-4096-4096-4096-4096_w8_th256_s0tn4096pr8so` (test accuracy 98.0 % in the training model)

```bash
python -m snn.train \
  --dataset nmnist32 \
  --hidden 4096,4096,4096,4096,4096 \
  --mode pipelined \
  --dt_us 20000 \
  --t_end_ms 420.0 \
  --epochs 5 \
  --lr 0.1 \
  --wbits_first 8 \
  --refractory_every 1 \
  --k_out 3 \
  --jitter_us 500.0 \
  --drop 0.1 \
  --spiking_out \
  --train_theta_out \
  --init_from results/models/nmnist32_pipelined1_dt20000_T21_h4096-4096-4096-4096-4096_w8_th256_s0tn4096pr8/best.pt \
  --log_every 100 \
  --tag tn4096pr8so \
  --out results/models/nmnist32_pipelined1_dt20000_T21_h4096-4096-4096-4096-4096_w8_th256_s0tn4096pr8so
```

### DVS Gesture, T = 20 ms

`dvsgesture_pipelined1_dt20000_T81_h4096-4096-4096-4096-4096_w8_th256_s0tn4096t20` (test accuracy 81.5 % in the training model)

```bash
python -m snn.train \
  --dataset dvsgesture \
  --hidden 4096,4096,4096,4096,4096 \
  --mode pipelined \
  --dt_us 20000 \
  --t_end_ms 1620.0 \
  --epochs 60 \
  --batch 64 \
  --lr 0.1 \
  --wbits_first 8 \
  --train_theta \
  --train_leak \
  --refractory_every 1 \
  --k_leak 3 \
  --k_out 3 \
  --spike_penalty 10.0 \
  --drop 0.1 \
  --core_prune 256 \
  --prune_epochs 6,12,18,24 \
  --tag tn4096t20 \
  --out results/models/dvsgesture_pipelined1_dt20000_T81_h4096-4096-4096-4096-4096_w8_th256_s0tn4096t20
```

`dvsgesture_pipelined1_dt20000_T81_h4096-4096-4096-4096-4096_w8_th256_s0tn4096t20so` (test accuracy 79.5 % in the training model)

```bash
python -m snn.train \
  --dataset dvsgesture \
  --hidden 4096,4096,4096,4096,4096 \
  --mode pipelined \
  --dt_us 20000 \
  --t_end_ms 1620.0 \
  --epochs 10 \
  --batch 64 \
  --lr 0.02 \
  --wbits_first 8 \
  --refractory_every 1 \
  --k_out 3 \
  --spike_penalty 10.0 \
  --drop 0.1 \
  --spiking_out \
  --train_theta_out \
  --init_from results/models/dvsgesture_pipelined1_dt20000_T81_h4096-4096-4096-4096-4096_w8_th256_s0tn4096t20/best.pt \
  --tag tn4096t20so \
  --out results/models/dvsgesture_pipelined1_dt20000_T81_h4096-4096-4096-4096-4096_w8_th256_s0tn4096t20so
```

### DVS Gesture, 1 ms

`dvsgesture_pipelined1_dt1000_T1620_h4096-4096-4096-4096-4096_w8_th256_s0tn4096t20to1ms` (test accuracy 76.5 % in the training model)

```bash
python -m snn.train \
  --dataset dvsgesture \
  --hidden 4096,4096,4096,4096,4096 \
  --mode pipelined \
  --t_end_ms 1620.0 \
  --epochs 4 \
  --batch 32 \
  --lr 0.01 \
  --wbits_first 8 \
  --train_theta \
  --train_leak \
  --refractory_every 1 \
  --snap_at 0.5 \
  --k_out 7 \
  --spike_penalty 10.0 \
  --drop 0.1 \
  --spiking_out \
  --train_theta_out \
  --init_from results/models/dvsgesture_pipelined1_dt20000_T81_h4096-4096-4096-4096-4096_w8_th256_s0tn4096t20so/best.pt \
  --tag tn4096t20to1ms \
  --out results/models/dvsgesture_pipelined1_dt1000_T1620_h4096-4096-4096-4096-4096_w8_th256_s0tn4096t20to1ms
```

### DVS Gesture, M2

`dvsgesture_async_dt5000_T324_h4096-4096-4096-4096-4096_w8_th256_s0tn4096t20m2` (test accuracy 80.2 % in the training model)

```bash
python -m snn.train \
  --dataset dvsgesture \
  --hidden 4096,4096,4096,4096,4096 \
  --dt_us 5000 \
  --t_end_ms 1620.0 \
  --epochs 6 \
  --batch 16 \
  --lr 0.01 \
  --wbits_first 8 \
  --leak_every 4 \
  --train_theta \
  --train_leak \
  --refractory_every 0 \
  --snap_at 0.5 \
  --k_out 3 \
  --spike_penalty 10.0 \
  --drop 0.1 \
  --spiking_out \
  --train_theta_out \
  --init_from results/models/dvsgesture_pipelined1_dt20000_T81_h4096-4096-4096-4096-4096_w8_th256_s0tn4096t20so/best.pt \
  --log_every 100 \
  --tag tn4096t20m2 \
  --out results/models/dvsgesture_async_dt5000_T324_h4096-4096-4096-4096-4096_w8_th256_s0tn4096t20m2
```

### DVS Gesture, M2r

`dvsgesture_async_dt5000_T324_h4096-4096-4096-4096-4096_w8_th256_s0tn4096t20m2r` (test accuracy 80.6 % in the training model)

```bash
python -m snn.train \
  --dataset dvsgesture \
  --hidden 4096,4096,4096,4096,4096 \
  --dt_us 5000 \
  --t_end_ms 1620.0 \
  --epochs 6 \
  --batch 16 \
  --lr 0.01 \
  --wbits_first 8 \
  --leak_every 4 \
  --train_theta \
  --train_leak \
  --refractory_every 4 \
  --snap_at 0.5 \
  --k_out 3 \
  --spike_penalty 10.0 \
  --drop 0.1 \
  --spiking_out \
  --train_theta_out \
  --init_from results/models/dvsgesture_pipelined1_dt20000_T81_h4096-4096-4096-4096-4096_w8_th256_s0tn4096t20so/best.pt \
  --tag tn4096t20m2r \
  --out results/models/dvsgesture_async_dt5000_T324_h4096-4096-4096-4096-4096_w8_th256_s0tn4096t20m2r
```

### N-TIDIGITS18, T = 4 ms

`ntidigits_pipelined1_dt4000_T381_h4096-4096-4096-4096-4096_w8_th256_s0tn4096t4` (test accuracy 73.6 % in the training model)

```bash
python -m snn.train \
  --dataset ntidigits \
  --hidden 4096,4096,4096,4096,4096 \
  --mode pipelined \
  --dt_us 4000 \
  --t_end_ms 1524.0 \
  --epochs 100 \
  --batch 64 \
  --lr 0.1 \
  --wbits_first 8 \
  --train_theta \
  --train_leak \
  --refractory_every 1 \
  --k_leak 5 \
  --k_out 5 \
  --spike_penalty 10.0 \
  --drop 0.1 \
  --core_prune 256 \
  --prune_epochs 10,20,30,40 \
  --tag tn4096t4 \
  --out results/models/ntidigits_pipelined1_dt4000_T381_h4096-4096-4096-4096-4096_w8_th256_s0tn4096t4
```

`ntidigits_pipelined1_dt4000_T381_h4096-4096-4096-4096-4096_w8_th256_s0tn4096t4so` (test accuracy 74.2 % in the training model)

```bash
python -m snn.train \
  --dataset ntidigits \
  --hidden 4096,4096,4096,4096,4096 \
  --mode pipelined \
  --dt_us 4000 \
  --t_end_ms 1524.0 \
  --epochs 10 \
  --batch 16 \
  --lr 0.02 \
  --wbits_first 8 \
  --refractory_every 1 \
  --k_out 5 \
  --spike_penalty 10.0 \
  --drop 0.1 \
  --spiking_out \
  --train_theta_out \
  --init_from results/models/ntidigits_pipelined1_dt4000_T381_h4096-4096-4096-4096-4096_w8_th256_s0tn4096t4/best.pt \
  --tag tn4096t4so \
  --out results/models/ntidigits_pipelined1_dt4000_T381_h4096-4096-4096-4096-4096_w8_th256_s0tn4096t4so
```

### N-TIDIGITS18, 1 ms

`ntidigits_pipelined1_dt1000_T1524_h4096-4096-4096-4096-4096_w8_th256_s0tn4096t4to1ms` (test accuracy 72.7 % in the training model)

```bash
python -m snn.train \
  --dataset ntidigits \
  --hidden 4096,4096,4096,4096,4096 \
  --mode pipelined \
  --t_end_ms 1524.0 \
  --epochs 6 \
  --batch 32 \
  --lr 0.02 \
  --wbits_first 8 \
  --train_theta \
  --train_leak \
  --refractory_every 1 \
  --snap_at 0.5 \
  --k_out 7 \
  --spike_penalty 10.0 \
  --drop 0.1 \
  --spiking_out \
  --train_theta_out \
  --init_from results/models/ntidigits_pipelined1_dt4000_T381_h4096-4096-4096-4096-4096_w8_th256_s0tn4096t4so/best.pt \
  --tag tn4096t4to1ms \
  --out results/models/ntidigits_pipelined1_dt1000_T1524_h4096-4096-4096-4096-4096_w8_th256_s0tn4096t4to1ms
```

### N-TIDIGITS18, M2

`ntidigits_async_dt1000_T1524_h4096-4096-4096-4096-4096_w8_th256_s0tn4096t4m2` (test accuracy 73.1 % in the training model)

```bash
python -m snn.train \
  --dataset ntidigits \
  --hidden 4096,4096,4096,4096,4096 \
  --t_end_ms 1524.0 \
  --epochs 6 \
  --batch 32 \
  --lr 0.02 \
  --wbits_first 8 \
  --leak_every 4 \
  --train_theta \
  --train_leak \
  --refractory_every 0 \
  --snap_at 0.5 \
  --k_out 5 \
  --spike_penalty 10.0 \
  --drop 0.1 \
  --spiking_out \
  --train_theta_out \
  --init_from results/models/ntidigits_pipelined1_dt4000_T381_h4096-4096-4096-4096-4096_w8_th256_s0tn4096t4so/best.pt \
  --tag tn4096t4m2 \
  --out results/models/ntidigits_async_dt1000_T1524_h4096-4096-4096-4096-4096_w8_th256_s0tn4096t4m2
```

### N-TIDIGITS18, M2r

`ntidigits_async_dt1000_T1524_h4096-4096-4096-4096-4096_w8_th256_s0tn4096t4m2r` (test accuracy 72.2 % in the training model)

```bash
python -m snn.train \
  --dataset ntidigits \
  --hidden 4096,4096,4096,4096,4096 \
  --t_end_ms 1524.0 \
  --epochs 6 \
  --batch 32 \
  --lr 0.02 \
  --wbits_first 8 \
  --leak_every 4 \
  --train_theta \
  --train_leak \
  --refractory_every 4 \
  --snap_at 0.5 \
  --k_out 5 \
  --spike_penalty 10.0 \
  --drop 0.1 \
  --spiking_out \
  --train_theta_out \
  --init_from results/models/ntidigits_pipelined1_dt4000_T381_h4096-4096-4096-4096-4096_w8_th256_s0tn4096t4so/best.pt \
  --tag tn4096t4m2r \
  --out results/models/ntidigits_async_dt1000_T1524_h4096-4096-4096-4096-4096_w8_th256_s0tn4096t4m2r
```

### N-TIDIGITS18, input memory

`ntd_dly_int` (test accuracy 79.7 % in the training model)

```bash
python -m snn.train \
  --dataset ntidigits \
  --hidden 4096,4096,4096,4096,4096 \
  --mode pipelined \
  --dt_us 4000 \
  --t_end_ms 1524.0 \
  --epochs 100 \
  --batch 64 \
  --lr 0.1 \
  --wbits_first 8 \
  --train_theta \
  --train_leak \
  --refractory_every 1 \
  --theta_min 128.0 \
  --k_leak 5 \
  --k_out 5 \
  --spike_penalty 10.0 \
  --drop 0.1 \
  --core_prune 256 \
  --prune_epochs 10,20,30,40 \
  --in_delays_ms 32,96 \
  --tag tn4096t4d \
  --out results/models/ntd_dly_int
```

`ntd_dly_so` (test accuracy 80.4 % in the training model)

```bash
python -m snn.train \
  --dataset ntidigits \
  --hidden 4096,4096,4096,4096,4096 \
  --mode pipelined \
  --dt_us 4000 \
  --t_end_ms 1524.0 \
  --epochs 10 \
  --batch 16 \
  --lr 0.02 \
  --wbits_first 8 \
  --refractory_every 1 \
  --theta_min 128.0 \
  --k_out 5 \
  --spike_penalty 10.0 \
  --drop 0.1 \
  --in_delays_ms 32,96 \
  --spiking_out \
  --train_theta_out \
  --init_from results/models/ntd_dly_int/best.pt \
  --tag tn4096t4dso \
  --out results/models/ntd_dly_so
```

### SSC, T = 4 ms

`ssc_pipelined1_dt4000_T256_h4096-4096-4096-4096-4096_w8_th256_s0tn4096t4f5` (test accuracy 43.8 % in the training model)

```bash
python -m snn.train \
  --dataset ssc \
  --hidden 4096,4096,4096,4096,4096 \
  --mode pipelined \
  --dt_us 4000 \
  --t_end_ms 1024.0 \
  --epochs 15 \
  --lr 0.1 \
  --wbits_first 8 \
  --train_theta \
  --train_leak \
  --refractory_every 1 \
  --k_leak 5 \
  --k_out 5 \
  --spike_penalty 10.0 \
  --drop 0.1 \
  --core_prune 256 \
  --prune_epochs 2,4,6,8 \
  --in_fanout 5 \
  --tag tn4096t4f5 \
  --out results/models/ssc_pipelined1_dt4000_T256_h4096-4096-4096-4096-4096_w8_th256_s0tn4096t4f5
```

`ssc_pipelined1_dt4000_T256_h4096-4096-4096-4096-4096_w8_th256_s0tn4096t4f5so` (test accuracy 37.9 % in the training model)

```bash
python -m snn.train \
  --dataset ssc \
  --hidden 4096,4096,4096,4096,4096 \
  --mode pipelined \
  --dt_us 4000 \
  --t_end_ms 1024.0 \
  --epochs 3 \
  --lr 0.02 \
  --wbits_first 8 \
  --refractory_every 1 \
  --k_out 5 \
  --spike_penalty 10.0 \
  --drop 0.1 \
  --spiking_out \
  --train_theta_out \
  --init_from results/models/ssc_pipelined1_dt4000_T256_h4096-4096-4096-4096-4096_w8_th256_s0tn4096t4f5/best.pt \
  --tag tn4096t4f5so \
  --out results/models/ssc_pipelined1_dt4000_T256_h4096-4096-4096-4096-4096_w8_th256_s0tn4096t4f5so
```

### SSC, 1 ms

`ssc_pipelined1_dt1000_T1024_h4096-4096-4096-4096-4096_w8_th256_s0tn4096t4f5to1ms` (test accuracy 37.1 % in the training model)

```bash
python -m snn.train \
  --dataset ssc \
  --hidden 4096,4096,4096,4096,4096 \
  --mode pipelined \
  --t_end_ms 1024.0 \
  --epochs 2 \
  --batch 96 \
  --lr 0.02 \
  --wbits_first 8 \
  --train_theta \
  --train_leak \
  --refractory_every 1 \
  --snap_at 0.5 \
  --k_out 7 \
  --spike_penalty 10.0 \
  --drop 0.1 \
  --spiking_out \
  --train_theta_out \
  --init_from results/models/ssc_pipelined1_dt4000_T256_h4096-4096-4096-4096-4096_w8_th256_s0tn4096t4f5so/best.pt \
  --tag tn4096t4f5to1ms \
  --out results/models/ssc_pipelined1_dt1000_T1024_h4096-4096-4096-4096-4096_w8_th256_s0tn4096t4f5to1ms
```

### SSC, M2

`ssc_async_dt1000_T1024_h4096-4096-4096-4096-4096_w8_th256_s0tn4096t4f5m2` (test accuracy 35.4 % in the training model)

```bash
python -m snn.train \
  --dataset ssc \
  --hidden 4096,4096,4096,4096,4096 \
  --t_end_ms 1024.0 \
  --epochs 2 \
  --batch 96 \
  --lr 0.02 \
  --wbits_first 8 \
  --leak_every 4 \
  --train_theta \
  --train_leak \
  --refractory_every 0 \
  --snap_at 0.5 \
  --k_out 5 \
  --spike_penalty 10.0 \
  --drop 0.1 \
  --spiking_out \
  --train_theta_out \
  --init_from results/models/ssc_pipelined1_dt4000_T256_h4096-4096-4096-4096-4096_w8_th256_s0tn4096t4f5so/best.pt \
  --tag tn4096t4f5m2 \
  --out results/models/ssc_async_dt1000_T1024_h4096-4096-4096-4096-4096_w8_th256_s0tn4096t4f5m2
```

### SSC, M2r

`ssc_async_dt1000_T1024_h4096-4096-4096-4096-4096_w8_th256_s0tn4096t4f5m2r` (test accuracy 38.6 % in the training model)

```bash
python -m snn.train \
  --dataset ssc \
  --hidden 4096,4096,4096,4096,4096 \
  --t_end_ms 1024.0 \
  --epochs 2 \
  --batch 96 \
  --lr 0.02 \
  --wbits_first 8 \
  --leak_every 4 \
  --train_theta \
  --train_leak \
  --refractory_every 4 \
  --snap_at 0.5 \
  --k_out 5 \
  --spike_penalty 10.0 \
  --drop 0.1 \
  --spiking_out \
  --train_theta_out \
  --init_from results/models/ssc_pipelined1_dt4000_T256_h4096-4096-4096-4096-4096_w8_th256_s0tn4096t4f5so/best.pt \
  --tag tn4096t4f5m2r \
  --out results/models/ssc_async_dt1000_T1024_h4096-4096-4096-4096-4096_w8_th256_s0tn4096t4f5m2r
```

### SSC, input memory

`ssc_dly_int` (test accuracy 42.3 % in the training model)

```bash
python -m snn.train \
  --dataset ssc \
  --hidden 4096,4096,4096,4096,4096 \
  --mode pipelined \
  --dt_us 4000 \
  --t_end_ms 1024.0 \
  --epochs 24 \
  --lr 0.1 \
  --wbits_first 8 \
  --train_theta \
  --train_leak \
  --refractory_every 1 \
  --snap_at 0.8 \
  --theta_min 128.0 \
  --k_leak 5 \
  --k_out 5 \
  --spike_penalty 10.0 \
  --drop 0.1 \
  --core_prune 256 \
  --prune_epochs 3,8,13,18 \
  --in_fanout 3 \
  --in_delays_ms 32,96 \
  --in_fanout_delayed 1 \
  --tag tn4096t4d \
  --out results/models/ssc_dly_int
```

`ssc_dly_so` (test accuracy 39.8 % in the training model)

```bash
python -m snn.train \
  --dataset ssc \
  --hidden 4096,4096,4096,4096,4096 \
  --mode pipelined \
  --dt_us 4000 \
  --t_end_ms 1024.0 \
  --epochs 5 \
  --lr 0.02 \
  --wbits_first 8 \
  --refractory_every 1 \
  --theta_min 128.0 \
  --k_out 5 \
  --spike_penalty 10.0 \
  --drop 0.1 \
  --in_delays_ms 32,96 \
  --spiking_out \
  --train_theta_out \
  --init_from results/models/ssc_dly_int/best.pt \
  --tag tn4096t4dso \
  --out results/models/ssc_dly_so
```

## System-model runs

Every spike-count network (`model_int.npz`) is run in burst mode and in real time, on the full test set. Burst mode
(global tick, barrier and step-free execution plain, with the refractory period on all neurons, M1, and in the
hidden layers only, M1h):

```bash
python -m snn.system --model results/models/<name>/model_int.npz --dataset <dataset> \
  --configs step:aligned,step:barrier,event:wire --wire_ref --wire_ref_hidden \
  --speeds ip --in_period 256 --tick_us 1000 --f_mhz 67 --fifo 64 --port_buf 64 \
  --work_curves --grid_ms 0.1 --workers 6 --check_exact 0 \
  --dump_steps results/scaling/<name>_steps.npz --out results/system/<name>_ip.json
```

Real time (the barrier equals the global tick there):

```bash
python -m snn.system --model results/models/<name>/model_int.npz --dataset <dataset> \
  --configs step:aligned,event:wire --wire_ref --wire_ref_hidden \
  --speeds rt --f_mhz 67 --fifo 64 --port_buf 64 \
  --work_curves --grid_ms 2 --workers 6 --check_exact 0 --out results/system/<name>_rt.json
```

- `--grid_ms` sets only the time resolution of the accuracy-against-time curves.
- `--work_curves` records the operations against time (rows, output spikes, leak and threshold passes), from which
  the energy follows with the constants in `synthesis/core_power/RESULTS.md`.
- `--dump_steps` stores every sample's step lengths and its decision after every step, from which a machine of K
  copies under one barrier or global tick follows exactly (random batches of K test samples).
- `--random_subset --n_samples 5000` runs a fixed random subset of the test set.
- The SSC runs need about 20 GB of memory and one to two hours with 6 workers.

## Threshold granularity

Accuracy of a time-stepped network when every layer checks its thresholds k times per step on partial sums, with
and without a limit of one spike per neuron and step:

```bash
python -m snn.granularity --model results/models/<name>/model_int.npz --dataset <dataset> --k 1,2,4,8,16 --n_samples 2000
```

## RTL against the model

The RTL's output unit integrates, so it is checked with a network without `--spiking_out`, for example the first
stage of the N-MNIST network at 20 ms:

```bash
M=results/models/nmnist32_pipelined1_dt20000_T21_h4096-4096-4096-4096-4096_w8_th256_s0tn4096pr8/model_int.npz
python rtl/run_rtl.py export --model $M --dir rtl/build/pr8
python rtl/run_rtl.py build --dir rtl/build/pr8                      # Verilator 5 (VERILATOR=/path/to/verilator)
python rtl/run_rtl.py run --model $M --dir rtl/build/pr8 --samples 0,1,2 --schedule step
python rtl/run_rtl.py run --model $M --dir rtl/build/pr8 --samples 0,1,2 --schedule event
```

## Energy

`synthesis/core_power/RESULTS.md` describes the measurement: the ROM, stimuli and expected spikes from `gen_core.py`,
the RTL check `run_rtl.sh`, synthesis `synth.sh`, `flatten.sh`, gate-level simulation and OpenSTA power per scenario
`measure.sh`, hardening `harden25.sh`, sign-off `signoff67.sh`, the annotated runs on the layout `phase2.sh`, and the
summary `report.py`. Tool and PDK paths are set in `synthesis/core_power/env.sh`.
