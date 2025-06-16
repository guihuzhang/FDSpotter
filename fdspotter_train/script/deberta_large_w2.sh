export MASTER_PORT=9597
export MASTER_ADDR=localhost
source setup.sh

cat /proc/cpuinfo
nvidia-smi

python3 src/nli/training.py \
  --model_class_name "deberta-v3-large" \
  -n 1 \
  -g 1 \
  -nr 0 \
  --fp16 \
  --weight_decay 0.01 \
  --fp16_opt_level O2 \
  --max_length 384 \
  --epochs 2 \
  --gradient_accumulation_steps 2 \
  --per_gpu_train_batch_size 16 \
  --per_gpu_eval_batch_size 16 \
  --train_data \
snli_train:none,mnli_train:none,fever_train:none,anli1train:none,anli2train:none,anli3train:none,ling_train:none,wanli_train:none,webnlg20train:none,cnc_train:none,train_disc:none \
  --train_weights \
1,1,1,4,8,4,1,1,1,1,2 \
  --eval_data \
snli_dv:none,mnli_m:none,mnli_mm:none,anli1dv:none,anli2dv:none,anli3dv:none,ling_dv:none,webnlg20dv:none,wanli_test:none,cnc_dv:none,syn_tmp:none,syn_ctg:none,syn_cmp:none \
  --eval_frequency 7101 \
  --experiment_name "disc2le2"
