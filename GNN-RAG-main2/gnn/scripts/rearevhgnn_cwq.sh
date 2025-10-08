### ReaRevHGNN+SBERT training on CWQ
python ./main.py ReaRevHGNN \
--entity_dim 50 \
--num_epoch 100 \
--batch_size 8 \
--eval_every 2 \
--data_folder data/CWQ/ \
--lm sbert \
--num_iter 2 \
--num_ins 3 \
--num_gnn 3 \
--relation_word_emb True \
--load_experiment ReaRev_CWQ.ckpt \
--is_eval \
--name cwq \
# --- HGNN Hyperparameters ---
--num_relation_clusters 10 \
--num_node_clusters 15 \
--num_hyper_gnn_layers 3