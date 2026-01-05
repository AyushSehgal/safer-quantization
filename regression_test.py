import json
import random
from peft import PeftModel, LoraConfig, get_peft_model, LoraConfig
from tqdm import tqdm

def normalize_prompt(text: str) -> str:
    # 去掉首尾空白
    text = text.strip()
    # 将连续多行压缩成单行（保留空格）
    text = " ".join([line.strip() for line in text.splitlines() if line.strip() != ""])
    return text

with open("data.json", "r", encoding="utf-8") as f:
    data = json.load(f)

random.seed(443)
random.shuffle(data)


prompts = [d["prompt"] for d in data][:1000]
labels = [d["label"] for d in data][:1000]

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from utils import model_quantization, evaluate

# model_name = "Qwen/Qwen2.5-7B-Instruct"
# model_name = "meta-llama/Llama-2-7b-chat-hf"
model_name = "google/gemma-2-9b-it"
# model_name = "google/gemma-3-4b-it"  # 需要huggingface权限
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    output_hidden_states=True,   # 关键：要输出中间层
    # load_in_4bit=True,
    # torch_dtype=torch.bfloat16,
    torch_dtype=torch.float16,
    device_map="auto"
)
model.eval()


# lora_path = ""
# model = PeftModel.from_pretrained(model, lora_path, device_map='auto', torch_dtype=torch.bfloat16, output_hidden_states=True)
# model = model.merge_and_unload()

for param in model.parameters():
    param.requires_grad = False

model.config.use_cache = False


# model, qlinears = model_quantization(model, model_name, 8, 8)
# layernorm_inputs = []  # {layer_idx: [tensor, tensor, ...]}
# def make_ln_input_hook(layer_idx):
#     def hook(module, inputs, output):
#         x = inputs[0].detach()
#         layernorm_inputs.append(x)
#     return hook
# def make_ln_output_hook(layer_idx):
#     def hook(module, inputs, output):
#         x = output.detach()
#         layernorm_inputs.append(x)
#     return hook
# layers = model.model.layers
# for i, layer in enumerate(layers):
#     h = layer.input_layernorm.register_forward_hook(
#         make_ln_input_hook(i)
#     )
# layers[-1].mlp.register_forward_hook(make_ln_output_hook(len(layers)-1))


def get_sentence_embedding(model, tokenizer, sentence):
    sentence = sentence.strip().replace('"', "")
    word_embeddings = model.get_input_embeddings()

    # Embed the sentence
    tokenized = tokenizer(sentence, return_tensors="pt", add_special_tokens=False).to(
        'cuda:0'
    )
    embedded = word_embeddings(tokenized.input_ids)
    return embedded

def apply_chat_template(tokenizer, user_text: str) -> str:
    """
    统一把 harmful 指令包装成 llama-2-chat 风格的对话提示。
    如果 tokenizer 自带 chat_template，则使用；否则用一个简单模板。
    """
    try:
        if hasattr(tokenizer, "apply_chat_template") and tokenizer.chat_template:
            messages = [
                {"role": "system", "content": "You are a helpful, honest, and safe assistant."},
                # {"role": "system", "content": ""},
                {"role": "user", "content": user_text},
            ]
            return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    except Exception:
        pass
    # 退化模板
    return (
        "### System:\nYou are a helpful, honest, and safe assistant.\n\n"
        f"### User:\n{user_text}\n\n### Assistant:\n"
    )

def get_hidden_states(text, model, tokenizer, max_length=128):
    text = apply_chat_template(tokenizer, text)
    # text = normalize_prompt(text)
    inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=max_length, padding="max_length").to(model.device)
    with torch.no_grad():
        hidden_states = model(**inputs, output_hidden_states=True).hidden_states
        # hidden_states = model(**inputs)
    # outputs.hidden_states: tuple(length = n_layers+1), each shape (batch, seq_len, hidden_dim)
    return hidden_states, inputs["attention_mask"]

import numpy as np

all_layer_features = []  # shape: [num_samples, num_layers, hidden_dim]

tokenizer.pad_token = tokenizer.eos_token
for prompt in tqdm(prompts):
    hidden_states, attn_mask = get_hidden_states(prompt, model, tokenizer)
    seq_lengths = attn_mask.sum(dim=1) - 1  # 最后一个非padding位置
    last_token_features = []
    for layer_h in hidden_states[1:]:
    # for layer_h in layernorm_inputs[1:]:
        # layer_h: (1, seq_len, hidden_dim)
        last_vec = layer_h[0, seq_lengths, :].float().cpu().squeeze(0).numpy()
        # last_vec = layer_h[0, :, :].mean(0).cpu().squeeze(0).numpy()
        last_token_features.append(last_vec)
    all_layer_features.append(np.stack(last_token_features))
    layernorm_inputs = []



all_layer_features = np.stack(all_layer_features)
labels = np.array(labels)

# # import random
# # random.shuffle(labels)

print(all_layer_features.shape)  # (num_samples, num_layers+1, hidden_dim)


all_layer_features_t = torch.tensor(all_layer_features).to('cuda:0').float()
# #
# # 转为 tensor
labels_t = torch.tensor(labels).to('cuda:0')

# 找出 benign / harmful 样本的索引
benign_idx = (labels_t == 0)
harmful_idx = (labels_t == 1)

num_layers = all_layer_features_t.shape[1]

# inter_distances = []
# intra_distances = []
# separability_ratios = []
#
# eps = 1e-8  # 防止除零
#
# for layer in range(num_layers):
#     # (num_samples, hidden_dim)
#     layer_feats = all_layer_features_t[:, layer, :]
#
#     # -------- centers --------
#     benign_feats = layer_feats[benign_idx]
#     harmful_feats = layer_feats[harmful_idx]
#
#     benign_center = benign_feats.mean(dim=0)
#     harmful_center = harmful_feats.mean(dim=0)
#
#     # -------- inter-class distance --------
#     inter_dist = torch.norm(benign_center - harmful_center, p=2)
#
#     # -------- intra-class distance --------
#     benign_intra = torch.norm(benign_feats - benign_center, p=2, dim=1).mean()
#     harmful_intra = torch.norm(harmful_feats - harmful_center, p=2, dim=1).mean()
#
#     intra_dist = 0.5 * (benign_intra + harmful_intra)
#
#     # -------- ratio --------
#     ratio = inter_dist / (intra_dist + eps)
#
#     inter_distances.append(inter_dist.item())
#     intra_distances.append(intra_dist.item())
#     separability_ratios.append(ratio.item())
#
# for i in range(num_layers):
#     print(f"Layer {i}: Inter-class distance = {inter_distances[i]:.4f}, Intra-class distance = {intra_distances[i]:.4f}, Separability ratio = {separability_ratios[i]:.4f}")
#
# # # 输出结果
# # for i, d in enumerate(distances):
# #     print(f"Layer {i}: benign-harmful center distance = {d:.4f}")
#
# exit()


# def pca_svd_torch(x: torch.Tensor, n_components=2):
#     """
#     x: (n_samples, hidden_dim) torch.Tensor
#     """
#     x_mean = x.mean(dim=0, keepdim=True)
#
#     # 1️⃣ 去中心化（必须）
#     x_centered = x - x_mean
#
#     # 2️⃣ SVD分解
#     # full_matrices=False 提升效率
#     U, S, Vh = torch.linalg.svd(x_centered, full_matrices=False)
#
#     # 3️⃣ 取前 k 个主成分方向 (Vh 的前 k 行，对应最大奇异值)
#     V_k = Vh[:n_components, :].T   # shape (d, k)
#
#     # 4️⃣ 投影
#     reduced = x_centered @ V_k     # shape (n, k)
#     return reduced, V_k, x_mean
#
# import matplotlib.pyplot as plt
#
# unique_labels = torch.unique(torch.tensor(labels))
# # fig, axes = plt.subplots(4, 8, figsize=(16, 8))  # 32层
# # fig, axes = plt.subplots(6, 7, figsize=(16, 8))  # 42层
# fig, axes = plt.subplots(1, 5, figsize=(20, 3))
#
# # axes = axes.flatten()
# colors = plt.cm.get_cmap('tab10', len(unique_labels))
# labels = torch.tensor(labels)
#
# from matplotlib.lines import Line2D
# legend_elements = [
#     Line2D([0], [0], marker='o', color='w',
#            label='Benign', markerfacecolor=colors(0),
#            markersize=8, alpha=0.7),
#     Line2D([0], [0], marker='o', color='w',
#            label='Harmful', markerfacecolor=colors(1),
#            markersize=8, alpha=0.7),
# ]


# x = all_layer_features_t[:, 26, :]  # shape (200, 4096)
#
# reduced, V_k, x_mean = pca_svd_torch(x, n_components=2)
#
# reduced = reduced.cpu().numpy()
#
# # ax = axes[layer_idx - start]
#
# for i, label in enumerate(unique_labels):
#     mask = labels == label
#     x = reduced[mask, 0]
#     y = reduced[mask, 1]
#     plt.scatter(x, y,
#                s=15, color=colors(i), alpha=0.9, label=f"Harmful" if label == 1 else "Benign")
#
# # 设置 mean ± 3σ
# x_mean = reduced[:, 0].mean()
# x_std = reduced[:, 0].std()
# y_mean = reduced[:, 1].mean()
# y_std = reduced[:, 1].std()
#
# # print(x_mean - 2 * x_std, x_mean + 2 * x_std)
# # print(y_mean - 2 * y_std, y_mean + 2 * y_std)
#
# # plt.xlim(-53.92356958007815, 53.923567871093724)
# # plt.ylim(-27.652799728393575, 27.652803298950175)
#
# plt.xlim(x_mean - 2 * x_std, x_mean + 2 * x_std)
# plt.ylim(y_mean - 2 * y_std, y_mean + 2 * y_std)
#
# plt.title(f"Layer {26}", fontsize=16)
# plt.xticks([])
# plt.yticks([])
#
# # if layer_idx % 8 == 0:
# #     ax.legend(loc='upper right')
#
# plt.legend(
#     handles=legend_elements,
#     # loc='upper center',        # 常见：upper center / lower center
#     ncol=1,
#     fontsize=18,
#     # bbox_to_anchor=(0.5, 1.05)  # 放在 figure 上方
# )
#
# # title = 'Llama2-7b-chat Pre-trained HS=0.0'
# title = 'Gemma2-9b-it'
# # ✅ 然后再设置 suptitle
# # fig.suptitle(title, fontsize=25, y=0.98)
# plt.savefig(f"1.svg", dpi=300)
# exit()

# start = 26
# for layer_idx in range(start, all_layer_features_t.shape[1]-1):
#
#     x = all_layer_features_t[:, layer_idx, :]  # shape (200, 4096)
#
#     reduced, V_k, x_mean = pca_svd_torch(x, n_components=2)
#
#     reduced = reduced.cpu().numpy()
#
#     ax = axes[layer_idx - start]
#
#     for i, label in enumerate(unique_labels):
#         mask = labels == label
#         x = reduced[mask, 0]
#         y = reduced[mask, 1]
#         ax.scatter(x, y,
#                    s=15, color=colors(i), alpha=0.7, label=f"Harmful" if label == 1 else "Benign")
#
#     # 设置 mean ± 3σ
#     x_mean = reduced[:, 0].mean()
#     x_std = reduced[:, 0].std()
#     y_mean = reduced[:, 1].mean()
#     y_std = reduced[:, 1].std()
#
#     ax.set_xlim(x_mean - 2 * x_std, x_mean + 2 * x_std)
#     ax.set_ylim(y_mean - 2 * y_std, y_mean + 2 * y_std)
#
#     ax.set_title(f"Layer {layer_idx}", fontsize=16)
#     ax.set_xticks([])
#     ax.set_yticks([])
#
#     # if layer_idx % 8 == 0:
#     #     ax.legend(loc='upper right')
#
# fig.legend(
#     handles=legend_elements,
#     # loc='upper center',        # 常见：upper center / lower center
#     ncol=1,
#     fontsize=18,
#     # bbox_to_anchor=(0.5, 1.05)  # 放在 figure 上方
# )
#
# # title = 'Llama2-7b-chat Pre-trained HS=0.0'
# # title = 'Gemma2-9b-it'
# # ✅ 然后再设置 suptitle
# # fig.suptitle(title, fontsize=25, y=0.98)
# # plt.tight_layout(rect=[0, 0, 1, 0.95])
# # plt.savefig(f"{title}.jpg", dpi=300)
# plt.savefig(f"3-3.svg", dpi=300, bbox_inches='tight')
# exit()


# import numpy as np
#
# harm_idx = np.where(labels == 1)[0]
# benign_idx = np.where(labels == 0)[0]
#
# print(f"Harmful samples: {len(harm_idx)}")
# print(f"Benign samples: {len(benign_idx)}")
#
# def cosine_similarity(a, b):
#     # a, b: (hidden_dim,)
#     num = np.dot(a, b)
#     denom = np.linalg.norm(a) * np.linalg.norm(b)
#     return num / denom
#
# def angle_in_degrees(a, b):
#     cos_sim = cosine_similarity(a, b)
#     # clamp to avoid numerical issues
#     cos_sim = np.clip(cos_sim, -1.0, 1.0)
#     return np.degrees(np.arccos(cos_sim))
#
# max_per_class = 200
#
# harm_idx = harm_idx[:max_per_class]
# benign_idx = benign_idx[:max_per_class]
#
# num_layers = all_layer_features.shape[1]
# hidden_dim = all_layer_features.shape[2]
#
# hh_cos_means, bb_cos_means, hb_cos_means = [], [], []
# hh_angle_means, bb_angle_means, hb_angle_means = [], [], []
#
# for layer_idx in range(num_layers):
#     # 取该层的 feature
#     feats = all_layer_features[:, layer_idx, :]  # shape (num_samples, hidden_dim)
#     harm_feats = feats[harm_idx]
#     benign_feats = feats[benign_idx]
#
#     # ---------------- H-H pairs ----------------
#     hh_cos_sims = []
#     hh_angles = []
#     for i in range(len(harm_feats)):
#         for j in range(i+1, len(harm_feats)):
#             cos_sim = cosine_similarity(harm_feats[i], harm_feats[j])
#             angle = angle_in_degrees(harm_feats[i], harm_feats[j])
#             hh_cos_sims.append(cos_sim)
#             hh_angles.append(angle)
#     hh_cos_means.append(np.mean(hh_cos_sims))
#     hh_angle_means.append(np.mean(hh_angles))
#
#     # ---------------- B-B pairs ----------------
#     bb_cos_sims = []
#     bb_angles = []
#     for i in range(len(benign_feats)):
#         for j in range(i+1, len(benign_feats)):
#             cos_sim = cosine_similarity(benign_feats[i], benign_feats[j])
#             angle = angle_in_degrees(benign_feats[i], benign_feats[j])
#             bb_cos_sims.append(cos_sim)
#             bb_angles.append(angle)
#     bb_cos_means.append(np.mean(bb_cos_sims))
#     bb_angle_means.append(np.mean(bb_angles))
#
#     # ---------------- H-B pairs ----------------
#     hb_cos_sims = []
#     hb_angles = []
#     for i in range(len(harm_feats)):
#         for j in range(len(benign_feats)):
#             cos_sim = cosine_similarity(harm_feats[i], benign_feats[j])
#             angle = angle_in_degrees(harm_feats[i], benign_feats[j])
#             hb_cos_sims.append(cos_sim)
#             hb_angles.append(angle)
#     hb_cos_means.append(np.mean(hb_cos_sims))
#     hb_angle_means.append(np.mean(hb_angles))

# import matplotlib.pyplot as plt
#
# layers = np.arange(num_layers)  # 0 = embedding
#
# plt.figure(figsize=(12,5))
# plt.subplot(1,2,1)
# plt.plot(layers, hh_cos_means, label='H-H')
# plt.plot(layers, bb_cos_means, label='B-B')
# plt.plot(layers, hb_cos_means, label='H-B')
# plt.xlabel('Layer')
# plt.ylabel('Mean Cosine Similarity')
# plt.title('Cosine Similarity across Layers')
# plt.legend()
#
# plt.subplot(1,2,2)
# plt.plot(layers, hh_angle_means, label='H-H')
# plt.plot(layers, bb_angle_means, label='B-B')
# plt.plot(layers, hb_angle_means, label='H-B')
# plt.xlabel('Layer')
# plt.ylabel('Mean Angle (degrees)')
# plt.title('Angle across Layers')
# plt.legend()
#
# plt.tight_layout()
# plt.savefig('similarity_analysis.png')
# plt.show()

from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

indices = np.arange(len(all_layer_features))

idx_train, idx_test = train_test_split(
    indices, test_size=0.2, random_state=443
)

X_train, X_test = all_layer_features[idx_train], all_layer_features[idx_test]
y_train, y_test = labels[idx_train], labels[idx_test]
num_layers = all_layer_features.shape[1]
hidden_dim = all_layer_features.shape[2]

results = []

layer_params = {}
for layer_idx in range(num_layers):

    X_train_layer = X_train[:, layer_idx, :]
    X_test_layer = X_test[:, layer_idx, :]

    w_i = layer_params[layer_idx]['w']
    b_i = layer_params[layer_idx]['b']


    logits = X_test_layer @ w_i + b_i
    y_pred = (logits > 0).astype(int)

    # clf = LogisticRegression(
    #     penalty='l1',       # 稀疏化
    #     solver='saga',
    #     max_iter=500,
    #     C=1.0              # 控制稀疏度
    # )
    #
    # clf.fit(X_train_layer, y_train)
    # y_pred = clf.predict(X_test_layer)
    acc = accuracy_score(y_test, y_pred)
    results.append(acc)

    # # ==== 分别统计 label=0 / label=1 的平均 logit ====
    # mask_benign = (y_test == 0)
    # mask_harmful = (y_test == 1)
    #
    # benign_logits = logits[mask_benign]
    # harmful_logits = logits[mask_harmful]
    #
    # benign_avg = np.mean(benign_logits) if benign_logits.size > 0 else float('nan')
    # harmful_avg = np.mean(harmful_logits) if harmful_logits.size > 0 else float('nan')
    #
    # print(f"Layer {layer_idx}:")
    # print(f"  Accuracy: {acc:.4f}")
    # print(f"  Avg logit (label=0 benign):   {benign_avg:.4f}")
    # print(f"  Avg logit (label=1 harmful):  {harmful_avg:.4f}")
    # print(f"  Logit gap (benign - harmful): {benign_avg - harmful_avg:.4f}")

    # print('GT:', y_test)
    # print('Pred:', y_pred)
    # print("-----------------------------------")

    print(f"Layer {layer_idx}: Accuracy = {acc:.4f} Rate: {len(y_test) * acc}/{len(y_test)}", )

    # w = clf.coef_[0]  # shape: [hidden_dim]
    # b = clf.intercept_[0]

    # layer_params[layer_idx] = {"w": w, "b": b}

    # np.save("cls_finetuned_model_gemma-2-9b-it.npy", layer_params)