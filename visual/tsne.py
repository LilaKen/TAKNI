import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from math import sqrt
import math
from utils.seed import set_seeds

set_seeds(seed_value=2023)

class TriangularCausalMask():
    def __init__(self, B, L, device="cpu"):
        mask_shape = [B, 1, L, L]
        with torch.no_grad():
            self._mask = torch.triu(torch.ones(mask_shape, dtype=torch.bool), diagonal=1).to(device)

    @property
    def mask(self):
        return self._mask


class ProbMask():
    def __init__(self, B, H, L, index, scores, device="cpu"):
        _mask = torch.ones(L, scores.shape[-1], dtype=torch.bool).to(device).triu(1)
        _mask_ex = _mask[None, None, :].expand(B, H, L, scores.shape[-1])
        indicator = _mask_ex[torch.arange(B)[:, None, None],
                    torch.arange(H)[None, :, None],
                    index, :].to(device)
        self._mask = indicator.view(scores.shape).to(device)

    @property
    def mask(self):
        return self._mask


class PositionalEmbedding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super(PositionalEmbedding, self).__init__()
        # Compute the positional encodings once in log space.
        pe = torch.zeros(max_len, d_model).float()
        pe.require_grad = False

        position = torch.arange(0, max_len).float().unsqueeze(1)
        div_term = (torch.arange(0, d_model, 2).float() * -(math.log(10000.0) / d_model)).exp()

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        return self.pe[:, :x.size(1)]


class TokenEmbedding(nn.Module):
    def __init__(self, c_in, d_model):
        super(TokenEmbedding, self).__init__()
        padding = 1 if torch.__version__ >= '1.5.0' else 2
        self.tokenConv = nn.Conv1d(in_channels=c_in, out_channels=d_model,
                                   kernel_size=3, padding=padding, padding_mode='circular')
        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='leaky_relu')

    def forward(self, x):
        x = self.tokenConv(x).transpose(1, 2)
        return x


class FixedEmbedding(nn.Module):
    def __init__(self, c_in, d_model):
        super(FixedEmbedding, self).__init__()

        w = torch.zeros(c_in, d_model).float()
        w.require_grad = False

        position = torch.arange(0, c_in).float().unsqueeze(1)
        div_term = (torch.arange(0, d_model, 2).float() * -(math.log(10000.0) / d_model)).exp()

        w[:, 0::2] = torch.sin(position * div_term)
        w[:, 1::2] = torch.cos(position * div_term)

        self.emb = nn.Embedding(c_in, d_model)
        self.emb.weight = nn.Parameter(w, requires_grad=False)

    def forward(self, x):
        return self.emb(x).detach()


class TemporalEmbedding(nn.Module):
    def __init__(self, d_model, embed_type='fixed', freq='h'):
        super(TemporalEmbedding, self).__init__()

        minute_size = 4;
        hour_size = 24
        weekday_size = 7;
        day_size = 32;
        month_size = 13

        Embed = FixedEmbedding if embed_type == 'fixed' else nn.Embedding
        if freq == 't':
            self.minute_embed = Embed(minute_size, d_model)
        self.hour_embed = Embed(hour_size, d_model)
        self.weekday_embed = Embed(weekday_size, d_model)
        self.day_embed = Embed(day_size, d_model)
        self.month_embed = Embed(month_size, d_model)

    def forward(self, x):
        x = x.long()

        minute_x = self.minute_embed(x[:, :, 4]) if hasattr(self, 'minute_embed') else 0.
        hour_x = self.hour_embed(x[:, :, 3])
        weekday_x = self.weekday_embed(x[:, :, 2])
        day_x = self.day_embed(x[:, :, 1])
        month_x = self.month_embed(x[:, :, 0])

        return hour_x + weekday_x + day_x + month_x + minute_x


class TimeFeatureEmbedding(nn.Module):
    def __init__(self, d_model, embed_type='timeF', freq='h'):
        super(TimeFeatureEmbedding, self).__init__()

        freq_map = {'h': 4, 't': 5, 's': 6, 'm': 1, 'a': 1, 'w': 2, 'd': 3, 'b': 3}
        d_inp = freq_map[freq]
        self.embed = nn.Linear(d_inp, d_model)

    def forward(self, x):
        return self.embed(x)


class DataEmbedding(nn.Module):
    def __init__(self, c_in, d_model, embed_type='fixed', freq='h', dropout=0.1):
        super(DataEmbedding, self).__init__()

        self.value_embedding = TokenEmbedding(c_in=c_in, d_model=d_model)
        self.position_embedding = PositionalEmbedding(d_model=d_model)
        self.temporal_embedding = TemporalEmbedding(d_model=d_model, embed_type=embed_type,
                                                    freq=freq) if embed_type != 'timeF' else TimeFeatureEmbedding(
            d_model=d_model, embed_type=embed_type, freq=freq)

        self.dropout = nn.Dropout(p=dropout)

    def forward(self, x):
        x = self.value_embedding(x) + self.position_embedding(x)

        return self.dropout(x)


class FullAttention(nn.Module):
    def __init__(self, mask_flag=True, factor=5, scale=None, attention_dropout=0.1, output_attention=False):
        super(FullAttention, self).__init__()
        self.scale = scale
        self.mask_flag = mask_flag
        self.output_attention = output_attention
        self.dropout = nn.Dropout(attention_dropout)

    def forward(self, queries, keys, values, attn_mask):
        B, L, H, E = queries.shape
        _, S, _, D = values.shape
        scale = self.scale or 1. / sqrt(E)

        scores = torch.einsum("blhe,bshe->bhls", queries, keys)
        if self.mask_flag:
            if attn_mask is None:
                attn_mask = TriangularCausalMask(B, L, device=queries.device)

            scores.masked_fill_(attn_mask.mask, -np.inf)

        A = self.dropout(torch.softmax(scale * scores, dim=-1))
        V = torch.einsum("bhls,bshd->blhd", A, values)

        if self.output_attention:
            return (V.contiguous(), A)
        else:
            return (V.contiguous(), None)


class ProbAttention(nn.Module):
    def __init__(self, mask_flag=True, factor=5, scale=None, attention_dropout=0.1, output_attention=False):
        super(ProbAttention, self).__init__()
        self.factor = factor
        self.scale = scale
        self.mask_flag = mask_flag
        self.output_attention = output_attention
        self.dropout = nn.Dropout(attention_dropout)

    def _prob_QK(self, Q, K, sample_k, n_top):  # n_top: c*ln(L_q)
        # Q [B, H, L, D]
        B, H, L_K, E = K.shape
        _, _, L_Q, _ = Q.shape

        # calculate the sampled Q_K
        K_expand = K.unsqueeze(-3).expand(B, H, L_Q, L_K, E)
        index_sample = torch.randint(L_K, (L_Q, sample_k))  # real U = U_part(factor*ln(L_k))*L_q
        K_sample = K_expand[:, :, torch.arange(L_Q).unsqueeze(1), index_sample, :]
        Q_K_sample = torch.matmul(Q.unsqueeze(-2), K_sample.transpose(-2, -1)).squeeze(-2)

        # find the Top_k query with sparisty measurement
        M = Q_K_sample.max(-1)[0] - torch.div(Q_K_sample.sum(-1), L_K)
        M_top = M.topk(n_top, sorted=False)[1]

        # use the reduced Q to calculate Q_K
        Q_reduce = Q[torch.arange(B)[:, None, None],
                   torch.arange(H)[None, :, None],
                   M_top, :]  # factor*ln(L_q)
        Q_K = torch.matmul(Q_reduce, K.transpose(-2, -1))  # factor*ln(L_q)*L_k

        return Q_K, M_top

    def _get_initial_context(self, V, L_Q):
        B, H, L_V, D = V.shape
        if not self.mask_flag:
            # V_sum = V.sum(dim=-2)
            V_sum = V.mean(dim=-2)
            contex = V_sum.unsqueeze(-2).expand(B, H, L_Q, V_sum.shape[-1]).clone()
        else:  # use mask
            assert (L_Q == L_V)  # requires that L_Q == L_V, i.e. for self-attention only
            contex = V.cumsum(dim=-2)
        return contex

    def _update_context(self, context_in, V, scores, index, L_Q, attn_mask):
        B, H, L_V, D = V.shape

        if self.mask_flag:
            attn_mask = ProbMask(B, H, L_Q, index, scores, device=V.device)
            scores.masked_fill_(attn_mask.mask, -np.inf)

        attn = torch.softmax(scores, dim=-1)  # nn.Softmax(dim=-1)(scores)

        context_in[torch.arange(B)[:, None, None],
        torch.arange(H)[None, :, None],
        index, :] = torch.matmul(attn, V).type_as(context_in)
        if self.output_attention:
            attns = (torch.ones([B, H, L_V, L_V]) / L_V).type_as(attn).to(attn.device)
            attns[torch.arange(B)[:, None, None], torch.arange(H)[None, :, None], index, :] = attn
            return (context_in, attns)
        else:
            return (context_in, None)

    def forward(self, queries, keys, values, attn_mask):
        B, L_Q, H, D = queries.shape
        _, L_K, _, _ = keys.shape

        queries = queries.transpose(2, 1)
        keys = keys.transpose(2, 1)
        values = values.transpose(2, 1)

        U_part = self.factor * np.ceil(np.log(L_K)).astype('int').item()  # c*ln(L_k)
        u = self.factor * np.ceil(np.log(L_Q)).astype('int').item()  # c*ln(L_q)

        U_part = U_part if U_part < L_K else L_K
        u = u if u < L_Q else L_Q

        scores_top, index = self._prob_QK(queries, keys, sample_k=U_part, n_top=u)

        # add scale factor
        scale = self.scale or 1. / sqrt(D)
        if scale is not None:
            scores_top = scores_top * scale
        # get the context
        context = self._get_initial_context(values, L_Q)
        # update the context with selected top_k queries
        context, attn = self._update_context(context, values, scores_top, index, L_Q, attn_mask)

        return context.transpose(2, 1).contiguous(), attn


class AttentionLayer(nn.Module):
    def __init__(self, attention, d_model, n_heads,
                 d_keys=None, d_values=None, mix=False):
        super(AttentionLayer, self).__init__()

        d_keys = d_keys or (d_model // n_heads)
        d_values = d_values or (d_model // n_heads)

        self.inner_attention = attention
        self.query_projection = nn.Linear(d_model, d_keys * n_heads)
        self.key_projection = nn.Linear(d_model, d_keys * n_heads)
        self.value_projection = nn.Linear(d_model, d_values * n_heads)
        self.out_projection = nn.Linear(d_values * n_heads, d_model)
        self.n_heads = n_heads
        self.mix = mix

    def forward(self, queries, keys, values, attn_mask):
        B, L, _ = queries.shape
        _, S, _ = keys.shape
        H = self.n_heads

        queries = self.query_projection(queries).view(B, L, H, -1)
        keys = self.key_projection(keys).view(B, S, H, -1)
        values = self.value_projection(values).view(B, S, H, -1)

        out, attn = self.inner_attention(
            queries,
            keys,
            values,
            attn_mask
        )
        if self.mix:
            out = out.transpose(2, 1).contiguous()
        out = out.view(B, L, -1)

        return self.out_projection(out), attn


class ConvLayer(nn.Module):
    def __init__(self, c_in):
        super(ConvLayer, self).__init__()
        padding = 1 if torch.__version__ >= '1.5.0' else 2
        self.downConv = nn.Conv1d(in_channels=c_in,
                                  out_channels=c_in,
                                  kernel_size=3,
                                  padding=padding,
                                  padding_mode='circular')
        self.norm = nn.BatchNorm1d(c_in)
        self.activation = nn.ELU()
        self.maxPool = nn.MaxPool1d(kernel_size=3, stride=2, padding=1)

    def forward(self, x):
        x = self.downConv(x.permute(0, 2, 1))
        x = self.norm(x)
        x = self.activation(x)
        x = self.maxPool(x)
        x = x.transpose(1, 2)
        return x


class EncoderLayer(nn.Module):
    def __init__(self, attention, d_model, d_ff=None, dropout=0.1, activation="relu"):
        super(EncoderLayer, self).__init__()
        d_ff = d_ff or 4 * d_model
        self.attention = attention
        self.conv1 = nn.Conv1d(in_channels=d_model, out_channels=d_ff, kernel_size=1)
        self.conv2 = nn.Conv1d(in_channels=d_ff, out_channels=d_model, kernel_size=1)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        self.activation = F.relu if activation == "relu" else F.gelu

    def forward(self, x, attn_mask=None):
        # x [B, L, D]
        # x = x + self.dropout(self.attention(
        #     x, x, x,
        #     attn_mask = attn_mask
        # ))
        new_x, attn = self.attention(
            x, x, x,
            attn_mask=attn_mask
        )
        x = x + self.dropout(new_x)

        y = x = self.norm1(x)
        y = self.dropout(self.activation(self.conv1(y.transpose(-1, 1))))
        y = self.dropout(self.conv2(y).transpose(-1, 1))

        return self.norm2(x + y), attn


class Encoder(nn.Module):
    def __init__(self, attn_layers, conv_layers=None, norm_layer=None):
        super(Encoder, self).__init__()
        self.attn_layers = nn.ModuleList(attn_layers)
        self.conv_layers = nn.ModuleList(conv_layers) if conv_layers is not None else None
        self.norm = norm_layer

    def forward(self, x, attn_mask=None):
        # x [B, L, D]
        attns = []
        if self.conv_layers is not None:
            for attn_layer, conv_layer in zip(self.attn_layers, self.conv_layers):
                x, attn = attn_layer(x, attn_mask=attn_mask)
                x = conv_layer(x)
                attns.append(attn)
            x, attn = self.attn_layers[-1](x, attn_mask=attn_mask)
            attns.append(attn)
        else:
            for attn_layer in self.attn_layers:
                x, attn = attn_layer(x, attn_mask=attn_mask)
                attns.append(attn)

        if self.norm is not None:
            x = self.norm(x)

        return x, attns


class EncoderStack(nn.Module):
    def __init__(self, encoders, inp_lens):
        super(EncoderStack, self).__init__()
        self.encoders = nn.ModuleList(encoders)
        self.inp_lens = inp_lens

    def forward(self, x, attn_mask=None):
        # x [B, L, D]
        x_stack = [];
        attns = []
        for i_len, encoder in zip(self.inp_lens, self.encoders):
            inp_len = x.shape[1] // (2 ** i_len)
            x_s, attn = encoder(x[:, -inp_len:, :])
            x_stack.append(x_s);
            attns.append(attn)
        x_stack = torch.cat(x_stack, -2)

        return x_stack, attns


class InformerEncoder(nn.Module):
    def __init__(self, enc_in,
                 factor=5, d_model=512, n_heads=8, e_layers=3, d_ff=512,
                 dropout=0.0, attn='prob', embed='fixed', freq='h', activation='gelu',
                 output_attention=False, distil=True, pretrained=False):
        super(InformerEncoder, self).__init__()
        self.attn = attn
        self.output_attention = output_attention

        # Encoding
        self.enc_embedding = DataEmbedding(enc_in, d_model, embed, freq, dropout)
        # Attention
        Attn = ProbAttention if attn == 'prob' else FullAttention
        # Encoder
        self.encoder = Encoder(
            [
                EncoderLayer(
                    AttentionLayer(Attn(False, factor, attention_dropout=dropout, output_attention=output_attention),
                                   d_model, n_heads, mix=False),
                    d_model,
                    d_ff,
                    dropout=dropout,
                    activation=activation
                ) for l in range(e_layers)
            ],
            [
                ConvLayer(
                    d_model
                ) for l in range(e_layers - 1)
            ] if distil else None,
            norm_layer=torch.nn.LayerNorm(d_model)
        )

    def forward(self, x_enc, enc_self_mask=None):
        enc_out = self.enc_embedding(x_enc)
        enc_out, attns = self.encoder(enc_out, attn_mask=enc_self_mask)
        enc_out = enc_out.view(len(enc_out), -1)

        return enc_out


# convnet without the last layer
class informer_features_fft(nn.Module):
    def __init__(self, pretrained=False):
        super(informer_features_fft, self).__init__()
        self.model_informer = InformerEncoder(enc_in=1, d_model=16, n_heads=8, e_layers=6, d_ff=256)
        self.__in_features = 256

    def forward(self, x):
        x = self.model_informer(x)
        return x

    def output_num(self):
        return self.__in_features


# model = informer_features_fft()
# x = torch.randn((64, 1, 512))
# y = model(x)
# print(y.shape)

# 初始化模型
model = informer_features_fft(pretrained=False)

# /home/public/ken/UDTL/checkpoint/informer/CWRU/FFT/informer_features_fft_1d_CWRUFFT_0_to_1_2023_False_CDA+E_True_NWD_adam_step/60-1.0000-best_model.pth
# /home/public/ken/UDTL/checkpoint/informer/JNU/FFT/informer_features_fft_1d_JNUFFT_0_to_1_2023_False_CDA+E_True_NWD_adam_step/85-0.9983-best_model.pth
# /home/public/ken/UDTL/checkpoint/informer/PHM/FFT/informer_features_fft_1d_PHMFFT_0_to_1_2023_False_CDA+E_True_NWD_adam_step/157-0.5096-best_model.pth
# /home/public/ken/UDTL/checkpoint/informer/PU/FFT/informer_features_fft_1d_PUFFT_0_to_1_2023_False_CDA+E_True_NWD_adam_step/235-0.7055-best_model.pth
# /home/public/ken/UDTL/checkpoint/informer/SEU/FFT/informer_features_fft_1d_SEUFFT_0_to_1_2023_False_CDA+E_True_NWD_adam_step/61-0.6188-best_model.pth



# 加载权重文件
weights_path = '../checkpoint/informer/SEU/FFT/informer_features_fft_1d_SEUFFT_0_to_1_2023_False_CDA+E_True_NWD_adam_step/61-0.6188-best_model.pth'  # 权重文件路径

device = torch.device('cpu')  # 自动选择设备

# 将模型移动到指定设备
model = model.to(device)

bottleneck_layer = nn.Sequential(nn.Linear(model.output_num(), 256),
                                      nn.ReLU(inplace=True), nn.Dropout())
classifier_layer = nn.Linear(256, 9)

model_all = nn.Sequential(model, bottleneck_layer, classifier_layer)


# 加载权重
try:
    checkpoint = torch.load(weights_path, map_location=device)  # 加载权重文件
    model_all.load_state_dict(checkpoint)  # 加载权重到模型中
    print("模型权重加载成功！")
except FileNotFoundError:
    print(f"错误：权重文件 {weights_path} 未找到，请检查路径是否正确。")
except RuntimeError as e:
    print(f"错误：加载权重时发生问题，可能是模型结构与权重不匹配。\n详细信息：{e}")

# 设置模型为评估模式（如果仅用于推理）
model_all.eval()

import datasets

# Load the datasets
Dataset = getattr(datasets, 'SEUFFT')
datasets = {}
transfer_task = [[0], [1]]
if isinstance(transfer_task[0], str):
    # print(args.transfer_task)
    transfer_task = eval("".join(transfer_task))

datasets['source_train'], datasets['source_val'], datasets['target_train'], datasets[
    'target_val'] = Dataset('../dataset/SEU', transfer_task, '0-1').data_split(
    transfer_learning=True)

dataloaders = {x: torch.utils.data.DataLoader(datasets[x], batch_size=32,
                                                   shuffle=(True if x.split('_')[1] == 'train' else False),
                                                   num_workers=8,
                                                   pin_memory=(True if device == 'cuda' else False),
                                                   drop_last=(True if 1 and x.split('_')[
                                                       1] == 'train' else False))
                    for x in ['source_train', 'source_val', 'target_train', 'target_val']}

from sklearn.manifold import TSNE
import matplotlib
matplotlib.use('Agg')  # 或者使用 'Agg' 后端
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
import os
# 设置全局字体为 Times New Roman，字体大小默认可设（但我们会显式指定）
# 加载本地字体文件并注册
# 指向你实际存放字体的路径（注意：这里用 Regular 字体）
font_path = '/usr/share/fonts/truetype/times-new-roman/times.ttf'
font_name = 'Times New Roman'

if os.path.exists(font_path):
    try:
        # 注册字体到 Matplotlib
        fm.fontManager.addfont(font_path)
        # 设置全局字体（必须使用字体的 family name，通常为 "Times New Roman"）
        plt.rcParams['font.family'] = font_name
        plt.rcParams['axes.unicode_minus'] = False  # 正常显示负号
        print(f"✅ 成功加载并设置字体: {font_name}")
    except Exception as e:
        print(f"❌ 字体加载失败: {e}")
        print("⚠️  回退到默认字体")
        plt.rcParams['font.family'] = 'DejaVu Sans'
        plt.rcParams['axes.unicode_minus'] = False
else:
    print(f"❌ 字体文件未找到: {font_path}")
    print("⚠️  请确认字体已安装并执行 'fc-cache -fv'")
    plt.rcParams['font.family'] = 'DejaVu Sans'
    plt.rcParams['axes.unicode_minus'] = False
# 提取模型的特征提取部分（即 model）
feature_extractor = model_all[0]  # model_all 的第一个部分是 feature_extractor

# 设置为评估模式
feature_extractor.eval()

# 准备存储特征、真实标签和预测标签的列表
all_features = []
all_labels = []
all_predictions = []
# 初始化正确预测计数器
correct_count = 0
total_count = 0
# 使用目标验证集（target_val）提取特征并进行预测
with torch.no_grad():  # 关闭梯度计算以节省内存
    for data, labels in dataloaders['target_val']:  # 使用 target_val 数据集
        # 确保数据在 CPU 上运行
        features = model_all[0](data)  # 使用 feature_extractor 提取特征
        outputs = model_all(data)  # 使用整个模型进行预测
        _, preds = torch.max(outputs, 1)  # 获取预测类别

        # 统计正确预测的数量
        correct_count += (preds == labels).sum().item()
        total_count += labels.size(0)

        # 保存特征、真实标签和预测标签
        all_features.append(features.numpy())  # 直接转换为 numpy 数组
        all_labels.append(labels.numpy())
        all_predictions.append(preds.numpy())

# 将所有特征、真实标签和预测标签拼接成一个数组
all_features = np.vstack(all_features)
all_labels = np.concatenate(all_labels)
all_predictions = np.concatenate(all_predictions)
# 计算准确率
accuracy = correct_count / total_count if total_count > 0 else 0
print(f"模型在目标验证集上的准确率: {accuracy:.4f}")
# 使用 t-SNE 进行降维
tsne = TSNE(n_components=2, perplexity=30, random_state=42)
features_tsne = tsne.fit_transform(all_features)

# 可视化 t-SNE 结果，基于预测标签
plt.figure(figsize=(10, 6))

# 根据预测标签绘制散点图
for label in np.unique(all_predictions):
    indices = all_predictions == label
    plt.scatter(features_tsne[indices, 0], features_tsne[indices, 1])

# 高亮显示预测错误的样本（真实标签与预测标签不一致）
incorrect_indices = all_labels != all_predictions
plt.scatter(features_tsne[incorrect_indices, 0], features_tsne[incorrect_indices, 1],
            edgecolors='red', facecolors='none', s=80, label='Misclassified')

plt.legend(fontsize=32)
# plt.title("t-SNE Visualization of Classification Model Features", fontsize=24)
plt.xlabel("Latent Feature 1", fontsize=32)
plt.ylabel("Latent Feature 2", fontsize=32)

# 设置坐标轴刻度标签字体大小
plt.xticks(fontsize=32)
plt.yticks(fontsize=32)
plt.savefig("seu_0-1_2023_tsne.pdf", format='pdf', bbox_inches='tight')
plt.show()