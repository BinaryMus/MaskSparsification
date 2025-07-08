import torch
import torch.nn as nn
from transformers import LlamaForCausalLM, Qwen3ForCausalLM, AutoTokenizer, DynamicCache
from transformers.masking_utils import create_causal_mask

cfg = {
    'A' : [64,     'M', 128,      'M', 256, 256,           'M', 512, 512,           'M', 512, 512,           'M'],
    'B' : [64, 64, 'M', 128, 128, 'M', 256, 256,           'M', 512, 512,           'M', 512, 512,           'M'],
    'D' : [64, 64, 'M', 128, 128, 'M', 256, 256, 256,      'M', 512, 512, 512,      'M', 512, 512, 512,      'M'],
    'E' : [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 512, 512, 'M', 512, 512, 512, 512, 'M']
}

def vgg16(cutlayer, num_classes=10):
    model = VGG(make_layers(cfg['D'], batch_norm=True), num_class=num_classes)
    layers = [
        model.features[0:3],
        model.features[3:7],
        model.features[7:10],
        model.features[10:14],
        model.features[14:17],
        model.features[17:20],
        model.features[20:24],
        model.features[24:27],
        model.features[27:30],
        model.features[30:34],
        model.features[34:37],
        model.features[37:40],
        model.features[40:44],
        nn.Flatten(),
        nn.Sequential(*([nn.Flatten()] + list(model.classifier[0:3]))),
        model.classifier[3:6],
        model.classifier[6]
    ]
    c, s = split_vison_model(layers, cutlayer)
    optimizer_c = torch.optim.SGD(c.parameters(), lr=0.1, momentum=0.9, weight_decay=5e-4)
    optimizer_s = torch.optim.SGD(s.parameters(), lr=0.1, momentum=0.9, weight_decay=5e-4)
    scheduler_c = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer_c, T_max=200)
    scheduler_s = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer_s, T_max=200)
    return c, s, model, optimizer_c, optimizer_s, scheduler_c, scheduler_s


def resnet18(cutlayer, num_classes=100):
    model = ResNet(BasicBlock, [2, 2, 2, 2], num_classes=num_classes)
    layers = [
        model.conv1,
        model.conv2_x[0],
        model.conv2_x[1],
        model.conv3_x[0],
        model.conv3_x[1],
        model.conv4_x[0],
        model.conv4_x[1],
        model.conv5_x[0],
        model.conv5_x[1],
        nn.Sequential(model.avg_pool, nn.Flatten()),
        model.fc
    ]
    c, s = split_vison_model(layers, cutlayer)
    optimizer_c = torch.optim.SGD(c.parameters(), lr=0.1, momentum=0.9, weight_decay=5e-4)
    optimizer_s = torch.optim.SGD(s.parameters(), lr=0.1, momentum=0.9, weight_decay=5e-4)
    scheduler_c = torch.optim.lr_scheduler.MultiStepLR(optimizer_c, milestones=[60, 120, 160], gamma=0.2)
    scheduler_s = torch.optim.lr_scheduler.MultiStepLR(optimizer_s, milestones=[60, 120, 160], gamma=0.2)
    return c, s, model, optimizer_c, optimizer_s, scheduler_c, scheduler_s

def llama3(cutlayer, device):
    model_name = "unsloth/Llama-3.2-1B"
    model = LlamaForCausalLM.from_pretrained(model_name, local_files_only=True)
    model = model.to(device)
    tokenizer = AutoTokenizer.from_pretrained(model_name, local_files_only=True)
    layers = []
    embd = LlamaEmbedLayer(model.model.embed_tokens, model.model.config, model.model.rotary_emb)
    layers.append(embd)
    for decoder_layer in model.model.layers[: model.model.config.num_hidden_layers]:
        layers.append(LlamaDecoderLayer(decoder_layer))
    linear = LlamaFinalLayer(model.model.norm, model.lm_head)
    layers.append(linear)
    c, s = split_nlp_model(layers, cutlayer)
    optimizer_c = torch.optim.SGD(c.parameters(), lr=1e-4)
    optimizer_s = torch.optim.SGD(s.parameters(), lr=1e-4)
    return c, s, model, tokenizer, optimizer_c, optimizer_s

def qwen3(cutlayer, device):
    model_name = "Qwen/Qwen3-1.7B"
    model = Qwen3ForCausalLM.from_pretrained(model_name, local_files_only=True)
    model = model.to(device)
    tokenizer = AutoTokenizer.from_pretrained(model_name, local_files_only=True)
    layers = []
    embd = QwenEmbedLayer(model.model.embed_tokens, model.model.config, model.model.rotary_emb)
    layers.append(embd)
    for decoder_layer in model.model.layers[: model.model.config.num_hidden_layers]:
        layers.append(QwenDecoderLayer(decoder_layer))
    linear = QwenFinalLayer(model.model.norm, model.lm_head)
    layers.append(linear)
    c, s = split_nlp_model(layers, cutlayer)
    optimizer_c = torch.optim.SGD(c.parameters(), lr=1e-4)
    optimizer_s = torch.optim.SGD(s.parameters(), lr=1e-4)
    return c, s, model, tokenizer, optimizer_c, optimizer_s

def split_vison_model(layers, cutlayer):
    assert 0 < cutlayer < len(layers), f"cutlayer must be less than {len(layers)} and greater than 0"
    client_model = torch.nn.Sequential(*layers[:cutlayer])
    server_model = torch.nn.Sequential(*layers[cutlayer:])
    return client_model, server_model

def split_nlp_model(layers, cutlayer):
    assert 0 < cutlayer < len(layers) - 2, f"cutlayer must be less than {len(layers) - 2} and greater than 0"
    embd, linear = layers[0], layers[-1]
    client_model = NLP_client_model(embd, layers[1:cutlayer+1])
    server_model = NLP_server_model(layers[cutlayer+1:-1], linear)
    return client_model, server_model

class NLP_client_model(nn.Module):
    def __init__(self, embd, layers):
        super().__init__()
        self.embd = embd
        self.layers = nn.ModuleList(layers)

    def forward(self, input_ids, attention_mask):
        output = list(self.embd(input_ids, attention_mask))
        for i in self.layers:
            hidden_state = i(*output)
            output[0] = hidden_state
        return output

class NLP_server_model(nn.Module):
    def __init__(self, layers, linear):
        super().__init__()
        self.layers = nn.ModuleList(layers)
        self.linear = linear
    
    def forward(self, output):
        for i in self.layers:
            hidden_state = i(*output)
            output[0] = hidden_state
        logits = self.linear(output[0])
        return logits

class VGG(nn.Module):
    def __init__(self, features, num_class=100):
        super().__init__()
        self.features = features
        self.classifier = nn.Sequential(
            nn.Linear(512, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(4096, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(4096, num_class)
        )

    def forward(self, x):
        output = self.features(x)
        output = output.view(output.size()[0], -1)
        output = self.classifier(output)
        return output

def make_layers(cfg, batch_norm=False):
    layers = []
    input_channel = 3
    for l in cfg:
        if l == 'M':
            layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
            continue
        layers += [nn.Conv2d(input_channel, l, kernel_size=3, padding=1)]
        if batch_norm:
            layers += [nn.BatchNorm2d(l)]

        layers += [nn.ReLU(inplace=True)]
        input_channel = l
    return nn.Sequential(*layers)


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_channels, out_channels, stride=1):
        super().__init__()

        self.residual_function = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels * BasicBlock.expansion, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels * BasicBlock.expansion)
        )

        self.shortcut = nn.Sequential()

        if stride != 1 or in_channels != BasicBlock.expansion * out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels * BasicBlock.expansion, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels * BasicBlock.expansion)
            )

    def forward(self, x):
        return nn.ReLU(inplace=True)(self.residual_function(x) + self.shortcut(x))
    
class ResNet(nn.Module):

    def __init__(self, block, num_block, num_classes=100):
        super().__init__()

        self.in_channels = 64

        self.conv1 = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True))

        self.conv2_x = self._make_layer(block, 64, num_block[0], 1)
        self.conv3_x = self._make_layer(block, 128, num_block[1], 2)
        self.conv4_x = self._make_layer(block, 256, num_block[2], 2)
        self.conv5_x = self._make_layer(block, 512, num_block[3], 2)
        self.avg_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512 * block.expansion, num_classes)

    def _make_layer(self, block, out_channels, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_channels, out_channels, stride))
            self.in_channels = out_channels * block.expansion

        return nn.Sequential(*layers)

    def forward(self, x):
        output = self.conv1(x)
        output = self.conv2_x(output)
        output = self.conv3_x(output)
        output = self.conv4_x(output)
        output = self.conv5_x(output)
        output = self.avg_pool(output)
        output = output.view(output.size(0), -1)
        output = self.fc(output)
        return output
    

class LlamaEmbedLayer(nn.Module):
    def __init__(self, embed_tokens, config, rotary_emb):
        super().__init__()
        self.embed_tokens = embed_tokens
        self.config = config
        self.rotary_emb = rotary_emb

    def forward(self, input_ids, attention_mask):
        inputs_embeds = self.embed_tokens(input_ids)
        past_key_values = DynamicCache()
        cache_position = torch.arange(
            0, inputs_embeds.shape[1], device=inputs_embeds.device
        )
        position_ids = cache_position.unsqueeze(0)
        causal_mask = create_causal_mask(
            config=self.config,
            input_embeds=inputs_embeds,
            attention_mask=attention_mask,
            cache_position=cache_position,
            past_key_values=past_key_values,
        )
        hidden_states = inputs_embeds
        position_embeddings = self.rotary_emb(hidden_states, position_ids)
        return hidden_states, causal_mask, position_ids, past_key_values, cache_position, position_embeddings

class QwenEmbedLayer(nn.Module):
    def __init__(self, embed_tokens, config, rotary_emb):
        super().__init__()
        self.embed_tokens = embed_tokens
        self.config = config
        self.rotary_emb = rotary_emb

    def forward(self, input_ids, attention_mask):
        inputs_embeds = self.embed_tokens(input_ids)
        past_key_values = DynamicCache()
        cache_position = torch.arange(0, inputs_embeds.shape[1], device=inputs_embeds.device)
        position_ids = cache_position.unsqueeze(0)
        mask_kwargs = {
            "config": self.config,
            "input_embeds": inputs_embeds,
            "attention_mask": attention_mask,
            "cache_position": cache_position,
            "past_key_values": past_key_values,
        }
        causal_mask_mapping = {
            "full_attention": create_causal_mask(**mask_kwargs),
        }
        hidden_states = inputs_embeds
        position_embeddings = self.rotary_emb(hidden_states, position_ids)
        return hidden_states, causal_mask_mapping, position_ids, past_key_values, cache_position, position_embeddings

class LlamaDecoderLayer(nn.Module):
    def __init__(self, layer):
        super().__init__()
        self.layer = layer
    
    def forward(self, hidden_states, causal_mask, position_ids, past_key_values, cache_position, position_embeddings):
        layer_outputs = self.layer(
            hidden_states,
            attention_mask=causal_mask,
            position_ids=position_ids,
            past_key_value=past_key_values,
            cache_position=cache_position,
            position_embeddings=position_embeddings,
        )
        hidden_states = layer_outputs[0]
        return hidden_states

class QwenDecoderLayer(nn.Module):
    def __init__(self, layer):
        super().__init__()
        self.layer = layer
    
    def forward(self, hidden_states, causal_mask_mapping, position_ids, past_key_values, cache_position, position_embeddings):
        layer_outputs = self.layer(
            hidden_states,
            attention_mask=causal_mask_mapping[self.layer.attention_type],
            position_ids=position_ids,
            past_key_value=past_key_values,
            output_attentions=False,
            use_cache=True,
            cache_position=cache_position,
            position_embeddings=position_embeddings,
        )
        hidden_states = layer_outputs[0]
        return hidden_states
    
class LlamaFinalLayer(nn.Module):
    def __init__(self, norm, lm_head):
        super().__init__()
        self.norm = norm
        self.lm_head = lm_head

    def forward(self, hidden_states):
        hidden_states = self.norm(hidden_states)
        slice_indices = slice(0, None)
        logits = self.lm_head(hidden_states[:, slice_indices, :])
        return logits
    
class QwenFinalLayer(nn.Module):
    def __init__(self, norm, lm_head):
        super().__init__()
        self.norm = norm
        self.lm_head = lm_head
    
    def forward(self, hidden_states):
        hidden_states = self.norm(hidden_states)
        slice_indices = slice(0, None)
        logits = self.lm_head(hidden_states[:, slice_indices, :])
        return logits
    