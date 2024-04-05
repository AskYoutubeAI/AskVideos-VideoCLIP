import os
import glob
import torch
import time
from torch.nn import functional as F
from argparse import Namespace
from PIL import Image

from video_llama.common.config import Config
from video_llama.common.registry import registry
from video_llama.processors.video_processor import load_video, load_video_long, load_video_long_subset
from video_llama.processors import Blip2ImageEvalProcessor

DEVICE = 'cuda'

def init(args):
    cfg = Config(args)
    model_config = cfg.model_cfg
    model_config.device_8bit = args.gpu_id
    model_cls = registry.get_model_class(model_config.arch)
    model = model_cls.from_config(model_config).to(f'cuda:{args.gpu_id}')
    model.eval()
    vis_processor_cfg = cfg.datasets_cfg.webvid.vis_processor.train
    vis_processor = registry.get_processor_class(
        vis_processor_cfg.name).from_config(vis_processor_cfg)
    img_processor = Blip2ImageEvalProcessor()
    return model, vis_processor, img_processor


def load_model(eval_config):
    gpu_id = 0
    args = {'cfg_path': eval_config, 'gpu_id': gpu_id, 'options': None}
    args = Namespace(**args)

    cfg = Config(args)
    model_config = cfg.model_cfg
    model_config.device_8bit = args.gpu_id
    model_cls = registry.get_model_class(model_config.arch)
    model = model_cls.from_config(model_config).to(
        'cuda:{}'.format(args.gpu_id))
    model.eval()

    vis_processor_cfg = cfg.datasets_cfg.webvid.vis_processor.train
    vis_processor = registry.get_processor_class(
        vis_processor_cfg.name).from_config(vis_processor_cfg)

    return model, vis_processor

def upload_video(model, video_path, vis_processor):
    if not isinstance(video_path, str):
        raise NotImplementedError("Non-string video paths are not implemented.")

    video = load_video(
        video_path=video_path,
        n_frms=16,
        #n_frms=8,
        height=224,
        width=224,
        sampling="uniform", return_msg=False
    )
    video = vis_processor.transform(video).unsqueeze(0).to(DEVICE)

    return model.encode_videoQformer_visual(video)[-1].last_hidden_state


def upload_video_long_itg(model, prompt, video_path, vis_processor, clip_len=10):
    if not isinstance(video_path, str):
        raise NotImplementedError("Non-string video paths are not implemented.")

    clips = load_video_long(
        video_path=video_path,
        n_frms=16,
        #n_frms=8,
        height=224,
        width=224,
        clip_len=clip_len,
        sampling="uniform", return_msg=False
    )

    all_clips = []
    for clip in clips:
        clip = vis_processor.transform(clip).unsqueeze(0).to(DEVICE)
        video_query_tokens, past_key_values, frame_atts = model.encode_videoQformer_visual(clip, output_for_itg=True)
        print(' Past Key values: ', past_key_values)
        print('sise: ', past_key_values[0][0].size(), frame_atts.size(), video_query_tokens.size())
        clip = embed_text_itg(model, prompt, past_key_values, query_tokens=video_query_tokens, video_atts=frame_atts)
        print(clip.size())
        #clip = F.normalize(model.vision_proj(clip), dim=-1)
        all_clips.append(clip)

    return all_clips

def upload_video_itm(model, prompt, video_path, vis_processor):
    if not isinstance(video_path, str):
        raise NotImplementedError("Non-string video paths are not implemented.")

    video = load_video(
        video_path=video_path,
        n_frms=16,
        #n_frms=8,
        height=224,
        width=224,
        #clip_len=clip_len,
        sampling="uniform", return_msg=False
    )
    video = vis_processor.transform(video).unsqueeze(0).to(DEVICE)

    video_query_tokens, frame_hidden_state, frame_atts = model.encode_videoQformer_visual(video, output_for_itm=True)
    video_emb = embed_text_itm(model, prompt, frame_hidden_state, query_tokens=video_query_tokens, video_atts=frame_atts)
    video_emb = F.normalize(model.vision_proj(video_emb), dim=-1)
    return video_emb


def upload_video_long_itm(model, prompt, video_path, vis_processor, clip_len=10):
    if not isinstance(video_path, str):
        raise NotImplementedError("Non-string video paths are not implemented.")

    clips = load_video_long(
        video_path=video_path,
        n_frms=16,
        #n_frms=8,
        height=224,
        width=224,
        clip_len=clip_len,
        sampling="uniform", return_msg=False
    )

    all_clips = []
    for clip in clips:
        clip = vis_processor.transform(clip).unsqueeze(0).to(DEVICE)
        # TODO: ITM Here.
        video_query_tokens, frame_hidden_state, frame_atts = model.encode_videoQformer_visual(clip, output_for_itm=True)
        clip = embed_text_itm(model, prompt, frame_hidden_state, query_tokens=video_query_tokens, video_atts=frame_atts)
        print(clip.size())
        #clip = F.normalize(model.vision_proj(clip), dim=-1)
        all_clips.append(clip)

    return all_clips

def upload_video_long(model, video_path, vis_processor, clip_len=10):
    if not isinstance(video_path, str):
        raise NotImplementedError("Non-string video paths are not implemented.")

    clips = load_video_long(
        video_path=video_path,
        n_frms=16,
        #n_frms=8,
        height=224,
        width=224,
        clip_len=clip_len,
        sampling="uniform", return_msg=False
    )

    all_clips = []
    for clip in clips:
        clip = vis_processor.transform(clip).unsqueeze(0).to(DEVICE)
        clip = model.encode_videoQformer_visual(clip)[-1].last_hidden_state
        clip = F.normalize(model.vision_proj(clip), dim=-1)
        all_clips.append(clip)

    return all_clips

def upload_video_long_subsets(model, video_path, vis_processor, subsets, orig_clip_len, clip_len=10):
    if not isinstance(video_path, str):
        raise NotImplementedError("Non-string video paths are not implemented.")

    clips = load_video_long_subset(
        video_path,
        subsets,
        n_frms=16,
        #n_frms=8,
        height=224,
        width=224,
        orig_clip_len=orig_clip_len,
        clip_len=clip_len,
        sampling="uniform", return_msg=False
    )

    all_clips = []
    for clip in clips:
        clip = vis_processor.transform(clip).unsqueeze(0).to(DEVICE)
        clip = model.encode_videoQformer_visual(clip)[-1].last_hidden_state
        clip = F.normalize(model.vision_proj(clip), dim=-1)
        all_clips.append(clip)

    return all_clips

def upload_image_itg(model, prompt, image_path, vis_processor):
    if not isinstance(image_path, str):
        raise NotImplementedError("Non-string image paths are not implemented.")
    
    raw_image = Image.open(image_path).convert('RGB')
    image = vis_processor(raw_image).unsqueeze(0).unsqueeze(2).to(DEVICE)
    video_query_tokens, past_key_values, frame_atts = model.encode_videoQformer_visual(image, output_for_itg=True)
    image_emb = embed_text_itg(model, prompt, past_key_values, query_tokens=video_query_tokens, video_atts=frame_atts)

    #image_emb = F.normalize(model.vision_proj(image_emb), dim=-1)
    return image_emb

def upload_image_itm(model, prompt, image_path, vis_processor):
    if not isinstance(image_path, str):
        raise NotImplementedError("Non-string image paths are not implemented.")
    
    raw_image = Image.open(image_path).convert('RGB')
    image = vis_processor(raw_image).unsqueeze(0).unsqueeze(2).to(DEVICE)
    video_query_tokens, frame_hidden_state, frame_atts = model.encode_videoQformer_visual(image, output_for_itm=True)
    image_emb = embed_text_itm(model, prompt, frame_hidden_state, query_tokens=video_query_tokens, video_atts=frame_atts)

    #image_emb = F.normalize(model.vision_proj(image_emb), dim=-1)
    return image_emb


def upload_image(model, image_path, vis_processor):
    if not isinstance(image_path, str):
        raise NotImplementedError("Non-string image paths are not implemented.")
    
    raw_image = Image.open(image_path).convert('RGB')
    image = vis_processor(raw_image).unsqueeze(0).unsqueeze(2).to(DEVICE)
    image_emb = model.encode_videoQformer_visual(image)[-1].last_hidden_state
    image_emb = F.normalize(model.vision_proj(image_emb), dim=-1)
    return image_emb



def get_all_video_embeddings(video_paths, model, vis_processor):
    video_embs = []
    for video_path in video_paths:
        embs = upload_video(
            model, video_path, vis_processor)
        embs = F.normalize(model.vision_proj(embs), dim=-1)
        video_embs.append(embs)
    return video_embs

def get_all_video_embeddings_long_video(video_path, model, vis_processor, clip_len=10):
    clip_embs = upload_video_long(
        model, video_path, vis_processor, clip_len=clip_len)
    return clip_embs

def get_all_video_embeddings_long_video_itg(video_path, prompt, model, vis_processor, clip_len=10):
    clip_embs = upload_video_long_itg(
        model, prompt, video_path, vis_processor, clip_len=clip_len)
    return clip_embs

def get_all_video_embeddings_long_video_itm(video_path, prompt, model, vis_processor, clip_len=10):
    clip_embs = upload_video_long_itm(
        model, prompt, video_path, vis_processor, clip_len=clip_len)
    return clip_embs

def get_all_video_embeddings_long_video_subsets(video_path, model, vis_processor, subsets, orig_clip_len=30, clip_len=10):
    clip_embs = upload_video_long_subsets(
        model, video_path, vis_processor, subsets,  orig_clip_len=orig_clip_len, clip_len=clip_len)
    return clip_embs

def get_image_embeddings_itg(image_paths, model, prompt, vis_processor):
    image_embs = []
    for image_path in image_paths:
        image_emb = upload_image_itg(
            model, prompt, image_path, vis_processor)
        image_embs.append(image_emb)
    return image_embs

def get_image_embeddings_itm(image_paths, model, prompt, vis_processor):
    image_embs = []
    for image_path in image_paths:
        image_emb = upload_image_itm(
            model, prompt, image_path, vis_processor)
        image_embs.append(image_emb)
    return image_embs

def get_image_embeddings(image_paths, model, vis_processor):
    image_embs = []
    for image_path in image_paths:
        image_emb = upload_image(
            model, image_path, vis_processor)
        image_embs.append(image_emb)
    return image_embs


def embed_text_itc(model, prompt):
    inputs = model.tokenizer(prompt,
            padding='max_length',
            truncation=True,
            max_length=320,
            return_tensors='pt').to('cuda')
    embds = model.video_Qformer.bert(
            inputs.input_ids,
            inputs.attention_mask,
            return_dict=True).last_hidden_state
    embds = F.normalize(model.text_proj(embds[:, 0, :]), dim=-1)
    return embds

def compute_itc(model, prompts, video_emb):
    text_embs = []
    for prompt in prompts:
        text_embs.append(embed_text_itc(model, prompt))
    text_embs = torch.cat(text_embs)

    sim_q2t = torch.einsum("iqf,tf->itq", video_emb, text_embs)
    sim, _ = sim_q2t.max(dim=-1)
    return sim

def embed_text_itg(model, prompt, past_key_values, query_tokens=None, video_atts=None):
    inputs = model.tokenizer(prompt,
            padding='max_length',
            truncation=True,
            max_length=320,
            return_tensors='pt').to('cuda')

    query_atts = torch.ones(query_tokens.size()[:-1], dtype=torch.long).to(query_tokens.device)

    attention_mask = torch.cat([query_atts, inputs.attention_mask], dim=1)

    output_itg = model.video_Qformer(
        inputs.input_ids,
        attention_mask=attention_mask,
        past_key_values=past_key_values,
        use_cache=True,
        output_hidden_states=True,
        return_dict=True,
    )

    last_hidden_state = output_itg.hidden_states[-1]
    itg_embeddings = last_hidden_state[:, : query_tokens.size(1), :]
    #itg_logit = itg_embeddings.mean(dim=1)
    itg_logit = itg_embeddings[:, -1, :]
    #itm_embeddings = output_itm.last_hidden_state[:, : query_tokens.size(1), :]
    #itm_logit = itm_embeddings.mean(dim=1)

    return itg_logit

def embed_text_itm(model, prompt, video_emb, query_tokens=None, video_atts=None):
    inputs = model.tokenizer(prompt,
            padding='max_length',
            truncation=True,
            max_length=320,
            return_tensors='pt').to('cuda')

    if query_tokens is None:
        query_tokens = model.video_query_tokens.expand(video_emb.shape[0], -1, -1)
    query_atts = torch.ones(query_tokens.size()[:-1], dtype=torch.long).to(video_emb.device)

    attention_mask = torch.cat([query_atts, inputs.attention_mask], dim=1)
    if video_atts is None:
        video_atts = torch.ones(video_emb.size()[:-1], dtype=torch.long).to(video_emb.device)

    output_itm = model.video_Qformer.bert(
            inputs.input_ids,
            query_embeds=query_tokens,
            attention_mask=attention_mask,
            encoder_hidden_states=video_emb,
            encoder_attention_mask=video_atts,
            return_dict=True)
    itm_embeddings = output_itm.last_hidden_state[:, :query_tokens.size(1), :]
    #itm_logit = itm_embeddings.mean(dim=1)
    itm_logit = itm_embeddings[:, -1, :]

    return itm_logit


def compute_itm(model, prompts, video_emb):
    text_embs = []
    for prompt in prompts:
        text_embs.append(embed_text_itm(model, prompt, video_emb))
    text_embs = torch.cat(text_embs)

    sim_q2t = torch.einsum("iqf,tf->itq", video_emb, text_embs)
    sim, _ = sim_q2t.max(-1)
    return sim

def compute_sim(model, prompts, video_embs, mode='itc'):
    sims = []
    for video_emb in video_embs:
        if mode == 'itc':
            sim = compute_itc(model, prompts, video_emb)
        if mode == 'itm':
            sim = compute_itm(model, prompts, video_emb)
        sims.append(sim)
    return sims

def compute_video_emb_dist(query_video_emb, video_emb):
    # Try dot product similarity.
    #sim = torch.einsum("iqf,iqf->iq", query_video_emb, video_emb)
    #sim, _ = sim.max(-1)
    #return sim
    return torch.linalg.norm(query_video_emb-video_emb)

def compute_dist_videoq(model, query_video_emb, video_embs):
    dists = []
    for video_emb in video_embs:
        dist = compute_video_emb_dist(query_video_emb, video_emb)
        dists.append(dist)
    return dists

def rank_matches(prompts, video_paths, model, vis_processor):
    video_embs = get_all_video_embeddings(video_paths, model, vis_processor)
    sims = compute_sim(model, prompts, video_embs)

    sorted_sims = sorted(list(zip(video_paths, sims)), key=lambda x: x[1])
    for video_path, sims in sorted_sims:
        print(video_path, sims.cpu().detach().numpy().item())

def rank_matches_videoq(query_video_path, video_paths, model, vis_processor):
    query_video_embs = get_all_video_embeddings([query_video_path], model, vis_processor)
    video_embs = get_all_video_embeddings(video_paths, model, vis_processor)

    dists = compute_dist_videoq(model, query_video_embs[0], video_embs)
    sorted_dists = sorted(list(zip(video_paths, dists)), key=lambda x: x[1])
    for video_path, dists in sorted_dists:
        print(video_path, dists.cpu().detach().numpy().item())

if __name__ == '__main__':
    eval_config = 'eval_configs/video_clip_v0.2.yaml'

    gpu_id = 0
    args = {'cfg_path': eval_config, 'gpu_id': gpu_id,
            'options': None}
    args = Namespace(**args)
    model, vis_processor, img_processor = init(args)

    video_paths = glob.glob('data/*.mp4')

    prompt = 'zebra'
    start = time.time()
    #rank_matches_videoq(video_paths[1], video_paths, model, vis_processor)
    rank_matches([prompt], video_paths, model, vis_processor)
    print(time.time() - start)
