"""
 Copyright (c) 2022, salesforce.com, inc.
 All rights reserved.
 SPDX-License-Identifier: BSD-3-Clause
 For full license text, see the LICENSE file in the repo root or https://opensource.org/licenses/BSD-3-Clause
"""

import os
import glob
from video_llama.datasets.datasets.base_dataset import BaseDataset
from video_llama.datasets.datasets.caption_datasets import CaptionDataset
#from video_llama.conversation.conversation_video import Conversation, SeparatorStyle
from video_llama.processors import transforms_video,AlproVideoTrainProcessor
from video_llama.processors.video_processor import ToTHWC,ToUint8,load_video
from typing import Dict, Optional, Sequence
import pandas as pd
import decord
from decord import VideoReader
import random
import torch
from torch.utils.data.dataloader import default_collate
import json
import transformers
from transformers import AutoTokenizer, AutoModelForCausalLM, LlamaTokenizer
import copy

class AskYoutubeDataset(BaseDataset):
    def __init__(self, vis_processor, text_processor, vis_root, ann_root):
        """
        vis_root (string): Root directory of video (e.g. webvid_eval/video/)
        ann_root (string): Root directory of annotations (e.g. webvid_eval/annotations/)
        split (string): val or test
        """
        super().__init__(vis_processor=vis_processor, text_processor=text_processor)

        ts_df = []

        self.video_paths = self._load_video_paths(vis_root)
        self.annotation = self._load_annotations(ann_root) 
        self.vis_root = vis_root
        self.resize_size = 224
        self.num_frm = self.vis_processor.n_frms
        self.frm_sampling_strategy = 'headtail'

    def _load_video_paths(self, vis_root):
        return dict([(path.split('/')[-1], path) for path in glob.glob(os.path.join(vis_root, '*'))])

    def _load_annotations(self, ann_root):
        captions_files = glob.glob(os.path.join(ann_root, '*', 'chunked_captions_30s.json'))
        all_captions = []
        for captions_file in captions_files:
            video_id = os.path.dirname(captions_file).split('/')[-1]
            if video_id not in self.video_paths:
                continue
            with open(captions_file, 'r') as f:
                captions = json.load(f)
            # combine captions.
            for i, chunk in enumerate(captions):
                #cap = '\n'.join(['\n'.join([s['utf8'] for s in seg['segs']]) for seg in chunk])
                cap = ''
                for seg in chunk:
                    if "dDurationMs" not in seg or ('aAppend' in seg and seg['aAppend']):
                        continue
                    cap += '\n' + ''.join([s['utf8'] for s in seg['segs']])
                cap = cap.replace('\n\n\n', '\n')
                cap = cap.replace('\n\n', '\n')
                m = {'video_id': video_id, 'seq_num': i, 'caption': cap}
                if not self._check_video_chunk_exists(video_id, i):
                    continue
                all_captions.append({'video_id': video_id, 'seq_num': i, 'caption': cap})
        return pd.DataFrame(all_captions)

    def _check_video_chunk_exists(self, video_id, i):
        video_path = os.path.join(self.video_paths[video_id], f'chunk_{i}.mp4')
        return os.path.exists(video_path)

    def __getitem__(self, index):
        #index = index % 2
        num_retries = 10  # skip error videos
        for _ in range(num_retries):
            sample = self.annotation.iloc[index]
            sample_dict = sample.to_dict()
            video_id = sample_dict['video_id']

            if 'caption' in sample_dict.keys():
                text = sample_dict['caption'].strip()
                text = text.replace('\n', ' ')
                #print(f'Caption {index}: ', text)
            else:
                raise NotImplementedError("Un-supported text annotation format.")

            # fetch video
            i = sample_dict['seq_num']
            #print(index, self.video_paths[video_id], i)
            video_path = os.path.join(self.video_paths[video_id], f'chunk_{i}.mp4')
            # if os.path.exists(video_path):
            try:
                video = self.vis_processor(video_path)
                #print('video: ', video)
            except:
                print(f"Failed to load examples with video: {video_path}. "
                            f"Will randomly sample an example as a replacement.")
                index = random.randint(0, len(self) - 1)
                continue
            caption = self.text_processor(text)

            # print(video.size())
            if video is None or caption is None \
                    or video.size()!=torch.Size([3,self.vis_processor.n_frms,224,224]):
                print(f"Failed to load examples with video: {video_path}. "
                            f"Will randomly sample an example as a replacement.")
                index = random.randint(0, len(self) - 1)
                continue
            else:
                break
        else:  
            raise RuntimeError(f"Failed to fetch video after {num_retries} retries.")
        # "image_id" is kept to stay compatible with the COCO evaluation format
        return {
            "image": video,
            "text_input": caption,
            'texts': text,
            "type":'video',
        }

    def __len__(self):
        return len(self.annotation)

    # def collater(self, samples):
    #     new_result = {}
    #     new_result['image'] = default_collate( [sample["image"] for sample in samples])
    #     new_result['text_input'] = default_collate( [sample["text_input"] for sample in samples])
    #     return new_result


class AskYoutubeInstructDataset(BaseDataset):
    def __init__(self, vis_processor, text_processor, vis_root, ann_root, num_video_query_token=32, tokenizer_name = '/mnt/workspace/ckpt/vicuna-13b/', data_type = 'video', model_type='vicuna'):
        """
        vis_root (string): Root directory of video (e.g. webvid_eval/video/)
        ann_root (string): Root directory of annotations (e.g. webvid_eval/annotations/)
        split (string): val or test
        """
        super().__init__(vis_processor=vis_processor, text_processor=text_processor)
        self.use_transcripts = False
        self.video_paths = dict([(path.split('/')[-1], path) for path in glob.glob(os.path.join(vis_root, '*'))])
        ts_df = []
        joined_captions = []
        captions_files = glob.glob(os.path.join(ann_root, '*', 'qa_captions_30s_vicuna1.5_8bit_13b_instruct.json'))
        for captions_file in captions_files:
            video_id = os.path.dirname(captions_file).split('/')[-1]
            if video_id not in self.video_paths:
                continue
            captions = json.load(open(captions_file, 'r'))
            # combine captions.
            for i, chunk in enumerate(captions):
                transcript = chunk["transcript"]
                qa = chunk["qa"]
                # TODO: Make separate questions and add all.
                m = {'video_id': video_id, 'seq_num': i, 'transcript': transcript, "qa": qa}
                joined_captions.append(pd.DataFrame.from_dict([m]))
        # for video_id in os.listdir(ann_root):
        #     captions_file = os.path.join(ann_root, video_id, 'qa_captions_30s_vicuna1.5_8bit_13b_instruct.json')
        #     if not os.path.exists(captions_file):
        #         continue
        #     captions = json.load(open(captions_file, 'r'))
        #     # combine captions.
        #     for i, chunk in enumerate(captions):
        #         transcript = chunk["transcript"]
        #         qa = chunk["qa"]
        #         m = {'video_id': video_id, 'seq_num': i, 'transcript': transcript, "qa": qa}
        #         joined_captions.append(pd.DataFrame.from_dict([m]))


        merged_df = pd.concat(joined_captions)
        self.annotation = merged_df
        self.vis_root = vis_root
        self.resize_size = 224
        self.num_frm = self.vis_processor.n_frms
        self.num_video_query_token = num_video_query_token
        self.tokenizer = LlamaTokenizer.from_pretrained(tokenizer_name, use_fast=False)
        self.tokenizer.pad_token = self.tokenizer.unk_token
        self.tokenizer.add_tokens([DEFAULT_IMAGE_PATCH_TOKEN], special_tokens=True)
        self.IMAGE_PATCH_TOKEN_ID = self.tokenizer.get_vocab()[DEFAULT_IMAGE_PATCH_TOKEN]

        self.transform = AlproVideoTrainProcessor(
            image_size=self.resize_size, n_frms = self.num_frm
        ).transform
        self.data_type = data_type
        self.model_type = model_type


    def _get_video_path(self, sample):
        video_id, i, transcript, qa = sample.values()
        video_glob = os.path.join(self.video_paths[video_id], f'chunk_{i}.mp4')
        #video_glob = os.path.join(self.vis_root, video_id, f'chunk_{i}.mp4')
        #if not os.path.exists(video_path):
        if len(glob.glob(video_glob)) == 0:
            print(f"Video path: {video_glob} doesn't exist")
            return None
        video_path = glob.glob(video_glob)[0]
        return video_path
    
    def create_conversation_list(self, sample_dict, use_transcripts=True):
        conversation_list = []
        transcript = sample_dict['transcript']
        qa = sample_dict['qa']['qa'] if 'qa' in sample_dict['qa'] else sample_dict['qa']
        if use_transcripts:
            prompt = f"Answer the questions given the following transcript: {transcript}\n\n"
        else:
            prompt = f"Answer the questions from the video:\n\n"
        if len(qa) == 0:
            #print('Empty QA')
            return []
        qa[0]['q'] = prompt + qa[0]['q']
        return qa

    def __getitem__(self, index):
        num_retries = 100  # skip error videos
        for _ in range(num_retries):
            sample = self.annotation.iloc[index]
            sample_dict = sample.to_dict()
            video_id = sample_dict['video_id']
            data_dict = dict()

            # fetch video
            #print(sample_dict)
            # if os.path.exists(video_path):
            try:
                video_path = self._get_video_path(sample_dict)
                if video_path is None:
                    index = random.randint(0, len(self) - 1)
                    #print('Empty video path')
                    continue
                conversation_list = self.create_conversation_list(sample_dict, self.use_transcripts)
                if len(conversation_list) == 0:
                    index = random.randint(0, len(self) - 1)
                    #print('Empty conversation list')
                    continue
                video, msg = load_video(
                    video_path=video_path,
                    n_frms=self.num_frm,
                    height=self.resize_size,
                    width=self.resize_size,
                    sampling ="uniform", return_msg = True
                )
                #print('Loaded video')
                video = self.transform(video)
                #print('Transformed video')
                sources = preprocess_multimodal(copy.deepcopy(conversation_list), None, cur_token_len=self.num_video_query_token, msg = msg)
                #print('Processed multimedia')
                new_sources = convert_source_vicuna_format(sources)
                
                #print('Converted sources')
                if self.model_type =='vicuna':
                    data_dict = preprocess(
                        new_sources,
                        self.tokenizer)
                elif self.model_type =='llama_v2':
                    data_dict = preprocess_for_llama_v2(
                        new_sources,
                        self.tokenizer)
                
                #print('Preprocessed', data_dict.keys(), video is None)
                data_dict = dict(input_ids=data_dict["input_ids"][0],
                                texts=data_dict["texts"][0],
                                labels=data_dict["labels"][0])
                #print('Converted to data_dict: ', video)
                # image exist in the data
                data_dict['image'] = video
                #print('Added video to data_dict: ', video)
                #print(new_sources)
            except Exception as e:
                print(e)
                print(f"Failed to load examples with video: {video_path}. Exception"
                            f"Will randomly sample an example as a replacement.")
                index = random.randint(0, len(self) - 1)
                continue

            if video is None \
                    or video.size()!=torch.Size([3,self.vis_processor.n_frms,224,224]):
                print(f"Failed to load examples with video: {video_path}. Video is None. "
                            f"Will randomly sample an example as a replacement.")
                print(video.size(), self.vis_processor.n_frms)
                index = random.randint(0, len(self) - 1)
                continue
            else:
                break
        else:  
            raise RuntimeError(f"Failed to fetch video after {num_retries} retries.")
        # "image_id" is kept to stay compatible with the COCO evaluation format
        #print(data_dict["texts"])
        return {
            "image": video,
            "text_input": data_dict["input_ids"],
            "labels": data_dict["labels"],
            "texts": data_dict["texts"],
            "type":'video',
        }

    def __len__(self):
        return len(self.annotation)


    def collater(self, instances):
        input_ids, labels = tuple([instance[key] for instance in instances]
                                  for key in ("text_input", "labels"))
        input_ids = torch.nn.utils.rnn.pad_sequence(
            input_ids,
            batch_first=True,
            padding_value=self.tokenizer.pad_token_id)
        labels = torch.nn.utils.rnn.pad_sequence(labels,
                                                 batch_first=True,
                                                 padding_value=IGNORE_INDEX)
        batch = dict(
            input_ids=input_ids,
            labels=labels,
            texts=[instance['texts'] for instance in instances],
            attention_mask=input_ids.ne(self.tokenizer.pad_token_id),
        )

        if 'image' in instances[0]:
            images = [instance['image'] for instance in instances]
            if all(x is not None and x.shape == images[0].shape for x in images):
                batch['images'] = torch.stack(images)
            else:
                batch['images'] = images
        batch['conv_type'] = 'multi'
        return batch

def convert_source_vicuna_format(sources):
    new_sources = []
    for source in sources:
        new_source = []
        for i, sentence in enumerate(source):
            role_0_msg = sentence['q']
            role_1_msg = sentence['a']
            new_source.append({
                'from':'human',
                'value': role_0_msg,
            })
            new_source.append({
                'from':'gpt',
                'value': role_1_msg,
            })
        new_sources.append(new_source)
    return new_sources

# DEFAULT_IMAGE_PATCH_TOKEN = '<ImageHere>'
# video_conversation = Conversation(
#     system="",
#     roles=("Human", "Assistant"),
#     messages=[],
#     offset=0,
#     sep_style=SeparatorStyle.SINGLE,
#     sep="###",
# )
# IGNORE_INDEX = -100

def _add_speaker_and_signal(header, source, get_conversation=True):
    """Add speaker and start/end signal on each round."""
    BEGIN_SIGNAL = "###"
    END_SIGNAL = "\n"
    conversation = header
    for sentence in source:
        from_str = sentence["from"]
        if from_str.lower() == "human":
            from_str = video_conversation.roles[0]
        elif from_str.lower() == "gpt":
            from_str = video_conversation.roles[1]
        else:
            from_str = 'unknown'
        sentence["value"] = (BEGIN_SIGNAL + from_str + ": " +
                             sentence["value"] + END_SIGNAL)
        if get_conversation:
            conversation += sentence["value"]
    conversation += BEGIN_SIGNAL
    return conversation

def _tokenize_fn(strings: Sequence[str],
                tokenizer: transformers.PreTrainedTokenizer) -> Dict:
    """Tokenize a list of strings."""
    tokenized_list = [
        tokenizer(
            text,
            return_tensors="pt",
            padding="longest",
            max_length=512,
            truncation=True,
        ) for text in strings
    ]
    input_ids = labels = [
        tokenized.input_ids[0] for tokenized in tokenized_list
    ]
    input_ids_lens = labels_lens = [
        tokenized.input_ids.ne(tokenizer.pad_token_id).sum().item()
        for tokenized in tokenized_list
    ]
    texts = tokenizer.batch_decode(input_ids)
    return dict(
        input_ids=input_ids,
        labels=labels,
        input_ids_lens=input_ids_lens,
        labels_lens=labels_lens,
        texts=texts,
    )

def _mask_targets(target, tokenized_lens, speakers):
    # cur_idx = 0
    cur_idx = tokenized_lens[0]
    tokenized_lens = tokenized_lens[1:]
    target[:cur_idx] = IGNORE_INDEX
    for tokenized_len, speaker in zip(tokenized_lens, speakers):
        if speaker == "human":
            target[cur_idx+2:cur_idx + tokenized_len] = IGNORE_INDEX
        cur_idx += tokenized_len

def preprocess(
    sources: Sequence[str],
    tokenizer: transformers.PreTrainedTokenizer,
) -> Dict:
    """
    Given a list of sources, each is a conversation list. This transform:
    1. Add signal '### ' at the beginning each sentence, with end signal '\n';
    2. Concatenate conversations together;
    3. Tokenize the concatenated conversation;
    4. Make a deepcopy as the target. Mask human words with IGNORE_INDEX.
    """
    # add end signal and concatenate together
    conversations = []
    for source in sources:
        # TODO: Add the transcript here
        #header = f"Answer the questions given the following transcript: {transcript}\n\n"
        header = f"{video_conversation.system}\n\n"
        conversation = _add_speaker_and_signal(header, source)
        conversations.append(conversation)
    # tokenize conversations
    conversations_tokenized = _tokenize_fn(conversations, tokenizer)
    input_ids = conversations_tokenized["input_ids"]
    texts = conversations_tokenized["texts"]
    targets = copy.deepcopy(input_ids)
    for target, source in zip(targets, sources):
        tokenized_lens = _tokenize_fn([header] + [s["value"] for s in source],
                                      tokenizer)["input_ids_lens"]
        speakers = [sentence["from"] for sentence in source]
        _mask_targets(target, tokenized_lens, speakers)

    return dict(input_ids=input_ids, labels=targets, texts=texts)

def preprocess_multimodal(
    conversation_list: Sequence[str],
    multimodal_cfg: dict,
    cur_token_len: int,
    msg=''
) -> Dict:
    is_multimodal = True
    image_token_len = cur_token_len
    conversation_list[0]["q"] = "<Video>"+DEFAULT_IMAGE_PATCH_TOKEN * image_token_len +"</Video> " + msg + conversation_list[0]["q"]
    return [conversation_list]

class WebvidDatasetEvalDataset(BaseDataset):
    def __init__(self, vis_processor, text_processor, vis_root, ann_paths):
        """
        vis_root (string): Root directory of images (e.g. coco/images/)
        ann_root (string): directory to store the annotation file
        split (string): val or test
        """
        super().__init__(vis_processor, text_processor, vis_root, ann_paths)

    def __getitem__(self, index):

        ann = self.annotation[index]

        vname = ann["video"]
        video_path = os.path.join(self.vis_root, vname)

        video = self.vis_processor(video_path)

        return {
            "video": video,
            "image_id": ann["image_id"],
            "instance_id": ann["instance_id"],
        }


if __name__ == '__main__':
    from video_llama.processors.base_processor import BaseProcessor
    vis_root = '/mnt/g/video_caption_dataset/*/*/data/chunked_videos_30s/'
    ann_root = '/mnt/g/video_caption_dataset/*/*/data/captions/'
    vis_processor = BaseProcessor()
    text_processor = BaseProcessor()
    dataset = AskYoutubeDataset(vis_processor, text_processor, vis_root, ann_root)
