from dataclasses import dataclass, field


@dataclass
class VLMConfig:
    vit_hidden_dim: int = 768
    vit_inter_dim: int = 4 * vit_hidden_dim
    vit_patch_size: int = 16
    vit_img_size: int = 512
    vit_n_heads: int = 12
    vit_dropout: float = 0.0
    vit_n_blocks: int = 12
    vit_ln_eps: float = 1e-6
    vit_cls_flag: bool = False
    vit_model_type: str = 'google/siglip2-base-patch16-512'

    lm_hidden_dim: int = 960
    lm_inter_dim: int = 2560
    lm_rms_eps: float = 1e-5
    lm_re_base: int = 100000
    lm_max_position_embeddings: int = 8192
    lm_base_vocab_size: int = 49152
    extra_token_amount: int = 17  # Number of extra tokens for the VLM (image start, image end, image token)
    lm_vocab_size: int = lm_base_vocab_size + extra_token_amount # Not a great way to do this, but it works for now (vlm_extra_tokens cannot be a dict, since this is mutable, and a Field has no len() function)
    lm_n_heads: int = 15
    lm_n_kv_heads: int = 5
    lm_dropout: float = 0.0
    lm_n_blocks: int = 32
    lm_attn_scaling: float = 1.0
    lm_max_length: int = 1024
    lm_use_tokens: bool = False # Decide if the LM expects tokens or embeddings as input (if using as a backbone for the VLM, set to False)
    lm_tie_weights: bool = True # Decide if you want to tie the LM Head weight to the token embedding weights
    lm_model_type: str = 'HuggingFaceTB/SmolLM2-360M-Instruct'
    lm_tokenizer: str = 'HuggingFaceTB/SmolLM2-360M-Instruct'
    lm_chat_template: str = "{% for message in messages %}{{'<|im_start|>' + message['role'] + '\n' + message['content'] + '<|im_end|>' + '\n'}}{% endfor %}{% if add_generation_prompt %}{{ '<|im_start|>assistant\n' }}{% endif %}"

    mp_pixel_shuffle_factor: int = 4
    mp_image_token_length: int = 64

    max_img_size: int = 1024

    vlm_extra_tokens: dict[str, str] = field(default_factory=lambda: {"image_token": "<|image|>",
      "r1c1": "<row_1_col_1>", "r1c2": "<row_1_col_2>", "r1c3": "<row_1_col_3>", "r1c4": "<row_1_col_4>",
      "r2c1": "<row_2_col_1>", "r2c2": "<row_2_col_2>", "r2c3": "<row_2_col_3>", "r2c4": "<row_2_col_4>",
      "r3c1": "<row_3_col_1>", "r3c2": "<row_3_col_2>", "r3c3": "<row_3_col_3>", "r3c4": "<row_3_col_4>",
      "r4c1": "<row_4_col_1>", "r4c2": "<row_4_col_2>", "r4c3": "<row_4_col_3>", "r4c4": "<row_4_col_4>"})
    vlm_load_backbone_weights: bool = True
    vlm_checkpoint_path: str = 'checkpoints'
    hf_repo_name: str = 'nanoVLM'


@dataclass
class TrainConfig:
    lr_mp: float = 0.00512
    lr_backbones: float = 5e-5
    data_cutoff_idx: int = None
    val_ratio: float = 0.05
    batch_size: int = 8
    gradient_accumulation_steps: int = 8
    max_grad_norm: float = 1.0
    eval_in_epochs: bool = True
    eval_interval: int = gradient_accumulation_steps * 100
    stats_log_interval: int = gradient_accumulation_steps * 25
    max_training_steps: int = 20000
    max_images_per_example: int = 4
    max_images_per_knapsack: int = 18
    max_sample_length: int = 1024
    compile: bool = False
    resume_from_vlm_checkpoint: bool = False # Indicate if the training should be resumed from a checkpoint of the whole VLM or you want to start from scratch
    train_dataset_path: str = 'HuggingFaceM4/cauldron_v3_test'
    # train_dataset_name: tuple[str, ...] = ("all", )
    train_dataset_name: tuple[str, ...] = ('CoSyn_400k_chart', 'CoSyn_400k_chemical', 'CoSyn_400k_circuit', 'CoSyn_400k_diagram', 
                                           'CoSyn_400k_document', 'CoSyn_400k_graphic', 'CoSyn_400k_math', 'CoSyn_400k_music', 
                                           'CoSyn_400k_nutrition', 'CoSyn_400k_table', 'a_okvqa', 'ai2d', 'ai2d(gpt4v)', 'ai2d(internvl)', 
                                           'aokvqa', 'arxivqa', 'blockdiagramcomputerized', 'blockdiagramhandwritten', 
                                           'cambrian(filtered)_processed', 'chart2text', 'chart2text(cauldron)', 'chartqa', 
                                           'chrome_writting', 'clevr', 'clevr_math(mathv360k)', 'cocoqa', 'diagram_image_to_text', 
                                           'docvqa', 'drivelm', 'dvqa', 'figureqa', 'figureqa(mathv360k)', 'finqa', 'funsd', 'geo170k(align)', 
                                           'geo170k(qa)', 'geo3k', 'geometry3k(mathv360k)', 'geomverse', 'geos(mathv360k)', 'groundui', 
                                           'hateful_memes', 'hitab', 'hme100k', 'iam', 'iconqa', 'iconqa(mathv360k)', 'idk', 'iiit5k', 
                                           'image_textualization(filtered)', 'imgur5k', 'infographic(gpt4v)', 'infographic_vqa', 
                                           'infographic_vqa_llava_format', 'intergps', 'latex_handwritten', 'latexformulas', 'llavar_gpt4_20k', 
                                           'lnqa', 'localized_narratives', 'lrv_chart', 'lrv_normal(filtered)', 'lvis_instruct4v', 'mapqa', 
                                           'mapqa(mathv360k)', 'maptext', 'mavis_math_metagen', 'mavis_math_rule_geo', 'memotion', 
                                           'mimic_cgd', 'mmra', 'mmsoc_memotion', 'multihiertt', 'nlvr2', 'ocrvqa', 'oodvqa', 'orand_car_a', 
                                           'pathvqa', 'pdfvqa', 'plotqa', 'pmc_vqa(mathv360k)', 'raven', 'rendered_text', 'robut_sqa', 
                                           'robut_wikisql', 'robut_wtq', 'scienceqa', 'scienceqa(nona_context)', 'screen2words', 'screenqa', 
                                           'sharegpt4o', 'sharegpt4v(coco)', 'sharegpt4v(knowledge)', 'sharegpt4v(llava)', 'sharegpt4v(sam)', 
                                           'sketchyvqa', 'slidevqa', 'spark', 'spatialsense', 'spot_the_diff', 'sroie', 'st_vqa', 
                                           'super_clevr(mathv360k)', 'synthdog', 'tabmwp', 'tabmwp(mathv360k)', 'tallyqa', 'tat_qa', 
                                           'textcaps', 'textocr(gpt4v)', 'textvqa', 'tqa', 'ureader_cap', 'ureader_ie', 'ureader_kg_processed', 
                                           'vision_flan(filtered)', 'vistext', 'vistext(cauldron)', 'visual7w', 'visualmrc', 
                                           'visualwebinstruct(filtered)', 'vizwiz(mathv360k)', 'vqarad', 'vqav2', 'vsr', 'websight', 'wildvision', 'wordart', 'yesbut')
    wandb_entity: str = "HuggingFace" # Indicate the entity to log to in wandb
    log_wandb: bool = True
    use_lmms_eval: bool = True # Use lmms-eval for evaluation
    lmms_eval_tasks: str = 'mmstar,mmmu,ocrbench,textvqa' # Pass additional task as one string, seperated by commas without spaces (e.g. 'mmstar,mmmu,ocrbench')
    lmms_eval_limit: int = None
    lmms_eval_batch_size: int = 128