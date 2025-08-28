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
    extra_token_amount: int = 65  # Number of extra tokens for the VLM (image start, image end, image token)
    lm_vocab_size: int = lm_base_vocab_size + extra_token_amount # Not a great way to do this, but it works for now (vlm_extra_tokens cannot be a dict, since this is mutable, and a Field has no len() function)
    lm_n_heads: int = 15
    lm_n_kv_heads: int = 5
    lm_dropout: float = 0.0
    lm_n_blocks: int = 32
    lm_attn_scaling: float = 1.0
    lm_max_length: int = 4096
    lm_use_tokens: bool = False # Decide if the LM expects tokens or embeddings as input (if using as a backbone for the VLM, set to False)
    lm_tie_weights: bool = True # Decide if you want to tie the LM Head weight to the token embedding weights
    lm_model_type: str = 'HuggingFaceTB/SmolLM2-360M-Instruct'
    lm_tokenizer: str = 'HuggingFaceTB/SmolLM2-360M-Instruct'
    lm_chat_template: str = "{% for message in messages %}{{'<|im_start|>' + message['role'] + '\n' + message['content'] + '<|im_end|>' + '\n'}}{% endfor %}{% if add_generation_prompt %}{{ '<|im_start|>assistant\n' }}{% endif %}"

    mp_pixel_shuffle_factor: int = 4
    mp_image_token_length: int = 64

    max_img_size: int = 2048 #1536
    resize_to_max_side_len: bool = False

    vlm_extra_tokens: dict[str, str] = field(default_factory=lambda: {"image_token": "<|image|>",
      "r1c1": "<row_1_col_1>", "r1c2": "<row_1_col_2>", "r1c3": "<row_1_col_3>", "r1c4": "<row_1_col_4>", "r1c5": "<row_1_col_5>", "r1c6": "<row_1_col_6>", "r1c7": "<row_1_col_7>", "r1c8": "<row_1_col_8>",
      "r2c1": "<row_2_col_1>", "r2c2": "<row_2_col_2>", "r2c3": "<row_2_col_3>", "r2c4": "<row_2_col_4>", "r2c5": "<row_2_col_5>", "r2c6": "<row_2_col_6>", "r2c7": "<row_2_col_7>", "r2c8": "<row_2_col_8>",
      "r3c1": "<row_3_col_1>", "r3c2": "<row_3_col_2>", "r3c3": "<row_3_col_3>", "r3c4": "<row_3_col_4>", "r3c5": "<row_3_col_5>", "r3c6": "<row_3_col_6>", "r3c7": "<row_3_col_7>", "r3c8": "<row_3_col_8>",
      "r4c1": "<row_4_col_1>", "r4c2": "<row_4_col_2>", "r4c3": "<row_4_col_3>", "r4c4": "<row_4_col_4>", "r4c5": "<row_4_col_5>", "r4c6": "<row_4_col_6>", "r4c7": "<row_4_col_7>", "r4c8": "<row_4_col_8>",
      "r5c1": "<row_5_col_1>", "r5c2": "<row_5_col_2>", "r5c3": "<row_5_col_3>", "r5c4": "<row_5_col_4>", "r5c5": "<row_5_col_5>", "r5c6": "<row_5_col_6>", "r5c7": "<row_5_col_7>", "r5c8": "<row_5_col_8>",
      "r6c1": "<row_6_col_1>", "r6c2": "<row_6_col_2>", "r6c3": "<row_6_col_3>", "r6c4": "<row_6_col_4>", "r6c5": "<row_6_col_5>", "r6c6": "<row_6_col_6>", "r6c7": "<row_6_col_7>", "r6c8": "<row_6_col_8>",
      "r7c1": "<row_7_col_1>", "r7c2": "<row_7_col_2>", "r7c3": "<row_7_col_3>", "r7c4": "<row_7_col_4>", "r7c5": "<row_7_col_5>", "r7c6": "<row_7_col_6>", "r7c7": "<row_7_col_7>", "r7c8": "<row_7_col_8>",
      "r8c1": "<row_8_col_1>", "r8c2": "<row_8_col_2>", "r8c3": "<row_8_col_3>", "r8c4": "<row_8_col_4>", "r8c5": "<row_8_col_5>", "r8c6": "<row_8_col_6>", "r8c7": "<row_8_col_7>", "r8c8": "<row_8_col_8>"})
    vlm_load_backbone_weights: bool = True
    vlm_checkpoint_path: str = 'checkpoints'
    hf_repo_name: str = 'nanoVLM'


@dataclass
class TrainConfig:
    lr_mp: float = 0.00512
    lr_vision_backbone: float = 5e-5
    lr_language_backbone: float = 5e-5
    data_cutoff_idx: int = None
    val_ratio: float = 0.005
    batch_size: int = 2
    gradient_accumulation_steps: int = 8
    max_grad_norm: float = 1.0
    eval_in_epochs: bool = True
    eval_interval: int = 600
    stats_log_interval: int = 100
    max_training_steps: int = 50100
    max_images_per_example: int = 4
    max_images_per_knapsack: int = 18
    max_sample_length: int = 4096
    compile: bool = False
    resume_from_vlm_checkpoint: bool = False # Indicate if the training should be resumed from a checkpoint of the whole VLM or you want to start from scratch
    train_dataset_path: str = '/scratch/cache/asterix_deduplicated' #'/fsx/luis_wiedmann/datasets/asterix_rated'
    train_dataset_name: tuple[str, ...] = ('CoSyn_400k_chart', 'CoSyn_400k_chemical', 'CoSyn_400k_circuit', 'CoSyn_400k_diagram', 'CoSyn_400k_document', 'CoSyn_400k_graphic', 'CoSyn_400k_math', 'CoSyn_400k_music', 'CoSyn_400k_nutrition', 'CoSyn_400k_table', 'DoclingMatix', 'LLaVA_Instruct_150K', 'SynthChartNet', 'SynthCodeNet', 'SynthFormulaNet', 'Unichart', 'a_okvqa', 'aguvis-stage-1', 'ai2d_merged', 'alfworldgpt', 'allava_laion', 'allava_vflan', 'aokvqa', 'art', 'arxivqa', 'bentham', 'blockdiagramcomputerized', 'blockdiagramhandwritten', 'cambrian(filtered)_processed', 'captcha', 'chart2text', 'chartqa', 'chinesememe', 'chrome_writting', 'clevr', 'clevr_math', 'clevr_math(mathv360k)', 'coco_colors', 'cocoqa', 'cocotext', 'ctw', 'datik', 'datikz', 'densefusion_1m', 'diagram_image_to_text', 'docvqa', 'drivelm', 'dvqa', 'est_vqa', 'face_emotion', 'figureqa', 'figureqa(mathv360k)', 'finqa', 'funsd', 'geo170k(align)', 'geo170k(qa)', 'geo3k', 'geometry3k(mathv360k)', 'geomverse', 'geoqa+(mathv360k)', 'geos(mathv360k)', 'google_landmarks', 'groundui', 'handwriting_forms', 'hateful_memes', 'hitab', 'hme100k', 'hw_squad', 'iam', 'iconqa', 'iconqa(mathv360k)', 'idk', 'iiit5k', 'image_textualization(filtered)', 'imgur5k', 'indoor_qa', 'infographic(gpt4v)', 'infographic_vqa', 'infographic_vqa_llava_format', 'intergps', 'invoices_receipts', 'k12_printing', 'laion_gpt4v', 'latex_handwritten', 'latexformulas', 'llavar_gpt4_20k', 'lnqa', 'localized_narratives', 'lrv_chart', 'lrv_normal(filtered)', 'lvis_instruct4v', 'mapqa', 'mapqa(mathv360k)', 'maptext', 'mathwriting-google', 'mavis_math_metagen', 'mavis_math_rule_geo', 'memotion', 'mimic_cgd', 'mmc_instruct', 'mmevol', 'mmra', 'mmsoc_memotion', 'multihiertt', 'nlvr2', 'objects365_qa', 'ocrvqa', 'olmOCR-mix-0225-books', 'olmOCR-mix-0225-documents', 'oodvqa', 'orand_car_a', 'pathvqa', 'pdfvqa', 'plotqa', 'pmc_vqa(mathv360k)', 'raven', 'rendered_text', 'robut_sqa', 'robut_wikisql', 'robut_wtq', 'scienceqa', 'scienceqa(nona_context)', 'screen2words', 'screenqa', 'sharegpt4o', 'sharegpt4v(coco)', 'sharegpt4v(knowledge)', 'sharegpt4v(llava)', 'sharegpt4v(sam)', 'sketchyvqa', 'slidevqa', 'spark', 'spatialsense', 'spot_the_diff', 'sroie', 'st_vqa', 'sujet_finance', 'super_clevr(mathv360k)', 'svrd', 'synthdog', 'tabmwp', 'tabmwp(mathv360k)', 'tal_ocr_eng', 'tallyqa', 'tat_dqa', 'tat_qa', 'text_OpenMathInstruct-2', 'text_code_feedback', 'text_codefeedback_filtered_instruction', 'text_infinitymath', 'text_mathinstruct', 'text_mathqa', 'text_mathstepdpo10k', 'text_numinamath_cot', 'text_openhermes_2_5', 'text_openorca', 'text_orcamath', 'text_pythoncode25k', 'text_pythoncodealpaca', 'text_ruozhiba', 'text_theoremqa', 'text_wizardlm_evol', 'textcaps', 'textocr(gpt4v)', 'textvqa', 'tqa', 'unigeo(mathv360k)', 'ureader_cap', 'ureader_ie', 'ureader_kg_processed', 'ureader_qa_processed', 'vision_flan(filtered)', 'vistext', 'visual7w', 'visualmrc', 'visualwebinstruct(filtered)', 'vizwiz(mathv360k)', 'vqaonbd', 'vqarad', 'vqav2', 'vsr', 'websight', 'wildvision', 'wordart', 'yesbut')
    relevance_min_rating: int = 1
    image_correspondence_min_rating: int = 1
    visual_dependency_min_rating: int = 1
    formatting_min_rating: int = 1
    wandb_entity: str = "HuggingFace" # Indicate the entity to log to in wandb
    log_wandb: bool = True
    use_lmms_eval: bool = True # Use lmms-eval for evaluation
    lmms_eval_tasks: str = 'mmstar,mmmu,ocrbench,textvqa,docvqa,scienceqa,mme,infovqa' # Pass additional task as one string, seperated by commas without spaces (e.g. 'mmstar,mmmu,ocrbench')
    lmms_eval_limit: float = None
    lmms_eval_batch_size: int = 64
