import { useParams, Link } from 'react-router-dom'
import { lazy, Suspense } from 'react'
import { motion } from 'framer-motion'
import { getCurriculumById, getChapterById, getSectionById, getAdjacentSections, resolveBuildsOn } from '../subjects/index.js'
import DifficultyBadge from '../components/navigation/DifficultyBadge.jsx'
import PrevNextNav from '../components/navigation/PrevNextNav.jsx'
import Breadcrumbs from '../components/layout/Breadcrumbs.jsx'
import useProgress from '../hooks/useProgress.js'

// Registry of sections that have full content pages written.
// Each entry lazy-loads a section's content component.
const CONTENT_REGISTRY = {
  // 01-text-fundamentals
  '01-text-fundamentals/c1-text-representation/s1-tokenization': lazy(() => import('../subjects/01-text-fundamentals/c1-text-representation/s1-tokenization.jsx')),
  '01-text-fundamentals/c1-text-representation/s2-unicode-encoding': lazy(() => import('../subjects/01-text-fundamentals/c1-text-representation/s2-unicode-encoding.jsx')),
  '01-text-fundamentals/c1-text-representation/s3-bow-tfidf': lazy(() => import('../subjects/01-text-fundamentals/c1-text-representation/s3-bow-tfidf.jsx')),
  '01-text-fundamentals/c1-text-representation/s4-ngrams': lazy(() => import('../subjects/01-text-fundamentals/c1-text-representation/s4-ngrams.jsx')),
  '01-text-fundamentals/c2-classical-nlp/s1-pos-tagging': lazy(() => import('../subjects/01-text-fundamentals/c2-classical-nlp/s1-pos-tagging.jsx')),
  '01-text-fundamentals/c2-classical-nlp/s2-ner': lazy(() => import('../subjects/01-text-fundamentals/c2-classical-nlp/s2-ner.jsx')),
  '01-text-fundamentals/c2-classical-nlp/s3-dependency-parsing': lazy(() => import('../subjects/01-text-fundamentals/c2-classical-nlp/s3-dependency-parsing.jsx')),
  '01-text-fundamentals/c2-classical-nlp/s4-sentiment-basics': lazy(() => import('../subjects/01-text-fundamentals/c2-classical-nlp/s4-sentiment-basics.jsx')),
  '01-text-fundamentals/c3-language-modeling/s1-what-is-lm': lazy(() => import('../subjects/01-text-fundamentals/c3-language-modeling/s1-what-is-lm.jsx')),
  '01-text-fundamentals/c3-language-modeling/s2-markov-models': lazy(() => import('../subjects/01-text-fundamentals/c3-language-modeling/s2-markov-models.jsx')),
  '01-text-fundamentals/c3-language-modeling/s3-perplexity': lazy(() => import('../subjects/01-text-fundamentals/c3-language-modeling/s3-perplexity.jsx')),
  '01-text-fundamentals/c3-language-modeling/s4-statistical-vs-neural': lazy(() => import('../subjects/01-text-fundamentals/c3-language-modeling/s4-statistical-vs-neural.jsx')),
  '01-text-fundamentals/c4-text-preprocessing/s1-cleaning': lazy(() => import('../subjects/01-text-fundamentals/c4-text-preprocessing/s1-cleaning.jsx')),
  '01-text-fundamentals/c4-text-preprocessing/s2-stemming': lazy(() => import('../subjects/01-text-fundamentals/c4-text-preprocessing/s2-stemming.jsx')),
  '01-text-fundamentals/c4-text-preprocessing/s3-stopwords': lazy(() => import('../subjects/01-text-fundamentals/c4-text-preprocessing/s3-stopwords.jsx')),
  '01-text-fundamentals/c4-text-preprocessing/s4-data-pipelines': lazy(() => import('../subjects/01-text-fundamentals/c4-text-preprocessing/s4-data-pipelines.jsx')),
  // 02-embeddings
  '02-embeddings/c1-distributed-representations/s1-one-hot-to-dense': lazy(() => import('../subjects/02-embeddings/c1-distributed-representations/s1-one-hot-to-dense.jsx')),
  '02-embeddings/c1-distributed-representations/s2-word2vec': lazy(() => import('../subjects/02-embeddings/c1-distributed-representations/s2-word2vec.jsx')),
  '02-embeddings/c1-distributed-representations/s3-negative-sampling': lazy(() => import('../subjects/02-embeddings/c1-distributed-representations/s3-negative-sampling.jsx')),
  '02-embeddings/c1-distributed-representations/s4-visualizing-embeddings': lazy(() => import('../subjects/02-embeddings/c1-distributed-representations/s4-visualizing-embeddings.jsx')),
  '02-embeddings/c2-advanced-embeddings/s1-glove': lazy(() => import('../subjects/02-embeddings/c2-advanced-embeddings/s1-glove.jsx')),
  '02-embeddings/c2-advanced-embeddings/s2-fasttext': lazy(() => import('../subjects/02-embeddings/c2-advanced-embeddings/s2-fasttext.jsx')),
  '02-embeddings/c2-advanced-embeddings/s3-elmo': lazy(() => import('../subjects/02-embeddings/c2-advanced-embeddings/s3-elmo.jsx')),
  '02-embeddings/c2-advanced-embeddings/s4-sentence-embeddings': lazy(() => import('../subjects/02-embeddings/c2-advanced-embeddings/s4-sentence-embeddings.jsx')),
  '02-embeddings/c3-embedding-properties/s1-analogies': lazy(() => import('../subjects/02-embeddings/c3-embedding-properties/s1-analogies.jsx')),
  '02-embeddings/c3-embedding-properties/s2-bias': lazy(() => import('../subjects/02-embeddings/c3-embedding-properties/s2-bias.jsx')),
  '02-embeddings/c3-embedding-properties/s3-evaluation': lazy(() => import('../subjects/02-embeddings/c3-embedding-properties/s3-evaluation.jsx')),
  '02-embeddings/c3-embedding-properties/s4-domain-specific': lazy(() => import('../subjects/02-embeddings/c3-embedding-properties/s4-domain-specific.jsx')),
  // 03-neural-nlp
  '03-neural-nlp/c1-sequence-models/s1-rnn': lazy(() => import('../subjects/03-neural-nlp/c1-sequence-models/s1-rnn.jsx')),
  '03-neural-nlp/c1-sequence-models/s2-vanishing-gradients': lazy(() => import('../subjects/03-neural-nlp/c1-sequence-models/s2-vanishing-gradients.jsx')),
  '03-neural-nlp/c1-sequence-models/s3-lstm': lazy(() => import('../subjects/03-neural-nlp/c1-sequence-models/s3-lstm.jsx')),
  '03-neural-nlp/c1-sequence-models/s4-gru-bidirectional': lazy(() => import('../subjects/03-neural-nlp/c1-sequence-models/s4-gru-bidirectional.jsx')),
  '03-neural-nlp/c2-seq2seq/s1-encoder-decoder': lazy(() => import('../subjects/03-neural-nlp/c2-seq2seq/s1-encoder-decoder.jsx')),
  '03-neural-nlp/c2-seq2seq/s2-teacher-forcing': lazy(() => import('../subjects/03-neural-nlp/c2-seq2seq/s2-teacher-forcing.jsx')),
  '03-neural-nlp/c2-seq2seq/s3-bahdanau-attention': lazy(() => import('../subjects/03-neural-nlp/c2-seq2seq/s3-bahdanau-attention.jsx')),
  '03-neural-nlp/c2-seq2seq/s4-luong-attention': lazy(() => import('../subjects/03-neural-nlp/c2-seq2seq/s4-luong-attention.jsx')),
  '03-neural-nlp/c3-cnn-for-text/s1-1d-convolutions': lazy(() => import('../subjects/03-neural-nlp/c3-cnn-for-text/s1-1d-convolutions.jsx')),
  '03-neural-nlp/c3-cnn-for-text/s2-textcnn': lazy(() => import('../subjects/03-neural-nlp/c3-cnn-for-text/s2-textcnn.jsx')),
  '03-neural-nlp/c3-cnn-for-text/s3-dilated-conv': lazy(() => import('../subjects/03-neural-nlp/c3-cnn-for-text/s3-dilated-conv.jsx')),
  '03-neural-nlp/c3-cnn-for-text/s4-cnn-vs-rnn': lazy(() => import('../subjects/03-neural-nlp/c3-cnn-for-text/s4-cnn-vs-rnn.jsx')),
  '03-neural-nlp/c4-training-essentials/s1-loss-functions': lazy(() => import('../subjects/03-neural-nlp/c4-training-essentials/s1-loss-functions.jsx')),
  '03-neural-nlp/c4-training-essentials/s2-optimizers': lazy(() => import('../subjects/03-neural-nlp/c4-training-essentials/s2-optimizers.jsx')),
  '03-neural-nlp/c4-training-essentials/s3-lr-schedules': lazy(() => import('../subjects/03-neural-nlp/c4-training-essentials/s3-lr-schedules.jsx')),
  '03-neural-nlp/c4-training-essentials/s4-gradient-clipping': lazy(() => import('../subjects/03-neural-nlp/c4-training-essentials/s4-gradient-clipping.jsx')),
  // 04-transformer-architecture
  '04-transformer-architecture/c1-self-attention/s1-qkv': lazy(() => import('../subjects/04-transformer-architecture/c1-self-attention/s1-qkv.jsx')),
  '04-transformer-architecture/c1-self-attention/s2-scaled-dot-product': lazy(() => import('../subjects/04-transformer-architecture/c1-self-attention/s2-scaled-dot-product.jsx')),
  '04-transformer-architecture/c1-self-attention/s3-multihead': lazy(() => import('../subjects/04-transformer-architecture/c1-self-attention/s3-multihead.jsx')),
  '04-transformer-architecture/c1-self-attention/s4-attention-masking': lazy(() => import('../subjects/04-transformer-architecture/c1-self-attention/s4-attention-masking.jsx')),
  '04-transformer-architecture/c2-building-blocks/s1-positional-encoding': lazy(() => import('../subjects/04-transformer-architecture/c2-building-blocks/s1-positional-encoding.jsx')),
  '04-transformer-architecture/c2-building-blocks/s2-ffn': lazy(() => import('../subjects/04-transformer-architecture/c2-building-blocks/s2-ffn.jsx')),
  '04-transformer-architecture/c2-building-blocks/s3-layer-norm': lazy(() => import('../subjects/04-transformer-architecture/c2-building-blocks/s3-layer-norm.jsx')),
  '04-transformer-architecture/c2-building-blocks/s4-residual': lazy(() => import('../subjects/04-transformer-architecture/c2-building-blocks/s4-residual.jsx')),
  '04-transformer-architecture/c3-full-architecture/s1-encoder': lazy(() => import('../subjects/04-transformer-architecture/c3-full-architecture/s1-encoder.jsx')),
  '04-transformer-architecture/c3-full-architecture/s2-decoder': lazy(() => import('../subjects/04-transformer-architecture/c3-full-architecture/s2-decoder.jsx')),
  '04-transformer-architecture/c3-full-architecture/s3-enc-dec': lazy(() => import('../subjects/04-transformer-architecture/c3-full-architecture/s3-enc-dec.jsx')),
  '04-transformer-architecture/c3-full-architecture/s4-attention-is-all': lazy(() => import('../subjects/04-transformer-architecture/c3-full-architecture/s4-attention-is-all.jsx')),
  '04-transformer-architecture/c4-efficient-attention/s1-complexity': lazy(() => import('../subjects/04-transformer-architecture/c4-efficient-attention/s1-complexity.jsx')),
  '04-transformer-architecture/c4-efficient-attention/s2-sparse-attention': lazy(() => import('../subjects/04-transformer-architecture/c4-efficient-attention/s2-sparse-attention.jsx')),
  '04-transformer-architecture/c4-efficient-attention/s3-linear-attention': lazy(() => import('../subjects/04-transformer-architecture/c4-efficient-attention/s3-linear-attention.jsx')),
  '04-transformer-architecture/c4-efficient-attention/s4-flash-attention': lazy(() => import('../subjects/04-transformer-architecture/c4-efficient-attention/s4-flash-attention.jsx')),
  '04-transformer-architecture/c5-positional-deep-dive/s1-sinusoidal': lazy(() => import('../subjects/04-transformer-architecture/c5-positional-deep-dive/s1-sinusoidal.jsx')),
  '04-transformer-architecture/c5-positional-deep-dive/s2-rope': lazy(() => import('../subjects/04-transformer-architecture/c5-positional-deep-dive/s2-rope.jsx')),
  '04-transformer-architecture/c5-positional-deep-dive/s3-alibi': lazy(() => import('../subjects/04-transformer-architecture/c5-positional-deep-dive/s3-alibi.jsx')),
  '04-transformer-architecture/c5-positional-deep-dive/s4-context-extrapolation': lazy(() => import('../subjects/04-transformer-architecture/c5-positional-deep-dive/s4-context-extrapolation.jsx')),
  // 05-pretraining
  '05-pretraining/c1-masked-lm/s1-bert': lazy(() => import('../subjects/05-pretraining/c1-masked-lm/s1-bert.jsx')),
  '05-pretraining/c1-masked-lm/s2-masked-prediction': lazy(() => import('../subjects/05-pretraining/c1-masked-lm/s2-masked-prediction.jsx')),
  '05-pretraining/c1-masked-lm/s3-nsp': lazy(() => import('../subjects/05-pretraining/c1-masked-lm/s3-nsp.jsx')),
  '05-pretraining/c1-masked-lm/s4-whole-word-masking': lazy(() => import('../subjects/05-pretraining/c1-masked-lm/s4-whole-word-masking.jsx')),
  '05-pretraining/c2-causal-lm/s1-autoregressive': lazy(() => import('../subjects/05-pretraining/c2-causal-lm/s1-autoregressive.jsx')),
  '05-pretraining/c2-causal-lm/s2-gpt-objective': lazy(() => import('../subjects/05-pretraining/c2-causal-lm/s2-gpt-objective.jsx')),
  '05-pretraining/c2-causal-lm/s3-next-token': lazy(() => import('../subjects/05-pretraining/c2-causal-lm/s3-next-token.jsx')),
  '05-pretraining/c2-causal-lm/s4-curriculum': lazy(() => import('../subjects/05-pretraining/c2-causal-lm/s4-curriculum.jsx')),
  '05-pretraining/c3-pretraining-scale/s1-data-curation': lazy(() => import('../subjects/05-pretraining/c3-pretraining-scale/s1-data-curation.jsx')),
  '05-pretraining/c3-pretraining-scale/s2-data-quality': lazy(() => import('../subjects/05-pretraining/c3-pretraining-scale/s2-data-quality.jsx')),
  '05-pretraining/c3-pretraining-scale/s3-tokenizer-training': lazy(() => import('../subjects/05-pretraining/c3-pretraining-scale/s3-tokenizer-training.jsx')),
  '05-pretraining/c3-pretraining-scale/s4-scaling-laws': lazy(() => import('../subjects/05-pretraining/c3-pretraining-scale/s4-scaling-laws.jsx')),
  '05-pretraining/c4-infrastructure/s1-distributed-training': lazy(() => import('../subjects/05-pretraining/c4-infrastructure/s1-distributed-training.jsx')),
  '05-pretraining/c4-infrastructure/s2-mixed-precision': lazy(() => import('../subjects/05-pretraining/c4-infrastructure/s2-mixed-precision.jsx')),
  '05-pretraining/c4-infrastructure/s3-checkpointing': lazy(() => import('../subjects/05-pretraining/c4-infrastructure/s3-checkpointing.jsx')),
  '05-pretraining/c4-infrastructure/s4-training-stability': lazy(() => import('../subjects/05-pretraining/c4-infrastructure/s4-training-stability.jsx')),
  '05-pretraining/c5-architectures-compared/s1-encoder-only': lazy(() => import('../subjects/05-pretraining/c5-architectures-compared/s1-encoder-only.jsx')),
  '05-pretraining/c5-architectures-compared/s2-decoder-only': lazy(() => import('../subjects/05-pretraining/c5-architectures-compared/s2-decoder-only.jsx')),
  '05-pretraining/c5-architectures-compared/s3-enc-dec-family': lazy(() => import('../subjects/05-pretraining/c5-architectures-compared/s3-enc-dec-family.jsx')),
  '05-pretraining/c5-architectures-compared/s4-when-to-use': lazy(() => import('../subjects/05-pretraining/c5-architectures-compared/s4-when-to-use.jsx')),
  // 06-finetuning
  '06-finetuning/c1-full-finetuning/s1-transfer-learning': lazy(() => import('../subjects/06-finetuning/c1-full-finetuning/s1-transfer-learning.jsx')),
  '06-finetuning/c1-full-finetuning/s2-task-heads': lazy(() => import('../subjects/06-finetuning/c1-full-finetuning/s2-task-heads.jsx')),
  '06-finetuning/c1-full-finetuning/s3-catastrophic-forgetting': lazy(() => import('../subjects/06-finetuning/c1-full-finetuning/s3-catastrophic-forgetting.jsx')),
  '06-finetuning/c1-full-finetuning/s4-multitask': lazy(() => import('../subjects/06-finetuning/c1-full-finetuning/s4-multitask.jsx')),
  '06-finetuning/c2-peft/s1-lora': lazy(() => import('../subjects/06-finetuning/c2-peft/s1-lora.jsx')),
  '06-finetuning/c2-peft/s2-adapters': lazy(() => import('../subjects/06-finetuning/c2-peft/s2-adapters.jsx')),
  '06-finetuning/c2-peft/s3-prefix-tuning': lazy(() => import('../subjects/06-finetuning/c2-peft/s3-prefix-tuning.jsx')),
  '06-finetuning/c2-peft/s4-comparison': lazy(() => import('../subjects/06-finetuning/c2-peft/s4-comparison.jsx')),
  '06-finetuning/c3-instruction-tuning/s1-instruction-datasets': lazy(() => import('../subjects/06-finetuning/c3-instruction-tuning/s1-instruction-datasets.jsx')),
  '06-finetuning/c3-instruction-tuning/s2-chat-templates': lazy(() => import('../subjects/06-finetuning/c3-instruction-tuning/s2-chat-templates.jsx')),
  '06-finetuning/c3-instruction-tuning/s3-sft': lazy(() => import('../subjects/06-finetuning/c3-instruction-tuning/s3-sft.jsx')),
  '06-finetuning/c3-instruction-tuning/s4-data-quality': lazy(() => import('../subjects/06-finetuning/c3-instruction-tuning/s4-data-quality.jsx')),
  '06-finetuning/c4-alignment/s1-rlhf': lazy(() => import('../subjects/06-finetuning/c4-alignment/s1-rlhf.jsx')),
  '06-finetuning/c4-alignment/s2-reward-modeling': lazy(() => import('../subjects/06-finetuning/c4-alignment/s2-reward-modeling.jsx')),
  '06-finetuning/c4-alignment/s3-ppo': lazy(() => import('../subjects/06-finetuning/c4-alignment/s3-ppo.jsx')),
  '06-finetuning/c4-alignment/s4-dpo-orpo': lazy(() => import('../subjects/06-finetuning/c4-alignment/s4-dpo-orpo.jsx')),
  // 07-landmark-models
  '07-landmark-models/c1-gpt-lineage/s1-gpt1-2-3': lazy(() => import('../subjects/07-landmark-models/c1-gpt-lineage/s1-gpt1-2-3.jsx')),
  '07-landmark-models/c1-gpt-lineage/s2-instructgpt': lazy(() => import('../subjects/07-landmark-models/c1-gpt-lineage/s2-instructgpt.jsx')),
  '07-landmark-models/c1-gpt-lineage/s3-gpt4': lazy(() => import('../subjects/07-landmark-models/c1-gpt-lineage/s3-gpt4.jsx')),
  '07-landmark-models/c1-gpt-lineage/s4-architectural-innovations': lazy(() => import('../subjects/07-landmark-models/c1-gpt-lineage/s4-architectural-innovations.jsx')),
  '07-landmark-models/c2-opensource-pioneers/s1-llama': lazy(() => import('../subjects/07-landmark-models/c2-opensource-pioneers/s1-llama.jsx')),
  '07-landmark-models/c2-opensource-pioneers/s2-mistral': lazy(() => import('../subjects/07-landmark-models/c2-opensource-pioneers/s2-mistral.jsx')),
  '07-landmark-models/c2-opensource-pioneers/s3-falcon-mpt': lazy(() => import('../subjects/07-landmark-models/c2-opensource-pioneers/s3-falcon-mpt.jsx')),
  '07-landmark-models/c2-opensource-pioneers/s4-phi': lazy(() => import('../subjects/07-landmark-models/c2-opensource-pioneers/s4-phi.jsx')),
  '07-landmark-models/c3-latest-generation/s1-llama3': lazy(() => import('../subjects/07-landmark-models/c3-latest-generation/s1-llama3.jsx')),
  '07-landmark-models/c3-latest-generation/s2-gemma': lazy(() => import('../subjects/07-landmark-models/c3-latest-generation/s2-gemma.jsx')),
  '07-landmark-models/c3-latest-generation/s3-qwen': lazy(() => import('../subjects/07-landmark-models/c3-latest-generation/s3-qwen.jsx')),
  '07-landmark-models/c3-latest-generation/s4-deepseek': lazy(() => import('../subjects/07-landmark-models/c3-latest-generation/s4-deepseek.jsx')),
  '07-landmark-models/c3-latest-generation/s5-command-r': lazy(() => import('../subjects/07-landmark-models/c3-latest-generation/s5-command-r.jsx')),
  '07-landmark-models/c4-moe/s1-moe-fundamentals': lazy(() => import('../subjects/07-landmark-models/c4-moe/s1-moe-fundamentals.jsx')),
  '07-landmark-models/c4-moe/s2-switch-transformer': lazy(() => import('../subjects/07-landmark-models/c4-moe/s2-switch-transformer.jsx')),
  '07-landmark-models/c4-moe/s3-mixtral-arch': lazy(() => import('../subjects/07-landmark-models/c4-moe/s3-mixtral-arch.jsx')),
  '07-landmark-models/c4-moe/s4-load-balancing': lazy(() => import('../subjects/07-landmark-models/c4-moe/s4-load-balancing.jsx')),
  '07-landmark-models/c5-reasoning/s1-cot-emergence': lazy(() => import('../subjects/07-landmark-models/c5-reasoning/s1-cot-emergence.jsx')),
  '07-landmark-models/c5-reasoning/s2-o1-o3': lazy(() => import('../subjects/07-landmark-models/c5-reasoning/s2-o1-o3.jsx')),
  '07-landmark-models/c5-reasoning/s3-deepseek-r1': lazy(() => import('../subjects/07-landmark-models/c5-reasoning/s3-deepseek-r1.jsx')),
  '07-landmark-models/c5-reasoning/s4-extended-thinking': lazy(() => import('../subjects/07-landmark-models/c5-reasoning/s4-extended-thinking.jsx')),
  // 08-vision-language
  '08-vision-language/c1-vision-foundations/s1-vit': lazy(() => import('../subjects/08-vision-language/c1-vision-foundations/s1-vit.jsx')),
  '08-vision-language/c1-vision-foundations/s2-image-tokenization': lazy(() => import('../subjects/08-vision-language/c1-vision-foundations/s2-image-tokenization.jsx')),
  '08-vision-language/c1-vision-foundations/s3-clip': lazy(() => import('../subjects/08-vision-language/c1-vision-foundations/s3-clip.jsx')),
  '08-vision-language/c1-vision-foundations/s4-siglip': lazy(() => import('../subjects/08-vision-language/c1-vision-foundations/s4-siglip.jsx')),
  '08-vision-language/c2-multimodal-arch/s1-fusion': lazy(() => import('../subjects/08-vision-language/c2-multimodal-arch/s1-fusion.jsx')),
  '08-vision-language/c2-multimodal-arch/s2-llava': lazy(() => import('../subjects/08-vision-language/c2-multimodal-arch/s2-llava.jsx')),
  '08-vision-language/c2-multimodal-arch/s3-flamingo': lazy(() => import('../subjects/08-vision-language/c2-multimodal-arch/s3-flamingo.jsx')),
  '08-vision-language/c2-multimodal-arch/s4-qwen-vl': lazy(() => import('../subjects/08-vision-language/c2-multimodal-arch/s4-qwen-vl.jsx')),
  '08-vision-language/c3-document-ocr/s1-layoutlm': lazy(() => import('../subjects/08-vision-language/c3-document-ocr/s1-layoutlm.jsx')),
  '08-vision-language/c3-document-ocr/s2-ocr-free': lazy(() => import('../subjects/08-vision-language/c3-document-ocr/s2-ocr-free.jsx')),
  '08-vision-language/c3-document-ocr/s3-chart-table': lazy(() => import('../subjects/08-vision-language/c3-document-ocr/s3-chart-table.jsx')),
  '08-vision-language/c3-document-ocr/s4-multipage': lazy(() => import('../subjects/08-vision-language/c3-document-ocr/s4-multipage.jsx')),
  '08-vision-language/c4-image-generation/s1-diffusion': lazy(() => import('../subjects/08-vision-language/c4-image-generation/s1-diffusion.jsx')),
  '08-vision-language/c4-image-generation/s2-stable-diffusion': lazy(() => import('../subjects/08-vision-language/c4-image-generation/s2-stable-diffusion.jsx')),
  '08-vision-language/c4-image-generation/s3-sdxl-flux': lazy(() => import('../subjects/08-vision-language/c4-image-generation/s3-sdxl-flux.jsx')),
  '08-vision-language/c4-image-generation/s4-pixart': lazy(() => import('../subjects/08-vision-language/c4-image-generation/s4-pixart.jsx')),
  '08-vision-language/c5-video-generation/s1-video-diffusion': lazy(() => import('../subjects/08-vision-language/c5-video-generation/s1-video-diffusion.jsx')),
  '08-vision-language/c5-video-generation/s2-svd': lazy(() => import('../subjects/08-vision-language/c5-video-generation/s2-svd.jsx')),
  '08-vision-language/c5-video-generation/s3-sora': lazy(() => import('../subjects/08-vision-language/c5-video-generation/s3-sora.jsx')),
  '08-vision-language/c5-video-generation/s4-cogvideo': lazy(() => import('../subjects/08-vision-language/c5-video-generation/s4-cogvideo.jsx')),
  '08-vision-language/c5-video-generation/s5-animatediff': lazy(() => import('../subjects/08-vision-language/c5-video-generation/s5-animatediff.jsx')),
  '08-vision-language/c6-finetuning-generative/s1-dreambooth': lazy(() => import('../subjects/08-vision-language/c6-finetuning-generative/s1-dreambooth.jsx')),
  '08-vision-language/c6-finetuning-generative/s2-lora-sd': lazy(() => import('../subjects/08-vision-language/c6-finetuning-generative/s2-lora-sd.jsx')),
  '08-vision-language/c6-finetuning-generative/s3-textual-inversion': lazy(() => import('../subjects/08-vision-language/c6-finetuning-generative/s3-textual-inversion.jsx')),
  '08-vision-language/c6-finetuning-generative/s4-animatediff-lora': lazy(() => import('../subjects/08-vision-language/c6-finetuning-generative/s4-animatediff-lora.jsx')),
  '08-vision-language/c6-finetuning-generative/s5-comfyui-testing': lazy(() => import('../subjects/08-vision-language/c6-finetuning-generative/s5-comfyui-testing.jsx')),
  '08-vision-language/c7-comfyui/s1-comfyui-intro': lazy(() => import('../subjects/08-vision-language/c7-comfyui/s1-comfyui-intro.jsx')),
  '08-vision-language/c7-comfyui/s2-core-concepts': lazy(() => import('../subjects/08-vision-language/c7-comfyui/s2-core-concepts.jsx')),
  '08-vision-language/c7-comfyui/s3-txt2img': lazy(() => import('../subjects/08-vision-language/c7-comfyui/s3-txt2img.jsx')),
  '08-vision-language/c7-comfyui/s4-img2img': lazy(() => import('../subjects/08-vision-language/c7-comfyui/s4-img2img.jsx')),
  '08-vision-language/c7-comfyui/s5-controlnet': lazy(() => import('../subjects/08-vision-language/c7-comfyui/s5-controlnet.jsx')),
  '08-vision-language/c7-comfyui/s6-lora-stacking': lazy(() => import('../subjects/08-vision-language/c7-comfyui/s6-lora-stacking.jsx')),
  '08-vision-language/c7-comfyui/s7-sdxl-flux-workflows': lazy(() => import('../subjects/08-vision-language/c7-comfyui/s7-sdxl-flux-workflows.jsx')),
  '08-vision-language/c7-comfyui/s8-video-nodes': lazy(() => import('../subjects/08-vision-language/c7-comfyui/s8-video-nodes.jsx')),
  '08-vision-language/c7-comfyui/s9-upscaling': lazy(() => import('../subjects/08-vision-language/c7-comfyui/s9-upscaling.jsx')),
  '08-vision-language/c7-comfyui/s10-custom-nodes': lazy(() => import('../subjects/08-vision-language/c7-comfyui/s10-custom-nodes.jsx')),
  '08-vision-language/c7-comfyui/s11-api-mode': lazy(() => import('../subjects/08-vision-language/c7-comfyui/s11-api-mode.jsx')),
  '08-vision-language/c7-comfyui/s12-performance': lazy(() => import('../subjects/08-vision-language/c7-comfyui/s12-performance.jsx')),
  // 09-tabular-models
  '09-tabular-models/c1-llms-tabular/s1-serialization': lazy(() => import('../subjects/09-tabular-models/c1-llms-tabular/s1-serialization.jsx')),
  '09-tabular-models/c1-llms-tabular/s2-tabllm': lazy(() => import('../subjects/09-tabular-models/c1-llms-tabular/s2-tabllm.jsx')),
  '09-tabular-models/c1-llms-tabular/s3-text-to-sql': lazy(() => import('../subjects/09-tabular-models/c1-llms-tabular/s3-text-to-sql.jsx')),
  '09-tabular-models/c1-llms-tabular/s4-table-qa': lazy(() => import('../subjects/09-tabular-models/c1-llms-tabular/s4-table-qa.jsx')),
  '09-tabular-models/c2-specialized-tabular/s1-tapas': lazy(() => import('../subjects/09-tabular-models/c2-specialized-tabular/s1-tapas.jsx')),
  '09-tabular-models/c2-specialized-tabular/s2-tablegpt': lazy(() => import('../subjects/09-tabular-models/c2-specialized-tabular/s2-tablegpt.jsx')),
  '09-tabular-models/c2-specialized-tabular/s3-spreadsheet': lazy(() => import('../subjects/09-tabular-models/c2-specialized-tabular/s3-spreadsheet.jsx')),
  '09-tabular-models/c2-specialized-tabular/s4-schema-linking': lazy(() => import('../subjects/09-tabular-models/c2-specialized-tabular/s4-schema-linking.jsx')),
  '09-tabular-models/c3-structured-output/s1-json-generation': lazy(() => import('../subjects/09-tabular-models/c3-structured-output/s1-json-generation.jsx')),
  '09-tabular-models/c3-structured-output/s2-function-calling': lazy(() => import('../subjects/09-tabular-models/c3-structured-output/s2-function-calling.jsx')),
  '09-tabular-models/c3-structured-output/s3-grammars': lazy(() => import('../subjects/09-tabular-models/c3-structured-output/s3-grammars.jsx')),
  '09-tabular-models/c3-structured-output/s4-constrained-decoding': lazy(() => import('../subjects/09-tabular-models/c3-structured-output/s4-constrained-decoding.jsx')),
  '09-tabular-models/c4-knowledge-graphs/s1-kg-augmented': lazy(() => import('../subjects/09-tabular-models/c4-knowledge-graphs/s1-kg-augmented.jsx')),
  '09-tabular-models/c4-knowledge-graphs/s2-entity-linking': lazy(() => import('../subjects/09-tabular-models/c4-knowledge-graphs/s2-entity-linking.jsx')),
  '09-tabular-models/c4-knowledge-graphs/s3-relation-extraction': lazy(() => import('../subjects/09-tabular-models/c4-knowledge-graphs/s3-relation-extraction.jsx')),
  '09-tabular-models/c4-knowledge-graphs/s4-graph-text': lazy(() => import('../subjects/09-tabular-models/c4-knowledge-graphs/s4-graph-text.jsx')),
  // 10-efficient-models
  '10-efficient-models/c1-compression/s1-distillation': lazy(() => import('../subjects/10-efficient-models/c1-compression/s1-distillation.jsx')),
  '10-efficient-models/c1-compression/s2-pruning': lazy(() => import('../subjects/10-efficient-models/c1-compression/s2-pruning.jsx')),
  '10-efficient-models/c1-compression/s3-weight-sharing': lazy(() => import('../subjects/10-efficient-models/c1-compression/s3-weight-sharing.jsx')),
  '10-efficient-models/c1-compression/s4-architecture-search': lazy(() => import('../subjects/10-efficient-models/c1-compression/s4-architecture-search.jsx')),
  '10-efficient-models/c2-quantization/s1-ptq': lazy(() => import('../subjects/10-efficient-models/c2-quantization/s1-ptq.jsx')),
  '10-efficient-models/c2-quantization/s2-qat': lazy(() => import('../subjects/10-efficient-models/c2-quantization/s2-qat.jsx')),
  '10-efficient-models/c2-quantization/s3-gptq-awq': lazy(() => import('../subjects/10-efficient-models/c2-quantization/s3-gptq-awq.jsx')),
  '10-efficient-models/c2-quantization/s4-extreme-quant': lazy(() => import('../subjects/10-efficient-models/c2-quantization/s4-extreme-quant.jsx')),
  '10-efficient-models/c3-efficient-architectures/s1-mamba': lazy(() => import('../subjects/10-efficient-models/c3-efficient-architectures/s1-mamba.jsx')),
  '10-efficient-models/c3-efficient-architectures/s2-rwkv': lazy(() => import('../subjects/10-efficient-models/c3-efficient-architectures/s2-rwkv.jsx')),
  '10-efficient-models/c3-efficient-architectures/s3-hyena': lazy(() => import('../subjects/10-efficient-models/c3-efficient-architectures/s3-hyena.jsx')),
  '10-efficient-models/c3-efficient-architectures/s4-jamba': lazy(() => import('../subjects/10-efficient-models/c3-efficient-architectures/s4-jamba.jsx')),
  '10-efficient-models/c4-edge-llms/s1-small-models': lazy(() => import('../subjects/10-efficient-models/c4-edge-llms/s1-small-models.jsx')),
  '10-efficient-models/c4-edge-llms/s2-mobile-deployment': lazy(() => import('../subjects/10-efficient-models/c4-edge-llms/s2-mobile-deployment.jsx')),
  '10-efficient-models/c4-edge-llms/s3-speculative-decoding': lazy(() => import('../subjects/10-efficient-models/c4-edge-llms/s3-speculative-decoding.jsx')),
  '10-efficient-models/c4-edge-llms/s4-latency-optimization': lazy(() => import('../subjects/10-efficient-models/c4-edge-llms/s4-latency-optimization.jsx')),
  // 11-practical-finetuning
  '11-practical-finetuning/c1-environment-setup/s1-gpu-requirements': lazy(() => import('../subjects/11-practical-finetuning/c1-environment-setup/s1-gpu-requirements.jsx')),
  '11-practical-finetuning/c1-environment-setup/s2-cuda-setup': lazy(() => import('../subjects/11-practical-finetuning/c1-environment-setup/s2-cuda-setup.jsx')),
  '11-practical-finetuning/c1-environment-setup/s3-python-env': lazy(() => import('../subjects/11-practical-finetuning/c1-environment-setup/s3-python-env.jsx')),
  '11-practical-finetuning/c1-environment-setup/s4-key-libraries': lazy(() => import('../subjects/11-practical-finetuning/c1-environment-setup/s4-key-libraries.jsx')),
  '11-practical-finetuning/c2-finetuning-types/s1-full-vs-peft': lazy(() => import('../subjects/11-practical-finetuning/c2-finetuning-types/s1-full-vs-peft.jsx')),
  '11-practical-finetuning/c2-finetuning-types/s2-lora-deep-dive': lazy(() => import('../subjects/11-practical-finetuning/c2-finetuning-types/s2-lora-deep-dive.jsx')),
  '11-practical-finetuning/c2-finetuning-types/s3-qlora-deep-dive': lazy(() => import('../subjects/11-practical-finetuning/c2-finetuning-types/s3-qlora-deep-dive.jsx')),
  '11-practical-finetuning/c2-finetuning-types/s4-lora-variants': lazy(() => import('../subjects/11-practical-finetuning/c2-finetuning-types/s4-lora-variants.jsx')),
  '11-practical-finetuning/c2-finetuning-types/s5-adapter-comparison': lazy(() => import('../subjects/11-practical-finetuning/c2-finetuning-types/s5-adapter-comparison.jsx')),
  '11-practical-finetuning/c3-unsloth/s1-why-unsloth': lazy(() => import('../subjects/11-practical-finetuning/c3-unsloth/s1-why-unsloth.jsx')),
  '11-practical-finetuning/c3-unsloth/s2-installing-unsloth': lazy(() => import('../subjects/11-practical-finetuning/c3-unsloth/s2-installing-unsloth.jsx')),
  '11-practical-finetuning/c3-unsloth/s3-finetune-llama': lazy(() => import('../subjects/11-practical-finetuning/c3-unsloth/s3-finetune-llama.jsx')),
  '11-practical-finetuning/c3-unsloth/s4-finetune-mistral': lazy(() => import('../subjects/11-practical-finetuning/c3-unsloth/s4-finetune-mistral.jsx')),
  '11-practical-finetuning/c3-unsloth/s5-custom-datasets': lazy(() => import('../subjects/11-practical-finetuning/c3-unsloth/s5-custom-datasets.jsx')),
  '11-practical-finetuning/c3-unsloth/s6-monitoring': lazy(() => import('../subjects/11-practical-finetuning/c3-unsloth/s6-monitoring.jsx')),
  '11-practical-finetuning/c4-trl/s1-sft-trainer': lazy(() => import('../subjects/11-practical-finetuning/c4-trl/s1-sft-trainer.jsx')),
  '11-practical-finetuning/c4-trl/s2-dpo-training': lazy(() => import('../subjects/11-practical-finetuning/c4-trl/s2-dpo-training.jsx')),
  '11-practical-finetuning/c4-trl/s3-reward-modeling': lazy(() => import('../subjects/11-practical-finetuning/c4-trl/s3-reward-modeling.jsx')),
  '11-practical-finetuning/c4-trl/s4-orpo': lazy(() => import('../subjects/11-practical-finetuning/c4-trl/s4-orpo.jsx')),
  '11-practical-finetuning/c4-trl/s5-full-walkthrough': lazy(() => import('../subjects/11-practical-finetuning/c4-trl/s5-full-walkthrough.jsx')),
  '11-practical-finetuning/c5-axolotl-llamafactory/s1-axolotl': lazy(() => import('../subjects/11-practical-finetuning/c5-axolotl-llamafactory/s1-axolotl.jsx')),
  '11-practical-finetuning/c5-axolotl-llamafactory/s2-llama-factory': lazy(() => import('../subjects/11-practical-finetuning/c5-axolotl-llamafactory/s2-llama-factory.jsx')),
  '11-practical-finetuning/c5-axolotl-llamafactory/s3-multi-gpu': lazy(() => import('../subjects/11-practical-finetuning/c5-axolotl-llamafactory/s3-multi-gpu.jsx')),
  '11-practical-finetuning/c5-axolotl-llamafactory/s4-framework-comparison': lazy(() => import('../subjects/11-practical-finetuning/c5-axolotl-llamafactory/s4-framework-comparison.jsx')),
  '11-practical-finetuning/c6-dataset-preparation/s1-formats': lazy(() => import('../subjects/11-practical-finetuning/c6-dataset-preparation/s1-formats.jsx')),
  '11-practical-finetuning/c6-dataset-preparation/s2-building-datasets': lazy(() => import('../subjects/11-practical-finetuning/c6-dataset-preparation/s2-building-datasets.jsx')),
  '11-practical-finetuning/c6-dataset-preparation/s3-cleaning': lazy(() => import('../subjects/11-practical-finetuning/c6-dataset-preparation/s3-cleaning.jsx')),
  '11-practical-finetuning/c6-dataset-preparation/s4-synthetic-data': lazy(() => import('../subjects/11-practical-finetuning/c6-dataset-preparation/s4-synthetic-data.jsx')),
  '11-practical-finetuning/c6-dataset-preparation/s5-size-guidelines': lazy(() => import('../subjects/11-practical-finetuning/c6-dataset-preparation/s5-size-guidelines.jsx')),
  '11-practical-finetuning/c7-model-formats/s1-safetensors': lazy(() => import('../subjects/11-practical-finetuning/c7-model-formats/s1-safetensors.jsx')),
  '11-practical-finetuning/c7-model-formats/s2-gguf': lazy(() => import('../subjects/11-practical-finetuning/c7-model-formats/s2-gguf.jsx')),
  '11-practical-finetuning/c7-model-formats/s3-gptq-awq-export': lazy(() => import('../subjects/11-practical-finetuning/c7-model-formats/s3-gptq-awq-export.jsx')),
  '11-practical-finetuning/c7-model-formats/s4-merging-lora': lazy(() => import('../subjects/11-practical-finetuning/c7-model-formats/s4-merging-lora.jsx')),
  '11-practical-finetuning/c7-model-formats/s5-converting-formats': lazy(() => import('../subjects/11-practical-finetuning/c7-model-formats/s5-converting-formats.jsx')),
  '11-practical-finetuning/c7-model-formats/s6-huggingface-hub': lazy(() => import('../subjects/11-practical-finetuning/c7-model-formats/s6-huggingface-hub.jsx')),
  '11-practical-finetuning/c8-evaluation-iteration/s1-perplexity': lazy(() => import('../subjects/11-practical-finetuning/c8-evaluation-iteration/s1-perplexity.jsx')),
  '11-practical-finetuning/c8-evaluation-iteration/s2-benchmarks': lazy(() => import('../subjects/11-practical-finetuning/c8-evaluation-iteration/s2-benchmarks.jsx')),
  '11-practical-finetuning/c8-evaluation-iteration/s3-human-eval': lazy(() => import('../subjects/11-practical-finetuning/c8-evaluation-iteration/s3-human-eval.jsx')),
  '11-practical-finetuning/c8-evaluation-iteration/s4-ab-testing': lazy(() => import('../subjects/11-practical-finetuning/c8-evaluation-iteration/s4-ab-testing.jsx')),
  '11-practical-finetuning/c8-evaluation-iteration/s5-failure-modes': lazy(() => import('../subjects/11-practical-finetuning/c8-evaluation-iteration/s5-failure-modes.jsx')),
  '11-practical-finetuning/c8-evaluation-iteration/s6-retrain-vs-adjust': lazy(() => import('../subjects/11-practical-finetuning/c8-evaluation-iteration/s6-retrain-vs-adjust.jsx')),
  '11-practical-finetuning/c9-image-video-finetuning/s1-hardware-requirements': lazy(() => import('../subjects/11-practical-finetuning/c9-image-video-finetuning/s1-hardware-requirements.jsx')),
  '11-practical-finetuning/c9-image-video-finetuning/s2-dreambooth-training': lazy(() => import('../subjects/11-practical-finetuning/c9-image-video-finetuning/s2-dreambooth-training.jsx')),
  '11-practical-finetuning/c9-image-video-finetuning/s3-lora-sd-training': lazy(() => import('../subjects/11-practical-finetuning/c9-image-video-finetuning/s3-lora-sd-training.jsx')),
  '11-practical-finetuning/c9-image-video-finetuning/s4-lora-flux': lazy(() => import('../subjects/11-practical-finetuning/c9-image-video-finetuning/s4-lora-flux.jsx')),
  '11-practical-finetuning/c9-image-video-finetuning/s5-textual-inversion': lazy(() => import('../subjects/11-practical-finetuning/c9-image-video-finetuning/s5-textual-inversion.jsx')),
  '11-practical-finetuning/c9-image-video-finetuning/s6-dataset-prep': lazy(() => import('../subjects/11-practical-finetuning/c9-image-video-finetuning/s6-dataset-prep.jsx')),
  '11-practical-finetuning/c9-image-video-finetuning/s7-animatediff-lora': lazy(() => import('../subjects/11-practical-finetuning/c9-image-video-finetuning/s7-animatediff-lora.jsx')),
  '11-practical-finetuning/c9-image-video-finetuning/s8-cogvideox': lazy(() => import('../subjects/11-practical-finetuning/c9-image-video-finetuning/s8-cogvideox.jsx')),
  '11-practical-finetuning/c9-image-video-finetuning/s9-hyperparameter-tuning': lazy(() => import('../subjects/11-practical-finetuning/c9-image-video-finetuning/s9-hyperparameter-tuning.jsx')),
  // 12-inference-serving
  '12-inference-serving/c1-decoding/s1-greedy-beam': lazy(() => import('../subjects/12-inference-serving/c1-decoding/s1-greedy-beam.jsx')),
  '12-inference-serving/c1-decoding/s2-sampling': lazy(() => import('../subjects/12-inference-serving/c1-decoding/s2-sampling.jsx')),
  '12-inference-serving/c1-decoding/s3-temperature': lazy(() => import('../subjects/12-inference-serving/c1-decoding/s3-temperature.jsx')),
  '12-inference-serving/c1-decoding/s4-structured-gen': lazy(() => import('../subjects/12-inference-serving/c1-decoding/s4-structured-gen.jsx')),
  '12-inference-serving/c2-inference-optimization/s1-kv-cache': lazy(() => import('../subjects/12-inference-serving/c2-inference-optimization/s1-kv-cache.jsx')),
  '12-inference-serving/c2-inference-optimization/s2-continuous-batching': lazy(() => import('../subjects/12-inference-serving/c2-inference-optimization/s2-continuous-batching.jsx')),
  '12-inference-serving/c2-inference-optimization/s3-speculative-decoding': lazy(() => import('../subjects/12-inference-serving/c2-inference-optimization/s3-speculative-decoding.jsx')),
  '12-inference-serving/c2-inference-optimization/s4-tensor-parallelism': lazy(() => import('../subjects/12-inference-serving/c2-inference-optimization/s4-tensor-parallelism.jsx')),
  '12-inference-serving/c3-ollama/s1-what-is-ollama': lazy(() => import('../subjects/12-inference-serving/c3-ollama/s1-what-is-ollama.jsx')),
  '12-inference-serving/c3-ollama/s2-installation': lazy(() => import('../subjects/12-inference-serving/c3-ollama/s2-installation.jsx')),
  '12-inference-serving/c3-ollama/s3-pulling-models': lazy(() => import('../subjects/12-inference-serving/c3-ollama/s3-pulling-models.jsx')),
  '12-inference-serving/c3-ollama/s4-model-library': lazy(() => import('../subjects/12-inference-serving/c3-ollama/s4-model-library.jsx')),
  '12-inference-serving/c3-ollama/s5-modelfile': lazy(() => import('../subjects/12-inference-serving/c3-ollama/s5-modelfile.jsx')),
  '12-inference-serving/c3-ollama/s6-custom-models': lazy(() => import('../subjects/12-inference-serving/c3-ollama/s6-custom-models.jsx')),
  '12-inference-serving/c3-ollama/s7-import-gguf': lazy(() => import('../subjects/12-inference-serving/c3-ollama/s7-import-gguf.jsx')),
  '12-inference-serving/c3-ollama/s8-import-lora': lazy(() => import('../subjects/12-inference-serving/c3-ollama/s8-import-lora.jsx')),
  '12-inference-serving/c3-ollama/s9-quantization': lazy(() => import('../subjects/12-inference-serving/c3-ollama/s9-quantization.jsx')),
  '12-inference-serving/c3-ollama/s10-gpu-acceleration': lazy(() => import('../subjects/12-inference-serving/c3-ollama/s10-gpu-acceleration.jsx')),
  '12-inference-serving/c3-ollama/s11-multi-model': lazy(() => import('../subjects/12-inference-serving/c3-ollama/s11-multi-model.jsx')),
  '12-inference-serving/c3-ollama/s12-rest-api': lazy(() => import('../subjects/12-inference-serving/c3-ollama/s12-rest-api.jsx')),
  '12-inference-serving/c3-ollama/s13-streaming': lazy(() => import('../subjects/12-inference-serving/c3-ollama/s13-streaming.jsx')),
  '12-inference-serving/c3-ollama/s14-vision-models': lazy(() => import('../subjects/12-inference-serving/c3-ollama/s14-vision-models.jsx')),
  '12-inference-serving/c3-ollama/s15-embeddings': lazy(() => import('../subjects/12-inference-serving/c3-ollama/s15-embeddings.jsx')),
  '12-inference-serving/c3-ollama/s16-remote-serving': lazy(() => import('../subjects/12-inference-serving/c3-ollama/s16-remote-serving.jsx')),
  '12-inference-serving/c3-ollama/s17-comparison': lazy(() => import('../subjects/12-inference-serving/c3-ollama/s17-comparison.jsx')),
  '12-inference-serving/c4-open-webui/s1-what-is-openwebui': lazy(() => import('../subjects/12-inference-serving/c4-open-webui/s1-what-is-openwebui.jsx')),
  '12-inference-serving/c4-open-webui/s2-installation': lazy(() => import('../subjects/12-inference-serving/c4-open-webui/s2-installation.jsx')),
  '12-inference-serving/c4-open-webui/s3-ollama-backend': lazy(() => import('../subjects/12-inference-serving/c4-open-webui/s3-ollama-backend.jsx')),
  '12-inference-serving/c4-open-webui/s4-openai-apis': lazy(() => import('../subjects/12-inference-serving/c4-open-webui/s4-openai-apis.jsx')),
  '12-inference-serving/c4-open-webui/s5-multi-backend': lazy(() => import('../subjects/12-inference-serving/c4-open-webui/s5-multi-backend.jsx')),
  '12-inference-serving/c4-open-webui/s6-user-management': lazy(() => import('../subjects/12-inference-serving/c4-open-webui/s6-user-management.jsx')),
  '12-inference-serving/c4-open-webui/s7-chat-features': lazy(() => import('../subjects/12-inference-serving/c4-open-webui/s7-chat-features.jsx')),
  '12-inference-serving/c4-open-webui/s8-model-params': lazy(() => import('../subjects/12-inference-serving/c4-open-webui/s8-model-params.jsx')),
  '12-inference-serving/c4-open-webui/s9-system-prompts': lazy(() => import('../subjects/12-inference-serving/c4-open-webui/s9-system-prompts.jsx')),
  '12-inference-serving/c4-open-webui/s10-rag': lazy(() => import('../subjects/12-inference-serving/c4-open-webui/s10-rag.jsx')),
  '12-inference-serving/c4-open-webui/s11-web-search': lazy(() => import('../subjects/12-inference-serving/c4-open-webui/s11-web-search.jsx')),
  '12-inference-serving/c4-open-webui/s12-tools': lazy(() => import('../subjects/12-inference-serving/c4-open-webui/s12-tools.jsx')),
  '12-inference-serving/c4-open-webui/s13-pipelines': lazy(() => import('../subjects/12-inference-serving/c4-open-webui/s13-pipelines.jsx')),
  '12-inference-serving/c4-open-webui/s14-image-gen': lazy(() => import('../subjects/12-inference-serving/c4-open-webui/s14-image-gen.jsx')),
  '12-inference-serving/c4-open-webui/s15-voice': lazy(() => import('../subjects/12-inference-serving/c4-open-webui/s15-voice.jsx')),
  '12-inference-serving/c4-open-webui/s16-admin': lazy(() => import('../subjects/12-inference-serving/c4-open-webui/s16-admin.jsx')),
  '12-inference-serving/c4-open-webui/s17-theming': lazy(() => import('../subjects/12-inference-serving/c4-open-webui/s17-theming.jsx')),
  '12-inference-serving/c4-open-webui/s18-api-access': lazy(() => import('../subjects/12-inference-serving/c4-open-webui/s18-api-access.jsx')),
  '12-inference-serving/c5-serving-frameworks/s1-vllm': lazy(() => import('../subjects/12-inference-serving/c5-serving-frameworks/s1-vllm.jsx')),
  '12-inference-serving/c5-serving-frameworks/s2-tgi': lazy(() => import('../subjects/12-inference-serving/c5-serving-frameworks/s2-tgi.jsx')),
  '12-inference-serving/c5-serving-frameworks/s3-tensorrt': lazy(() => import('../subjects/12-inference-serving/c5-serving-frameworks/s3-tensorrt.jsx')),
  '12-inference-serving/c5-serving-frameworks/s4-llamacpp-server': lazy(() => import('../subjects/12-inference-serving/c5-serving-frameworks/s4-llamacpp-server.jsx')),
  '12-inference-serving/c5-serving-frameworks/s5-litellm': lazy(() => import('../subjects/12-inference-serving/c5-serving-frameworks/s5-litellm.jsx')),
  '12-inference-serving/c5-serving-frameworks/s6-comparison': lazy(() => import('../subjects/12-inference-serving/c5-serving-frameworks/s6-comparison.jsx')),
  '12-inference-serving/c6-apis-orchestration/s1-openai-compatible': lazy(() => import('../subjects/12-inference-serving/c6-apis-orchestration/s1-openai-compatible.jsx')),
  '12-inference-serving/c6-apis-orchestration/s2-streaming': lazy(() => import('../subjects/12-inference-serving/c6-apis-orchestration/s2-streaming.jsx')),
  '12-inference-serving/c6-apis-orchestration/s3-rate-limiting': lazy(() => import('../subjects/12-inference-serving/c6-apis-orchestration/s3-rate-limiting.jsx')),
  '12-inference-serving/c6-apis-orchestration/s4-cost-optimization': lazy(() => import('../subjects/12-inference-serving/c6-apis-orchestration/s4-cost-optimization.jsx')),
  // 13-rag
  '13-rag/c1-rag-fundamentals/s1-why-rag': lazy(() => import('../subjects/13-rag/c1-rag-fundamentals/s1-why-rag.jsx')),
  '13-rag/c1-rag-fundamentals/s2-pipeline': lazy(() => import('../subjects/13-rag/c1-rag-fundamentals/s2-pipeline.jsx')),
  '13-rag/c1-rag-fundamentals/s3-embedding-models': lazy(() => import('../subjects/13-rag/c1-rag-fundamentals/s3-embedding-models.jsx')),
  '13-rag/c1-rag-fundamentals/s4-vector-databases': lazy(() => import('../subjects/13-rag/c1-rag-fundamentals/s4-vector-databases.jsx')),
  '13-rag/c2-chunking/s1-chunking-strategies': lazy(() => import('../subjects/13-rag/c2-chunking/s1-chunking-strategies.jsx')),
  '13-rag/c2-chunking/s2-hierarchical-indexing': lazy(() => import('../subjects/13-rag/c2-chunking/s2-hierarchical-indexing.jsx')),
  '13-rag/c2-chunking/s3-metadata-filtering': lazy(() => import('../subjects/13-rag/c2-chunking/s3-metadata-filtering.jsx')),
  '13-rag/c2-chunking/s4-hybrid-search': lazy(() => import('../subjects/13-rag/c2-chunking/s4-hybrid-search.jsx')),
  '13-rag/c3-advanced-rag/s1-query-transformation': lazy(() => import('../subjects/13-rag/c3-advanced-rag/s1-query-transformation.jsx')),
  '13-rag/c3-advanced-rag/s2-reranking': lazy(() => import('../subjects/13-rag/c3-advanced-rag/s2-reranking.jsx')),
  '13-rag/c3-advanced-rag/s3-multi-hop': lazy(() => import('../subjects/13-rag/c3-advanced-rag/s3-multi-hop.jsx')),
  '13-rag/c3-advanced-rag/s4-agentic-rag': lazy(() => import('../subjects/13-rag/c3-advanced-rag/s4-agentic-rag.jsx')),
  '13-rag/c4-rag-evaluation/s1-metrics': lazy(() => import('../subjects/13-rag/c4-rag-evaluation/s1-metrics.jsx')),
  '13-rag/c4-rag-evaluation/s2-ragas': lazy(() => import('../subjects/13-rag/c4-rag-evaluation/s2-ragas.jsx')),
  '13-rag/c4-rag-evaluation/s3-context-optimization': lazy(() => import('../subjects/13-rag/c4-rag-evaluation/s3-context-optimization.jsx')),
  '13-rag/c4-rag-evaluation/s4-when-not-rag': lazy(() => import('../subjects/13-rag/c4-rag-evaluation/s4-when-not-rag.jsx')),
  // 14-agents
  '14-agents/c1-reasoning-engine/s1-cot': lazy(() => import('../subjects/14-agents/c1-reasoning-engine/s1-cot.jsx')),
  '14-agents/c1-reasoning-engine/s2-react': lazy(() => import('../subjects/14-agents/c1-reasoning-engine/s2-react.jsx')),
  '14-agents/c1-reasoning-engine/s3-planning': lazy(() => import('../subjects/14-agents/c1-reasoning-engine/s3-planning.jsx')),
  '14-agents/c1-reasoning-engine/s4-self-reflection': lazy(() => import('../subjects/14-agents/c1-reasoning-engine/s4-self-reflection.jsx')),
  '14-agents/c2-tool-use/s1-tool-definitions': lazy(() => import('../subjects/14-agents/c2-tool-use/s1-tool-definitions.jsx')),
  '14-agents/c2-tool-use/s2-parallel-sequential': lazy(() => import('../subjects/14-agents/c2-tool-use/s2-parallel-sequential.jsx')),
  '14-agents/c2-tool-use/s3-error-handling': lazy(() => import('../subjects/14-agents/c2-tool-use/s3-error-handling.jsx')),
  '14-agents/c2-tool-use/s4-custom-tools': lazy(() => import('../subjects/14-agents/c2-tool-use/s4-custom-tools.jsx')),
  '14-agents/c3-agent-frameworks/s1-langchain': lazy(() => import('../subjects/14-agents/c3-agent-frameworks/s1-langchain.jsx')),
  '14-agents/c3-agent-frameworks/s2-claude-agent-sdk': lazy(() => import('../subjects/14-agents/c3-agent-frameworks/s2-claude-agent-sdk.jsx')),
  '14-agents/c3-agent-frameworks/s3-autonomous-agents': lazy(() => import('../subjects/14-agents/c3-agent-frameworks/s3-autonomous-agents.jsx')),
  '14-agents/c3-agent-frameworks/s4-multi-agent': lazy(() => import('../subjects/14-agents/c3-agent-frameworks/s4-multi-agent.jsx')),
  '14-agents/c4-coding-agents/s1-code-generation': lazy(() => import('../subjects/14-agents/c4-coding-agents/s1-code-generation.jsx')),
  '14-agents/c4-coding-agents/s2-repo-understanding': lazy(() => import('../subjects/14-agents/c4-coding-agents/s2-repo-understanding.jsx')),
  '14-agents/c4-coding-agents/s3-tdd-llm': lazy(() => import('../subjects/14-agents/c4-coding-agents/s3-tdd-llm.jsx')),
  '14-agents/c4-coding-agents/s4-claude-code': lazy(() => import('../subjects/14-agents/c4-coding-agents/s4-claude-code.jsx')),
  // 15-evaluation-safety
  '15-evaluation-safety/c1-evaluation/s1-benchmarks': lazy(() => import('../subjects/15-evaluation-safety/c1-evaluation/s1-benchmarks.jsx')),
  '15-evaluation-safety/c1-evaluation/s2-llm-judge': lazy(() => import('../subjects/15-evaluation-safety/c1-evaluation/s2-llm-judge.jsx')),
  '15-evaluation-safety/c1-evaluation/s3-arena': lazy(() => import('../subjects/15-evaluation-safety/c1-evaluation/s3-arena.jsx')),
  '15-evaluation-safety/c1-evaluation/s4-task-eval': lazy(() => import('../subjects/15-evaluation-safety/c1-evaluation/s4-task-eval.jsx')),
  '15-evaluation-safety/c2-prompt-engineering/s1-zero-few-shot': lazy(() => import('../subjects/15-evaluation-safety/c2-prompt-engineering/s1-zero-few-shot.jsx')),
  '15-evaluation-safety/c2-prompt-engineering/s2-system-prompts': lazy(() => import('../subjects/15-evaluation-safety/c2-prompt-engineering/s2-system-prompts.jsx')),
  '15-evaluation-safety/c2-prompt-engineering/s3-prompt-chaining': lazy(() => import('../subjects/15-evaluation-safety/c2-prompt-engineering/s3-prompt-chaining.jsx')),
  '15-evaluation-safety/c2-prompt-engineering/s4-prompt-injection': lazy(() => import('../subjects/15-evaluation-safety/c2-prompt-engineering/s4-prompt-injection.jsx')),
  '15-evaluation-safety/c3-safety/s1-jailbreaking': lazy(() => import('../subjects/15-evaluation-safety/c3-safety/s1-jailbreaking.jsx')),
  '15-evaluation-safety/c3-safety/s2-constitutional-ai': lazy(() => import('../subjects/15-evaluation-safety/c3-safety/s2-constitutional-ai.jsx')),
  '15-evaluation-safety/c3-safety/s3-guardrails': lazy(() => import('../subjects/15-evaluation-safety/c3-safety/s3-guardrails.jsx')),
  '15-evaluation-safety/c3-safety/s4-responsible-disclosure': lazy(() => import('../subjects/15-evaluation-safety/c3-safety/s4-responsible-disclosure.jsx')),
  '15-evaluation-safety/c4-ethics/s1-bias-fairness': lazy(() => import('../subjects/15-evaluation-safety/c4-ethics/s1-bias-fairness.jsx')),
  '15-evaluation-safety/c4-ethics/s2-copyright': lazy(() => import('../subjects/15-evaluation-safety/c4-ethics/s2-copyright.jsx')),
  '15-evaluation-safety/c4-ethics/s3-environmental': lazy(() => import('../subjects/15-evaluation-safety/c4-ethics/s3-environmental.jsx')),
  '15-evaluation-safety/c4-ethics/s4-governance': lazy(() => import('../subjects/15-evaluation-safety/c4-ethics/s4-governance.jsx')),
}

function CheckIcon() {
  return (
    <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="3" strokeLinecap="round" strokeLinejoin="round" aria-hidden="true">
      <polyline points="20 6 9 17 4 12" />
    </svg>
  )
}

function ClockIcon() {
  return (
    <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round" aria-hidden="true">
      <circle cx="12" cy="12" r="10" />
      <polyline points="12 6 12 12 16 14" />
    </svg>
  )
}

function BookIcon() {
  return (
    <svg width="48" height="48" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.5" strokeLinecap="round" strokeLinejoin="round" className="text-emerald-300 dark:text-emerald-700" aria-hidden="true">
      <path d="M2 3h6a4 4 0 0 1 4 4v14a3 3 0 0 0-3-3H2z" />
      <path d="M22 3h-6a4 4 0 0 0-4 4v14a3 3 0 0 1 3-3h7z" />
    </svg>
  )
}

function ComingSoonPlaceholder({ section }) {
  return (
    <motion.div
      className="flex flex-col items-center gap-6 rounded-2xl border border-dashed border-emerald-200 bg-emerald-50/50 px-8 py-16 text-center dark:border-emerald-800/40 dark:bg-emerald-950/10"
      initial={{ opacity: 0, scale: 0.97 }}
      animate={{ opacity: 1, scale: 1 }}
      transition={{ duration: 0.4, delay: 0.1 }}
    >
      <BookIcon />
      <div className="space-y-2">
        <h2 className="text-xl font-bold text-gray-900 dark:text-white">
          Content Coming Soon
        </h2>
        <p className="max-w-md text-sm text-gray-500 dark:text-gray-400 leading-relaxed">
          The interactive content for{' '}
          <strong className="font-semibold text-gray-700 dark:text-gray-300">
            {section.title}
          </strong>{' '}
          is being prepared. It will include detailed concepts, code examples,
          architectural diagrams, and hands-on exercises.
        </p>
      </div>
      <div className="flex flex-wrap justify-center gap-2">
        {['Concepts', 'Code Examples', 'Diagrams', 'Exercises'].map((tag) => (
          <span
            key={tag}
            className="rounded-full bg-emerald-100 px-3 py-1 text-xs font-medium text-emerald-600 dark:bg-emerald-900/30 dark:text-emerald-400"
          >
            {tag}
          </span>
        ))}
      </div>
    </motion.div>
  )
}

function PrerequisiteBanner({ section, subjectId, chapterId }) {
  if (!section?.buildsOn) return null;
  const prereq = resolveBuildsOn(section.buildsOn);
  if (!prereq) return null;

  const isSameSubject = prereq.subjectId === subjectId;
  const href = `/subjects/${prereq.subjectId}/${prereq.chapterId}/${prereq.sectionId}`;

  return (
    <div className="mb-6 flex items-start gap-3 rounded-lg border border-amber-200 bg-amber-50/60 px-4 py-3 dark:border-amber-800/40 dark:bg-amber-950/20">
      <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round" className="mt-0.5 shrink-0 text-amber-600 dark:text-amber-400" aria-hidden="true">
        <path d="M2 3h6a4 4 0 0 1 4 4v14a3 3 0 0 0-3-3H2z" />
        <path d="M22 3h-6a4 4 0 0 0-4 4v14a3 3 0 0 1 3-3h7z" />
      </svg>
      <div className="text-sm leading-relaxed text-amber-900 dark:text-amber-200">
        <span className="font-medium">Builds on: </span>
        <Link
          to={href}
          className="underline decoration-amber-400/60 underline-offset-2 hover:decoration-amber-600 dark:decoration-amber-600/60 dark:hover:decoration-amber-400 transition-colors"
        >
          {prereq.title}
        </Link>
        {!isSameSubject && (
          <span className="ml-1 text-amber-700 dark:text-amber-400/80">
            ({prereq.subjectTitle})
          </span>
        )}
      </div>
    </div>
  );
}

function SectionContent({ subjectId, chapterId, sectionId, section }) {
  const key = `${subjectId}/${chapterId}/${sectionId}`
  const ContentComponent = CONTENT_REGISTRY[key]
  if (ContentComponent) {
    return (
      <Suspense fallback={<div className="py-16 text-center text-gray-400">Loading content…</div>}>
        <ContentComponent />
      </Suspense>
    )
  }
  return <ComingSoonPlaceholder section={section} />
}

export default function SectionPage() {
  const { subjectId, chapterId, sectionId } = useParams()
  const { isComplete, markComplete } = useProgress()

  const subject = getCurriculumById(subjectId)
  const chapter = getChapterById(subjectId, chapterId)
  const section = getSectionById(subjectId, chapterId, sectionId)
  const done = isComplete(subjectId, chapterId, sectionId)

  if (!subject || !chapter || !section) {
    return (
      <div className="flex min-h-[60vh] flex-col items-center justify-center gap-4 px-6 text-center">
        <div className="text-5xl" aria-hidden="true">∅</div>
        <h1 className="text-2xl font-bold text-gray-900 dark:text-white">Section Not Found</h1>
        <p className="text-gray-500 dark:text-gray-400">
          Could not find section "{sectionId}".
        </p>
        <Link
          to="/"
          className="rounded-lg bg-emerald-600 px-5 py-2 text-sm font-semibold text-white hover:bg-emerald-700 focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-emerald-500"
        >
          Back to Home
        </Link>
      </div>
    )
  }

  const { prev, next } = getAdjacentSections(subjectId, chapterId, sectionId)

  const breadcrumbs = [
    { label: 'Home', href: '/' },
    { label: subject.title, href: `/subjects/${subjectId}` },
    { label: chapter.title, href: `/subjects/${subjectId}/${chapterId}` },
    { label: section.title },
  ]

  function handleMarkComplete() {
    if (!done) {
      markComplete(subjectId, chapterId, sectionId)
    }
  }

  return (
    <div className="min-h-screen">
      {/* Section Header */}
      <div
        className="relative border-b border-gray-200 dark:border-gray-800"
        style={{ background: `linear-gradient(135deg, ${subject.colorHex}10 0%, transparent 50%)` }}
      >
        <div
          className="absolute left-0 top-0 h-full w-1.5"
          style={{ backgroundColor: subject.colorHex }}
          aria-hidden="true"
        />

        <div className="mx-auto max-w-3xl px-6 py-8 pl-10">
          <Breadcrumbs items={breadcrumbs} />

          <motion.div
            className="mt-4"
            initial={{ opacity: 0, y: 10 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ duration: 0.4 }}
          >
            <h1 className="text-2xl font-extrabold text-gray-900 dark:text-white sm:text-3xl leading-snug">
              {section.title}
            </h1>

            <div className="mt-3 flex flex-wrap items-center gap-3">
              <DifficultyBadge level={section.difficulty} />
              {section.readingMinutes && (
                <span className="flex items-center gap-1.5 text-sm text-gray-500 dark:text-gray-400">
                  <ClockIcon />
                  {section.readingMinutes} min read
                </span>
              )}
              {done && (
                <span className="flex items-center gap-1.5 rounded-full bg-emerald-100 px-2.5 py-0.5 text-xs font-semibold text-emerald-700 dark:bg-emerald-900/30 dark:text-emerald-400">
                  <CheckIcon />
                  Completed
                </span>
              )}
            </div>

            {section.description && (
              <p className="mt-3 text-gray-600 dark:text-gray-400 leading-relaxed">
                {section.description}
              </p>
            )}
          </motion.div>
        </div>
      </div>

      {/* Main content area */}
      <div className="mx-auto max-w-3xl px-6 py-12">
        {/* Prerequisite context for progressive learning */}
        <PrerequisiteBanner section={section} subjectId={subjectId} chapterId={chapterId} />

        {/* Dynamically loaded content or "Coming Soon" */}
        <SectionContent
          subjectId={subjectId}
          chapterId={chapterId}
          sectionId={sectionId}
          section={section}
        />

        {/* Mark as complete */}
        <div className="mt-8 flex justify-center">
          <button
            type="button"
            onClick={handleMarkComplete}
            disabled={done}
            className={`inline-flex items-center gap-2 rounded-xl px-6 py-3 text-sm font-semibold transition-all focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-emerald-500 focus-visible:ring-offset-2 ${
              done
                ? 'cursor-default bg-emerald-100 text-emerald-700 dark:bg-emerald-900/30 dark:text-emerald-400'
                : 'bg-emerald-600 text-white hover:bg-emerald-700 shadow-md hover:shadow-lg'
            }`}
            aria-label={done ? 'Section already marked complete' : 'Mark this section as complete'}
          >
            {done ? (
              <>
                <CheckIcon />
                Marked as Complete
              </>
            ) : (
              'Mark as Complete'
            )}
          </button>
        </div>

        {/* Prev / Next navigation */}
        <PrevNextNav prev={prev} next={next} />
      </div>
    </div>
  )
}
