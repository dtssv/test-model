# 多模态大模型系统设计文档

## 一、系统概述

### 1.1 项目目标

构建一个端到端的多模态大语言模型（Multimodal Large Language Model, MLLM）系统，支持文本、图像、音频、视频四种模态的理解与生成。系统涵盖完整的数据管线（采集、打标、清洗）、模型训练管线（预训练、指令微调、RLHF对齐）、以及对外API服务层。支持Dense模型与Mixture-of-Experts（MoE）稀疏模型，可灵活配置不同激活参数规模（1B/7B/13B/34B/72B）。

### 1.2 技术选型总览

| 层级 | 技术栈 |
|------|--------|
| 编程语言 | Python 3.11+, C++/CUDA (算子优化) |
| 深度学习框架 | PyTorch 2.3+ |
| 分布式训练 | DeepSpeed ZeRO-3, Megatron-Core (3D并行) |
| 视觉编码器 | EVA-CLIP ViT-G/14, SigLIP SO400M |
| 音频编码器 | Whisper-Large-V3 |
| 视频编码器 | InternVideo2 |
| 语言模型骨干 | Llama-3.1 / Qwen-2.5 架构 (可切换) |
| 数据处理 | data-juicer, MinerU, Label Studio |
| 模型推理 | vLLM 0.6+, TensorRT-LLM |
| API框架 | FastAPI + Uvicorn |
| 容器化 | Docker + Kubernetes |
| 监控 | Prometheus + Grafana |
| 存储 | MinIO (对象存储), PostgreSQL (元数据), Redis (缓存) |

### 1.3 系统架构总览

```
┌─────────────────────────────────────────────────────────────────────┐
│                         API Gateway (Nginx)                         │
├─────────────────────────────────────────────────────────────────────┤
│                    API Service Layer (FastAPI)                       │
│  ┌──────────┐  ┌──────────┐  ┌──────────┐  ┌───────────────────┐   │
│  │ Chat API │  │Vision API│  │Audio API │  │ Multimodal API    │   │
│  └──────────┘  └──────────┘  └──────────┘  └───────────────────┘   │
├─────────────────────────────────────────────────────────────────────┤
│                  Inference Engine (vLLM / TRT-LLM)                  │
├─────────────────────────────────────────────────────────────────────┤
│                    Model Registry & Versioning                      │
├─────────────────────────────────────────────────────────────────────┤
│              Training Pipeline (DeepSpeed + Megatron)                │
│  ┌──────────┐  ┌──────────┐  ┌──────────┐  ┌──────────┐           │
│  │Pretrain  │  │SFT       │  │DPO/RLHF  │  │MoE Train │           │
│  └──────────┘  └──────────┘  └──────────┘  └──────────┘           │
├─────────────────────────────────────────────────────────────────────┤
│                     Data Pipeline Layer                              │
│  ┌──────────┐  ┌──────────┐  ┌──────────┐  ┌──────────┐           │
│  │Collection│  │Labeling  │  │Cleaning  │  │Tokenizer │           │
│  └──────────┘  └──────────┘  └──────────┘  └──────────┘           │
├─────────────────────────────────────────────────────────────────────┤
│                     Storage Layer                                    │
│  ┌──────────┐  ┌──────────┐  ┌──────────┐                          │
│  │MinIO     │  │PostgreSQL│  │Redis     │                          │
│  └──────────┘  └──────────┘  └──────────┘                          │
└─────────────────────────────────────────────────────────────────────┘
```

---

## 二、数据管线模块（Data Pipeline）

### 2.1 模块总览

数据管线分为三个核心子模块：**数据采集（Collection）**、**数据打标（Labeling）**、**数据清洗（Cleaning）**，以及辅助的 **Tokenization** 模块。

```
data_pipeline/
├── __init__.py
├── collection/                    # 数据采集模块
│   ├── __init__.py
│   ├── base_collector.py          # 采集器基类
│   ├── text_collector.py          # 文本数据采集器
│   ├── image_text_collector.py    # 图文对数据采集器
│   ├── video_collector.py         # 视频数据采集器
│   ├── audio_collector.py         # 音频数据采集器
│   ├── web_crawler.py             # 通用网页爬取引擎
│   ├── api_fetcher.py             # 公开API数据拉取
│   ├── dataset_downloader.py      # 公开数据集下载器
│   └── deduplicator.py            # 数据去重器
├── labeling/                      # 数据打标模块
│   ├── __init__.py
│   ├── base_labeler.py            # 打标器基类
│   ├── caption_labeler.py         # 图像描述自动生成
│   ├── qa_labeler.py              # QA对自动生成
│   ├── classification_labeler.py  # 分类标签生成
│   ├── quality_scorer.py          # 数据质量评分器
│   ├── safety_labeler.py          # 安全标签标注
│   ├── human_review.py            # 人工审核接口(Label Studio)
│   └── label_studio_bridge.py     # Label Studio集成桥接
├── cleaning/                      # 数据清洗模块
│   ├── __init__.py
│   ├── base_cleaner.py            # 清洗器基类
│   ├── text_cleaner.py            # 文本清洗器
│   ├── image_cleaner.py           # 图像清洗器
│   ├── audio_cleaner.py           # 音频清洗器
│   ├── video_cleaner.py           # 视频清洗器
│   ├── dedup_engine.py            # 去重引擎(MinHash/SimHash)
│   ├── pii_remover.py             # 个人隐私信息移除
│   ├── toxicity_filter.py         # 有毒内容过滤
│   ├── quality_filter.py          # 质量过滤器
│   └── data_juicer_wrapper.py     # data-juicer框架封装
├── tokenization/                  # Tokenization模块
│   ├── __init__.py
│   ├── tokenizer_trainer.py       # BPE Tokenizer训练
│   ├── multimodal_tokenizer.py    # 多模态Token化
│   └── image_tokenizer.py         # 图像Token化(VQVAE)
├── storage/                       # 存储接口
│   ├── __init__.py
│   ├── minio_client.py            # MinIO对象存储客户端
│   ├── metadata_store.py          # PostgreSQL元数据管理
│   └── dataset_registry.py        # 数据集注册表
├── pipeline/                      # 管线编排
│   ├── __init__.py
│   ├── pipeline_executor.py       # 管线执行引擎
│   ├── dag_builder.py             # DAG构建器
│   └── pipeline_config.py         # 管线配置
└── config/
    ├── collection_config.yaml
    ├── labeling_config.yaml
    └── cleaning_config.yaml
```

### 2.2 数据采集模块详细设计

#### 2.2.1 `base_collector.py` — 采集器基类

```python
class BaseCollector(ABC):
    """
    所有数据采集器的抽象基类。
    定义采集、存储、上报进度的标准接口。
    """
    def __init__(self, config: CollectionConfig, storage: MinIOClient,
                 metadata_store: MetadataStore)
    
    @abstractmethod
    async def collect(self, source: DataSource) -> AsyncIterator[RawDataItem]:
        """从指定数据源异步采集原始数据"""
    
    async def save_batch(self, items: List[RawDataItem]) -> BatchSaveResult:
        """批量保存到MinIO，同时写入元数据"""
    
    def report_progress(self, collected: int, total: int, errors: int) -> None:
        """上报采集进度到监控系统"""
    
    def validate_item(self, item: RawDataItem) -> bool:
        """基础数据校验(格式、大小、完整性)"""
```

#### 2.2.2 `text_collector.py` — 文本数据采集器

```python
class TextCollector(BaseCollector):
    """
    大规模文本语料采集器。
    数据来源：Common Crawl、Wikipedia、arXiv、GitHub代码、
             书籍语料(Project Gutenberg)、新闻语料。
    """
    def __init__(self, config: TextCollectionConfig, ...)
    
    async def collect_common_crawl(self, warc_paths: List[str]) -> AsyncIterator[RawDataItem]:
        """从Common Crawl WARC文件中提取文本。使用warcio库解析。"""
    
    async def collect_wikipedia(self, lang: str, dump_date: str) -> AsyncIterator[RawDataItem]:
        """下载并解析Wikipedia dump。使用mwparserfromhell提取纯文本。"""
    
    async def collect_arxiv(self, categories: List[str]) -> AsyncIterator[RawDataItem]:
        """通过arXiv Bulk Data Access获取论文全文。使用MinerU提取PDF文本。"""
    
    async def collect_github_code(self, languages: List[str],
                                   min_stars: int) -> AsyncIterator[RawDataItem]:
        """通过GitHub Archive或BigQuery公开数据集获取开源代码。"""
    
    async def collect_books(self) -> AsyncIterator[RawDataItem]:
        """从Project Gutenberg等公开书籍源获取文本。"""
    
    def extract_text_from_html(self, html: str) -> str:
        """使用trafilatura库从HTML中提取正文文本。"""
    
    def detect_language(self, text: str) -> str:
        """使用fasttext lid模型检测文本语言。"""
```

#### 2.2.3 `image_text_collector.py` — 图文对数据采集器

```python
class ImageTextCollector(BaseCollector):
    """
    图文对数据采集器。
    数据来源：LAION-5B子集、CC3M/CC12M、DataComp、
             ShareGPT4V、ALLaVA等公开图文数据集。
    """
    def __init__(self, config: ImageTextCollectionConfig, ...)
    
    async def collect_laion(self, subset: str, 
                             aesthetic_score_min: float) -> AsyncIterator[RawDataItem]:
        """从LAION-5B下载图文对，通过img2dataset工具批量下载。
        使用CLIP aesthetic score进行初步筛选。"""
    
    async def collect_datacomp(self, pool_size: str) -> AsyncIterator[RawDataItem]:
        """从DataComp数据池获取高质量图文对。"""
    
    async def collect_sharegpt4v(self) -> AsyncIterator[RawDataItem]:
        """下载ShareGPT4V详细图像描述数据集。"""
    
    async def download_image(self, url: str, timeout: int) -> Optional[bytes]:
        """异步下载单张图像，带重试和超时机制。使用aiohttp。"""
    
    def verify_image(self, image_data: bytes) -> bool:
        """验证图像完整性(使用Pillow尝试打开并检查)。"""
    
    def compute_clip_score(self, image: Image, text: str) -> float:
        """计算图文对的CLIP相似度分数，用于质量筛选。"""
```

#### 2.2.4 `video_collector.py` — 视频数据采集器

```python
class VideoCollector(BaseCollector):
    """
    视频数据采集器。
    数据来源：WebVid-10M、InternVid、HD-VILA-100M、
             Pexels/Pixabay等CC协议视频。
    """
    def __init__(self, config: VideoCollectionConfig, ...)
    
    async def collect_webvid(self, split: str) -> AsyncIterator[RawDataItem]:
        """下载WebVid-10M视频-文本对数据。"""
    
    async def collect_pexels(self, query: str, 
                              max_duration: int) -> AsyncIterator[RawDataItem]:
        """通过Pexels API获取CC0协议视频。"""
    
    def extract_keyframes(self, video_path: str, 
                           fps: float) -> List[np.ndarray]:
        """使用OpenCV提取视频关键帧。"""
    
    def extract_audio_track(self, video_path: str) -> str:
        """使用ffmpeg分离视频音轨。"""
    
    def get_video_metadata(self, video_path: str) -> VideoMetadata:
        """提取视频元数据(分辨率、帧率、时长、编码格式)。"""
```

#### 2.2.5 `audio_collector.py` — 音频数据采集器

```python
class AudioCollector(BaseCollector):
    """
    音频数据采集器。
    数据来源：LibriSpeech、GigaSpeech、WenetSpeech、
             Common Voice、VoxPopuli。
    """
    def __init__(self, config: AudioCollectionConfig, ...)
    
    async def collect_librispeech(self, subset: str) -> AsyncIterator[RawDataItem]:
        """下载LibriSpeech ASR语料(960h英文)。"""
    
    async def collect_wenetspeech(self, subset: str) -> AsyncIterator[RawDataItem]:
        """下载WenetSpeech中文语音数据(10000h+)。"""
    
    async def collect_common_voice(self, lang: str, 
                                    version: str) -> AsyncIterator[RawDataItem]:
        """下载Mozilla Common Voice多语言语音数据。"""
    
    def convert_audio_format(self, audio_path: str, target_sr: int,
                              target_format: str) -> str:
        """使用soundfile/librosa统一音频格式和采样率。"""
    
    def compute_snr(self, audio: np.ndarray) -> float:
        """计算信噪比(SNR)用于质量筛选。"""
```

#### 2.2.6 `web_crawler.py` — 通用网页爬取引擎

```python
class WebCrawler:
    """
    通用网页爬取引擎，基于Scrapy + Playwright。
    支持JavaScript渲染页面、robots.txt遵守、频率限制。
    """
    def __init__(self, config: CrawlerConfig)
    
    async def crawl(self, seed_urls: List[str], 
                     max_pages: int, depth: int) -> AsyncIterator[CrawledPage]:
        """从种子URL开始深度爬取，返回解析后的页面。"""
    
    def respect_robots_txt(self, url: str) -> bool:
        """检查URL是否允许爬取(遵守robots.txt)。"""
    
    def extract_content(self, html: str, url: str) -> ExtractedContent:
        """使用trafilatura提取页面主体内容，去除导航栏/广告等。"""
    
    async def render_js_page(self, url: str) -> str:
        """使用Playwright渲染JavaScript动态页面。"""
```

#### 2.2.7 `dataset_downloader.py` — 公开数据集下载器

```python
class DatasetDownloader:
    """
    公开数据集统一下载器。
    支持HuggingFace Hub、学术数据集镜像等数据源。
    """
    def __init__(self, cache_dir: str, storage: MinIOClient)
    
    def download_from_huggingface(self, repo_id: str, 
                                   subset: Optional[str],
                                   revision: str) -> str:
        """从HuggingFace Hub下载数据集。使用huggingface_hub库。"""
    
    def download_from_url(self, url: str, 
                           checksum: Optional[str]) -> str:
        """从直接URL下载文件，支持断点续传和校验。"""
    
    def extract_archive(self, archive_path: str, 
                         target_dir: str) -> str:
        """解压tar/zip/gz等格式的压缩包。"""
    
    def verify_checksum(self, file_path: str, 
                         expected_hash: str, algo: str) -> bool:
        """校验文件完整性(SHA256/MD5)。"""
```

#### 2.2.8 `deduplicator.py` — 数据去重器

```python
class Deduplicator:
    """
    大规模数据去重引擎。
    支持精确去重(SHA256)和模糊去重(MinHash LSH, SimHash)。
    """
    def __init__(self, config: DedupConfig)
    
    def exact_dedup(self, items: Iterator[RawDataItem]) -> Iterator[RawDataItem]:
        """基于SHA256哈希的精确去重。使用布隆过滤器加速。"""
    
    def minhash_dedup(self, items: Iterator[RawDataItem], 
                       threshold: float, num_perm: int) -> Iterator[RawDataItem]:
        """基于MinHash LSH的模糊文本去重。使用datasketch库。"""
    
    def simhash_dedup(self, items: Iterator[RawDataItem],
                       hamming_distance: int) -> Iterator[RawDataItem]:
        """基于SimHash的近似去重，适用于网页文本。"""
    
    def image_dedup(self, items: Iterator[RawDataItem],
                     threshold: float) -> Iterator[RawDataItem]:
        """基于pHash感知哈希的图像去重。"""
```

### 2.3 数据打标模块详细设计

#### 2.3.1 `base_labeler.py` — 打标器基类

```python
class BaseLabeler(ABC):
    """
    数据打标器基类。支持自动打标和人工打标两种模式。
    自动打标使用预训练模型生成标签，人工打标通过Label Studio平台完成。
    """
    def __init__(self, config: LabelingConfig, model_registry: ModelRegistry)
    
    @abstractmethod
    def label(self, items: List[RawDataItem]) -> List[LabeledDataItem]:
        """对数据批量打标"""
    
    def validate_labels(self, items: List[LabeledDataItem]) -> ValidationReport:
        """标签质量校验(一致性、覆盖率、分布)"""
    
    def export_labels(self, items: List[LabeledDataItem], 
                       format: str) -> str:
        """导出标签为指定格式(JSON/Parquet/JSONL)"""
```

#### 2.3.2 `caption_labeler.py` — 图像描述自动生成

```python
class CaptionLabeler(BaseLabeler):
    """
    使用预训练视觉语言模型自动生成图像描述。
    策略：多模型投票 + 质量评分筛选。
    使用模型：InternVL2, CogVLM2, ShareCaptioner。
    """
    def __init__(self, config: CaptionLabelingConfig, 
                 models: Dict[str, VLModel])
    
    def generate_short_caption(self, image: Image) -> str:
        """生成简短描述(一句话，< 77 tokens)。用于预训练阶段。"""
    
    def generate_detailed_caption(self, image: Image) -> str:
        """生成详细描述(多句话，包含空间关系、属性、动作)。
        用于指令微调阶段。"""
    
    def generate_multi_model_caption(self, image: Image) -> CaptionResult:
        """多模型并行生成描述，取投票结果或最高评分。"""
    
    def score_caption(self, image: Image, caption: str) -> float:
        """使用CLIP/BLIP-2对生成的描述评分。"""
    
    def label(self, items: List[RawDataItem]) -> List[LabeledDataItem]:
        """批量处理，生成short+detailed两种粒度的描述。"""
```

#### 2.3.3 `qa_labeler.py` — QA对自动生成

```python
class QALabeler(BaseLabeler):
    """
    基于多模态内容自动生成问答对。
    策略：使用强模型(如GPT-4o/Qwen-VL-Max)生成，弱模型验证。
    生成多种类型QA：描述型、推理型、计数型、OCR型、空间关系型。
    """
    def __init__(self, config: QALabelingConfig, 
                 generator_model: str, verifier_model: str)
    
    def generate_descriptive_qa(self, image: Image, 
                                 caption: str) -> List[QAPair]:
        """生成描述型问答对(What/Who/Where类问题)。"""
    
    def generate_reasoning_qa(self, image: Image, 
                               caption: str) -> List[QAPair]:
        """生成推理型问答对(Why/How/What if类问题)。"""
    
    def generate_counting_qa(self, image: Image) -> List[QAPair]:
        """生成计数型问答对(How many类问题)。"""
    
    def generate_ocr_qa(self, image: Image) -> List[QAPair]:
        """生成OCR类问答对(针对图中文字)。"""
    
    def verify_qa_quality(self, image: Image, 
                           qa_pair: QAPair) -> QAVerificationResult:
        """使用验证模型检查QA对的正确性和相关性。"""
    
    def label(self, items: List[RawDataItem]) -> List[LabeledDataItem]:
        """综合生成多类型QA对并验证。"""
```

#### 2.3.4 `quality_scorer.py` — 数据质量评分器

```python
class QualityScorer:
    """
    多维度数据质量评分器。
    文本维度：流畅度、信息密度、教育价值(参考fineweb-edu)。
    图像维度：美学评分、清晰度、NSFW概率。
    音频维度：信噪比、语音清晰度、说话人质量。
    """
    def __init__(self, config: QualityScorerConfig)
    
    def score_text_quality(self, text: str) -> TextQualityScore:
        """文本质量评分。使用fasttext分类器评估教育价值(0-5分)，
        使用perplexity评估流畅度。"""
    
    def score_image_quality(self, image: Image) -> ImageQualityScore:
        """图像质量评分。使用CLIP aesthetic predictor、
        LAION-Aesthetics、NIMA模型。"""
    
    def score_audio_quality(self, audio: np.ndarray, sr: int) -> AudioQualityScore:
        """音频质量评分。SNR + PESQ + 语音活动检测(VAD)。"""
    
    def score_alignment(self, modality_a: Any, modality_b: Any,
                         modality_type: str) -> float:
        """跨模态对齐质量评分。使用CLIP/CLAP模型。"""
    
    def compute_composite_score(self, scores: Dict[str, float],
                                 weights: Dict[str, float]) -> float:
        """加权综合评分。"""
```

#### 2.3.5 `safety_labeler.py` — 安全标签标注

```python
class SafetyLabeler(BaseLabeler):
    """
    安全标签标注器。
    检测NSFW内容、暴力、仇恨言论、个人隐私(PII)等。
    """
    def __init__(self, config: SafetyLabelingConfig)
    
    def detect_nsfw_image(self, image: Image) -> NSFWResult:
        """NSFW图像检测。使用开源NSFW检测模型(如safety-checker)。"""
    
    def detect_toxic_text(self, text: str) -> ToxicityResult:
        """有毒文本检测。使用Perspective API或开源toxicity模型。"""
    
    def detect_pii(self, text: str) -> List[PIIEntity]:
        """个人隐私信息检测。使用presidio库检测姓名/邮箱/电话/地址等。"""
    
    def detect_violence(self, image: Image) -> ViolenceResult:
        """暴力图像检测。"""
    
    def label(self, items: List[RawDataItem]) -> List[LabeledDataItem]:
        """综合安全标注。"""
```

#### 2.3.6 `label_studio_bridge.py` — Label Studio集成

```python
class LabelStudioBridge:
    """
    Label Studio平台集成桥接层。
    用于人工标注任务的创建、分配、结果回收。
    """
    def __init__(self, api_url: str, api_key: str, project_config: dict)
    
    def create_project(self, name: str, 
                        label_config_xml: str) -> int:
        """在Label Studio中创建标注项目。"""
    
    def import_tasks(self, project_id: int, 
                      tasks: List[dict]) -> ImportResult:
        """批量导入标注任务。"""
    
    def export_annotations(self, project_id: int,
                            format: str) -> List[dict]:
        """导出已完成的标注结果。"""
    
    def get_project_stats(self, project_id: int) -> ProjectStats:
        """获取项目标注进度统计。"""
    
    def compute_inter_annotator_agreement(self, project_id: int) -> float:
        """计算标注者间一致性(Cohen's Kappa / Fleiss' Kappa)。"""
```

### 2.4 数据清洗模块详细设计

#### 2.4.1 `text_cleaner.py` — 文本清洗器

```python
class TextCleaner(BaseCleaner):
    """
    文本数据清洗器。执行多阶段清洗流程：
    规则清洗 → 语言过滤 → 质量过滤 → 去重 → PII移除 → 安全过滤。
    """
    def __init__(self, config: TextCleaningConfig)
    
    def remove_boilerplate(self, text: str) -> str:
        """移除网页模板文本(导航、页脚、广告)。
        使用trafilatura + 自定义规则。"""
    
    def normalize_unicode(self, text: str) -> str:
        """Unicode标准化(NFKC)，全角转半角，统一标点。"""
    
    def remove_html_tags(self, text: str) -> str:
        """清除残留HTML标签和实体。"""
    
    def filter_by_language(self, text: str, 
                            allowed_langs: List[str]) -> bool:
        """使用fasttext lid.176.bin模型过滤非目标语言。"""
    
    def filter_by_length(self, text: str, min_len: int, 
                          max_len: int) -> bool:
        """过滤过短或过长的文本。"""
    
    def filter_by_perplexity(self, text: str, 
                              max_ppl: float) -> bool:
        """使用KenLM语言模型计算困惑度，过滤低质量文本。"""
    
    def remove_repetitions(self, text: str, 
                            max_char_repeat: int,
                            max_word_repeat: int) -> str:
        """移除字符级和词级重复(如"哈哈哈哈哈哈"、连续重复句子)。"""
    
    def filter_by_education_score(self, text: str,
                                    min_score: float) -> bool:
        """使用fineweb-edu风格的教育价值分类器评分并过滤。"""
    
    def clean(self, items: List[RawDataItem]) -> List[CleanedDataItem]:
        """执行完整清洗流水线。"""
```

#### 2.4.2 `image_cleaner.py` — 图像清洗器

```python
class ImageCleaner(BaseCleaner):
    """
    图像数据清洗器。
    清洗维度：格式校验、分辨率过滤、美学评分、
    NSFW过滤、水印检测、重复检测。
    """
    def __init__(self, config: ImageCleaningConfig)
    
    def validate_format(self, image_data: bytes) -> bool:
        """验证图像格式(JPEG/PNG/WebP)和完整性。"""
    
    def filter_by_resolution(self, image: Image, 
                              min_size: int, max_size: int) -> bool:
        """过滤分辨率过低(< 256px)或过大的图像。"""
    
    def filter_by_aspect_ratio(self, image: Image,
                                max_ratio: float) -> bool:
        """过滤宽高比异常的图像(如极窄的Banner)。"""
    
    def filter_nsfw(self, image: Image, threshold: float) -> bool:
        """使用NSFW检测模型过滤不当图像。"""
    
    def detect_watermark(self, image: Image) -> float:
        """水印检测置信度评分。"""
    
    def filter_by_aesthetic_score(self, image: Image,
                                   min_score: float) -> bool:
        """使用LAION-Aesthetics V2模型评估美学分数。"""
    
    def filter_by_clip_score(self, image: Image, text: str,
                              min_score: float) -> bool:
        """过滤图文对齐度低的样本。"""
    
    def clean(self, items: List[RawDataItem]) -> List[CleanedDataItem]:
        """执行完整图像清洗流水线。"""
```

#### 2.4.3 `audio_cleaner.py` — 音频清洗器

```python
class AudioCleaner(BaseCleaner):
    """
    音频数据清洗器。
    清洗维度：格式标准化、信噪比过滤、静音检测、
    语音活动检测、说话人分离质量。
    """
    def __init__(self, config: AudioCleaningConfig)
    
    def standardize_format(self, audio_path: str) -> str:
        """统一为16kHz, 16bit, mono WAV格式。使用sox/soundfile。"""
    
    def filter_by_snr(self, audio: np.ndarray, 
                       min_snr: float) -> bool:
        """过滤信噪比低于阈值的音频。"""
    
    def detect_silence(self, audio: np.ndarray, sr: int,
                        max_silence_ratio: float) -> bool:
        """检测静音占比，过滤以静音为主的音频。"""
    
    def voice_activity_detection(self, audio: np.ndarray,
                                  sr: int) -> List[Tuple[float, float]]:
        """使用Silero VAD进行语音活动检测，返回语音片段时间戳。"""
    
    def filter_by_duration(self, audio: np.ndarray, sr: int,
                            min_dur: float, max_dur: float) -> bool:
        """过滤时长过短或过长的音频。"""
    
    def clean(self, items: List[RawDataItem]) -> List[CleanedDataItem]:
        """执行完整音频清洗流水线。"""
```

#### 2.4.4 `video_cleaner.py` — 视频清洗器

```python
class VideoCleaner(BaseCleaner):
    """
    视频数据清洗器。
    清洗维度：格式校验、分辨率过滤、帧质量评估、
    场景切换检测、重复视频检测。
    """
    def __init__(self, config: VideoCleaningConfig)
    
    def validate_video(self, video_path: str) -> bool:
        """验证视频文件完整性和可解码性。使用ffprobe。"""
    
    def filter_by_resolution(self, video_path: str,
                              min_height: int) -> bool:
        """过滤分辨率过低的视频。"""
    
    def filter_by_duration(self, video_path: str,
                            min_dur: float, max_dur: float) -> bool:
        """过滤时长不在范围内的视频。"""
    
    def compute_frame_quality(self, frames: List[np.ndarray]) -> float:
        """评估视频帧质量(清晰度、曝光)。"""
    
    def detect_scene_changes(self, video_path: str) -> List[float]:
        """使用PySceneDetect检测场景切换时间点。"""
    
    def detect_duplicate_video(self, video_path: str) -> Optional[str]:
        """基于视频指纹(VideoHash)检测重复视频。"""
    
    def clean(self, items: List[RawDataItem]) -> List[CleanedDataItem]:
        """执行完整视频清洗流水线。"""
```

#### 2.4.5 `dedup_engine.py` — 去重引擎

```python
class DedupEngine:
    """
    大规模数据去重引擎。支持十亿级文档去重。
    算法：MinHash LSH (文本)、pHash (图像)、VideoHash (视频)。
    使用datasketch库进行MinHash计算，使用Redis存储LSH索引。
    """
    def __init__(self, config: DedupConfig, redis_client: Redis)
    
    def build_minhash_index(self, corpus: Iterator[str],
                             num_perm: int, threshold: float) -> None:
        """构建MinHash LSH索引。分布式构建，存入Redis。"""
    
    def query_duplicates(self, text: str) -> List[str]:
        """查询文本的近似重复项。"""
    
    def build_phash_index(self, images: Iterator[Image]) -> None:
        """构建图像感知哈希索引。"""
    
    def batch_dedup(self, items: Iterator[RawDataItem],
                     modality: str) -> Iterator[RawDataItem]:
        """批量去重处理。根据modality选择对应算法。"""
```

#### 2.4.6 `pii_remover.py` — 个人隐私信息移除

```python
class PIIRemover:
    """
    个人可识别信息(PII)检测与移除。
    使用Microsoft Presidio框架 + 自定义中文规则。
    检测类型：姓名、邮箱、电话、身份证号、银行卡号、地址、IP地址。
    """
    def __init__(self, config: PIIConfig)
    
    def detect_pii(self, text: str) -> List[PIIEntity]:
        """检测文本中的PII实体。返回实体类型、位置、置信度。"""
    
    def anonymize(self, text: str, 
                   strategy: str) -> str:
        """匿名化处理。strategy: 'replace'(替换为占位符) 
        / 'mask'(部分遮盖) / 'hash'(哈希替换)。"""
    
    def detect_chinese_pii(self, text: str) -> List[PIIEntity]:
        """使用正则+NER模型检测中文PII(手机号、身份证、地址)。"""
```

#### 2.4.7 `data_juicer_wrapper.py` — data-juicer框架封装

```python
class DataJuicerWrapper:
    """
    阿里开源data-juicer框架的封装层。
    data-juicer提供了100+个数据处理算子(Operator)，
    支持文本/图像/音频/视频的统一处理。
    """
    def __init__(self, config_path: str)
    
    def load_config(self, config_path: str) -> dict:
        """加载data-juicer YAML配置文件。"""
    
    def run_pipeline(self, input_path: str, 
                      output_path: str) -> PipelineResult:
        """执行data-juicer处理流水线。"""
    
    def analyze_data(self, input_path: str) -> DataAnalysisReport:
        """数据分析：统计分布、质量指标、异常检测。"""
    
    def get_available_operators(self) -> List[OperatorInfo]:
        """列出所有可用的处理算子。"""
    
    def compose_recipe(self, operators: List[dict]) -> str:
        """组合算子生成处理方案(recipe)。"""
```

### 2.5 Tokenization模块详细设计

#### 2.5.1 `tokenizer_trainer.py` — Tokenizer训练

```python
class TokenizerTrainer:
    """
    BPE Tokenizer训练器。
    基于SentencePiece / HuggingFace Tokenizers库训练多语言Tokenizer。
    """
    def __init__(self, config: TokenizerConfig)
    
    def train_bpe(self, corpus_files: List[str], 
                   vocab_size: int, 
                   special_tokens: List[str]) -> None:
        """训练BPE Tokenizer。使用HuggingFace tokenizers库。"""
    
    def evaluate_fertility(self, tokenizer, 
                            test_texts: Dict[str, List[str]]) -> Dict[str, float]:
        """评估Tokenizer在不同语言上的fertility(tokens/word)。"""
    
    def add_special_tokens(self, tokenizer, 
                            tokens: List[str]) -> None:
        """添加多模态特殊token: <image>, <audio>, <video>, 
        <img_start>, <img_end>等。"""
    
    def export_tokenizer(self, output_dir: str) -> None:
        """导出训练好的Tokenizer(tokenizer.json + 配置文件)。"""
```

#### 2.5.2 `multimodal_tokenizer.py` — 多模态Token化

```python
class MultimodalTokenizer:
    """
    多模态统一Token化器。
    将文本、图像、音频、视频统一映射到token序列。
    文本：BPE tokens
    图像：视觉编码器特征 → 投影器 → visual tokens
    音频：Whisper编码器特征 → 投影器 → audio tokens
    视频：帧采样 → 视觉编码器 → temporal pooling → video tokens
    """
    def __init__(self, text_tokenizer, image_processor, 
                 audio_processor, video_processor)
    
    def tokenize_text(self, text: str) -> List[int]:
        """文本BPE编码。"""
    
    def tokenize_image(self, image: Image, 
                        resolution: int) -> torch.Tensor:
        """图像处理为模型输入tensor。
        支持动态分辨率(anyres策略)。"""
    
    def tokenize_audio(self, audio: np.ndarray, sr: int) -> torch.Tensor:
        """音频处理为Whisper特征输入。"""
    
    def tokenize_video(self, video_path: str,
                        max_frames: int, fps: float) -> torch.Tensor:
        """视频均匀采样帧，处理为tensor序列。"""
    
    def build_multimodal_input(self, conversation: List[dict]) -> ModelInput:
        """将多模态对话构建为模型输入。
        处理<image>/<audio>/<video>占位符，
        将对应模态特征插入到token序列中。"""
```

### 2.6 存储与管线编排

#### 2.6.1 `minio_client.py` — MinIO对象存储客户端

```python
class MinIOClient:
    """
    MinIO对象存储客户端封装。
    用于存储原始数据、清洗后数据、模型checkpoint等大文件。
    """
    def __init__(self, endpoint: str, access_key: str, 
                 secret_key: str, secure: bool)
    
    def upload_file(self, bucket: str, object_name: str, 
                     file_path: str) -> str:
        """上传文件到MinIO，返回对象URL。"""
    
    def upload_bytes(self, bucket: str, object_name: str, 
                      data: bytes) -> str:
        """上传字节数据到MinIO。"""
    
    def download_file(self, bucket: str, object_name: str,
                       file_path: str) -> None:
        """从MinIO下载文件。"""
    
    def list_objects(self, bucket: str, 
                      prefix: str) -> List[ObjectInfo]:
        """列出桶中指定前缀的对象。"""
    
    def ensure_bucket(self, bucket: str) -> None:
        """确保桶存在，不存在则创建。"""
```

#### 2.6.2 `metadata_store.py` — 元数据管理

```python
class MetadataStore:
    """
    基于PostgreSQL的数据集元数据管理。
    记录每条数据的来源、处理状态、质量分数、标签等。
    """
    def __init__(self, dsn: str)
    
    def register_dataset(self, dataset: DatasetInfo) -> int:
        """注册新数据集，返回数据集ID。"""
    
    def insert_items(self, dataset_id: int, 
                      items: List[DataItemMeta]) -> None:
        """批量插入数据项元数据。"""
    
    def update_item_status(self, item_id: str, 
                            status: str, metadata: dict) -> None:
        """更新数据项的处理状态和元数据。"""
    
    def query_items(self, filters: Dict[str, Any],
                     limit: int, offset: int) -> List[DataItemMeta]:
        """按条件查询数据项。支持按质量分数、状态、模态等过滤。"""
    
    def get_dataset_statistics(self, dataset_id: int) -> DatasetStats:
        """获取数据集统计信息(数量、模态分布、质量分布等)。"""
```

#### 2.6.3 `pipeline_executor.py` — 管线执行引擎

```python
class PipelineExecutor:
    """
    数据管线执行引擎。
    支持DAG式任务编排，断点续跑，进度监控。
    """
    def __init__(self, config: PipelineConfig)
    
    def build_dag(self, stages: List[PipelineStage]) -> DAG:
        """构建处理流程DAG。"""
    
    async def execute(self, dag: DAG, 
                       resume_from: Optional[str]) -> PipelineResult:
        """执行DAG，支持从指定节点恢复。"""
    
    def checkpoint(self, stage_name: str, state: dict) -> None:
        """保存检查点用于断点续跑。"""
    
    def get_progress(self) -> Dict[str, StageProgress]:
        """获取各阶段执行进度。"""
    
    def handle_failure(self, stage_name: str, 
                        error: Exception) -> FailureAction:
        """失败处理策略：重试/跳过/终止。"""
```

---

## 三、模型架构模块（Model Architecture）

### 3.1 模块总览

模型架构采用标准的 MLLM 五组件设计：**模态编码器 → 输入投影器 → 语言模型骨干 → 输出投影器 → 模态生成器**。支持 Dense 和 MoE 两种模型结构，以及多种参数规模。

```
model/
├── __init__.py
├── architecture/                      # 模型架构
│   ├── __init__.py
│   ├── model_config.py                # 统一模型配置
│   ├── multimodal_model.py            # 多模态模型主类
│   ├── model_factory.py               # 模型工厂(按配置创建模型)
│   ├── encoders/                      # 模态编码器
│   │   ├── __init__.py
│   │   ├── vision_encoder.py          # 视觉编码器(EVA-CLIP/SigLIP)
│   │   ├── audio_encoder.py           # 音频编码器(Whisper)
│   │   ├── video_encoder.py           # 视频编码器(InternVideo2)
│   │   └── encoder_factory.py         # 编码器工厂
│   ├── projectors/                    # 投影器/连接器
│   │   ├── __init__.py
│   │   ├── mlp_projector.py           # MLP投影器
│   │   ├── qformer_projector.py       # Q-Former投影器
│   │   ├── resampler_projector.py     # Perceiver Resampler
│   │   └── dynamic_resolution.py      # 动态分辨率处理
│   ├── backbone/                      # 语言模型骨干
│   │   ├── __init__.py
│   │   ├── transformer_block.py       # Transformer Block
│   │   ├── attention.py               # 多头注意力(GQA/MHA)
│   │   ├── feed_forward.py            # FFN (SwiGLU)
│   │   ├── rope_embedding.py          # RoPE位置编码
│   │   ├── rmsnorm.py                 # RMSNorm
│   │   ├── dense_backbone.py          # Dense模型骨干
│   │   └── moe_backbone.py            # MoE模型骨干
│   ├── moe/                           # MoE专用组件
│   │   ├── __init__.py
│   │   ├── moe_layer.py              # MoE层实现
│   │   ├── expert.py                  # 专家网络
│   │   ├── router.py                  # 路由器(Top-K/Expert Choice)
│   │   ├── load_balancing.py          # 负载均衡损失
│   │   └── expert_parallel.py         # 专家并行通信
│   └── heads/                         # 输出头
│       ├── __init__.py
│       ├── lm_head.py                 # 语言模型头
│       └── multimodal_head.py         # 多模态生成头
├── scaling/                           # 模型规模配置
│   ├── __init__.py
│   ├── model_scales.py                # 预定义模型规模(1B-72B)
│   └── scaling_laws.py                # Scaling Law工具
└── utils/
    ├── __init__.py
    ├── checkpoint_utils.py            # Checkpoint工具
    └── model_utils.py                 # 模型工具函数
```

### 3.2 模型配置体系

#### 3.2.1 `model_config.py` — 统一模型配置

```python
@dataclass
class MultimodalModelConfig:
    """多模态大模型统一配置。"""
    # === 语言模型骨干配置 ===
    hidden_size: int                      # 隐藏层维度
    num_hidden_layers: int                # Transformer层数
    num_attention_heads: int              # 注意力头数
    num_key_value_heads: int              # KV头数(GQA)
    intermediate_size: int                # FFN中间层维度
    vocab_size: int                       # 词表大小
    max_position_embeddings: int          # 最大位置长度
    rope_theta: float                     # RoPE base frequency
    rms_norm_eps: float                   # RMSNorm epsilon
    activation_function: str              # 激活函数(silu/gelu)
    tie_word_embeddings: bool             # 是否共享输入输出embedding
    
    # === MoE配置 ===
    use_moe: bool                         # 是否使用MoE
    num_experts: int                      # 专家总数
    num_experts_per_tok: int              # 每个token激活的专家数(Top-K)
    moe_layer_freq: int                   # MoE层间隔(每隔N层放一个MoE层)
    router_aux_loss_coef: float           # 路由辅助损失系数
    expert_capacity_factor: float         # 专家容量因子
    moe_routing_type: str                 # 路由类型(top_k/expert_choice)
    shared_expert_num: int                # 共享专家数量(DeepSeek-V2风格)
    
    # === 视觉编码器配置 ===
    vision_encoder_type: str              # eva_clip/siglip
    vision_encoder_path: str              # 预训练权重路径
    image_size: int                       # 输入图像尺寸
    patch_size: int                       # patch大小
    vision_hidden_size: int               # 视觉编码器隐藏维度
    use_dynamic_resolution: bool          # 动态分辨率
    max_tiles: int                        # 最大tile数量
    
    # === 音频编码器配置 ===
    audio_encoder_type: str               # whisper
    audio_encoder_path: str               # 预训练权重路径
    audio_sample_rate: int                # 采样率(16000)
    audio_max_length: int                 # 最大音频长度(秒)
    
    # === 视频编码器配置 ===
    video_encoder_type: str               # internvideo2
    video_max_frames: int                 # 最大帧数
    video_fps: float                      # 采样帧率
    
    # === 投影器配置 ===
    projector_type: str                   # mlp/qformer/resampler
    projector_hidden_size: int            # 投影器隐藏维度
    num_query_tokens: int                 # Q-Former/Resampler查询token数
    
    # === 训练配置引用 ===
    dtype: str                            # bf16/fp16/fp32
```

#### 3.2.2 `model_scales.py` — 预定义模型规模

```python
MODEL_SCALES = {
    "1B-Dense": MultimodalModelConfig(
        hidden_size=2048, num_hidden_layers=24, num_attention_heads=16,
        num_key_value_heads=8, intermediate_size=5504, vocab_size=152064,
        max_position_embeddings=32768, use_moe=False,
        vision_encoder_type="siglip", projector_type="mlp",
        # ... 总参数量 ~1.5B (含视觉编码器)
    ),
    "7B-Dense": MultimodalModelConfig(
        hidden_size=4096, num_hidden_layers=32, num_attention_heads=32,
        num_key_value_heads=8, intermediate_size=11008, vocab_size=152064,
        max_position_embeddings=131072, use_moe=False,
        vision_encoder_type="siglip", projector_type="mlp",
        # ... 总参数量 ~8B (含视觉编码器)
    ),
    "13B-Dense": MultimodalModelConfig(
        hidden_size=5120, num_hidden_layers=40, num_attention_heads=40,
        num_key_value_heads=8, intermediate_size=13824, vocab_size=152064,
        max_position_embeddings=131072, use_moe=False,
        vision_encoder_type="eva_clip", projector_type="mlp",
        # ... 总参数量 ~14B
    ),
    "34B-Dense": MultimodalModelConfig(
        hidden_size=6656, num_hidden_layers=60, num_attention_heads=52,
        num_key_value_heads=8, intermediate_size=17920, vocab_size=152064,
        max_position_embeddings=131072, use_moe=False,
        vision_encoder_type="eva_clip", projector_type="mlp",
        # ... 总参数量 ~35B
    ),
    "72B-Dense": MultimodalModelConfig(
        hidden_size=8192, num_hidden_layers=80, num_attention_heads=64,
        num_key_value_heads=8, intermediate_size=29568, vocab_size=152064,
        max_position_embeddings=131072, use_moe=False,
        vision_encoder_type="eva_clip", projector_type="mlp",
        # ... 总参数量 ~73B
    ),
    # === MoE模型 (激活参数/总参数) ===
    "7B-MoE-A2B": MultimodalModelConfig(
        hidden_size=2048, num_hidden_layers=28, num_attention_heads=16,
        num_key_value_heads=8, intermediate_size=5504, vocab_size=152064,
        max_position_embeddings=131072, use_moe=True,
        num_experts=64, num_experts_per_tok=6, moe_layer_freq=1,
        shared_expert_num=2, moe_routing_type="top_k",
        # ... 总参数量 ~7B, 激活参数 ~2B (DeepSeek-V2 Lite风格)
    ),
    "34B-MoE-A7B": MultimodalModelConfig(
        hidden_size=4096, num_hidden_layers=32, num_attention_heads=32,
        num_key_value_heads=8, intermediate_size=11008, vocab_size=152064,
        max_position_embeddings=131072, use_moe=True,
        num_experts=8, num_experts_per_tok=2, moe_layer_freq=1,
        shared_expert_num=0, moe_routing_type="top_k",
        # ... 总参数量 ~34B, 激活参数 ~7B (Mixtral风格)
    ),
    "200B-MoE-A34B": MultimodalModelConfig(
        hidden_size=6656, num_hidden_layers=60, num_attention_heads=52,
        num_key_value_heads=8, intermediate_size=17920, vocab_size=152064,
        max_position_embeddings=131072, use_moe=True,
        num_experts=128, num_experts_per_tok=6, moe_layer_freq=1,
        shared_expert_num=2, moe_routing_type="top_k",
        # ... 总参数量 ~200B, 激活参数 ~34B (DeepSeek-V2风格)
    ),
}
```

### 3.3 编码器详细设计

#### 3.3.1 `vision_encoder.py` — 视觉编码器

```python
class VisionEncoder(nn.Module):
    """
    视觉编码器。支持EVA-CLIP ViT-G/14和SigLIP SO400M。
    使用预训练权重初始化，训练时可选择冻结或部分微调。
    """
    def __init__(self, config: MultimodalModelConfig)
    
    def load_pretrained(self, path: str) -> None:
        """加载预训练视觉编码器权重。"""
    
    def forward(self, pixel_values: torch.Tensor) -> torch.Tensor:
        """
        输入: pixel_values [B, C, H, W] 或 [B, N_tiles, C, H, W]
        输出: image_features [B, N_patches, vision_hidden_size]
        """
    
    def extract_features(self, pixel_values: torch.Tensor,
                          select_layer: int) -> torch.Tensor:
        """从指定层提取特征(通常倒数第二层效果最好)。"""
    
    def freeze(self) -> None:
        """冻结编码器参数。"""
    
    def unfreeze_last_n_layers(self, n: int) -> None:
        """解冻最后N层用于微调。"""
```

#### 3.3.2 `audio_encoder.py` — 音频编码器

```python
class AudioEncoder(nn.Module):
    """
    音频编码器。基于OpenAI Whisper-Large-V3。
    提取音频特征序列用于后续投影。
    """
    def __init__(self, config: MultimodalModelConfig)
    
    def load_pretrained(self, path: str) -> None:
        """加载Whisper预训练编码器权重。"""
    
    def forward(self, audio_features: torch.Tensor) -> torch.Tensor:
        """
        输入: audio_features [B, T, mel_bins] (Mel频谱)
        输出: audio_hidden [B, T', audio_hidden_size]
        """
    
    def preprocess_audio(self, waveform: torch.Tensor,
                          sample_rate: int) -> torch.Tensor:
        """预处理音频波形为Mel频谱特征。"""
    
    def freeze(self) -> None:
        """冻结编码器参数。"""
```

#### 3.3.3 `video_encoder.py` — 视频编码器

```python
class VideoEncoder(nn.Module):
    """
    视频编码器。可复用视觉编码器逐帧编码 + 时序池化，
    或使用InternVideo2进行视频级编码。
    """
    def __init__(self, config: MultimodalModelConfig, 
                 vision_encoder: Optional[VisionEncoder])
    
    def forward(self, video_frames: torch.Tensor) -> torch.Tensor:
        """
        输入: video_frames [B, T, C, H, W]
        输出: video_features [B, N_tokens, hidden_size]
        """
    
    def temporal_pooling(self, frame_features: torch.Tensor,
                          method: str) -> torch.Tensor:
        """时序特征池化。method: 'mean'/'attention'/'adaptive'。"""
    
    def uniform_sample_frames(self, video_path: str, 
                                num_frames: int) -> torch.Tensor:
        """均匀采样视频帧。"""
```

### 3.4 投影器详细设计

#### 3.4.1 `mlp_projector.py` — MLP投影器

```python
class MLPProjector(nn.Module):
    """
    两层MLP投影器(LLaVA风格)。
    将视觉/音频特征映射到语言模型的隐藏空间。
    结构: Linear → GELU → Linear
    """
    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        输入: [B, N, input_dim]
        输出: [B, N, output_dim]
        """
```

#### 3.4.2 `resampler_projector.py` — Perceiver Resampler

```python
class PerceiverResampler(nn.Module):
    """
    Perceiver Resampler投影器(Flamingo/Qwen-VL风格)。
    使用可学习查询向量对视觉token进行压缩，
    将可变数量的视觉token压缩为固定数量的查询token。
    """
    def __init__(self, input_dim: int, output_dim: int,
                 num_queries: int, num_heads: int, num_layers: int)
    
    def forward(self, visual_features: torch.Tensor,
                 attention_mask: Optional[torch.Tensor]) -> torch.Tensor:
        """
        输入: visual_features [B, N_visual, input_dim]
        输出: query_output [B, num_queries, output_dim]
        通过交叉注意力将N个视觉token压缩为num_queries个。
        """
```

#### 3.4.3 `dynamic_resolution.py` — 动态分辨率处理

```python
class DynamicResolutionProcessor:
    """
    动态分辨率处理器(AnyRes策略)。
    将高分辨率图像切分为多个tile，
    每个tile单独编码后拼接，同时保留一个全局缩略图。
    """
    def __init__(self, base_resolution: int, max_tiles: int,
                 possible_resolutions: List[Tuple[int, int]])
    
    def select_best_resolution(self, original_size: Tuple[int, int]) -> Tuple[int, int]:
        """选择最佳分辨率网格(如2x2, 1x3等)。"""
    
    def split_into_tiles(self, image: Image, 
                          target_resolution: Tuple[int, int]) -> List[Image]:
        """将图像切分为tile列表。"""
    
    def create_thumbnail(self, image: Image) -> Image:
        """创建全局缩略图。"""
    
    def process(self, image: Image) -> Tuple[torch.Tensor, int]:
        """完整处理: 切分 + 缩略图 + 返回tensor和实际tile数。"""
```

### 3.5 语言模型骨干与MoE详细设计

#### 3.5.1 `transformer_block.py` — Transformer Block

```python
class TransformerBlock(nn.Module):
    """
    标准Transformer解码器块。
    结构: RMSNorm → Attention → Residual → RMSNorm → FFN → Residual
    支持Pre-Norm和Post-Norm两种模式。
    """
    def __init__(self, config: MultimodalModelConfig, layer_idx: int)
    
    def forward(self, hidden_states: torch.Tensor,
                 attention_mask: torch.Tensor,
                 position_ids: torch.Tensor,
                 past_key_value: Optional[Tuple],
                 use_cache: bool) -> Tuple[torch.Tensor, ...]:
        """前向传播。"""
```

#### 3.5.2 `attention.py` — 注意力机制

```python
class GroupedQueryAttention(nn.Module):
    """
    分组查询注意力(GQA)。
    Q头数 > KV头数，减少KV缓存内存占用。
    支持FlashAttention-2加速。
    """
    def __init__(self, hidden_size: int, num_heads: int,
                 num_kv_heads: int, head_dim: int)
    
    def forward(self, hidden_states: torch.Tensor,
                 attention_mask: torch.Tensor,
                 position_ids: torch.Tensor,
                 past_key_value: Optional[Tuple],
                 use_cache: bool) -> Tuple[torch.Tensor, Optional[Tuple]]:
        """
        使用FlashAttention-2计算注意力。
        支持causal mask和cross-attention mask。
        """
    
    def _apply_rotary_embedding(self, q: torch.Tensor, 
                                  k: torch.Tensor,
                                  position_ids: torch.Tensor) -> Tuple:
        """应用RoPE旋转位置编码。"""
```

#### 3.5.3 `feed_forward.py` — FFN

```python
class SwiGLUFeedForward(nn.Module):
    """
    SwiGLU前馈网络。
    结构: gate_proj(x) * SiLU(up_proj(x)) → down_proj
    参数量: 3 * hidden_size * intermediate_size (相比标准FFN多50%)
    """
    def __init__(self, hidden_size: int, intermediate_size: int)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """SwiGLU前向计算。"""
```

#### 3.5.4 `moe_layer.py` — MoE层

```python
class MoELayer(nn.Module):
    """
    Mixture of Experts层。
    替代标准Transformer中的FFN层。
    支持Top-K路由和Expert Choice路由。
    支持共享专家(Shared Expert, DeepSeek-V2风格)。
    """
    def __init__(self, config: MultimodalModelConfig, layer_idx: int)
    
    @property
    def num_active_params(self) -> int:
        """返回每个token激活的参数量。"""
    
    def forward(self, hidden_states: torch.Tensor) -> Tuple[torch.Tensor, dict]:
        """
        输入: hidden_states [B, T, D]
        输出: (output [B, T, D], aux_losses dict)
        流程: 
        1. Router计算每个token的专家选择概率
        2. Top-K选择激活的专家
        3. 将token分发给对应专家计算
        4. 加权聚合专家输出
        5. (可选) 加上共享专家的输出
        """
    
    def _dispatch_tokens(self, hidden_states: torch.Tensor,
                          routing_weights: torch.Tensor,
                          expert_indices: torch.Tensor) -> List[torch.Tensor]:
        """将token分发到对应专家。"""
    
    def _combine_outputs(self, expert_outputs: List[torch.Tensor],
                          routing_weights: torch.Tensor) -> torch.Tensor:
        """加权组合专家输出。"""
```

#### 3.5.5 `router.py` — 路由器

```python
class TopKRouter(nn.Module):
    """
    Top-K路由器。每个token选择得分最高的K个专家。
    使用可学习的线性层计算路由概率。
    支持容量因子(capacity factor)限制每个专家处理的token数。
    """
    def __init__(self, hidden_size: int, num_experts: int,
                 top_k: int, capacity_factor: float)
    
    def forward(self, hidden_states: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        输入: hidden_states [B*T, D]
        输出: (routing_weights [B*T, K], expert_indices [B*T, K])
        """
    
    def _add_noise(self, logits: torch.Tensor) -> torch.Tensor:
        """训练时添加噪声增加探索(Noisy Top-K)。"""

class ExpertChoiceRouter(nn.Module):
    """
    Expert Choice路由器。每个专家选择得分最高的C个token。
    保证完美的负载均衡，但token可能被多个或零个专家选中。
    """
    def __init__(self, hidden_size: int, num_experts: int,
                 capacity_factor: float)
    
    def forward(self, hidden_states: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        输入: hidden_states [B*T, D]
        输出: (dispatch_mask [E, C, B*T], combine_weights [E, C, B*T])
        """
```

#### 3.5.6 `load_balancing.py` — 负载均衡

```python
class LoadBalancingLoss:
    """
    MoE负载均衡辅助损失。
    防止token集中到少数专家(专家坍塌问题)。
    支持多种均衡策略：Switch Loss、Z-Loss、Expert-Level Balance Loss。
    """
    def __init__(self, num_experts: int, 
                 aux_loss_coef: float, z_loss_coef: float)
    
    def compute_switch_loss(self, routing_logits: torch.Tensor,
                             expert_indices: torch.Tensor) -> torch.Tensor:
        """Switch Transformer风格的均衡损失。f_i * P_i的乘积。"""
    
    def compute_z_loss(self, routing_logits: torch.Tensor) -> torch.Tensor:
        """Z-Loss: 惩罚路由logits过大，防止数值不稳定。"""
    
    def compute_expert_balance_loss(self, expert_load: torch.Tensor) -> torch.Tensor:
        """专家级负载均衡损失。惩罚负载方差。"""
    
    def forward(self, routing_logits: torch.Tensor,
                 expert_indices: torch.Tensor) -> torch.Tensor:
        """综合计算辅助损失。"""
```

#### 3.5.7 `multimodal_model.py` — 多模态模型主类

```python
class MultimodalModel(nn.Module):
    """
    多模态大模型主类。整合所有组件。
    支持多模态输入(文本+图像+音频+视频)的理解与生成。
    """
    def __init__(self, config: MultimodalModelConfig)
    
    @property
    def total_params(self) -> int:
        """总参数量。"""
    
    @property
    def active_params(self) -> int:
        """每个token的激活参数量(Dense=总参数, MoE<总参数)。"""
    
    def encode_images(self, pixel_values: torch.Tensor) -> torch.Tensor:
        """编码图像: VisionEncoder → Projector → image_tokens"""
    
    def encode_audio(self, audio_values: torch.Tensor) -> torch.Tensor:
        """编码音频: AudioEncoder → Projector → audio_tokens"""
    
    def encode_video(self, video_values: torch.Tensor) -> torch.Tensor:
        """编码视频: VideoEncoder → Projector → video_tokens"""
    
    def prepare_inputs_embeds(self, input_ids: torch.Tensor,
                               pixel_values: Optional[torch.Tensor],
                               audio_values: Optional[torch.Tensor],
                               video_values: Optional[torch.Tensor]) -> torch.Tensor:
        """准备多模态输入embedding。
        将文本token embedding与多模态token在序列维度拼接。"""
    
    def forward(self, input_ids: torch.Tensor,
                 attention_mask: torch.Tensor,
                 pixel_values: Optional[torch.Tensor],
                 audio_values: Optional[torch.Tensor],
                 video_values: Optional[torch.Tensor],
                 labels: Optional[torch.Tensor],
                 use_cache: bool) -> CausalLMOutput:
        """
        完整前向传播。
        返回: loss (如果提供labels), logits, past_key_values, aux_losses
        """
    
    @torch.no_grad()
    def generate(self, **kwargs) -> torch.Tensor:
        """自回归生成。委托给HuggingFace GenerationMixin。"""
```

#### 3.5.8 `model_factory.py` — 模型工厂

```python
class ModelFactory:
    """
    模型工厂。根据配置创建对应规模和类型的模型。
    """
    @staticmethod
    def create_model(scale: str, 
                      custom_config: Optional[dict]) -> MultimodalModel:
        """
        根据预定义规模(1B-Dense, 7B-Dense, 34B-MoE-A7B等)
        或自定义配置创建模型。
        """
    
    @staticmethod
    def from_pretrained(model_path: str, 
                         device_map: str) -> MultimodalModel:
        """从预训练checkpoint加载模型。"""
    
    @staticmethod
    def convert_dense_to_moe(dense_model: MultimodalModel,
                              moe_config: dict) -> MultimodalModel:
        """将Dense模型的FFN层转换为MoE层(Upcycling)。
        每个FFN复制为多个expert的初始化。"""
```

---

## 四、训练管线模块（Training Pipeline）

### 4.1 模块总览

训练管线支持三阶段训练流程：**多模态预训练 → 指令微调(SFT) → 人类偏好对齐(DPO/RLHF)**，以及MoE专用训练流程。

```
training/
├── __init__.py
├── pretrain/                          # 预训练
│   ├── __init__.py
│   ├── pretrain_runner.py             # 预训练主入口
│   ├── pretrain_dataset.py            # 预训练数据集
│   ├── pretrain_config.py             # 预训练配置
│   └── curriculum_learning.py         # 课程学习策略
├── sft/                               # 指令微调
│   ├── __init__.py
│   ├── sft_runner.py                  # SFT主入口
│   ├── sft_dataset.py                 # SFT数据集(多模态对话格式)
│   ├── sft_config.py                  # SFT配置
│   └── conversation_template.py       # 对话模板管理
├── alignment/                         # 偏好对齐
│   ├── __init__.py
│   ├── dpo_trainer.py                 # DPO训练器
│   ├── rlhf_trainer.py                # RLHF训练器(PPO)
│   ├── reward_model.py                # 奖励模型
│   ├── preference_dataset.py          # 偏好数据集
│   └── alignment_config.py            # 对齐配置
├── distributed/                       # 分布式训练
│   ├── __init__.py
│   ├── deepspeed_config.py            # DeepSpeed配置生成
│   ├── megatron_config.py             # Megatron-Core配置
│   ├── parallel_strategy.py           # 并行策略选择
│   ├── expert_parallel.py             # 专家并行(MoE专用)
│   └── communication.py               # 通信优化
├── optimizer/                         # 优化器
│   ├── __init__.py
│   ├── adamw_optimizer.py             # AdamW优化器封装
│   ├── lr_scheduler.py                # 学习率调度器
│   └── gradient_utils.py              # 梯度裁剪/累积工具
├── callbacks/                         # 训练回调
│   ├── __init__.py
│   ├── checkpoint_callback.py         # Checkpoint保存
│   ├── logging_callback.py            # 日志记录(WandB/TensorBoard)
│   ├── evaluation_callback.py         # 训练中评估
│   └── early_stopping.py              # 早停
├── evaluation/                        # 模型评估
│   ├── __init__.py
│   ├── benchmark_runner.py            # 评测集运行器
│   ├── vqa_evaluator.py               # VQA评测
│   ├── caption_evaluator.py           # 图像描述评测
│   ├── mmlu_evaluator.py              # MMLU文本评测
│   └── metrics.py                     # 评测指标
└── config/
    ├── pretrain_7b_dense.yaml
    ├── pretrain_34b_moe.yaml
    ├── sft_7b.yaml
    └── deepspeed_zero3.json
```

### 4.2 预训练模块详细设计

#### 4.2.1 `pretrain_runner.py` — 预训练主入口

```python
class PretrainRunner:
    """
    多模态预训练主入口。
    阶段1: 冻结视觉编码器和LLM，仅训练投影器(模态对齐)。
    阶段2: 解冻LLM，联合训练投影器和LLM(知识注入)。
    阶段3: (可选) 解冻视觉编码器最后N层，全量微调。
    """
    def __init__(self, config: PretrainConfig)
    
    def setup_model(self) -> MultimodalModel:
        """初始化模型，加载预训练权重(视觉编码器+LLM骨干)。"""
    
    def setup_optimizer(self, model: nn.Module) -> Optimizer:
        """创建优化器。不同参数组使用不同学习率。
        投影器lr > LLM lr > 视觉编码器lr。"""
    
    def setup_lr_scheduler(self, optimizer: Optimizer,
                            num_training_steps: int) -> LRScheduler:
        """学习率调度：线性warmup + 余弦衰减。"""
    
    def setup_data(self, stage: int) -> DataLoader:
        """根据训练阶段创建数据加载器。
        阶段1: 大规模图文对(CC3M/LAION等)
        阶段2: 混合数据(图文+文本+音频+视频)"""
    
    def setup_distributed(self) -> None:
        """初始化分布式训练环境。
        DeepSpeed ZeRO-3 或 Megatron 3D并行。"""
    
    def train_stage(self, stage: int) -> Dict[str, float]:
        """执行指定阶段的训练。"""
    
    def run(self) -> None:
        """执行完整多阶段预训练。"""
    
    def save_checkpoint(self, step: int, metrics: dict) -> str:
        """保存训练checkpoint到MinIO。"""
    
    def resume_from_checkpoint(self, checkpoint_path: str) -> int:
        """从checkpoint恢复训练。"""
```

#### 4.2.2 `pretrain_dataset.py` — 预训练数据集

```python
class PretrainDataset(IterableDataset):
    """
    预训练数据集。支持多模态数据混合采样。
    数据格式: WebDataset (tar shards) 或 JSONL + 文件引用。
    使用流式加载避免内存不足。
    """
    def __init__(self, config: PretrainDataConfig, 
                 tokenizer: MultimodalTokenizer)
    
    def __iter__(self) -> Iterator[Dict[str, torch.Tensor]]:
        """流式迭代数据。每次返回一个样本dict。"""
    
    def mix_datasets(self, datasets: Dict[str, IterableDataset],
                      weights: Dict[str, float]) -> IterableDataset:
        """按权重混合多个数据集。
        如: 文本50% + 图文30% + 音频10% + 视频10%。"""
    
    def process_image_text_pair(self, sample: dict) -> dict:
        """处理图文对样本。构建"<image>\n描述这张图片"格式。"""
    
    def process_interleaved(self, sample: dict) -> dict:
        """处理图文交错数据(如MMC4格式)。"""
    
    def dynamic_padding(self, batch: List[dict]) -> dict:
        """动态padding到batch内最大长度。"""

class PretrainDataConfig:
    """预训练数据配置。"""
    data_sources: Dict[str, DataSourceConfig]  # 数据源配置
    mix_weights: Dict[str, float]              # 混合权重
    max_seq_length: int                        # 最大序列长度
    image_resolution: int                      # 图像分辨率
    num_workers: int                           # 数据加载线程数
    prefetch_factor: int                       # 预取因子
```

#### 4.2.3 `curriculum_learning.py` — 课程学习

```python
class CurriculumLearning:
    """
    课程学习策略。按难度渐进式训练。
    维度：数据质量(低→高)、序列长度(短→长)、
    模态复杂度(单模态→多模态)、图像分辨率(低→高)。
    """
    def __init__(self, config: CurriculumConfig)
    
    def get_current_curriculum(self, global_step: int) -> CurriculumState:
        """根据当前训练步数返回课程状态。"""
    
    def adjust_data_mix(self, global_step: int) -> Dict[str, float]:
        """动态调整数据混合比例。"""
    
    def adjust_sequence_length(self, global_step: int) -> int:
        """渐进式增加序列长度。"""
    
    def adjust_image_resolution(self, global_step: int) -> int:
        """渐进式增加图像分辨率。"""
```

### 4.3 指令微调(SFT)模块详细设计

#### 4.3.1 `sft_runner.py` — SFT主入口

```python
class SFTRunner:
    """
    监督指令微调(Supervised Fine-Tuning)主入口。
    使用多模态对话数据训练模型遵循指令。
    支持全参数微调和LoRA/QLoRA高效微调。
    """
    def __init__(self, config: SFTConfig)
    
    def setup_model(self) -> MultimodalModel:
        """加载预训练checkpoint，配置微调策略。"""
    
    def setup_lora(self, model: nn.Module, 
                    lora_config: LoRAConfig) -> nn.Module:
        """应用LoRA适配器。使用peft库。
        target_modules: q_proj, k_proj, v_proj, o_proj, 
        gate_proj, up_proj, down_proj。"""
    
    def setup_qlora(self, model: nn.Module,
                     qlora_config: QLoRAConfig) -> nn.Module:
        """应用QLoRA(4-bit量化+LoRA)。使用bitsandbytes。"""
    
    def compute_loss(self, logits: torch.Tensor,
                      labels: torch.Tensor) -> torch.Tensor:
        """计算交叉熵损失。仅计算assistant回复部分的loss。"""
    
    def run(self) -> None:
        """执行SFT训练。"""

class LoRAConfig:
    """LoRA配置。"""
    r: int                    # LoRA秩(8/16/32/64)
    alpha: int                # LoRA alpha
    dropout: float            # LoRA dropout
    target_modules: List[str] # 目标模块
    task_type: str            # CAUSAL_LM
```

#### 4.3.2 `sft_dataset.py` — SFT数据集

```python
class SFTDataset(Dataset):
    """
    指令微调数据集。
    数据格式: 多轮对话，每轮包含role(system/user/assistant)和content。
    content支持文本+图像+音频+视频混合。
    """
    def __init__(self, data_path: str, tokenizer: MultimodalTokenizer,
                 max_length: int, template: ConversationTemplate)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """返回单条样本的input_ids, labels, attention_mask, 
        pixel_values等。"""
    
    def apply_template(self, conversation: List[dict]) -> str:
        """应用对话模板(ChatML/Llama-3等格式)。"""
    
    def mask_instruction_tokens(self, input_ids: torch.Tensor,
                                  labels: torch.Tensor) -> torch.Tensor:
        """将指令部分(system+user)的label设为-100(不计算loss)。"""
    
    def load_and_process_media(self, media_path: str,
                                media_type: str) -> torch.Tensor:
        """加载并预处理多模态文件。"""
```

#### 4.3.3 `conversation_template.py` — 对话模板

```python
class ConversationTemplate:
    """
    对话模板管理。支持多种格式。
    """
    TEMPLATES = {
        "chatml": {
            "system": "<|im_start|>system\n{content}<|im_end|>\n",
            "user": "<|im_start|>user\n{content}<|im_end|>\n",
            "assistant": "<|im_start|>assistant\n{content}<|im_end|>\n",
        },
        "llama3": {
            "system": "<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n\n{content}<|eot_id|>",
            "user": "<|start_header_id|>user<|end_header_id|>\n\n{content}<|eot_id|>",
            "assistant": "<|start_header_id|>assistant<|end_header_id|>\n\n{content}<|eot_id|>",
        },
    }
    
    def __init__(self, template_name: str)
    
    def format_conversation(self, messages: List[dict]) -> str:
        """将消息列表格式化为模板字符串。"""
    
    def get_response_start_tokens(self) -> List[int]:
        """返回assistant回复开始的token IDs(用于loss masking)。"""
```

### 4.4 偏好对齐模块详细设计

#### 4.4.1 `dpo_trainer.py` — DPO训练器

```python
class DPOTrainer:
    """
    Direct Preference Optimization训练器。
    无需训练单独的奖励模型，直接优化策略模型。
    loss = -log(sigmoid(beta * (log_pi(chosen) - log_pi(rejected) 
           - log_ref(chosen) + log_ref(rejected))))
    """
    def __init__(self, config: DPOConfig, model: MultimodalModel,
                 ref_model: MultimodalModel)
    
    def compute_dpo_loss(self, chosen_logps: torch.Tensor,
                          rejected_logps: torch.Tensor,
                          ref_chosen_logps: torch.Tensor,
                          ref_rejected_logps: torch.Tensor) -> torch.Tensor:
        """计算DPO损失。"""
    
    def get_batch_logps(self, logits: torch.Tensor,
                         labels: torch.Tensor) -> torch.Tensor:
        """计算batch中每个样本的log概率。"""
    
    def train_step(self, batch: dict) -> Dict[str, float]:
        """单步DPO训练。"""
    
    def run(self) -> None:
        """执行DPO训练。"""

class DPOConfig:
    """DPO配置。"""
    beta: float                  # KL散度惩罚系数(0.1-0.5)
    learning_rate: float         # 学习率(通常5e-7)
    max_length: int              # 最大序列长度
    max_prompt_length: int       # 最大prompt长度
    num_epochs: int              # 训练轮数
    label_smoothing: float       # 标签平滑
    loss_type: str               # sigmoid/hinge/ipo
```

#### 4.4.2 `reward_model.py` — 奖励模型

```python
class RewardModel(nn.Module):
    """
    奖励模型。用于RLHF训练。
    基于多模态模型骨干，将LM Head替换为奖励头(输出标量)。
    """
    def __init__(self, base_model: MultimodalModel)
    
    def forward(self, input_ids: torch.Tensor,
                 attention_mask: torch.Tensor,
                 pixel_values: Optional[torch.Tensor]) -> torch.Tensor:
        """
        输出: reward_scores [B] 每个样本的奖励分数
        取最后一个非padding token的隐藏状态通过reward head。
        """
    
    def compute_reward_loss(self, chosen_rewards: torch.Tensor,
                             rejected_rewards: torch.Tensor) -> torch.Tensor:
        """Bradley-Terry偏好损失: -log(sigmoid(r_chosen - r_rejected))。"""
```

#### 4.4.3 `preference_dataset.py` — 偏好数据集

```python
class PreferenceDataset(Dataset):
    """
    偏好数据集。每条样本包含一个prompt和两个回复(chosen/rejected)。
    数据来源：人工标注偏好对、AI辅助构造偏好对。
    """
    def __init__(self, data_path: str, tokenizer: MultimodalTokenizer,
                 max_length: int)
    
    def __getitem__(self, idx: int) -> dict:
        """返回: prompt_ids, chosen_ids, rejected_ids, 
        pixel_values(如有图像)。"""
    
    def build_preference_pair(self, prompt: str, chosen: str,
                                rejected: str) -> dict:
        """构建偏好对样本。"""
```

### 4.5 分布式训练详细设计

#### 4.5.1 `parallel_strategy.py` — 并行策略

```python
class ParallelStrategy:
    """
    分布式训练并行策略选择器。
    根据模型大小和GPU资源自动推荐最优并行策略。
    
    支持的并行维度:
    - DP (Data Parallelism): 数据并行
    - TP (Tensor Parallelism): 张量并行，切分Attention和FFN
    - PP (Pipeline Parallelism): 流水线并行，切分Transformer层
    - EP (Expert Parallelism): 专家并行(MoE专用)
    - SP (Sequence Parallelism): 序列并行，切分序列维度
    - ZeRO (Zero Redundancy Optimizer): 优化器状态/梯度/参数分片
    """
    def __init__(self, num_gpus: int, gpu_memory_gb: float,
                 model_config: MultimodalModelConfig)
    
    def recommend_strategy(self) -> ParallelConfig:
        """根据模型大小和GPU资源推荐最优并行策略。
        
        推荐规则:
        - 1B模型: DP only (ZeRO-2)
        - 7B模型: DP + ZeRO-3, 或 TP=2 + DP
        - 13B模型: TP=2/4 + DP + ZeRO-3
        - 34B模型: TP=4 + PP=2 + DP, 或 TP=8 + DP
        - 72B模型: TP=8 + PP=4 + DP + ZeRO-3
        - MoE模型: 上述 + EP (专家并行)
        """
    
    def estimate_memory(self, config: ParallelConfig) -> MemoryEstimate:
        """估算每个GPU的显存占用(参数+梯度+优化器状态+激活值)。"""
    
    def validate_strategy(self, config: ParallelConfig) -> bool:
        """验证并行策略的可行性(GPU数量整除、显存够用等)。"""

@dataclass
class ParallelConfig:
    """并行配置。"""
    dp_size: int              # 数据并行度
    tp_size: int              # 张量并行度
    pp_size: int              # 流水线并行度
    ep_size: int              # 专家并行度(MoE)
    sp_enabled: bool          # 是否启用序列并行
    zero_stage: int           # ZeRO阶段(0/1/2/3)
    gradient_accumulation: int # 梯度累积步数
    micro_batch_size: int     # 微批量大小
```

#### 4.5.2 `deepspeed_config.py` — DeepSpeed配置

```python
class DeepSpeedConfigGenerator:
    """
    DeepSpeed配置文件生成器。
    根据模型规模和训练需求自动生成最优配置。
    """
    def __init__(self, model_config: MultimodalModelConfig,
                 parallel_config: ParallelConfig)
    
    def generate_zero3_config(self) -> dict:
        """生成ZeRO-3配置。
        包含: optimizer分片、gradient分片、parameter分片、
        offload策略、通信优化等。"""
    
    def generate_moe_config(self) -> dict:
        """生成MoE专用配置。
        包含: EP并行度、MoE通信方案、专家容量等。"""
    
    def generate_activation_checkpointing_config(self) -> dict:
        """生成激活检查点配置。减少显存占用。"""
    
    def generate_mixed_precision_config(self) -> dict:
        """生成混合精度配置(BF16)。"""
    
    def export_config(self, output_path: str) -> None:
        """导出为JSON配置文件。"""
```

### 4.6 评估模块详细设计

#### 4.6.1 `benchmark_runner.py` — 评测集运行器

```python
class BenchmarkRunner:
    """
    多模态模型评测运行器。
    支持的评测集:
    视觉问答: VQAv2, GQA, TextVQA, DocVQA, ChartQA
    图像描述: COCO Caption, NoCaps
    多模态推理: MMMU, MathVista, MMBench, MMStar, SEED-Bench
    视频理解: Video-MME, MVBench
    文本: MMLU, HumanEval, GSM8K
    """
    def __init__(self, model: MultimodalModel, 
                 tokenizer: MultimodalTokenizer)
    
    def run_benchmark(self, benchmark_name: str,
                       split: str) -> BenchmarkResult:
        """运行指定评测集，返回评测结果。"""
    
    def run_all_benchmarks(self, benchmarks: List[str]) -> Dict[str, BenchmarkResult]:
        """运行所有评测集。"""
    
    def generate_report(self, results: Dict[str, BenchmarkResult]) -> str:
        """生成评测报告(Markdown格式)。"""
```

---

## 五、推理引擎模块（Inference Engine）

### 5.1 模块总览

推理引擎负责模型的高效推理服务，支持vLLM和TensorRT-LLM两种后端。

```
inference/
├── __init__.py
├── engine/                            # 推理引擎
│   ├── __init__.py
│   ├── base_engine.py                 # 引擎基类
│   ├── vllm_engine.py                 # vLLM推理引擎
│   ├── trtllm_engine.py              # TensorRT-LLM推理引擎
│   └── engine_factory.py             # 引擎工厂
├── serving/                           # 推理服务
│   ├── __init__.py
│   ├── model_loader.py                # 模型加载器
│   ├── batch_processor.py             # 批处理器
│   ├── streaming_handler.py           # 流式输出处理
│   └── kv_cache_manager.py            # KV缓存管理
├── optimization/                      # 推理优化
│   ├── __init__.py
│   ├── quantization.py                # 模型量化(AWQ/GPTQ/FP8)
│   ├── speculative_decoding.py        # 推测解码
│   └── continuous_batching.py         # 连续批处理
└── config/
    ├── vllm_config.yaml
    └── trtllm_config.yaml
```

### 5.2 推理引擎详细设计

#### 5.2.1 `vllm_engine.py` — vLLM推理引擎

```python
class VLLMEngine(BaseEngine):
    """
    基于vLLM的高性能推理引擎。
    核心优化：PagedAttention、Continuous Batching、
    前缀缓存(Prefix Caching)、推测解码。
    支持多模态输入(图像/音频/视频)。
    """
    def __init__(self, config: VLLMConfig)
    
    def load_model(self, model_path: str, 
                    quantization: Optional[str]) -> None:
        """加载模型。支持AWQ/GPTQ/FP8量化模型。
        使用vLLM的AsyncLLMEngine。"""
    
    async def generate(self, prompt: str,
                        images: Optional[List[Image]],
                        audio: Optional[np.ndarray],
                        sampling_params: SamplingParams) -> GenerationResult:
        """异步生成文本。支持多模态输入。"""
    
    async def generate_stream(self, prompt: str,
                               images: Optional[List[Image]],
                               sampling_params: SamplingParams) -> AsyncIterator[str]:
        """流式生成文本。逐token返回。"""
    
    def get_model_info(self) -> ModelInfo:
        """获取加载的模型信息(参数量、显存占用等)。"""
    
    def get_engine_stats(self) -> EngineStats:
        """获取引擎统计(吞吐量、延迟、队列长度等)。"""

class VLLMConfig:
    """vLLM配置。"""
    model_path: str                    # 模型路径
    tensor_parallel_size: int          # 张量并行度
    max_model_len: int                 # 最大模型长度
    gpu_memory_utilization: float      # GPU显存利用率(0.9)
    quantization: Optional[str]        # 量化方式(awq/gptq/fp8/None)
    enable_prefix_caching: bool        # 前缀缓存
    max_num_seqs: int                  # 最大并发序列数
    dtype: str                         # 计算精度(auto/bfloat16/float16)
    trust_remote_code: bool            # 信任远程代码
    enable_chunked_prefill: bool       # 分块预填充
```

#### 5.2.2 `quantization.py` — 模型量化

```python
class ModelQuantizer:
    """
    模型量化工具。
    支持训练后量化(PTQ)和量化感知训练(QAT)。
    """
    def __init__(self, config: QuantizationConfig)
    
    def quantize_awq(self, model_path: str, 
                      calibration_data: List[str],
                      output_path: str, bits: int) -> None:
        """AWQ量化(Activation-aware Weight Quantization)。
        使用autoawq库。支持4-bit量化。"""
    
    def quantize_gptq(self, model_path: str,
                       calibration_data: List[str],
                       output_path: str, bits: int) -> None:
        """GPTQ量化。使用auto-gptq库。"""
    
    def quantize_fp8(self, model_path: str,
                      output_path: str) -> None:
        """FP8量化。适用于H100等支持FP8的GPU。"""
    
    def evaluate_quantized_model(self, original_path: str,
                                   quantized_path: str,
                                   benchmark: str) -> QuantizationReport:
        """评估量化前后模型质量变化。"""
```

---

## 六、API服务模块（API Service）

### 6.1 模块总览

对外API服务采用FastAPI框架，兼容OpenAI API格式，支持文本、视觉、音频、多模态等多种接口。

```
api/
├── __init__.py
├── app.py                             # FastAPI应用主入口
├── routes/                            # API路由
│   ├── __init__.py
│   ├── chat_completions.py            # /v1/chat/completions
│   ├── completions.py                 # /v1/completions
│   ├── embeddings.py                  # /v1/embeddings
│   ├── models.py                      # /v1/models
│   ├── audio.py                       # /v1/audio/transcriptions
│   └── health.py                      # /health
├── schemas/                           # 请求/响应模型
│   ├── __init__.py
│   ├── chat_schema.py                 # Chat API模型
│   ├── completion_schema.py           # Completion API模型
│   ├── embedding_schema.py            # Embedding API模型
│   ├── audio_schema.py                # Audio API模型
│   └── common_schema.py               # 通用模型(Usage/Error等)
├── middleware/                        # 中间件
│   ├── __init__.py
│   ├── auth_middleware.py             # API Key认证
│   ├── rate_limiter.py                # 速率限制
│   ├── request_logger.py             # 请求日志
│   └── cors_middleware.py             # CORS配置
├── services/                          # 业务逻辑
│   ├── __init__.py
│   ├── chat_service.py                # 聊天服务
│   ├── multimodal_service.py          # 多模态处理服务
│   ├── embedding_service.py           # 嵌入向量服务
│   ├── audio_service.py               # 音频处理服务
│   └── model_manager.py               # 模型管理服务
├── utils/                             # 工具
│   ├── __init__.py
│   ├── token_counter.py               # Token计数
│   ├── image_utils.py                 # 图像处理(Base64/URL)
│   └── audio_utils.py                 # 音频处理
└── config/
    ├── api_config.yaml
    └── model_registry.yaml
```

### 6.2 API路由详细设计

#### 6.2.1 `chat_completions.py` — Chat API (兼容OpenAI格式)

```python
router = APIRouter()

@router.post("/v1/chat/completions")
async def create_chat_completion(
    request: ChatCompletionRequest,
    api_key: str = Depends(verify_api_key),
    rate_limiter: RateLimiter = Depends(get_rate_limiter)
) -> Union[ChatCompletionResponse, StreamingResponse]:
    """
    多模态聊天补全API。兼容OpenAI Chat Completions API格式。
    支持文本、图像(Base64/URL)、音频、视频输入。
    支持流式和非流式输出。
    
    请求示例:
    {
        "model": "multimodal-7b",
        "messages": [
            {"role": "system", "content": "你是一个有帮助的助手。"},
            {"role": "user", "content": [
                {"type": "text", "text": "描述这张图片"},
                {"type": "image_url", "image_url": {"url": "data:image/png;base64,..."}}
            ]}
        ],
        "max_tokens": 1024,
        "temperature": 0.7,
        "stream": true
    }
    """
```

#### 6.2.2 `audio.py` — Audio API

```python
router = APIRouter()

@router.post("/v1/audio/transcriptions")
async def create_transcription(
    file: UploadFile,
    model: str = Form("whisper-large-v3"),
    language: Optional[str] = Form(None),
    api_key: str = Depends(verify_api_key)
) -> TranscriptionResponse:
    """
    音频转文字API。
    支持音频文件上传，返回转录文本。
    """

@router.post("/v1/audio/chat")
async def audio_chat(
    request: AudioChatRequest,
    api_key: str = Depends(verify_api_key)
) -> ChatCompletionResponse:
    """
    音频对话API。
    接收音频输入，返回文本回复。
    自动进行语音识别后送入多模态模型。
    """
```

### 6.3 请求/响应模型详细设计

#### 6.3.1 `chat_schema.py` — Chat API数据模型

```python
class ChatCompletionRequest(BaseModel):
    """聊天补全请求模型。兼容OpenAI格式。"""
    model: str                                      # 模型名称
    messages: List[ChatMessage]                      # 消息列表
    max_tokens: Optional[int] = 2048                 # 最大生成token数
    temperature: Optional[float] = 0.7               # 温度
    top_p: Optional[float] = 0.9                     # 核采样
    top_k: Optional[int] = 50                        # Top-K采样
    frequency_penalty: Optional[float] = 0.0         # 频率惩罚
    presence_penalty: Optional[float] = 0.0          # 存在惩罚
    repetition_penalty: Optional[float] = 1.0        # 重复惩罚
    stop: Optional[List[str]] = None                 # 停止词
    stream: Optional[bool] = False                   # 是否流式
    seed: Optional[int] = None                       # 随机种子
    n: Optional[int] = 1                             # 生成数量

class ChatMessage(BaseModel):
    """聊天消息。"""
    role: Literal["system", "user", "assistant"]
    content: Union[str, List[ContentPart]]           # 纯文本或多模态内容

class ContentPart(BaseModel):
    """多模态内容块。"""
    type: Literal["text", "image_url", "audio_url", "video_url"]
    text: Optional[str] = None
    image_url: Optional[ImageURL] = None
    audio_url: Optional[AudioURL] = None
    video_url: Optional[VideoURL] = None

class ImageURL(BaseModel):
    """图像URL。"""
    url: str                          # Base64或HTTP URL
    detail: Optional[str] = "auto"    # auto/low/high

class ChatCompletionResponse(BaseModel):
    """聊天补全响应。"""
    id: str                                          # 唯一请求ID
    object: str = "chat.completion"
    created: int                                     # 创建时间戳
    model: str                                       # 使用的模型
    choices: List[ChatChoice]                         # 选择列表
    usage: UsageInfo                                  # Token用量

class ChatChoice(BaseModel):
    """单个选择。"""
    index: int
    message: ChatMessage
    finish_reason: Optional[str]                      # stop/length/tool_calls

class UsageInfo(BaseModel):
    """Token用量信息。"""
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int
```

### 6.4 中间件详细设计

#### 6.4.1 `auth_middleware.py` — API Key认证

```python
class APIKeyAuth:
    """
    API Key认证中间件。
    支持Bearer Token和自定义Header两种方式。
    API Key存储在PostgreSQL中，支持过期时间和权限控制。
    """
    def __init__(self, db: MetadataStore, redis: Redis)
    
    async def verify_api_key(self, api_key: str) -> APIKeyInfo:
        """验证API Key有效性。先查Redis缓存，未命中查数据库。"""
    
    async def create_api_key(self, user_id: str, 
                              permissions: List[str],
                              expires_at: Optional[datetime]) -> str:
        """创建新的API Key。"""
    
    async def revoke_api_key(self, api_key: str) -> None:
        """撤销API Key。"""
    
    async def get_usage(self, api_key: str) -> APIKeyUsage:
        """获取API Key的使用统计。"""
```

#### 6.4.2 `rate_limiter.py` — 速率限制

```python
class RateLimiter:
    """
    API速率限制器。基于Redis滑动窗口算法。
    支持按API Key、按IP、按模型进行限速。
    """
    def __init__(self, redis: Redis, config: RateLimitConfig)
    
    async def check_rate_limit(self, key: str, 
                                 limit: int, window: int) -> bool:
        """检查是否超过速率限制。使用Redis ZRANGEBYSCORE。"""
    
    async def record_request(self, key: str) -> None:
        """记录一次请求。"""
    
    def get_limit_headers(self, key: str) -> Dict[str, str]:
        """返回速率限制相关的HTTP响应头。
        X-RateLimit-Limit, X-RateLimit-Remaining, X-RateLimit-Reset。"""
```

### 6.5 服务层详细设计

#### 6.5.1 `chat_service.py` — 聊天服务

```python
class ChatService:
    """
    聊天业务逻辑服务。
    处理请求解析、多模态内容处理、模型调用、响应构建。
    """
    def __init__(self, engine: BaseEngine, 
                 tokenizer: MultimodalTokenizer,
                 model_manager: ModelManager)
    
    async def process_chat(self, request: ChatCompletionRequest) -> ChatCompletionResponse:
        """处理非流式聊天请求。"""
    
    async def process_chat_stream(self, request: ChatCompletionRequest) -> AsyncIterator[str]:
        """处理流式聊天请求。返回SSE格式的chunk流。"""
    
    async def _extract_multimodal_content(self, messages: List[ChatMessage]) -> ProcessedInput:
        """从消息中提取并预处理多模态内容。
        下载URL图像、解码Base64、处理音频等。"""
    
    def _build_prompt(self, messages: List[ChatMessage],
                       template: ConversationTemplate) -> str:
        """使用对话模板构建prompt。"""
    
    def _count_tokens(self, text: str) -> int:
        """计算token数量。"""
    
    def _build_response(self, generated_text: str, 
                          usage: UsageInfo,
                          request_id: str) -> ChatCompletionResponse:
        """构建API响应。"""
```

#### 6.5.2 `model_manager.py` — 模型管理服务

```python
class ModelManager:
    """
    模型管理服务。
    管理多个模型的加载、卸载、版本切换。
    支持A/B测试、灰度发布。
    """
    def __init__(self, config: ModelManagerConfig, 
                 storage: MinIOClient)
    
    def load_model(self, model_name: str, 
                    version: str) -> BaseEngine:
        """加载指定版本的模型到推理引擎。"""
    
    def unload_model(self, model_name: str) -> None:
        """卸载模型释放GPU资源。"""
    
    def list_models(self) -> List[ModelInfo]:
        """列出所有可用模型。"""
    
    def get_model_engine(self, model_name: str) -> BaseEngine:
        """获取模型对应的推理引擎实例。"""
    
    def switch_model_version(self, model_name: str, 
                               new_version: str) -> None:
        """切换模型版本(热更新)。"""
    
    def register_model(self, model_info: ModelInfo,
                        model_path: str) -> None:
        """注册新模型版本到模型仓库。"""
```

### 6.6 FastAPI应用主入口

#### 6.6.1 `app.py` — 应用入口

```python
def create_app(config: APIConfig) -> FastAPI:
    """
    创建FastAPI应用实例。
    """
    app = FastAPI(
        title="Multimodal LLM API",
        version="1.0.0",
        description="多模态大语言模型推理服务API"
    )
    
    # 注册中间件
    app.add_middleware(CORSMiddleware, ...)
    app.add_middleware(RequestLoggerMiddleware)
    
    # 注册路由
    app.include_router(chat_router, prefix="/v1")
    app.include_router(completion_router, prefix="/v1")
    app.include_router(embedding_router, prefix="/v1")
    app.include_router(audio_router, prefix="/v1")
    app.include_router(model_router, prefix="/v1")
    app.include_router(health_router)
    
    # 启动事件: 加载模型
    @app.on_event("startup")
    async def startup():
        model_manager.load_default_models()
    
    # 关闭事件: 清理资源
    @app.on_event("shutdown")
    async def shutdown():
        model_manager.unload_all_models()
    
    return app
```

---

## 七、部署与运维模块（Deployment & Operations）

### 7.1 模块总览

```
deployment/
├── docker/                            # Docker配置
│   ├── Dockerfile.training            # 训练镜像
│   ├── Dockerfile.inference           # 推理镜像
│   ├── Dockerfile.api                 # API服务镜像
│   └── docker-compose.yaml            # 本地开发编排
├── kubernetes/                        # Kubernetes部署
│   ├── namespace.yaml
│   ├── inference/
│   │   ├── deployment.yaml            # 推理服务部署
│   │   ├── service.yaml               # Service
│   │   ├── hpa.yaml                   # 自动扩缩容
│   │   └── gpu-resource-quota.yaml    # GPU资源配额
│   ├── api/
│   │   ├── deployment.yaml            # API网关部署
│   │   ├── service.yaml
│   │   └── ingress.yaml               # Ingress配置
│   └── monitoring/
│       ├── prometheus.yaml            # Prometheus
│       └── grafana.yaml               # Grafana
├── monitoring/                        # 监控配置
│   ├── __init__.py
│   ├── metrics_collector.py           # 指标采集器
│   ├── alert_rules.py                 # 告警规则
│   ├── dashboards/
│   │   ├── inference_dashboard.json   # 推理监控面板
│   │   └── training_dashboard.json    # 训练监控面板
│   └── logging_config.py             # 日志配置
└── scripts/
    ├── setup_environment.sh           # 环境初始化
    ├── start_training.sh              # 启动训练
    ├── start_inference.sh             # 启动推理
    └── deploy_api.sh                  # 部署API
```

### 7.2 监控系统详细设计

#### 7.2.1 `metrics_collector.py` — 指标采集

```python
class MetricsCollector:
    """
    Prometheus指标采集器。
    采集推理、训练、系统三个维度的指标。
    """
    def __init__(self):
        # 推理指标
        self.request_count = Counter("api_request_total", 
                                      "Total API requests",
                                      ["method", "endpoint", "status"])
        self.request_latency = Histogram("api_request_duration_seconds",
                                          "Request latency",
                                          ["method", "endpoint"])
        self.tokens_per_second = Gauge("inference_tokens_per_second",
                                        "Tokens generated per second")
        self.batch_size = Histogram("inference_batch_size",
                                     "Inference batch size")
        self.queue_length = Gauge("inference_queue_length",
                                   "Pending requests in queue")
        
        # GPU指标
        self.gpu_utilization = Gauge("gpu_utilization_percent",
                                      "GPU utilization", ["gpu_id"])
        self.gpu_memory_used = Gauge("gpu_memory_used_bytes",
                                      "GPU memory used", ["gpu_id"])
        self.gpu_temperature = Gauge("gpu_temperature_celsius",
                                      "GPU temperature", ["gpu_id"])
        
        # 训练指标
        self.training_loss = Gauge("training_loss", "Training loss")
        self.training_step = Gauge("training_global_step", "Global step")
        self.learning_rate = Gauge("training_learning_rate", "Learning rate")
    
    def record_request(self, method: str, endpoint: str, 
                        status: int, duration: float) -> None:
        """记录API请求指标。"""
    
    def record_inference(self, tokens_generated: int, 
                          latency: float, batch_size: int) -> None:
        """记录推理指标。"""
    
    def collect_gpu_metrics(self) -> None:
        """采集GPU指标。使用pynvml库。"""
```

### 7.3 Kubernetes部署配置

#### 推理服务部署 (deployment.yaml 核心配置)

```yaml
# 推理服务部署配置要点:
# - 使用GPU节点(nvidia.com/gpu资源)
# - 配置健康检查(liveness/readiness probe)
# - 设置资源限制(GPU数量、内存)
# - 使用PVC挂载模型权重
# - HPA基于GPU利用率和队列长度自动扩缩容
#
# 不同模型规模的资源需求:
# 1B模型:  1x A100-40G (或 2x A10)
# 7B模型:  1x A100-80G (FP16) 或 1x A100-40G (INT4)
# 13B模型: 2x A100-80G (FP16) 或 1x A100-80G (INT4)
# 34B模型: 4x A100-80G (FP16) 或 2x A100-80G (INT4)
# 72B模型: 8x A100-80G (FP16) 或 4x A100-80G (INT4)
# MoE模型: 按激活参数量参考上述Dense模型
```

---

## 八、完整数据流程

### 8.1 训练阶段数据流

```
原始数据源                     数据采集                  数据清洗
┌──────────┐              ┌──────────┐            ┌──────────┐
│Common    │──┐           │Text      │──┐         │Text      │
│Crawl     │  │           │Collector │  │         │Cleaner   │
├──────────┤  │           ├──────────┤  │         ├──────────┤
│LAION-5B  │──┼──采集───→ │ImageText │──┼──清洗──→│Image     │
├──────────┤  │           │Collector │  │         │Cleaner   │
│LibriSpeech│──┤           ├──────────┤  │         ├──────────┤
├──────────┤  │           │Audio     │──┤         │Audio     │
│WebVid    │──┘           │Collector │──┘         │Cleaner   │
└──────────┘              └──────────┘            └──────────┘
                                                       │
                          ┌──────────┐                 │
                          │打标模块   │                 │
                          │Caption   │←────打标────────┘
                          │QA       │                  │
                          │Safety   │                  │
                          └──────────┘                 │
                               │                       │
                               ▼                       ▼
                          ┌──────────┐          ┌──────────┐
                          │MinIO     │          │PostgreSQL│
                          │(对象存储) │          │(元数据)   │
                          └──────────┘          └──────────┘
                               │
                               ▼
            ┌──────────────────────────────────────┐
            │         训练管线 (Training Pipeline)    │
            │                                        │
            │  阶段1: 模态对齐预训练                    │
            │    冻结LLM+视觉编码器, 训练投影器          │
            │    数据: 大规模图文对                      │
            │                                        │
            │  阶段2: 联合预训练                        │
            │    解冻LLM, 训练投影器+LLM               │
            │    数据: 混合多模态数据                    │
            │                                        │
            │  阶段3: 指令微调(SFT)                    │
            │    全参数/LoRA微调                       │
            │    数据: 多模态对话数据                    │
            │                                        │
            │  阶段4: 偏好对齐(DPO/RLHF)              │
            │    优化人类偏好                           │
            │    数据: 偏好对(chosen/rejected)          │
            └──────────────────────────────────────┘
                               │
                               ▼
                          ┌──────────┐
                          │模型仓库   │
                          │(MinIO)   │
                          └──────────┘
```

### 8.2 推理阶段数据流

```
用户请求                API服务               推理引擎              模型
┌────────┐         ┌──────────┐         ┌──────────┐       ┌──────────┐
│HTTP    │──请求──→│FastAPI   │──解析──→│vLLM      │──推理→│MLLM      │
│Request │         │中间件链   │         │引擎      │       │(GPU)     │
│(多模态) │         │Auth/Rate │         │PagedAttn │       │          │
│        │←─响应──│/Log/CORS │←─生成──│Cont.Batch│←─输出│          │
└────────┘         └──────────┘         └──────────┘       └──────────┘
                        │                     │
                        ▼                     ▼
                   ┌──────────┐         ┌──────────┐
                   │Redis     │         │Prometheus│
                   │(缓存/限流)│         │(监控指标) │
                   └──────────┘         └──────────┘
```

---

## 九、关键技术决策总结

### 9.1 模型架构选择依据

| 决策项 | 选择 | 依据 |
|--------|------|------|
| 视觉编码器 | EVA-CLIP ViT-G/14 + SigLIP | EVA-CLIP在大分辨率上表现优异；SigLIP适合轻量模型，Sigmoid Loss避免全局对比学习的大batch需求 |
| 投影器 | MLP (默认) / Resampler (高压缩) | MLP简单高效，信息保留度高；Resampler适合需要压缩视觉token数量的场景 |
| LLM激活函数 | SwiGLU | 相比GELU/ReLU，SwiGLU在同等参数量下性能更优(PaLM验证) |
| 注意力机制 | GQA + FlashAttention-2 | GQA减少KV缓存显存；FlashAttention-2减少HBM访问 |
| 位置编码 | RoPE | 支持长度外推，广泛验证 |
| MoE路由 | Top-K + 共享专家 | Top-K路由成熟稳定；共享专家确保基础能力(DeepSeek-V2验证) |

### 9.2 训练策略选择依据

| 决策项 | 选择 | 依据 |
|--------|------|------|
| 分阶段训练 | 3阶段(对齐→预训练→SFT) | 先对齐模态避免视觉信号破坏LLM能力，业界标准做法(LLaVA/InternVL) |
| 分布式策略 | DeepSpeed ZeRO-3 + Megatron TP/PP | ZeRO-3通用性强；超大模型需3D并行 |
| MoE训练 | DeepSpeed-MoE + Expert Parallelism | 成熟的MoE训练框架，支持EP通信优化 |
| 偏好对齐 | DPO优先于PPO | DPO无需单独的奖励模型，更稳定，资源消耗更低 |
| 高效微调 | LoRA (r=64, alpha=128) | 在保持性能的同时减少90%+可训练参数 |

### 9.3 数据管线选择依据

| 决策项 | 选择 | 依据 |
|--------|------|------|
| 文本清洗 | data-juicer + 自定义规则 | data-juicer提供100+算子，覆盖全面 |
| 文本去重 | MinHash LSH | 十亿级文档去重的工业标准方案(datasketch库) |
| 图像描述 | 多模型投票(InternVL2+CogVLM2) | 多模型降低单模型偏差，提升标注质量 |
| 质量评分 | fineweb-edu风格分类器 | 教育价值评分在预训练数据筛选中效果显著(HuggingFace验证) |
| PII检测 | Microsoft Presidio + 自定义规则 | 工业级PII检测框架，支持中英文 |
| 人工标注 | Label Studio | 开源最佳标注平台，支持多模态 |

### 9.4 推理与部署选择依据

| 决策项 | 选择 | 依据 |
|--------|------|------|
| 推理引擎 | vLLM | PagedAttention极大提升显存利用率；Continuous Batching提升吞吐 |
| 量化方式 | AWQ 4-bit | 精度损失最小的4-bit量化方案 |
| API框架 | FastAPI | 异步高性能，自动OpenAPI文档，生态丰富 |
| API格式 | 兼容OpenAI格式 | 行业标准，便于迁移和集成 |
| 存储 | MinIO | S3兼容的高性能对象存储，适合大文件 |

---

## 十、项目依赖清单

### 10.1 核心依赖

```
# 深度学习
torch>=2.3.0
transformers>=4.45.0
accelerate>=0.34.0
deepspeed>=0.15.0
flash-attn>=2.6.0
peft>=0.13.0
bitsandbytes>=0.44.0
tokenizers>=0.20.0
sentencepiece>=0.2.0

# 推理
vllm>=0.6.0

# 数据处理
data-juicer>=0.2.0
datasketch>=1.6.0
trafilatura>=1.12.0
img2dataset>=1.45.0
warcio>=1.7.4
mwparserfromhell>=0.6.6
Pillow>=10.4.0
opencv-python>=4.10.0
librosa>=0.10.2
soundfile>=0.12.1
ffmpeg-python>=0.2.0

# API服务
fastapi>=0.115.0
uvicorn>=0.32.0
pydantic>=2.9.0
python-multipart>=0.0.12
aiohttp>=3.10.0

# 存储
minio>=7.2.0
psycopg2-binary>=2.9.9
redis>=5.2.0
sqlalchemy>=2.0.35

# 标注
label-studio-sdk>=1.0.0
presidio-analyzer>=2.2.0
presidio-anonymizer>=2.2.0

# 监控
prometheus-client>=0.21.0

# 工具
pynvml>=11.5.0
wandb>=0.18.0
tqdm>=4.66.0
pyyaml>=6.0.2
```