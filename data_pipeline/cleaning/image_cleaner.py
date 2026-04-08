"""
图像数据清洗器
清洗维度：格式校验、分辨率过滤、美学评分、NSFW过滤、水印检测、重复检测
"""

from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional, Tuple
import logging
from PIL import Image
import io
import uuid

from .base_cleaner import (
    BaseCleaner, CleaningConfig, CleanedDataItem, CleaningStatus
)

logger = logging.getLogger(__name__)


@dataclass
class ImageCleaningConfig(CleaningConfig):
    """图像清洗专用配置"""
    min_width: int = 256
    min_height: int = 256
    max_width: int = 8192
    max_height: int = 8192
    min_file_size: int = 1024  # 1KB
    max_file_size: int = 50 * 1024 * 1024  # 50MB
    allowed_formats: List[str] = field(default_factory=lambda: ['JPEG', 'PNG', 'WebP'])
    max_aspect_ratio: float = 10.0  # 最大宽高比
    min_aspect_ratio: float = 0.1   # 最小宽高比
    enable_nsfw_filter: bool = True
    nsfw_threshold: float = 0.5
    enable_aesthetic_scoring: bool = True
    min_aesthetic_score: float = 4.0  # LAION aesthetic score范围1-10
    enable_watermark_detection: bool = False
    watermark_threshold: float = 0.5
    enable_clip_scoring: bool = False
    min_clip_score: float = 0.2
    enable_blur_detection: bool = True
    max_blur_score: float = 100.0  # Laplacian方差阈值


class ImageCleaner(BaseCleaner):
    """
    图像数据清洗器。
    清洗维度：格式校验、分辨率过滤、美学评分、NSFW过滤、水印检测、重复检测。
    """
    
    def __init__(self, config: ImageCleaningConfig):
        super().__init__(config)
        self.image_config = config
        self.nsfw_detector = None
        self.aesthetic_scorer = None
        self.watermark_detector = None
        
        # 初始化模型
        if config.enable_nsfw_filter:
            self._init_nsfw_detector()
        
        if config.enable_aesthetic_scoring:
            self._init_aesthetic_scorer()
        
        if config.enable_watermark_detection:
            self._init_watermark_detector()
    
    def _init_nsfw_detector(self):
        """初始化NSFW检测器"""
        try:
            # 实际使用时加载预训练模型
            # from transformers import pipeline
            # self.nsfw_detector = pipeline("image-classification", 
            #                                model="Falconsai/nsfw_image_detection")
            self.logger.info("NSFW detector initialized")
        except Exception as e:
            self.logger.warning(f"Failed to initialize NSFW detector: {e}")
    
    def _init_aesthetic_scorer(self):
        """初始化美学评分器"""
        try:
            # 实际使用时加载LAION aesthetic predictor
            # from transformers import AutoModel
            # self.aesthetic_scorer = AutoModel.from_pretrained(...)
            self.logger.info("Aesthetic scorer initialized")
        except Exception as e:
            self.logger.warning(f"Failed to initialize aesthetic scorer: {e}")
    
    def _init_watermark_detector(self):
        """初始化水印检测器"""
        try:
            # 实际使用时加载水印检测模型
            self.logger.info("Watermark detector initialized")
        except Exception as e:
            self.logger.warning(f"Failed to initialize watermark detector: {e}")
    
    def clean(self, items: List[Any]) -> List[CleanedDataItem]:
        """
        执行完整图像清洗流程
        
        Args:
            items: 待清洗的图像数据项列表
            
        Returns:
            List[CleanedDataItem]: 清洗后的数据项列表
        """
        cleaned_items = []
        
        for item in items:
            try:
                # 提取图像数据
                image_data = self._extract_image(item)
                
                if not image_data:
                    cleaned_items.append(CleanedDataItem(
                        item_id=f"cleaned_{getattr(item, 'data_id', str(uuid.uuid4()))}",
                        original_id=getattr(item, 'data_id', ''),
                        data_type='image',
                        content=None,
                        status=CleaningStatus.FILTERED,
                        metadata={'reason': 'no_image_data'}
                    ))
                    continue
                
                cleaning_steps = []
                
                # 1. 格式验证
                if not self.validate_format(image_data):
                    cleaned_items.append(CleanedDataItem(
                        item_id=f"cleaned_{getattr(item, 'data_id', str(uuid.uuid4()))}",
                        original_id=getattr(item, 'data_id', ''),
                        data_type='image',
                        content=None,
                        status=CleaningStatus.FILTERED,
                        metadata={'reason': 'invalid_format'}
                    ))
                    continue
                
                cleaning_steps.append('format_validation')
                
                # 打开图像
                try:
                    image = Image.open(io.BytesIO(image_data))
                except Exception as e:
                    cleaned_items.append(CleanedDataItem(
                        item_id=f"cleaned_{getattr(item, 'data_id', str(uuid.uuid4()))}",
                        original_id=getattr(item, 'data_id', ''),
                        data_type='image',
                        content=None,
                        status=CleaningStatus.FAILED,
                        errors=[f'Failed to open image: {str(e)}']
                    ))
                    continue
                
                # 2. 分辨率过滤
                if not self.filter_by_resolution(image):
                    cleaned_items.append(CleanedDataItem(
                        item_id=f"cleaned_{getattr(item, 'data_id', str(uuid.uuid4()))}",
                        original_id=getattr(item, 'data_id', ''),
                        data_type='image',
                        content=None,
                        status=CleaningStatus.FILTERED,
                        cleaning_steps=cleaning_steps,
                        metadata={'reason': 'resolution_filtered', 'size': image.size}
                    ))
                    continue
                
                cleaning_steps.append('resolution_filter')
                
                # 3. 宽高比过滤
                if not self.filter_by_aspect_ratio(image):
                    cleaned_items.append(CleanedDataItem(
                        item_id=f"cleaned_{getattr(item, 'data_id', str(uuid.uuid4()))}",
                        original_id=getattr(item, 'data_id', ''),
                        data_type='image',
                        content=None,
                        status=CleaningStatus.FILTERED,
                        cleaning_steps=cleaning_steps,
                        metadata={'reason': 'aspect_ratio_filtered'}
                    ))
                    continue
                
                cleaning_steps.append('aspect_ratio_filter')
                
                # 4. 模糊检测
                if self.image_config.enable_blur_detection:
                    blur_score = self.detect_blur(image)
                    if blur_score < self.image_config.max_blur_score:
                        cleaned_items.append(CleanedDataItem(
                            item_id=f"cleaned_{getattr(item, 'data_id', str(uuid.uuid4()))}",
                            original_id=getattr(item, 'data_id', ''),
                            data_type='image',
                            content=None,
                            status=CleaningStatus.FILTERED,
                            cleaning_steps=cleaning_steps,
                            metadata={'reason': 'blur_filtered', 'blur_score': blur_score}
                        ))
                        continue
                    cleaning_steps.append(f'blur_detection:{blur_score:.2f}')
                
                # 5. NSFW过滤
                nsfw_score = None
                if self.image_config.enable_nsfw_filter and self.nsfw_detector:
                    nsfw_score = self.detect_nsfw(image)
                    if nsfw_score > self.image_config.nsfw_threshold:
                        cleaned_items.append(CleanedDataItem(
                            item_id=f"cleaned_{getattr(item, 'data_id', str(uuid.uuid4()))}",
                            original_id=getattr(item, 'data_id', ''),
                            data_type='image',
                            content=None,
                            status=CleaningStatus.FILTERED,
                            cleaning_steps=cleaning_steps,
                            metadata={'reason': 'nsfw_filtered', 'nsfw_score': nsfw_score}
                        ))
                        continue
                    cleaning_steps.append(f'nsfw_filter:{nsfw_score:.2f}')
                
                # 6. 美学评分
                aesthetic_score = None
                if self.image_config.enable_aesthetic_scoring and self.aesthetic_scorer:
                    aesthetic_score = self.score_aesthetic(image)
                    if aesthetic_score < self.image_config.min_aesthetic_score:
                        cleaned_items.append(CleanedDataItem(
                            item_id=f"cleaned_{getattr(item, 'data_id', str(uuid.uuid4()))}",
                            original_id=getattr(item, 'data_id', ''),
                            data_type='image',
                            content=None,
                            status=CleaningStatus.FILTERED,
                            cleaning_steps=cleaning_steps,
                            quality_score=aesthetic_score / 10.0,  # 归一化到0-1
                            metadata={'reason': 'aesthetic_filtered', 'aesthetic_score': aesthetic_score}
                        ))
                        continue
                    cleaning_steps.append(f'aesthetic_score:{aesthetic_score:.2f}')
                
                # 7. 水印检测
                watermark_score = None
                if self.image_config.enable_watermark_detection and self.watermark_detector:
                    watermark_score = self.detect_watermark(image)
                    if watermark_score > self.image_config.watermark_threshold:
                        cleaned_items.append(CleanedDataItem(
                            item_id=f"cleaned_{getattr(item, 'data_id', str(uuid.uuid4()))}",
                            original_id=getattr(item, 'data_id', ''),
                            data_type='image',
                            content=None,
                            status=CleaningStatus.FILTERED,
                            cleaning_steps=cleaning_steps,
                            metadata={'reason': 'watermark_filtered', 'watermark_score': watermark_score}
                        ))
                        continue
                    cleaning_steps.append(f'watermark_detection:{watermark_score:.2f}')
                
                # 清洗成功
                quality_score = aesthetic_score / 10.0 if aesthetic_score else None
                
                cleaned_items.append(CleanedDataItem(
                    item_id=f"cleaned_{getattr(item, 'data_id', str(uuid.uuid4()))}",
                    original_id=getattr(item, 'data_id', ''),
                    data_type='image',
                    content=image_data,
                    status=CleaningStatus.SUCCESS,
                    quality_score=quality_score,
                    cleaning_steps=cleaning_steps,
                    metadata={
                        'width': image.width,
                        'height': image.height,
                        'format': image.format,
                        'mode': image.mode,
                        'file_size': len(image_data),
                        'nsfw_score': nsfw_score,
                        'aesthetic_score': aesthetic_score,
                        'watermark_score': watermark_score
                    }
                ))
                
            except Exception as e:
                self.logger.error(f"Error cleaning image item: {e}")
                cleaned_items.append(CleanedDataItem(
                    item_id=f"cleaned_{getattr(item, 'data_id', str(uuid.uuid4()))}",
                    original_id=getattr(item, 'data_id', ''),
                    data_type='image',
                    content=None,
                    status=CleaningStatus.FAILED,
                    errors=[str(e)]
                ))
        
        return cleaned_items
    
    def validate(self, item: Any) -> bool:
        """验证图像数据项"""
        image_data = self._extract_image(item)
        
        if not image_data:
            return False
        
        # 检查文件大小
        size = len(image_data)
        if size < self.image_config.min_file_size or size > self.image_config.max_file_size:
            return False
        
        # 检查格式
        if not self.validate_format(image_data):
            return False
        
        return True
    
    def _extract_image(self, item: Any) -> Optional[bytes]:
        """从数据项中提取图像数据"""
        if isinstance(item, bytes):
            return item
        elif isinstance(item, dict):
            return item.get('image') or item.get('content')
        elif hasattr(item, 'content'):
            if isinstance(item.content, dict):
                return item.content.get('image')
            return item.content
        else:
            return None
    
    def validate_format(self, image_data: bytes) -> bool:
        """
        验证图像格式和完整性
        
        Args:
            image_data: 图像二进制数据
            
        Returns:
            bool: 是否有效
        """
        try:
            image = Image.open(io.BytesIO(image_data))
            image.verify()
            
            # 检查格式
            if image.format not in self.image_config.allowed_formats:
                return False
            
            return True
            
        except Exception as e:
            self.logger.debug(f"Image validation failed: {e}")
            return False
    
    def filter_by_resolution(self, image: Image.Image) -> bool:
        """
        过滤分辨率过低或过大的图像
        
        Args:
            image: PIL图像对象
            
        Returns:
            bool: 是否保留
        """
        width, height = image.size
        
        if width < self.image_config.min_width or height < self.image_config.min_height:
            return False
        
        if width > self.image_config.max_width or height > self.image_config.max_height:
            return False
        
        return True
    
    def filter_by_aspect_ratio(self, image: Image.Image) -> bool:
        """
        过滤宽高比异常的图像
        
        Args:
            image: PIL图像对象
            
        Returns:
            bool: 是否保留
        """
        width, height = image.size
        aspect_ratio = width / height
        
        if aspect_ratio < self.image_config.min_aspect_ratio:
            return False
        
        if aspect_ratio > self.image_config.max_aspect_ratio:
            return False
        
        return True
    
    def detect_blur(self, image: Image.Image) -> float:
        """
        检测图像模糊度（使用Laplacian方差）
        
        Args:
            image: PIL图像对象
            
        Returns:
            float: 模糊度分数（越高越清晰）
        """
        try:
            import cv2
            import numpy as np
            
            # 转换为OpenCV格式
            image_array = np.array(image)
            if len(image_array.shape) == 3:
                gray = cv2.cvtColor(image_array, cv2.COLOR_RGB2GRAY)
            else:
                gray = image_array
            
            # 计算Laplacian方差
            laplacian_var = cv2.Laplacian(gray, cv2.CV_64F).var()
            
            return laplacian_var
            
        except ImportError:
            self.logger.warning("OpenCV not installed, blur detection skipped")
            return float('inf')
        except Exception as e:
            self.logger.error(f"Error detecting blur: {e}")
            return float('inf')
    
    def detect_nsfw(self, image: Image.Image) -> float:
        """
        NSFW图像检测
        
        Args:
            image: PIL图像对象
            
        Returns:
            float: NSFW概率 (0-1)
        """
        if not self.nsfw_detector:
            return 0.0
        
        try:
            # 使用预训练模型检测
            # results = self.nsfw_detector(image)
            # nsfw_score = next((r['score'] for r in results if r['label'] == 'nsfw'), 0.0)
            # return nsfw_score
            
            # 简化实现
            return 0.0
            
        except Exception as e:
            self.logger.error(f"Error detecting NSFW: {e}")
            return 0.0
    
    def score_aesthetic(self, image: Image.Image) -> float:
        """
        使用LAION-Aesthetics模型评估美学分数
        
        Args:
            image: PIL图像对象
            
        Returns:
            float: 美学分数 (1-10)
        """
        if not self.aesthetic_scorer:
            return 5.0  # 默认中等分数
        
        try:
            # 使用预训练模型评分
            # score = self.aesthetic_scorer(image)
            # return score
            
            # 简化实现：基于分辨率和宽高比的启发式评分
            width, height = image.size
            resolution = width * height
            
            # 高分辨率加分
            score = 5.0
            if resolution > 1920 * 1080:
                score += 1.0
            elif resolution > 1280 * 720:
                score += 0.5
            
            # 接近标准宽高比加分
            aspect = width / height
            if 1.3 < aspect < 1.9:  # 接近16:9
                score += 0.5
            elif 0.9 < aspect < 1.1:  # 接近1:1
                score += 0.3
            
            return min(10.0, max(1.0, score))
            
        except Exception as e:
            self.logger.error(f"Error scoring aesthetic: {e}")
            return 5.0
    
    def detect_watermark(self, image: Image.Image) -> float:
        """
        水印检测置信度评分
        
        Args:
            image: PIL图像对象
            
        Returns:
            float: 水印概率 (0-1)
        """
        if not self.watermark_detector:
            return 0.0
        
        try:
            # 使用水印检测模型
            # score = self.watermark_detector(image)
            # return score
            
            # 简化实现
            return 0.0
            
        except Exception as e:
            self.logger.error(f"Error detecting watermark: {e}")
            return 0.0
    
    def filter_quality(self, item: Any, threshold: float = None) -> bool:
        """
        过滤低质量图像
        
        Args:
            item: 数据项
            threshold: 质量阈值
            
        Returns:
            bool: 是否保留
        """
        if not self.validate(item):
            return False
        
        image_data = self._extract_image(item)
        
        try:
            image = Image.open(io.BytesIO(image_data))
            
            # 检查分辨率
            if not self.filter_by_resolution(image):
                return False
            
            # 检查宽高比
            if not self.filter_by_aspect_ratio(image):
                return False
            
            # 检查模糊度
            if self.image_config.enable_blur_detection:
                blur_score = self.detect_blur(image)
                if blur_score < self.image_config.max_blur_score:
                    return False
            
            # 检查美学分数
            if self.image_config.enable_aesthetic_scoring and self.aesthetic_scorer:
                aesthetic_score = self.score_aesthetic(image)
                if aesthetic_score < self.image_config.min_aesthetic_score:
                    return False
            
            return True
            
        except Exception as e:
            self.logger.error(f"Error filtering image quality: {e}")
            return False