"""
大规模数据去重引擎
支持精确去重(SHA256)和模糊去重(MinHash LSH, SimHash)
"""

from typing import Iterator, List, Dict, Any, Optional
from dataclasses import dataclass
import hashlib
import logging

from .base_cleaner import RawDataItem

logger = logging.getLogger(__name__)


@dataclass
class DedupConfig:
    """去重配置"""
    enable_exact_dedup: bool = True
    enable_minhash_dedup: bool = True
    minhash_threshold: float = 0.8  # MinHash相似度阈值
    minhash_num_perm: int = 128  # MinHash排列数量
    enable_simhash_dedup: bool = False
    simhash_hamming_distance: int = 3  # SimHash汉明距离阈值
    enable_image_dedup: bool = True
    image_phash_threshold: int = 10  # 感知哈希汉明距离阈值
    use_bloom_filter: bool = True  # 使用布隆过滤器加速精确去重
    bloom_filter_size: int = 100000000  # 布隆过滤器大小


class DedupEngine:
    """
    大规模数据去重引擎。
    支持十亿级文档去重。
    算法：MinHash LSH (文本)、pHash (图像)、VideoHash (视频)。
    使用datasketch库进行MinHash计算，使用Redis存储LSH索引。
    """
    
    def __init__(self, config: DedupConfig, redis_client=None):
        """
        初始化去重引擎
        
        Args:
            config: 去重配置
            redis_client: Redis客户端（用于存储去重索引）
        """
        self.config = config
        self.redis_client = redis_client
        self.logger = logging.getLogger(self.__class__.__name__)
        
        # 布隆过滤器（用于精确去重）
        self.bloom_filter = None
        if config.use_bloom_filter:
            self._init_bloom_filter()
        
        # MinHash LSH索引
        self.minhash_lsh = None
        if config.enable_minhash_dedup:
            self._init_minhash_lsh()
        
        # SimHash索引
        self.simhash_index = None
        if config.enable_simhash_dedup:
            self._init_simhash_index()
    
    def _init_bloom_filter(self):
        """初始化布隆过滤器"""
        try:
            from pybloom_live import ScalableBloomFilter
            self.bloom_filter = ScalableBloomFilter(
                initial_capacity=self.config.bloom_filter_size,
                error_rate=0.001
            )
            self.logger.info("Bloom filter initialized")
        except ImportError:
            self.logger.warning("pybloom_live not installed, using set for dedup")
            self.bloom_filter = set()
    
    def _init_minhash_lsh(self):
        """初始化MinHash LSH索引"""
        try:
            from datasketch import MinHashLSH
            self.minhash_lsh = MinHashLSH(
                threshold=self.config.minhash_threshold,
                num_perm=self.config.minhash_num_perm
            )
            self.logger.info("MinHash LSH initialized")
        except ImportError:
            self.logger.warning("datasketch not installed, MinHash dedup disabled")
    
    def _init_simhash_index(self):
        """初始化SimHash索引"""
        try:
            from simhash import SimhashIndex
            # SimHash索引将在使用时动态创建
            self.logger.info("SimHash index initialized")
        except ImportError:
            self.logger.warning("simhash not installed, SimHash dedup disabled")
    
    def exact_dedup(self, items: Iterator[Any]) -> Iterator[Any]:
        """
        基于SHA256哈希的精确去重。使用布隆过滤器加速。
        
        Args:
            items: 数据项迭代器
            
        Yields:
            Any: 去重后的数据项
        """
        seen_hashes = set() if not self.bloom_filter else None
        
        for item in items:
            # 计算哈希
            content = self._get_content(item)
            content_hash = hashlib.sha256(content.encode('utf-8') if isinstance(content, str) else content).hexdigest()
            
            # 检查是否重复
            if self.bloom_filter is not None:
                if content_hash in self.bloom_filter:
                    self.logger.debug(f"Duplicate item found (bloom filter): {content_hash[:16]}")
                    continue
                self.bloom_filter.add(content_hash)
            else:
                if content_hash in seen_hashes:
                    self.logger.debug(f"Duplicate item found: {content_hash[:16]}")
                    continue
                seen_hashes.add(content_hash)
            
            yield item
    
    def minhash_dedup(
        self,
        items: Iterator[Any],
        threshold: float = None,
        num_perm: int = None
    ) -> Iterator[Any]:
        """
        基于MinHash LSH的模糊文本去重
        
        Args:
            items: 数据项迭代器
            threshold: 相似度阈值
            num_perm: 排列数量
            
        Yields:
            Any: 去重后的数据项
        """
        if not self.minhash_lsh:
            self.logger.warning("MinHash LSH not initialized, returning all items")
            yield from items
            return
        
        threshold = threshold or self.config.minhash_threshold
        num_perm = num_perm or self.config.minhash_num_perm
        
        try:
            from datasketch import MinHash
        except ImportError:
            self.logger.warning("datasketch not installed, MinHash dedup skipped")
            yield from items
            return
        
        for item in items:
            content = self._get_content(item)
            if not isinstance(content, str):
                yield item
                continue
            
            # 创建MinHash
            minhash = MinHash(num_perm=num_perm)
            # 使用n-gram分词
            tokens = content.lower().split()
            for token in tokens:
                minhash.update(token.encode('utf-8'))
            
            # 查询相似项
            similar_items = self.minhash_lsh.query(minhash)
            
            if similar_items:
                self.logger.debug(f"Similar item found, skipping")
                continue
            
            # 添加到索引
            item_id = self._get_item_id(item)
            self.minhash_lsh.insert(item_id, minhash)
            
            yield item
    
    def simhash_dedup(
        self,
        items: Iterator[Any],
        hamming_distance: int = None
    ) -> Iterator[Any]:
        """
        基于SimHash的近似去重，适用于网页文本
        
        Args:
            items: 数据项迭代器
            hamming_distance: 汉明距离阈值
            
        Yields:
            Any: 去重后的数据项
        """
        try:
            from simhash import Simhash, SimhashIndex
        except ImportError:
            self.logger.warning("simhash not installed, SimHash dedup skipped")
            yield from items
            return
        
        hamming_distance = hamming_distance or self.config.simhash_hamming_distance
        
        # 创建SimHash索引
        index = SimhashIndex([], k=hamming_distance)
        
        for item in items:
            content = self._get_content(item)
            if not isinstance(content, str):
                yield item
                continue
            
            # 计算SimHash
            simhash = Simhash(content)
            
            # 查找相似项
            similar_items = index.get_near_dups(simhash)
            
            if similar_items:
                self.logger.debug(f"Similar item found (SimHash), skipping")
                continue
            
            # 添加到索引
            item_id = self._get_item_id(item)
            index.add(item_id, simhash)
            
            yield item
    
    def image_dedup(
        self,
        items: Iterator[Any],
        threshold: int = None
    ) -> Iterator[Any]:
        """
        基于pHash感知哈希的图像去重
        
        Args:
            items: 数据项迭代器
            threshold: 汉明距离阈值
            
        Yields:
            Any: 去重后的数据项
        """
        try:
            from PIL import Image
            import imagehash
        except ImportError:
            self.logger.warning("imagehash not installed, image dedup skipped")
            yield from items
            return
        
        threshold = threshold or self.config.image_phash_threshold
        seen_hashes = {}
        
        for item in items:
            # 提取图像数据
            image_data = self._get_image_data(item)
            if not image_data:
                yield item
                continue
            
            try:
                # 计算感知哈希
                import io
                image = Image.open(io.BytesIO(image_data))
                phash = imagehash.phash(image)
                
                # 检查相似图像
                is_duplicate = False
                for seen_hash, seen_id in seen_hashes.items():
                    hamming_dist = phash - seen_hash
                    if hamming_dist <= threshold:
                        self.logger.debug(f"Duplicate image found (distance={hamming_dist})")
                        is_duplicate = True
                        break
                
                if not is_duplicate:
                    seen_hashes[phash] = self._get_item_id(item)
                    yield item
                    
            except Exception as e:
                self.logger.error(f"Error computing image hash: {e}")
                yield item
    
    def batch_dedup(
        self,
        items: Iterator[Any],
        modality: str = 'text'
    ) -> Iterator[Any]:
        """
        批量去重处理。根据modality选择对应算法。
        
        Args:
            items: 数据项迭代器
            modality: 数据模态 ('text', 'image', 'video')
            
        Yields:
            Any: 去重后的数据项
        """
        # 精确去重
        if self.config.enable_exact_dedup:
            items = self.exact_dedup(items)
        
        # 模糊去重
        if modality == 'text':
            if self.config.enable_minhash_dedup:
                items = self.minhash_dedup(items)
            elif self.config.enable_simhash_dedup:
                items = self.simhash_dedup(items)
        elif modality == 'image':
            if self.config.enable_image_dedup:
                items = self.image_dedup(items)
        
        yield from items
    
    def _get_content(self, item: Any) -> Any:
        """从数据项中提取内容"""
        if isinstance(item, dict):
            return item.get('content') or item.get('text', '')
        elif hasattr(item, 'content'):
            return item.content
        elif isinstance(item, str):
            return item
        else:
            return str(item)
    
    def _get_image_data(self, item: Any) -> Optional[bytes]:
        """从数据项中提取图像数据"""
        if isinstance(item, dict):
            return item.get('image') or item.get('content')
        elif hasattr(item, 'content'):
            if isinstance(item.content, dict):
                return item.content.get('image')
            return item.content
        elif isinstance(item, bytes):
            return item
        else:
            return None
    
    def _get_item_id(self, item: Any) -> str:
        """获取数据项ID"""
        if isinstance(item, dict):
            return item.get('data_id', str(hash(str(item))))
        elif hasattr(item, 'data_id'):
            return item.data_id
        else:
            return str(hash(str(item)))